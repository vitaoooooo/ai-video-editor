"""Executa o render de uma Timeline e valida a saída com ffprobe."""

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from ai_video_editor.core.timeline import Timeline
from ai_video_editor.renderer.executor import RenderExecutor, RenderProgress
from ai_video_editor.renderer.ffmpeg_locator import get_ffprobe_path
from ai_video_editor.renderer.loudness import (
    LoudnessResult,
    apply_filter,
    measure_file_command,
    parse_loudnorm_output,
)
from ai_video_editor.renderer.timeline_compiler import TimelineCompiler, TimelineRenderJob


@dataclass(frozen=True)
class TimelineRenderResult:
    """Resultado de um render de Timeline já validado."""

    output_path: Path
    duration_sec: float  # duração da trilha de vídeo medida
    frame_count: int
    resolution: Tuple[int, int]
    fps: float
    size_bytes: int
    command_executed: list[str]
    loudness: Optional[LoudnessResult] = None
    warnings: list[str] = field(default_factory=list)


class TimelineRenderExecutor:
    """Renderiza os pedaços em paralelo, monta o áudio, faz o mux e confere a duração."""

    def __init__(
        self,
        compiler: Optional[TimelineCompiler] = None,
        ffprobe_bin: Optional[Path] = None,
        max_workers: Optional[int] = None,
        temp_root: Optional[Path] = None,
    ):
        self.compiler = compiler or TimelineCompiler()
        self.ffprobe_bin = ffprobe_bin or get_ffprobe_path()
        self.max_workers = max_workers or min(8, os.cpu_count() or 2)
        self.temp_root = temp_root

    def probe_video(self, output_file: Path) -> tuple[float, int, Tuple[int, int], float]:
        """Duração, número de frames, resolução e fps da trilha de vídeo da saída."""
        args = [
            str(self.ffprobe_bin),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(output_file.resolve()),
        ]
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        stream = (json.loads(res.stdout).get("streams") or [{}])[0]
        num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        return (
            float(stream.get("duration", 0.0)),
            int(stream.get("nb_frames", 0)),
            (int(stream.get("width", 0)), int(stream.get("height", 0))),
            fps,
        )

    def _normalize_loudness(
        self, timeline: Timeline, job: TimelineRenderJob
    ) -> tuple[Optional[str], Optional[LoudnessResult]]:
        """Passada 1 do loudnorm sobre o áudio final; devolve o filtro da passada 2 (ou None)."""
        target = timeline.loudness
        if target is None:
            return None, None
        if not job.has_audio:
            return None, LoudnessResult(target=target, status="skipped", note="vídeo sem áudio")

        started = time.perf_counter()
        stderr = RenderExecutor._run_ffmpeg(measure_file_command(self.compiler.ffmpeg_bin, job.audio_path, target))
        measure_sec = round(time.perf_counter() - started, 3)
        try:
            measured = parse_loudnorm_output(stderr)
        except ValueError as exc:
            return None, LoudnessResult(target=target, status="skipped", note=str(exc), measure_sec=measure_sec)
        if measured.is_silent:
            return None, LoudnessResult(
                target=target,
                status="skipped",
                note=f"áudio sem conteúdo mensurável (loudness medido: {measured.input_i} LUFS)",
                measure_sec=measure_sec,
            )
        return apply_filter(target, measured), LoudnessResult(
            target=target, status="applied", measured=measured, measure_sec=measure_sec
        )

    def execute(self, timeline: Timeline, on_progress: Optional[RenderProgress] = None) -> TimelineRenderResult:
        """Renderiza a Timeline em `timeline.output_path`.

        Pedaços de vídeo (sem áudio) rodam em paralelo (até `max_workers`), junto com o único
        processo de áudio (principal recortado e concatenado + ambiente mixado); depois o áudio é
        medido pelo loudnorm (se pedido) e unido ao vídeo concatenado num mux final. Temporários
        apagados mesmo com erro.

        Raises:
            RuntimeError: falha do FFmpeg ou saída com duração/fps diferentes do esperado.
        """
        timeline.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.temp_root is not None:
            self.temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ave_timeline_", dir=self.temp_root) as tmp:
            job = self.compiler.compile(timeline, Path(tmp))
            job.audio_graph_path.write_text(job.audio_graph, encoding="utf-8")

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                # O áudio vai primeiro: é o processo mais longo e não depende dos pedaços de vídeo.
                commands = [job.audio_command, *job.piece_commands]
                futures = [pool.submit(RenderExecutor._run_ffmpeg, cmd) for cmd in commands]
                total = len(futures)
                if on_progress is not None:
                    on_progress(0, total)
                try:
                    for done, future in enumerate(as_completed(futures), start=1):
                        future.result()
                        if on_progress is not None:
                            on_progress(done, total)
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise

            job.video_list_path.write_text(job.video_list_content, encoding="utf-8")
            audio_filter, loudness = self._normalize_loudness(timeline, job)
            mux = self.compiler.mux_command(job, timeline.output_path, audio_filter)
            RenderExecutor._run_ffmpeg(mux)

        output = timeline.output_path
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"O render não gerou o arquivo de saída: {output}")

        duration, frames, resolution, fps = self.probe_video(output)
        frame_sec = 1 / timeline.fps
        if abs(duration - job.expected_duration_sec) > frame_sec + 1e-6:
            raise RuntimeError(
                f"Duração do vídeo renderizado ({duration:.3f}s) difere da timeline "
                f"({job.expected_duration_sec:.3f}s) em mais de 1 frame"
            )
        if abs(fps - timeline.fps) > 0.01:
            raise RuntimeError(f"FPS do vídeo renderizado ({fps:.3f}) difere do pedido ({timeline.fps})")

        return TimelineRenderResult(
            output_path=output.resolve(),
            duration_sec=duration,
            frame_count=frames,
            resolution=resolution,
            fps=fps,
            size_bytes=output.stat().st_size,
            command_executed=mux,
            loudness=loudness,
            warnings=job.warnings + (loudness.warnings if loudness else []),
        )
