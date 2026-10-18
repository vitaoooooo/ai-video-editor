"""Executes compiled FFmpeg commands and validates output results."""

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ai_video_editor.core.schema import EditPlan
from ai_video_editor.renderer.compiler import FFmpegCompiler
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path


@dataclass(frozen=True)
class RenderResult:
    """Outcome and metadata of a completed render."""

    output_path: Path
    duration_sec: float
    resolution: Tuple[int, int]
    size_bytes: int
    command_executed: list[str]


class RenderExecutor:
    """Coordinates compiling and running FFmpeg render jobs."""

    def __init__(
        self,
        compiler: Optional[FFmpegCompiler] = None,
        ffprobe_bin: Optional[Path] = None,
        max_workers: Optional[int] = None,
        temp_root: Optional[Path] = None,
    ):
        self.compiler = compiler or FFmpegCompiler()
        self.ffprobe_bin = ffprobe_bin or get_ffprobe_path()
        # Limita quantos ffmpeg rodam ao mesmo tempo (memória independe do total de segmentos)
        self.max_workers = max_workers or min(8, os.cpu_count() or 2)
        self.temp_root = temp_root

    def probe_output(self, output_file: Path) -> Tuple[float, Tuple[int, int]]:
        """Extracts verified duration and resolution from rendered video."""
        args = [
            str(self.ffprobe_bin),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height",
            "-of",
            "json",
            str(output_file.resolve()),
        ]
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        duration = float(data.get("format", {}).get("duration", 0.0))
        streams = data.get("streams", [])
        width, height = 0, 0
        for s in streams:
            if "width" in s and "height" in s:
                width = int(s["width"])
                height = int(s["height"])
                break

        return duration, (width, height)

    @staticmethod
    def _run_ffmpeg(cmd_args: list[str]) -> None:
        proc = subprocess.run(cmd_args, capture_output=True, text=True)
        if proc.returncode != 0:
            error_msg = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(
                f"FFmpeg execution failed with code {proc.returncode}:\n{error_msg}"
            )

    def execute(self, plan: EditPlan) -> RenderResult:
        """Renders the edit plan to disk and validates output integrity.

        Cada segmento vira um arquivo intermediário (em paralelo, até `max_workers`),
        depois os intermediários são unidos pelo concat demuxer. Os temporários são
        apagados ao final, mesmo em caso de erro.
        """
        # Ensure target directory exists
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.temp_root is not None:
            self.temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ave_render_", dir=self.temp_root) as tmp:
            job = self.compiler.compile_segmented(plan, Path(tmp))

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(self._run_ffmpeg, cmd) for cmd in job.segment_commands]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise

            job.concat_list_path.write_text(job.concat_list_content, encoding="utf-8")
            self._run_ffmpeg(job.concat_command)

        # Validate output file
        if not plan.output_path.is_file():
            raise FileNotFoundError(
                f"Expected render output file does not exist: {plan.output_path}"
            )

        file_size = plan.output_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Render output file is 0 bytes: {plan.output_path}")

        # Probe output video properties
        duration, resolution = self.probe_output(plan.output_path)

        return RenderResult(
            output_path=plan.output_path.resolve(),
            duration_sec=duration,
            resolution=resolution,
            size_bytes=file_size,
            command_executed=job.concat_command,
        )
