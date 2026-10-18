"""Compiles an EditPlan into deterministic, safe FFmpeg command-line arguments."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path

AUDIO_SAMPLE_RATE = 44100

# Parâmetros de vídeo idênticos em todos os segmentos: requisito para o concat com -c copy.
VIDEO_ENCODE_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p"]
FINAL_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_SAMPLE_RATE)]

# Folga de leitura após o fim do segmento; o corte exato é feito pelos filtros trim/atrim.
INPUT_READ_MARGIN_SEC = 0.25


@dataclass(frozen=True)
class SourceInfo:
    """Metadados de um arquivo de origem obtidos com uma única chamada ao ffprobe."""

    duration_sec: float
    has_audio: bool


def probe_source(video_path: Path, ffprobe_bin: Optional[Path] = None) -> SourceInfo:
    """Inspeciona duração e presença de áudio de um arquivo em uma só chamada ao ffprobe."""
    probe_cmd = ffprobe_bin or get_ffprobe_path()
    args = [
        str(probe_cmd),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(video_path),
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Falha ao inspecionar {video_path} com ffprobe:\n{res.stderr.strip()}")

    data = json.loads(res.stdout or "{}")
    try:
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return SourceInfo(duration_sec=duration, has_audio=has_audio)


@dataclass(frozen=True)
class SegmentTiming:
    """Posição de um segmento na origem e seu tamanho quantizado em frames e amostras."""

    start_sec: float
    end_sec: float
    frame_count: int
    sample_count: int
    duration_sec: float


@dataclass(frozen=True)
class SegmentedRender:
    """Comandos de um render segmentado: um ffmpeg por segmento e um concat final."""

    segment_commands: list[list[str]]
    segment_paths: list[Path]
    concat_list_path: Path
    concat_list_content: str
    concat_command: list[str]
    expected_duration_sec: float


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _video_filters(width: int, height: int, fps: int, frame_count: int) -> str:
    """Normaliza o vídeo e fixa a contagem exata de frames (clonando o último se faltar)."""
    return (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,"
        f"tpad=stop_mode=clone:stop=-1,"
        f"trim=end_frame={frame_count},setpts=PTS-STARTPTS"
    )


def _audio_filters(seg: ClipSegment, timing: SegmentTiming, fade_sec: float) -> str:
    """Normaliza o áudio, fixa a contagem exata de amostras e aplica os micro fades."""
    chain = (
        f"aresample={AUDIO_SAMPLE_RATE}:first_pts=0,"
        f"volume={seg.volume},"
        f"aformat=sample_fmts=s16:channel_layouts=stereo,"
        f"apad,atrim=end_sample={timing.sample_count},asetpts=PTS-STARTPTS"
    )
    fade = min(fade_sec, timing.sample_count / AUDIO_SAMPLE_RATE / 2)
    if fade > 0:
        fade_out_start = timing.sample_count / AUDIO_SAMPLE_RATE - fade
        chain += (
            f",afade=t=in:st=0:d={_fmt(fade)}"
            f",afade=t=out:st={_fmt(fade_out_start)}:d={_fmt(fade)}"
        )
    return chain


def _silence_source(timing: SegmentTiming) -> str:
    return (
        f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_SAMPLE_RATE},"
        f"aformat=sample_fmts=s16:channel_layouts=stereo,"
        f"atrim=end_sample={timing.sample_count}"
    )


class FFmpegCompiler:
    """Compiles an EditPlan into an argument array for subprocess execution."""

    def __init__(
        self,
        ffmpeg_bin: Optional[Path] = None,
        ffprobe_bin: Optional[Path] = None,
    ):
        self.ffmpeg_bin = ffmpeg_bin or get_ffmpeg_path()
        self.ffprobe_bin = ffprobe_bin or get_ffprobe_path()

    def probe_sources(self, plan: EditPlan) -> dict[Path, SourceInfo]:
        """Inspeciona cada arquivo de origem distinto do plano exatamente uma vez."""
        infos: dict[Path, SourceInfo] = {}
        for seg in plan.segments:
            if not seg.source_path.exists():
                raise FileNotFoundError(f"Source file not found: {seg.source_path}")
            key = seg.source_path.resolve()
            if key not in infos:
                infos[key] = probe_source(key, self.ffprobe_bin)
        return infos

    def plan_timeline(
        self, plan: EditPlan, infos: dict[Path, SourceInfo]
    ) -> list[SegmentTiming]:
        """Quantiza cada segmento em frames e amostras sem acumular desvio de A/V.

        As amostras são derivadas da posição acumulada em frames, então áudio e vídeo
        terminam juntos no fim de cada segmento, independentemente do número de cortes.
        """
        fps = plan.target_fps
        timings: list[SegmentTiming] = []
        total_frames = 0

        for idx, seg in enumerate(plan.segments):
            info = infos[seg.source_path.resolve()]
            end = seg.end_sec if seg.end_sec is not None else info.duration_sec
            if info.duration_sec > 0:
                end = min(end, info.duration_sec)
            if end <= seg.start_sec:
                raise ValueError(
                    f"Segmento {idx} começa em {seg.start_sec}s, depois do fim da origem "
                    f"({info.duration_sec:.3f}s): {seg.source_path}"
                )

            frame_count = max(1, round((end - seg.start_sec) * fps))
            first_sample = round(total_frames * AUDIO_SAMPLE_RATE / fps)
            total_frames += frame_count
            last_sample = round(total_frames * AUDIO_SAMPLE_RATE / fps)

            timings.append(
                SegmentTiming(
                    start_sec=seg.start_sec,
                    end_sec=end,
                    frame_count=frame_count,
                    sample_count=last_sample - first_sample,
                    duration_sec=frame_count / fps,
                )
            )
        return timings

    def compile_segmented(self, plan: EditPlan, work_dir: Path) -> SegmentedRender:
        """Gera um comando curto por segmento e um concat final (demuxer, -c:v copy).

        Cada segmento usa seek rápido (-ss antes do -i), então só decodifica a partir do
        keyframe anterior ao corte. O áudio intermediário é PCM, sem priming de encoder,
        e é codificado em AAC uma única vez no concat final.
        """
        width, height = plan.target_resolution
        fps = plan.target_fps
        infos = self.probe_sources(plan)
        timings = self.plan_timeline(plan, infos)

        segment_commands: list[list[str]] = []
        segment_paths: list[Path] = []
        list_lines = ["ffconcat version 1.0"]

        for idx, (seg, timing) in enumerate(zip(plan.segments, timings)):
            source = seg.source_path.resolve()
            seg_path = work_dir / f"seg_{idx:05d}.mov"
            read_duration = timing.end_sec - timing.start_sec + INPUT_READ_MARGIN_SEC

            v_chain = f"[0:v]{_video_filters(width, height, fps, timing.frame_count)}[v]"
            if infos[source].has_audio:
                a_chain = f"[0:a]{_audio_filters(seg, timing, plan.audio_fade_sec)}[a]"
            else:
                a_chain = f"{_silence_source(timing)}[a]"

            segment_commands.append([
                str(self.ffmpeg_bin),
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                _fmt(timing.start_sec),
                "-t",
                _fmt(read_duration),
                "-i",
                str(source),
                "-filter_complex",
                f"{v_chain};{a_chain}",
                "-map",
                "[v]",
                "-map",
                "[a]",
                *VIDEO_ENCODE_ARGS,
                "-r",
                str(fps),
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-ac",
                "2",
                str(seg_path),
            ])
            segment_paths.append(seg_path)
            list_lines.append(f"file '{seg_path.name}'")
            list_lines.append(f"duration {_fmt(timing.duration_sec)}")

        concat_list_path = work_dir / "concat.txt"
        concat_command = [
            str(self.ffmpeg_bin),
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            *FINAL_AUDIO_ARGS,
            "-movflags",
            "+faststart",
            str(plan.output_path.resolve()),
        ]

        return SegmentedRender(
            segment_commands=segment_commands,
            segment_paths=segment_paths,
            concat_list_path=concat_list_path,
            concat_list_content="\n".join(list_lines) + "\n",
            concat_command=concat_command,
            expected_duration_sec=sum(t.duration_sec for t in timings),
        )

    def compile(self, plan: EditPlan) -> list[str]:
        """Gera um único comando com filtergraph completo (render em uma passada).

        Adequado apenas para planos pequenos: o filtergraph cresce com o número de
        segmentos. O RenderExecutor usa `compile_segmented`.
        """
        width, height = plan.target_resolution
        fps = plan.target_fps
        infos = self.probe_sources(plan)
        timings = self.plan_timeline(plan, infos)

        args: list[str] = [str(self.ffmpeg_bin), "-y"]

        # Um -i por arquivo de origem distinto
        input_index: dict[Path, int] = {}
        for seg in plan.segments:
            key = seg.source_path.resolve()
            if key not in input_index:
                input_index[key] = len(input_index)
                args.extend(["-i", str(key)])

        filter_chains: list[str] = []
        concat_inputs: list[str] = []

        for idx, (seg, timing) in enumerate(zip(plan.segments, timings)):
            key = seg.source_path.resolve()
            src = input_index[key]
            trim = f"start={seg.start_sec}:end={timing.end_sec}"

            filter_chains.append(
                f"[{src}:v]trim={trim},setpts=PTS-STARTPTS,"
                f"{_video_filters(width, height, fps, timing.frame_count)}[v{idx}]"
            )
            if infos[key].has_audio:
                filter_chains.append(
                    f"[{src}:a]atrim={trim},asetpts=PTS-STARTPTS,"
                    f"{_audio_filters(seg, timing, plan.audio_fade_sec)}[a{idx}]"
                )
            else:
                filter_chains.append(f"{_silence_source(timing)}[a{idx}]")
            concat_inputs.append(f"[v{idx}][a{idx}]")

        filter_chains.append(
            f"{''.join(concat_inputs)}concat=n={len(plan.segments)}:v=1:a=1[outv][outa]"
        )

        args.extend([
            "-filter_complex",
            ";".join(filter_chains),
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            *VIDEO_ENCODE_ARGS,
            "-r",
            str(fps),
            *FINAL_AUDIO_ARGS,
            str(plan.output_path.resolve()),
        ])

        return args
