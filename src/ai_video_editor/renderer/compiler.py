"""Compiles an EditPlan into deterministic, safe FFmpeg command-line arguments."""

import json
import subprocess
from pathlib import Path
from typing import Optional

from ai_video_editor.core.schema import EditPlan
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path


def has_audio_stream(video_path: Path, ffprobe_bin: Optional[Path] = None) -> bool:
    """Checks whether the video file contains at least one audio stream."""
    probe_cmd = ffprobe_bin or get_ffprobe_path()
    args = [
        str(probe_cmd),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return "audio" in res.stdout.strip().lower()
    except Exception:
        return False


def get_video_duration(video_path: Path, ffprobe_bin: Optional[Path] = None) -> float:
    """Returns video duration in seconds via ffprobe."""
    probe_cmd = ffprobe_bin or get_ffprobe_path()
    args = [
        str(probe_cmd),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


class FFmpegCompiler:
    """Compiles an EditPlan into an argument array for subprocess execution."""

    def __init__(
        self,
        ffmpeg_bin: Optional[Path] = None,
        ffprobe_bin: Optional[Path] = None,
    ):
        self.ffmpeg_bin = ffmpeg_bin or get_ffmpeg_path()
        self.ffprobe_bin = ffprobe_bin or get_ffprobe_path()

    def compile(self, plan: EditPlan) -> list[str]:
        """Generates list of command arguments representing the edit plan."""
        width, height = plan.target_resolution
        fps = plan.target_fps
        num_segments = len(plan.segments)

        args: list[str] = [str(self.ffmpeg_bin), "-y"]

        # Add inputs
        for seg in plan.segments:
            if not seg.source_path.exists():
                raise FileNotFoundError(f"Source file not found: {seg.source_path}")
            args.extend(["-i", str(seg.source_path.resolve())])

        # Build filtergraph
        filter_chains: list[str] = []
        concat_inputs: list[str] = []

        for idx, seg in enumerate(plan.segments):
            # 1. Video filter chain
            trim_v = f"start={seg.start_sec}"
            if seg.end_sec is not None:
                trim_v += f":end={seg.end_sec}"

            v_chain = (
                f"[{idx}:v]trim={trim_v},setpts=PTS-STARTPTS,"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,fps={fps}[v{idx}]"
            )
            filter_chains.append(v_chain)

            # 2. Audio filter chain
            has_audio = has_audio_stream(seg.source_path, self.ffprobe_bin)
            if has_audio:
                trim_a = f"start={seg.start_sec}"
                if seg.end_sec is not None:
                    trim_a += f":end={seg.end_sec}"
                a_chain = (
                    f"[{idx}:a]atrim={trim_a},asetpts=PTS-STARTPTS,"
                    f"volume={seg.volume}[a{idx}]"
                )
            else:
                # Synthesize silence for the segment duration
                seg_duration = (
                    seg.end_sec - seg.start_sec
                    if seg.end_sec is not None
                    else max(0.1, get_video_duration(seg.source_path, self.ffprobe_bin) - seg.start_sec)
                )
                a_chain = f"anullsrc=channel_layout=stereo:sample_rate=44100:d={seg_duration:.3f}[a{idx}]"

            filter_chains.append(a_chain)
            concat_inputs.append(f"[v{idx}][a{idx}]")

        # 3. Concatenate all segments
        concat_str = f"{''.join(concat_inputs)}concat=n={num_segments}:v=1:a=1[outv][outa]"
        filter_chains.append(concat_str)

        full_filtergraph = ";".join(filter_chains)

        args.extend([
            "-filter_complex",
            full_filtergraph,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            str(plan.output_path.resolve()),
        ])

        return args
