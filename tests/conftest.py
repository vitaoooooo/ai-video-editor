"""Shared test fixtures and synthetic video generator."""

import subprocess
from pathlib import Path
from typing import Tuple
import pytest

from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path


def create_synthetic_video(
    output_path: Path,
    duration_sec: float = 3.0,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    audio_freq: int = 440,
) -> Path:
    """Generates a synthetic MP4 video using ffmpeg test sources (no external media needed)."""
    ffmpeg_bin = get_ffmpeg_path()
    args = [
        str(ffmpeg_bin),
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size={width}x{height}:rate={fps}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={audio_freq}:duration={duration_sec}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        str(output_path.resolve()),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return output_path


@pytest.fixture
def synthetic_video_pair(tmp_path: Path) -> Tuple[Path, Path]:
    """Provides two distinct synthetic test videos with audio."""
    v1 = tmp_path / "synthetic_clip_1.mp4"
    v2 = tmp_path / "synthetic_clip_2.mp4"
    create_synthetic_video(v1, duration_sec=3.0, audio_freq=440)
    create_synthetic_video(v2, duration_sec=3.0, audio_freq=880)
    return v1, v2
