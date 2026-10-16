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
    """Generates a synthetic MP4 video with audio using ffmpeg lavfi sources."""
    ffmpeg_bin = get_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def create_synthetic_video_no_audio(
    output_path: Path,
    duration_sec: float = 3.0,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
) -> Path:
    """Generates a synthetic MP4 video without any audio stream."""
    ffmpeg_bin = get_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(ffmpeg_bin),
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size={width}x{height}:rate={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path.resolve()),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return output_path


def create_synthetic_audio_only(
    output_path: Path,
    duration_sec: float = 3.0,
    audio_freq: int = 440,
) -> Path:
    """Generates an audio-only file (no video stream)."""
    ffmpeg_bin = get_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(ffmpeg_bin),
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={audio_freq}:duration={duration_sec}",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        str(output_path.resolve()),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return output_path


def create_corrupted_media_file(output_path: Path) -> Path:
    """Creates a corrupted file containing non-media garbage bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"CORRUPTED_NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_1234567890\x00\xff\xfe")
    return output_path


@pytest.fixture
def synthetic_video_pair(tmp_path: Path) -> Tuple[Path, Path]:
    """Provides two distinct synthetic test videos with audio."""
    v1 = tmp_path / "synthetic_clip_1.mp4"
    v2 = tmp_path / "synthetic_clip_2.mp4"
    create_synthetic_video(v1, duration_sec=3.0, audio_freq=440)
    create_synthetic_video(v2, duration_sec=3.0, audio_freq=880)
    return v1, v2


@pytest.fixture
def synthetic_video_with_audio(tmp_path: Path) -> Path:
    return create_synthetic_video(tmp_path / "clip_with_audio.mp4", duration_sec=3.0)


@pytest.fixture
def synthetic_video_without_audio(tmp_path: Path) -> Path:
    return create_synthetic_video_no_audio(tmp_path / "clip_no_audio.mp4", duration_sec=3.0)


@pytest.fixture
def synthetic_audio_only_file(tmp_path: Path) -> Path:
    return create_synthetic_audio_only(tmp_path / "audio_only.m4a", duration_sec=3.0)


@pytest.fixture
def corrupted_file(tmp_path: Path) -> Path:
    return create_corrupted_media_file(tmp_path / "corrupted_media.mp4")


@pytest.fixture
def short_video(tmp_path: Path) -> Path:
    """1.2 second video for edge case testing (interval does not divide evenly)."""
    return create_synthetic_video(tmp_path / "short_1_2s.mp4", duration_sec=1.2)


@pytest.fixture
def long_video(tmp_path: Path) -> Path:
    """8.5 second video for multi-frame testing."""
    return create_synthetic_video(tmp_path / "long_8_5s.mp4", duration_sec=8.5)
