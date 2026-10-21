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


# Fixtures de sessão: vídeos sintéticos são gerados uma única vez (por worker do xdist)
# e usados somente para leitura; as saídas dos testes vão sempre para tmp_path.
@pytest.fixture(scope="session")
def synthetic_video_pair(tmp_path_factory) -> Tuple[Path, Path]:
    """Provides two distinct synthetic test videos with audio."""
    root = tmp_path_factory.mktemp("synthetic_pair")
    v1 = root / "synthetic_clip_1.mp4"
    v2 = root / "synthetic_clip_2.mp4"
    create_synthetic_video(v1, duration_sec=3.0, audio_freq=440)
    create_synthetic_video(v2, duration_sec=3.0, audio_freq=880)
    return v1, v2


@pytest.fixture(scope="session")
def synthetic_video_with_audio(tmp_path_factory) -> Path:
    return create_synthetic_video(tmp_path_factory.mktemp("with_audio") / "clip_with_audio.mp4", duration_sec=3.0)


@pytest.fixture(scope="session")
def synthetic_video_without_audio(tmp_path_factory) -> Path:
    return create_synthetic_video_no_audio(tmp_path_factory.mktemp("no_audio") / "clip_no_audio.mp4", duration_sec=3.0)


@pytest.fixture(scope="session")
def synthetic_audio_only_file(tmp_path_factory) -> Path:
    return create_synthetic_audio_only(tmp_path_factory.mktemp("audio_only") / "audio_only.m4a", duration_sec=3.0)


@pytest.fixture(scope="session")
def corrupted_file(tmp_path_factory) -> Path:
    return create_corrupted_media_file(tmp_path_factory.mktemp("corrupted") / "corrupted_media.mp4")


@pytest.fixture(scope="session")
def short_video(tmp_path_factory) -> Path:
    """1.2 second video for edge case testing (interval does not divide evenly)."""
    return create_synthetic_video(tmp_path_factory.mktemp("short") / "short_1_2s.mp4", duration_sec=1.2)


@pytest.fixture(scope="session")
def long_video(tmp_path_factory) -> Path:
    """8.5 second video for multi-frame testing."""
    return create_synthetic_video(tmp_path_factory.mktemp("long") / "long_8_5s.mp4", duration_sec=8.5)


def create_tone_silence_video(
    output_path: Path,
    pattern: list[Tuple[str, float]],
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> Path:
    """Gera MP4 com vídeo testsrc e áudio concatenando trechos ('tone' | 'silence', duração)."""
    ffmpeg_bin = get_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(dur for _, dur in pattern)
    inputs = ["-f", "lavfi", "-i", f"testsrc=duration={total}:size={width}x{height}:rate={fps}"]
    for kind, dur in pattern:
        src = (
            f"sine=frequency=440:sample_rate=44100:duration={dur}"
            if kind == "tone"
            else f"anullsrc=r=44100:cl=mono:d={dur}"
        )
        inputs += ["-f", "lavfi", "-i", src]
    n = len(pattern)
    labels = "".join(f"[{i + 1}:a]" for i in range(n))
    args = [
        str(ffmpeg_bin),
        "-y",
        *inputs,
        "-filter_complex",
        f"{labels}concat=n={n}:v=0:a=1[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
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


def create_rotated_video(output_path: Path, rotation: int = 90, **kwargs) -> Path:
    """Gera um vídeo com áudio e grava metadado de rotação (display matrix) sem recodificar."""
    raw = output_path.with_name(output_path.stem + "_raw" + output_path.suffix)
    create_synthetic_video(raw, **kwargs)
    args = [
        str(get_ffmpeg_path()),
        "-y",
        "-display_rotation:v:0",
        str(rotation),
        "-i",
        str(raw.resolve()),
        "-c",
        "copy",
        str(output_path.resolve()),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return output_path


@pytest.fixture(scope="session")
def mixed_sources(tmp_path_factory) -> dict[str, Path]:
    """Clipes de formatos misturados para a Timeline (somente leitura).

    main: 640x360@30 com áudio (440 Hz); small: 320x240@25 com áudio (880 Hz);
    vertical: 360x640@60 exibido (640x360 codificado + rotação de 90°), com áudio (660 Hz);
    mute: 640x360@30 sem áudio. Todos com 3 s.
    """
    root = tmp_path_factory.mktemp("mixed_sources")
    return {
        "main": create_synthetic_video(root / "main.mp4", duration_sec=3.0, audio_freq=440),
        "small": create_synthetic_video(
            root / "small.mp4", duration_sec=3.0, width=320, height=240, fps=25, audio_freq=880
        ),
        "vertical": create_rotated_video(
            root / "vertical.mp4", duration_sec=3.0, width=640, height=360, fps=60, audio_freq=660
        ),
        "mute": create_synthetic_video_no_audio(root / "mute.mp4", duration_sec=3.0),
    }
