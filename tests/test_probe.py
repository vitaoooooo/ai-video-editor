"""Unit and integration tests for VideoProbe."""

from pathlib import Path
import pytest

from ai_video_editor.analyzer.exceptions import (
    InvalidMediaFormatError,
    NoVideoStreamError,
    VideoNotFoundError,
)
from ai_video_editor.analyzer.probe import VideoProbe, parse_fractional_fps


def test_parse_fractional_fps():
    """Validates safe parsing of various fps representations without eval()."""
    assert parse_fractional_fps("30/1") == 30.0
    assert parse_fractional_fps("25/1") == 25.0
    assert parse_fractional_fps("30000/1001") == pytest.approx(29.97003, rel=1e-4)
    assert parse_fractional_fps("24000/1001") == pytest.approx(23.97602, rel=1e-4)
    assert parse_fractional_fps("60") == 60.0
    assert parse_fractional_fps("29.97") == 29.97

    # Edge cases & invalid inputs
    assert parse_fractional_fps("0/0") == 0.0
    assert parse_fractional_fps("") == 0.0
    assert parse_fractional_fps(None) == 0.0
    assert parse_fractional_fps("not_a_number") == 0.0
    assert parse_fractional_fps("0/1") == 0.0


def test_probe_valid_video_with_audio(synthetic_video_with_audio: Path):
    """Probes valid synthetic MP4 containing both video and audio streams."""
    probe = VideoProbe()
    meta = probe.probe(synthetic_video_with_audio)

    assert meta.source_path == synthetic_video_with_audio.resolve()
    assert meta.duration_sec == pytest.approx(3.0, abs=0.2)
    assert meta.size_bytes > 0

    # Video stream
    assert meta.video_stream.codec_name in ("h264", "avc1")
    assert meta.video_stream.width == 640
    assert meta.video_stream.height == 360
    assert meta.video_stream.fps == 30.0

    # Audio stream
    assert meta.audio_stream is not None
    assert meta.audio_stream.codec_name == "aac"
    assert meta.audio_stream.sample_rate == 44100


def test_probe_valid_video_without_audio(synthetic_video_without_audio: Path):
    """Probes valid synthetic video without audio; audio_stream must be None."""
    probe = VideoProbe()
    meta = probe.probe(synthetic_video_without_audio)

    assert meta.video_stream.width == 640
    assert meta.video_stream.height == 360
    assert meta.audio_stream is None


def test_probe_audio_only_file_raises_no_video_stream(synthetic_audio_only_file: Path):
    """Media file without video stream raises NoVideoStreamError."""
    probe = VideoProbe()
    with pytest.raises(NoVideoStreamError):
        probe.probe(synthetic_audio_only_file)


def test_probe_corrupted_file_raises_invalid_format(corrupted_file: Path):
    """Corrupted garbage file raises InvalidMediaFormatError."""
    probe = VideoProbe()
    with pytest.raises(InvalidMediaFormatError):
        probe.probe(corrupted_file)


def test_probe_non_existent_file_raises_not_found(tmp_path: Path):
    """Missing file raises VideoNotFoundError."""
    probe = VideoProbe()
    with pytest.raises(VideoNotFoundError):
        probe.probe(tmp_path / "does_not_exist.mp4")
