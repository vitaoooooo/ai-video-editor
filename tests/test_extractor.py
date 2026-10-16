"""Unit and integration tests for FrameExtractor and timestamp calculations."""

from pathlib import Path
import pytest

from ai_video_editor.analyzer.exceptions import FrameExtractionError
from ai_video_editor.analyzer.extractor import FrameExtractor, calculate_target_timestamps
from ai_video_editor.analyzer.probe import VideoProbe
from ai_video_editor.analyzer.schema import FrameExtractionConfig


def test_calculate_target_timestamps_interval_uneven():
    """Interval that does not divide duration evenly stops at last valid step."""
    # 5.2s duration with 2.0s interval -> 0.0, 2.0, 4.0 (6.0 exceeds duration)
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=2.0)
    ts = calculate_target_timestamps(5.2, cfg)
    assert ts == [0.0, 2.0, 4.0]


def test_calculate_target_timestamps_short_video():
    """Short video (1.2s) with 0.5s interval produces 3 timestamps."""
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=0.5)
    ts = calculate_target_timestamps(1.2, cfg)
    assert ts == [0.0, 0.5, 1.0]


def test_calculate_target_timestamps_very_short_sub_interval():
    """Video shorter than interval still yields the initial 0.0s frame."""
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=2.0)
    ts = calculate_target_timestamps(0.8, cfg)
    assert ts == [0.0]


def test_calculate_target_timestamps_max_frames_cap():
    """Sampling is strictly bounded by max_frames ceiling."""
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=1.0, max_frames=5)
    ts = calculate_target_timestamps(100.0, cfg)
    assert len(ts) == 5
    assert ts == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_calculate_target_timestamps_fixed_count():
    """Fixed count distributes timestamps across video duration."""
    cfg = FrameExtractionConfig(strategy="fixed_count", target_count=5)
    ts = calculate_target_timestamps(10.0, cfg)
    assert len(ts) == 5
    assert ts[0] == 0.0
    # Last timestamp should be slightly before end of video
    assert ts[-1] <= 10.0


def test_extract_frames_short_video_integration(short_video: Path, tmp_path: Path):
    """Extracts frames from 1.2s video with 0.5s interval; verifies files on disk."""
    probe = VideoProbe()
    meta = probe.probe(short_video)

    out_dir = tmp_path / "frames_short"
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=0.5, image_format="jpg")

    extractor = FrameExtractor()
    planned_ts, frames = extractor.extract_frames(meta, out_dir, cfg)

    assert planned_ts == [0.0, 0.5, 1.0]
    assert len(frames) == 3

    for f in frames:
        assert f.image_path.is_file()
        assert f.size_bytes > 0
        assert f.width == 640
        assert f.height == 360
        assert f.target_timestamp_sec in (0.0, 0.5, 1.0)


def test_extract_frames_long_video_integration(long_video: Path, tmp_path: Path):
    """Extracts frames from 8.5s video with 3.0s interval (uneven division: 0, 3, 6)."""
    probe = VideoProbe()
    meta = probe.probe(long_video)

    out_dir = tmp_path / "frames_long"
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=3.0, image_format="jpg")

    extractor = FrameExtractor()
    planned_ts, frames = extractor.extract_frames(meta, out_dir, cfg)

    # 9.0s exceeds 8.5s duration, so exactly 3 frames (0.0, 3.0, 6.0)
    assert planned_ts == [0.0, 3.0, 6.0]
    assert len(frames) == 3
    for f in frames:
        assert f.image_path.is_file()
        assert f.size_bytes > 500


def test_extract_frames_fixed_count_png(short_video: Path, tmp_path: Path):
    """Extracts exactly target_count frames in PNG format."""
    probe = VideoProbe()
    meta = probe.probe(short_video)

    out_dir = tmp_path / "frames_png"
    cfg = FrameExtractionConfig(strategy="fixed_count", target_count=4, image_format="png")

    extractor = FrameExtractor()
    planned_ts, frames = extractor.extract_frames(meta, out_dir, cfg)

    assert len(frames) == 4
    for f in frames:
        assert f.image_path.suffix == ".png"
        assert f.image_path.is_file()


def test_directory_isolation_no_collisions(short_video: Path, tmp_path: Path):
    """Separate extraction runs into isolated directories do not collide."""
    probe = VideoProbe()
    meta = probe.probe(short_video)

    dir1 = tmp_path / "run_1"
    dir2 = tmp_path / "run_2"

    extractor = FrameExtractor()
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=0.5)

    _, frames1 = extractor.extract_frames(meta, dir1, cfg)
    _, frames2 = extractor.extract_frames(meta, dir2, cfg)

    # Check distinct paths
    paths1 = {f.image_path for f in frames1}
    paths2 = {f.image_path for f in frames2}
    assert paths1.isdisjoint(paths2)
    assert len(paths1) == 3
    assert len(paths2) == 3
