"""Integration tests for VideoIngestService and batch ingestion workflow."""

import json
from pathlib import Path
import pytest

from ai_video_editor.analyzer.schema import FrameExtractionConfig
from ai_video_editor.analyzer.service import VideoIngestService


def test_analyze_single_video(synthetic_video_with_audio: Path, tmp_path: Path):
    """Analyzes a single video and verifies structure and directory creation."""
    service = VideoIngestService(default_base_dir=tmp_path)
    out_dir = tmp_path / "single_test_out"
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=1.5)

    result = service.analyze_single(synthetic_video_with_audio, output_dir=out_dir, config=cfg)

    assert result.metadata.video_stream.width == 640
    assert result.metadata.audio_stream is not None
    assert len(result.frames) >= 2
    assert "does not guarantee exhaustive representation" in result.sampling_disclaimer
    assert result.output_dir == out_dir.resolve()
    for f in result.frames:
        assert f.image_path.is_file()


def test_analyze_batch_mixed_success_and_failures(
    synthetic_video_with_audio: Path,
    synthetic_video_without_audio: Path,
    synthetic_audio_only_file: Path,
    corrupted_file: Path,
    tmp_path: Path,
):
    """Batch analysis processes multiple files, isolating successes and recording failures."""
    missing_file = tmp_path / "ghost_file.mp4"

    video_inputs = [
        synthetic_video_with_audio,
        synthetic_video_without_audio,
        synthetic_audio_only_file,
        corrupted_file,
        missing_file,
    ]

    service = VideoIngestService(default_base_dir=tmp_path / "batch_out")
    cfg = FrameExtractionConfig(strategy="interval", interval_sec=1.5)

    batch_res = service.analyze_batch(video_inputs, config=cfg)

    # 5 files requested
    assert batch_res.total_requested == 5

    # 2 valid video files should succeed
    assert len(batch_res.successful) == 2
    success_paths = {res.metadata.source_path.name for res in batch_res.successful}
    assert synthetic_video_with_audio.name in success_paths
    assert synthetic_video_without_audio.name in success_paths

    # 3 invalid files should be recorded in failed dictionary
    assert len(batch_res.failed) == 3
    assert str(synthetic_audio_only_file) in batch_res.failed
    assert "No video stream found" in batch_res.failed[str(synthetic_audio_only_file)]

    assert str(corrupted_file) in batch_res.failed
    assert "Invalid or corrupted media file" in batch_res.failed[str(corrupted_file)]

    assert str(missing_file) in batch_res.failed
    assert "Video file not found" in batch_res.failed[str(missing_file)]

    # Verify session directory contains the saved JSON report
    report_path = batch_res.session_dir / "analysis_report.json"
    assert report_path.is_file()

    report_content = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_content["session_id"] == batch_res.session_id
    assert len(report_content["successful"]) == 2
    assert len(report_content["failed"]) == 3
