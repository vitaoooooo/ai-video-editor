"""Unit tests for analyzer schemas and serialization."""

from datetime import datetime
from pathlib import Path
import pytest
from pydantic import ValidationError

from ai_video_editor.analyzer.schema import (
    AudioStreamInfo,
    BatchAnalysisResult,
    ExtractedFrame,
    FrameExtractionConfig,
    VideoAnalysisResult,
    VideoMetadata,
    VideoStreamInfo,
)


def test_video_metadata_schema_valid():
    """Validates complete VideoMetadata construction."""
    meta = VideoMetadata(
        source_path=Path("sample.mp4"),
        duration_sec=12.5,
        size_bytes=1048576,
        bitrate=800000,
        video_stream=VideoStreamInfo(
            codec_name="h264",
            width=1920,
            height=1080,
            fps=29.97,
            bitrate=700000,
            pixel_format="yuv420p",
            duration_sec=12.5,
        ),
        audio_stream=AudioStreamInfo(
            codec_name="aac",
            sample_rate=48000,
            channels=2,
            bitrate=128000,
            duration_sec=12.5,
        ),
    )
    assert meta.video_stream.width == 1920
    assert meta.video_stream.fps == 29.97
    assert meta.audio_stream is not None
    assert meta.audio_stream.channels == 2


def test_video_metadata_without_audio():
    """Video without audio stream is valid."""
    meta = VideoMetadata(
        source_path=Path("no_audio.mp4"),
        duration_sec=5.0,
        size_bytes=500000,
        video_stream=VideoStreamInfo(
            codec_name="h264",
            width=1280,
            height=720,
            fps=30.0,
        ),
        audio_stream=None,
    )
    assert meta.audio_stream is None


def test_frame_extraction_config_defaults_and_limits():
    """Validates default ceiling (50 frames) and validation constraints."""
    cfg = FrameExtractionConfig()
    assert cfg.max_frames == 50
    assert cfg.interval_sec == 2.0
    assert cfg.strategy == "interval"
    assert cfg.image_format == "jpg"

    # Negative interval disallowed
    with pytest.raises(ValidationError):
        FrameExtractionConfig(interval_sec=-1.0)

    # Zero max_frames disallowed
    with pytest.raises(ValidationError):
        FrameExtractionConfig(max_frames=0)

    # Exceeding hard upper ceiling (500) disallowed
    with pytest.raises(ValidationError):
        FrameExtractionConfig(max_frames=600)


def test_analysis_result_json_roundtrip(tmp_path: Path):
    """Analysis result serializes to JSON with disclaimer intact."""
    meta = VideoMetadata(
        source_path=tmp_path / "vid.mp4",
        duration_sec=4.0,
        size_bytes=200000,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )
    frame = ExtractedFrame(
        frame_index=0,
        target_timestamp_sec=0.0,
        image_path=tmp_path / "frame_0000_000000ms.jpg",
        width=640,
        height=360,
        size_bytes=15000,
    )
    res = VideoAnalysisResult(
        metadata=meta,
        sampling_strategy_used="interval: every 2.00s",
        planned_timestamps_sec=[0.0, 2.0],
        frames=[frame],
        output_dir=tmp_path / "frames",
    )

    # Verify disclaimer presence
    assert "does not guarantee exhaustive representation" in res.sampling_disclaimer

    json_str = res.model_dump_json(indent=2)
    loaded = VideoAnalysisResult.model_validate_json(json_str)
    assert loaded.metadata.duration_sec == 4.0
    assert len(loaded.frames) == 1
    assert loaded.frames[0].width == 640


def test_batch_analysis_result_serializable(tmp_path: Path):
    """BatchAnalysisResult captures successes and failures cleanly."""
    batch = BatchAnalysisResult(
        session_id="test_session_1",
        session_dir=tmp_path,
        total_requested=2,
        successful=[],
        failed={"bad_video.mp4": "File not found"},
    )
    assert batch.failed["bad_video.mp4"] == "File not found"
    json_data = batch.model_dump_json()
    assert "bad_video.mp4" in json_data
