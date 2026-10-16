"""Pydantic schemas for technical video analysis and frame extraction."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class VideoStreamInfo(BaseModel):
    """Detailed technical properties of a primary video stream."""

    codec_name: str = Field(..., description="Video codec name (e.g. h264, vp9, hevc)")
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")
    fps: float = Field(..., ge=0.0, description="Frames per second")
    bitrate: Optional[int] = Field(default=None, ge=0, description="Stream bitrate in bits/sec")
    pixel_format: Optional[str] = Field(default=None, description="Pixel format (e.g. yuv420p)")
    duration_sec: Optional[float] = Field(default=None, ge=0.0, description="Stream duration in seconds")


class AudioStreamInfo(BaseModel):
    """Properties of an audio stream, if present."""

    codec_name: str = Field(..., description="Audio codec name (e.g. aac, mp3, opus)")
    sample_rate: Optional[int] = Field(default=None, gt=0, description="Sampling rate in Hz")
    channels: Optional[int] = Field(default=None, ge=1, description="Number of audio channels")
    bitrate: Optional[int] = Field(default=None, ge=0, description="Audio bitrate in bits/sec")
    duration_sec: Optional[float] = Field(default=None, ge=0.0, description="Audio duration in seconds")


class VideoMetadata(BaseModel):
    """Normalized technical metadata extracted from a video file."""

    source_path: Path = Field(..., description="Absolute path to the source video file")
    duration_sec: float = Field(..., ge=0.0, description="Total media duration in seconds")
    size_bytes: int = Field(..., ge=0, description="File size on disk in bytes")
    bitrate: Optional[int] = Field(default=None, ge=0, description="Overall container bitrate in bits/sec")
    video_stream: VideoStreamInfo = Field(..., description="Primary video stream metadata")
    audio_stream: Optional[AudioStreamInfo] = Field(
        default=None, description="Primary audio stream metadata (None if audio is absent)"
    )


SamplingStrategy = Literal["interval", "fixed_count"]
ImageFormat = Literal["jpg", "png"]


class FrameExtractionConfig(BaseModel):
    """Configuration governing representative frame extraction."""

    strategy: SamplingStrategy = Field(
        default="interval",
        description="Sampling strategy: 'interval' (every N seconds) or 'fixed_count' (evenly distributed)",
    )
    interval_sec: float = Field(
        default=2.0,
        gt=0.0,
        description="Interval in seconds between frames when strategy='interval'",
    )
    target_count: int = Field(
        default=5,
        gt=0,
        description="Number of frames to extract when strategy='fixed_count'",
    )
    max_frames: int = Field(
        default=50,
        gt=0,
        le=500,
        description="Configurable ceiling for maximum frames to extract per video",
    )
    image_format: ImageFormat = Field(
        default="jpg",
        description="Image format for extracted frames ('jpg' or 'png')",
    )


class ExtractedFrame(BaseModel):
    """Record of an individual extracted image frame with trace metadata."""

    frame_index: int = Field(..., ge=0, description="0-indexed sequence number of the extracted frame")
    target_timestamp_sec: float = Field(..., ge=0.0, description="Intended timestamp in video")
    image_path: Path = Field(..., description="Absolute path to the saved image file on disk")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    size_bytes: int = Field(..., gt=0, description="Image file size in bytes")


class VideoAnalysisResult(BaseModel):
    """Complete technical analysis output for a single video file."""

    metadata: VideoMetadata = Field(..., description="Normalized technical metadata")
    sampling_strategy_used: str = Field(..., description="Description of the sampling strategy applied")
    sampling_disclaimer: str = Field(
        default="Discrete sample of frames; does not guarantee exhaustive representation of all video content.",
        description="Explicit limitation notice regarding frame coverage",
    )
    planned_timestamps_sec: list[float] = Field(
        ..., description="List of target timestamps requested by the extraction strategy"
    )
    frames: list[ExtractedFrame] = Field(
        default_factory=list, description="Extracted frames successfully saved to disk"
    )
    output_dir: Path = Field(..., description="Unique output directory containing extracted frames")
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the analysis run",
    )


class BatchAnalysisResult(BaseModel):
    """Aggregated results for multiple analyzed video files."""

    session_id: str = Field(..., description="Unique run/session identifier")
    session_dir: Path = Field(..., description="Root directory for the session analysis artifacts")
    total_requested: int = Field(..., ge=0, description="Total number of video paths received")
    successful: list[VideoAnalysisResult] = Field(
        default_factory=list, description="List of successful video analysis results"
    )
    failed: dict[str, str] = Field(
        default_factory=dict, description="Mapping of failed file path -> error explanation"
    )
