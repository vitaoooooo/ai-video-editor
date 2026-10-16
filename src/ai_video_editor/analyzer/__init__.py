"""Video ingestion and technical analysis module."""

from ai_video_editor.analyzer.exceptions import (
    FrameExtractionError,
    InvalidMediaFormatError,
    NoVideoStreamError,
    VideoIngestError,
    VideoNotFoundError,
)
from ai_video_editor.analyzer.extractor import FrameExtractor, calculate_target_timestamps
from ai_video_editor.analyzer.probe import VideoProbe, parse_fractional_fps
from ai_video_editor.analyzer.schema import (
    AudioStreamInfo,
    BatchAnalysisResult,
    ExtractedFrame,
    FrameExtractionConfig,
    VideoAnalysisResult,
    VideoMetadata,
    VideoStreamInfo,
)
from ai_video_editor.analyzer.service import VideoIngestService

__all__ = [
    "AudioStreamInfo",
    "BatchAnalysisResult",
    "ExtractedFrame",
    "FrameExtractionConfig",
    "FrameExtractionError",
    "FrameExtractor",
    "InvalidMediaFormatError",
    "NoVideoStreamError",
    "VideoAnalysisResult",
    "VideoIngestError",
    "VideoIngestService",
    "VideoMetadata",
    "VideoNotFoundError",
    "VideoProbe",
    "VideoStreamInfo",
    "calculate_target_timestamps",
    "parse_fractional_fps",
]
