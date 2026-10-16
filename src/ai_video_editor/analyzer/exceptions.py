"""Domain exceptions for video ingestion and technical analysis."""

from pathlib import Path
from typing import Optional


class VideoIngestError(Exception):
    """Base exception for all video ingestion and analysis errors."""

    def __init__(self, message: str, file_path: Optional[Path] = None):
        super().__init__(message)
        self.file_path = file_path


class VideoNotFoundError(VideoIngestError):
    """Raised when the specified video file does not exist on disk."""

    def __init__(self, file_path: Path):
        super().__init__(f"Video file not found: {file_path}", file_path=file_path)


class InvalidMediaFormatError(VideoIngestError):
    """Raised when ffprobe fails to parse the media file or file is corrupted."""

    def __init__(self, file_path: Path, details: str = ""):
        msg = f"Invalid or corrupted media file: {file_path}"
        if details:
            msg += f" ({details})"
        super().__init__(msg, file_path=file_path)


class NoVideoStreamError(VideoIngestError):
    """Raised when the media file contains no video streams (e.g., audio-only)."""

    def __init__(self, file_path: Path):
        super().__init__(f"No video stream found in media file: {file_path}", file_path=file_path)


class FrameExtractionError(VideoIngestError):
    """Raised when frame extraction fails during FFmpeg processing."""

    def __init__(self, file_path: Path, timestamp_sec: float, details: str = ""):
        msg = f"Failed to extract frame from '{file_path}' at {timestamp_sec:.3f}s"
        if details:
            msg += f": {details}"
        super().__init__(msg, file_path=file_path)
        self.timestamp_sec = timestamp_sec
