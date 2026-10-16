"""Extracts technical metadata using ffprobe with safe type conversions."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from ai_video_editor.analyzer.exceptions import (
    InvalidMediaFormatError,
    NoVideoStreamError,
    VideoNotFoundError,
)
from ai_video_editor.analyzer.schema import (
    AudioStreamInfo,
    VideoMetadata,
    VideoStreamInfo,
)
from ai_video_editor.renderer.ffmpeg_locator import get_ffprobe_path


def parse_fractional_fps(fps_str: Optional[str]) -> float:
    """Parses fractional frame rates safely using fractions.Fraction without eval().

    Examples:
        '30/1' -> 30.0
        '30000/1001' -> 29.97002997...
        '25' -> 25.0
        '0/0', None, '' -> 0.0
    """
    if not fps_str or not isinstance(fps_str, str):
        return 0.0

    fps_clean = fps_str.strip()
    if not fps_clean:
        return 0.0

    try:
        frac = Fraction(fps_clean)
        if frac.denominator == 0:
            return 0.0
        return float(frac)
    except (ValueError, ZeroDivisionError):
        # Fallback for plain float string if fraction parsing failed
        try:
            return float(fps_clean)
        except ValueError:
            return 0.0


def _safe_int(val: Any) -> Optional[int]:
    """Converts a value to int if possible, else None."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> Optional[float]:
    """Converts a value to float if possible, else None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class VideoProbe:
    """Probes video and audio media streams using ffprobe."""

    def __init__(self, ffprobe_bin: Optional[Path] = None):
        self.ffprobe_bin = ffprobe_bin or get_ffprobe_path()

    def probe(self, file_path: Path) -> VideoMetadata:
        """Probes the specified media file and returns normalized VideoMetadata."""
        path = file_path.resolve()

        if not path.is_file():
            raise VideoNotFoundError(path)

        args = [
            str(self.ffprobe_bin),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]

        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except Exception as e:
            raise InvalidMediaFormatError(path, details=str(e))

        if proc.returncode != 0:
            err = proc.stderr.strip() or "ffprobe non-zero exit"
            raise InvalidMediaFormatError(path, details=err)

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise InvalidMediaFormatError(path, details=f"ffprobe returned non-JSON output: {e}")

        format_info = data.get("format", {})
        streams = data.get("streams", [])

        # 1. Locate primary video stream
        video_stream_raw = next(
            (s for s in streams if s.get("codec_type") == "video"),
            None,
        )
        if video_stream_raw is None:
            raise NoVideoStreamError(path)

        # 2. Extract video stream details
        fps = parse_fractional_fps(video_stream_raw.get("r_frame_rate"))
        if fps <= 0.0:
            fps = parse_fractional_fps(video_stream_raw.get("avg_frame_rate"))
        if fps <= 0.0:
            fps = 30.0  # sensible fallback if container lacks rate tags

        video_stream = VideoStreamInfo(
            codec_name=video_stream_raw.get("codec_name", "unknown"),
            width=int(video_stream_raw.get("width", 0)),
            height=int(video_stream_raw.get("height", 0)),
            fps=round(fps, 3),
            bitrate=_safe_int(video_stream_raw.get("bit_rate")),
            pixel_format=video_stream_raw.get("pix_fmt"),
            duration_sec=_safe_float(video_stream_raw.get("duration")),
        )

        # 3. Locate primary audio stream (if present)
        audio_stream_raw = next(
            (s for s in streams if s.get("codec_type") == "audio"),
            None,
        )
        audio_stream: Optional[AudioStreamInfo] = None
        if audio_stream_raw is not None:
            audio_stream = AudioStreamInfo(
                codec_name=audio_stream_raw.get("codec_name", "unknown"),
                sample_rate=_safe_int(audio_stream_raw.get("sample_rate")),
                channels=_safe_int(audio_stream_raw.get("channels")),
                bitrate=_safe_int(audio_stream_raw.get("bit_rate")),
                duration_sec=_safe_float(audio_stream_raw.get("duration")),
            )

        # 4. Resolve total media duration
        duration_sec = _safe_float(format_info.get("duration"))
        if duration_sec is None:
            duration_sec = video_stream.duration_sec
        if duration_sec is None and audio_stream:
            duration_sec = audio_stream.duration_sec
        if duration_sec is None or duration_sec < 0.0:
            duration_sec = 0.0

        # 5. File size
        size_bytes = _safe_int(format_info.get("size"))
        if size_bytes is None:
            size_bytes = path.stat().st_size

        return VideoMetadata(
            source_path=path,
            duration_sec=round(duration_sec, 3),
            size_bytes=size_bytes,
            bitrate=_safe_int(format_info.get("bit_rate")),
            video_stream=video_stream,
            audio_stream=audio_stream,
        )
