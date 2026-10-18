"""Módulo de análise de áudio: extração e detecção de silêncios."""

from ai_video_editor.audio.exceptions import (
    AudioAnalysisError,
    AudioExtractionError,
    MediaNotFoundError,
    NoAudioStreamError,
    SilenceDetectionError,
)
from ai_video_editor.audio.extractor import extract_audio
from ai_video_editor.audio.schema import (
    AudioAnalysisParams,
    AudioAnalysisResult,
    SilenceInterval,
)
from ai_video_editor.audio.silence import compute_speech, detect_silences, parse_silencedetect

__all__ = [
    "AudioAnalysisError",
    "AudioAnalysisParams",
    "AudioAnalysisResult",
    "AudioExtractionError",
    "MediaNotFoundError",
    "NoAudioStreamError",
    "SilenceDetectionError",
    "SilenceInterval",
    "compute_speech",
    "detect_silences",
    "extract_audio",
    "parse_silencedetect",
]
