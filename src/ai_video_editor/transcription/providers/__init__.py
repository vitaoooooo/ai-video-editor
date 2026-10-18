"""Provedores de transcrição implementados."""

from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider

__all__ = ["FasterWhisperProvider", "MockTranscriptionProvider"]
