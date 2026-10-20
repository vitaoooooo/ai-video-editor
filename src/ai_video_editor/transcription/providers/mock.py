"""Provedor de transcrição simulado, determinístico e sem dependências externas."""

from pathlib import Path
from typing import Optional

from ai_video_editor.transcription.exceptions import TranscriptionFailedError
from ai_video_editor.transcription.provider import TranscriptionProgress, TranscriptionProvider
from ai_video_editor.transcription.schema import (
    Transcript,
    TranscriptSegment,
    TranscriptionConfig,
    Word,
)

_DEFAULT_SENTENCES = [
    "olá pessoal bem vindo ao tutorial",
    "hoje vamos aprender a editar vídeo",
    "obrigado por assistir",
]


class MockTranscriptionProvider(TranscriptionProvider):
    """Gera uma transcrição fixa a partir de frases configuráveis (ou de um Transcript pronto)."""

    def __init__(
        self,
        sentences: Optional[list[str]] = None,
        word_duration_sec: float = 0.4,
        gap_sec: float = 0.5,
        language: str = "pt",
        custom_transcript: Optional[Transcript] = None,
        force_failure: bool = False,
    ):
        self.sentences = list(sentences) if sentences is not None else list(_DEFAULT_SENTENCES)
        self.word_duration_sec = word_duration_sec
        self.gap_sec = gap_sec
        self.language = language
        self.custom_transcript = custom_transcript
        self.force_failure = force_failure

    @property
    def provider_name(self) -> str:
        return "mock"

    def transcribe(
        self,
        media_path: Path,
        config: TranscriptionConfig,
        on_progress: Optional[TranscriptionProgress] = None,
    ) -> Transcript:
        if self.force_failure:
            raise TranscriptionFailedError(self.provider_name, details="Falha simulada.")
        if self.custom_transcript is not None:
            if on_progress is not None:
                total = self.custom_transcript.duration_sec
                for segment in self.custom_transcript.segments:
                    on_progress(min(segment.end_sec, total), total)
            return self.custom_transcript

        cursor = 0.0
        segments: list[TranscriptSegment] = []
        for sentence in self.sentences:
            seg_start = cursor
            words = []
            for text in sentence.split():
                words.append(
                    Word(text=text, start_sec=cursor, end_sec=cursor + self.word_duration_sec)
                )
                cursor += self.word_duration_sec
            if not words:
                continue
            segments.append(
                TranscriptSegment(
                    id=len(segments),
                    start_sec=seg_start,
                    end_sec=cursor,
                    text=sentence,
                    words=words,
                )
            )
            cursor += self.gap_sec

        if on_progress is not None:
            for segment in segments:
                on_progress(segment.end_sec, cursor)
        return Transcript(
            source_path=Path(media_path),
            language=config.language or self.language,
            duration_sec=cursor,
            model_used="mock",
            segments=segments,
        )
