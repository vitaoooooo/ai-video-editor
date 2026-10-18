"""Provedor de transcrição local baseado em faster-whisper (dependência opcional)."""

from pathlib import Path

from ai_video_editor.transcription.exceptions import (
    TranscriptionDependencyError,
    TranscriptionFailedError,
    TranscriptionMediaError,
)
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import (
    Transcript,
    TranscriptSegment,
    TranscriptionConfig,
    Word,
)

# Menor duração aceita para uma palavra; o Whisper às vezes devolve start == end.
_MIN_WORD_SEC = 0.001


class FasterWhisperProvider(TranscriptionProvider):
    """Transcreve com faster-whisper (word_timestamps=True), aceitando áudio ou vídeo."""

    @property
    def provider_name(self) -> str:
        return "faster_whisper"

    def _load_model(self, config: TranscriptionConfig):
        """Importa faster-whisper só quando necessário e instancia o modelo."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionDependencyError(self.provider_name) from exc
        return WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)

    def transcribe(self, media_path: Path, config: TranscriptionConfig) -> Transcript:
        media_path = Path(media_path)
        if not media_path.is_file():
            raise TranscriptionMediaError(self.provider_name, media_path, "Arquivo inexistente.")

        model = self._load_model(config)
        try:
            raw_segments, info = model.transcribe(
                str(media_path), language=config.language, word_timestamps=True
            )
            segments = self._convert_segments(raw_segments)
            language = info.language
            duration = info.duration
        except Exception as exc:
            raise TranscriptionFailedError(self.provider_name, details=str(exc)) from exc

        return Transcript(
            source_path=media_path,
            language=language,
            duration_sec=duration,
            model_used=f"faster-whisper:{config.model_size}",
            segments=segments,
        )

    @staticmethod
    def _convert_segments(raw_segments) -> list[TranscriptSegment]:
        """Converte os segmentos do faster-whisper (gerador) para o schema, com IDs 0..n-1."""
        segments: list[TranscriptSegment] = []
        for raw in raw_segments:
            words = [
                Word(
                    text=w.word.strip(),
                    start_sec=w.start,
                    end_sec=max(w.end, w.start + _MIN_WORD_SEC),
                    probability=w.probability,
                )
                for w in (getattr(raw, "words", None) or [])
            ]
            segments.append(
                TranscriptSegment(
                    id=len(segments),
                    start_sec=raw.start,
                    end_sec=max(raw.end, raw.start + _MIN_WORD_SEC),
                    text=raw.text.strip(),
                    words=words,
                )
            )
        return segments
