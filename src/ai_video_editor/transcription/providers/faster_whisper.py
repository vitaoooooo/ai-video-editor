"""Provedor de transcrição local baseado em faster-whisper (dependência opcional)."""

import subprocess
from pathlib import Path

from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
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

# O Whisper espera áudio mono em 16 kHz.
_SAMPLE_RATE = 16000


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

    def _decode_audio(self, media_path: Path):
        """Decodifica o áudio com FFmpeg para um array float32 mono 16 kHz (sem usar o PyAV)."""
        import numpy as np

        args = [
            str(get_ffmpeg_path()),
            "-nostdin",
            "-i",
            str(media_path.resolve()),
            "-vn",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "-",
        ]
        try:
            proc = subprocess.run(args, capture_output=True, shell=False)
        except OSError as exc:
            raise TranscriptionMediaError(
                self.provider_name, media_path, f"Não foi possível executar o FFmpeg: {exc}"
            ) from exc
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise TranscriptionMediaError(
                self.provider_name, media_path, err or "FFmpeg retornou erro ao decodificar o áudio."
            )
        usable = len(proc.stdout) - len(proc.stdout) % 4
        if usable == 0:
            raise TranscriptionMediaError(self.provider_name, media_path, "A mídia não possui áudio.")
        return np.frombuffer(proc.stdout[:usable], dtype=np.float32)

    def transcribe(self, media_path: Path, config: TranscriptionConfig) -> Transcript:
        media_path = Path(media_path)
        if not media_path.is_file():
            raise TranscriptionMediaError(self.provider_name, media_path, "Arquivo inexistente.")

        model = self._load_model(config)
        audio = self._decode_audio(media_path)
        duration = len(audio) / _SAMPLE_RATE
        try:
            raw_segments, info = model.transcribe(
                audio,
                language=config.language,
                word_timestamps=True,
                vad_filter=config.vad_filter,
            )
            segments = self._convert_segments(raw_segments)
            language = info.language
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
