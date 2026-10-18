"""Interface abstrata comum para provedores de transcrição."""

from abc import ABC, abstractmethod
from pathlib import Path

from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig


class TranscriptionProvider(ABC):
    """Contrato base para qualquer provedor de transcrição (faster-whisper, Mock, etc.)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Retorna o identificador único do provedor (ex: 'faster_whisper', 'mock')."""
        ...

    @abstractmethod
    def transcribe(self, media_path: Path, config: TranscriptionConfig) -> Transcript:
        """Transcreve um arquivo de áudio ou vídeo.

        Args:
            media_path: Caminho da mídia (áudio ou vídeo).
            config: Parâmetros do modelo e do idioma.

        Returns:
            Transcript validado, com IDs de segmento sequenciais a partir de 0.
        """
        ...
