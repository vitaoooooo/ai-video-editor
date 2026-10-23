"""Interface abstrata comum para provedores de transcrição."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor.transcription.normalize import SegmentAdjustments
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig, resolve_model_size

# (segundos de áudio já transcritos, duração total do áudio em segundos).
TranscriptionProgress = Callable[[float, float], None]


class TranscriptionProvider(ABC):
    """Contrato base para qualquer provedor de transcrição (faster-whisper, Mock, etc.)."""

    # O que a normalização (`transcription/normalize.py`) mudou na última transcrição; None = nada
    # foi normalizado (provedor que não usa a normalização ou nenhuma transcrição ainda).
    _last_adjustments: Optional[SegmentAdjustments] = None

    @property
    def last_adjustments(self) -> Optional[SegmentAdjustments]:
        """Ajustes da última chamada de `transcribe` (aviso no `edit`; não entra no Transcript)."""
        return self._last_adjustments

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Retorna o identificador único do provedor (ex: 'faster_whisper', 'mock')."""
        ...

    @abstractmethod
    def transcribe(
        self,
        media_path: Path,
        config: TranscriptionConfig,
        on_progress: Optional[TranscriptionProgress] = None,
    ) -> Transcript:
        """Transcreve um arquivo de áudio ou vídeo.

        Args:
            media_path: Caminho da mídia (áudio ou vídeo).
            config: Parâmetros do modelo e do idioma.
            on_progress: Chamado a cada segmento pronto com (fim do segmento, duração do áudio).

        Returns:
            Transcript validado, com IDs de segmento sequenciais a partir de 0.
        """
        ...

    def effective_settings(self, config: TranscriptionConfig) -> tuple[str, str]:
        """(dispositivo, compute_type) que `transcribe` usará com essa configuração.

        Entra no fingerprint do cache da transcrição; provedores que resolvem "auto" sobrescrevem.
        """
        return config.device, config.compute_type

    def effective_model(self, config: TranscriptionConfig) -> str:
        """Modelo que `transcribe` usará (resolve "auto" pelo dispositivo efetivo).

        Entra no fingerprint do cache da transcrição, junto com `effective_settings`.
        """
        return resolve_model_size(config.model_size, self.effective_settings(config)[0])
