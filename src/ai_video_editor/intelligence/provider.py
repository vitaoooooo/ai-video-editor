"""Interface abstrata comum para provedores de inteligência artificial multimodal."""

from abc import ABC, abstractmethod
from typing import Optional

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata
from ai_video_editor.intelligence.schema import AIAnalysisConfig, VisualAnalysisResponse


class MultimodalAIProvider(ABC):
    """Contrato base abstrato para qualquer provedor de IA multimodal (Gemini, OpenAI, Mock, etc.)."""

    # Uso de tokens da última chamada; provedores que não informam mantêm None.
    last_input_tokens: Optional[int] = None
    last_output_tokens: Optional[int] = None
    # Chamadas à API na última análise (>1 se houve nova tentativa por falha temporária).
    last_api_attempts: int = 1

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Retorna o identificador único do provedor (ex: 'gemini', 'mock')."""
        ...

    @abstractmethod
    def analyze_frames(
        self,
        metadata: VideoMetadata,
        frames: list[ExtractedFrame],
        config: AIAnalysisConfig,
    ) -> VisualAnalysisResponse:
        """Executa a análise multimodal sobre os frames enviados e metadados técnicos.

        Args:
            metadata: Informações técnicas do vídeo originadas da Fase 2.
            frames: Lista de frames (ou proxies) selecionados para o envio.
            config: Configuração com parâmetros de modelo, cotas e timeouts.

        Returns:
            VisualAnalysisResponse estruturado e validado.
        """
        ...
