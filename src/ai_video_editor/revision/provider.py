"""Interface abstrata do revisor de IA."""

from abc import ABC, abstractmethod
from typing import Optional

from ai_video_editor.director.schema import UserBrief
from ai_video_editor.llm.schema import ModelSwitch
from ai_video_editor.revision.schema import RevisionContext, RevisionDecision, SentenceState


class EditReviser(ABC):
    """Contrato para qualquer revisor (Gemini, Mock, etc.).

    Como o Diretor, o revisor só referencia IDs de frases; os tempos são calculados
    depois, de forma determinística.
    """

    last_input_tokens: Optional[int] = None
    last_output_tokens: Optional[int] = None
    # Chamadas à API na última revisão (>1 se houve nova tentativa por falha temporária).
    last_api_attempts: int = 1
    thinking_label: Optional[str] = None
    effective_thinking_label: Optional[str] = None
    # Cadeia pedida (`provedor:modelo`), modelo que respondeu e trocas automáticas de modelo.
    model_chain: list[str] = []
    effective_provider: Optional[str] = None
    effective_model_id: Optional[str] = None
    model_switches: list[ModelSwitch] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identificador do provedor (ex: 'gemini', 'mock')."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Modelo usado nas revisões."""
        ...

    @abstractmethod
    def revise(
        self, states: list[SentenceState], brief: UserBrief, context: RevisionContext
    ) -> RevisionDecision:
        """Devolve o novo conjunto de frases removidas para o pedido.

        Args:
            states: Todas as frases com o estado (mantida/removida e motivo) na versão base.
            brief: Briefing original da edição.
            context: Vídeo, pedido novo, histórico curto e erro da tentativa anterior.

        Raises:
            AIResponseValidationError: se a resposta do modelo não puder ser interpretada.
        """
        ...
