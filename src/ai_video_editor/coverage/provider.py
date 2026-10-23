"""Interface abstrata do Diretor de cobertura."""

from abc import ABC, abstractmethod
from typing import Optional

from ai_video_editor.coverage.schema import CoverageContext, CoverageDecision, TopicMap
from ai_video_editor.llm.schema import ModelSwitch


class CoverageDirector(ABC):
    """Contrato para qualquer Diretor de cobertura (cadeia de modelos, mock...).

    Ele só escolhe IDs (frases, tomadas) e cita evidências; os tempos saem do código. Duas chamadas: o
    mapa de assuntos (`map_topics`) e a cobertura (`decide`).
    """

    last_input_tokens: Optional[int] = None
    last_output_tokens: Optional[int] = None
    last_api_attempts: int = 1
    thinking_label: Optional[str] = None
    effective_thinking_label: Optional[str] = None
    model_chain: list[str] = []
    effective_provider: Optional[str] = None
    effective_model_id: Optional[str] = None
    model_switches: list[ModelSwitch] = []

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def map_topics(self, context: CoverageContext) -> TopicMap:
        """Divide as frases mantidas em blocos de assunto (ainda não validados); `context.previous_error` numa
        nova tentativa.

        Raises:
            AIResponseValidationError: se a resposta do modelo não puder ser interpretada.
        """
        ...

    @abstractmethod
    def decide(self, context: CoverageContext) -> CoverageDecision:
        """Escolhe coberturas e respiros com evidências (ainda não validados), sabendo o mapa de assuntos
        (`context.topics`); `context.previous_error` numa nova tentativa.

        Raises:
            AIResponseValidationError: se a resposta do modelo não puder ser interpretada.
        """
        ...
