"""Interface abstrata do Produtor de IA."""

from abc import ABC, abstractmethod
from typing import Optional

from ai_video_editor.llm.schema import ModelSwitch
from ai_video_editor.producer.schema import BriefingContent, MaterialSummary, ProducerContext, RecipeInfo


class BriefingProducer(ABC):
    """Contrato para qualquer Produtor (cadeia de modelos, mock).

    O Produtor só escolhe uma receita registrada e escreve metas e instruções; o código valida
    a resposta contra o registro de receitas e o material.
    """

    # Mesmos atributos de execução do Diretor (preenchidos por `ChainBackedStage`).
    last_input_tokens: Optional[int] = None
    last_output_tokens: Optional[int] = None
    last_api_attempts: int = 1
    model_chain: list[str] = []
    effective_provider: Optional[str] = None
    effective_model_id: Optional[str] = None
    model_switches: list[ModelSwitch] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identificador do provedor (ex: 'gemini', 'mock')."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Modelo pedido."""

    @abstractmethod
    def propose(
        self, request: str, material: MaterialSummary, recipes: list[RecipeInfo], context: ProducerContext
    ) -> BriefingContent:
        """Briefing para o pedido e o material (ainda não validado).

        Raises:
            AIResponseValidationError: resposta do modelo fora do schema.
            AIProviderError: falha de comunicação (sem chave, cota, timeout...).
        """
