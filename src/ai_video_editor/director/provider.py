"""Interface abstrata do Diretor de IA."""

from abc import ABC, abstractmethod
from typing import Optional

from ai_video_editor.director.schema import DirectorContext, DirectorDecision, Sentence, UserBrief


class EditDirector(ABC):
    """Contrato para qualquer Diretor (Gemini, Mock, etc.).

    O Diretor só escolhe IDs de frases a remover; os tempos de corte são calculados
    depois, de forma determinística, a partir das frases e dos silêncios detectados.
    """

    # Uso de tokens da última chamada; provedores que não informam mantêm None.
    last_input_tokens: Optional[int] = None
    last_output_tokens: Optional[int] = None
    # Descrição da configuração de raciocínio usada (vai para o relatório); None se não se aplica.
    thinking_label: Optional[str] = None
    # Nível efetivamente usado na última chamada (difere do pedido se a API recusou o nível).
    effective_thinking_label: Optional[str] = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identificador do provedor (ex: 'gemini', 'mock')."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Modelo usado nas decisões (entra no fingerprint do cache)."""
        ...

    @abstractmethod
    def decide(
        self, sentences: list[Sentence], brief: UserBrief, context: DirectorContext
    ) -> DirectorDecision:
        """Decide quais frases remover.

        Args:
            sentences: Frases da transcrição, com IDs sequenciais.
            brief: Pedido do usuário (intensidade, duração alvo, instruções).
            context: Dados do vídeo; `previous_error` vem preenchido numa nova tentativa.

        Returns:
            DirectorDecision (ainda não validada contra as frases).

        Raises:
            AIResponseValidationError: se a resposta do modelo não puder ser interpretada.
        """
        ...
