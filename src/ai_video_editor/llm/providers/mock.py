"""Provedor simulado da camada `llm`: respostas fixas, sem rede e sem chave."""

from typing import Any, Callable, Optional, Sequence, Union

from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import AIProviderError
from ai_video_editor.llm.client import StructuredLLMClient
from ai_video_editor.llm.schema import ImageInput, LLMCallConfig, T, ThinkingLevel

# Uma função recebe (prompt, imagens, schema) e devolve a resposta (ou lança a exceção).
MockResponder = Callable[[str, list[ImageInput], type[BaseModel]], Any]
MockLLMResponse = Union[BaseModel, dict, Exception, MockResponder]


class MockLLMClient(StructuredLLMClient):
    """Devolve as respostas da lista, uma por chamada (uma exceção na lista é lançada).

    A última resposta se repete quando a lista acaba. Sem lista, devolve o schema montado
    com `{"summary": "Resposta simulada."}`. Uma função na lista é chamada com
    (prompt, imagens, schema) e devolve a resposta: serve para respostas que dependem do
    pedido (ex: os IDs das tomadas de um lote). `has_key=False` simula provedor sem chave
    e `name` permite fingir outro provedor nos testes da cadeia.

    `calls` guarda (prompt, config) de cada chamada e `image_calls`, as imagens recebidas.
    """

    provider_name = "mock"
    default_model = "mock"

    def __init__(
        self,
        responses: Optional[list[MockLLMResponse]] = None,
        has_key: bool = True,
        name: str = "mock",
        **_: Any,
    ):
        super().__init__(client=object())
        self.responses = list(responses) if responses is not None else None
        self.has_key = has_key
        self.provider_name = name
        self.api_key_env = f"{name.upper()}_API_KEY"
        self.calls: list[tuple[str, LLMCallConfig]] = []
        self.image_calls: list[list[ImageInput]] = []

    def has_credentials(self) -> bool:
        return self.has_key

    def _create_client(self) -> Any:  # pragma: no cover — o cliente é sempre injetado
        return object()

    def thinking_setting(self, model_id: str, level: ThinkingLevel) -> tuple[str, Any]:
        return "thinking", level

    def _request(
        self,
        prompt: str,
        schema: type[BaseModel],
        config: LLMCallConfig,
        level: Optional[ThinkingLevel],
        images: Sequence[ImageInput] = (),
    ) -> Any:
        self.calls.append((prompt, config))
        self.image_calls.append(list(images))
        if self.responses is None:
            return {"summary": "Resposta simulada."}
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if callable(response) and not isinstance(response, (BaseModel, Exception)):
            response = response(prompt, list(images), schema)
        if isinstance(response, Exception):
            raise response
        return response

    def _usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        return None, None

    def _parse(self, response: Any, schema: type[T]) -> T:
        return response if isinstance(response, schema) else schema.model_validate(response)

    def map_error(self, error: Exception, timeout_sec: float) -> AIProviderError:
        return AIProviderError(str(error), provider=self.provider_name)
