"""Provedor simulado da camada `llm`: respostas fixas, sem rede e sem chave."""

from typing import Any, Optional, Union

from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import AIProviderError
from ai_video_editor.llm.client import StructuredLLMClient
from ai_video_editor.llm.schema import LLMCallConfig, T, ThinkingLevel

MockLLMResponse = Union[BaseModel, dict, Exception]


class MockLLMClient(StructuredLLMClient):
    """Devolve as respostas da lista, uma por chamada (uma exceção na lista é lançada).

    A última resposta se repete quando a lista acaba. Sem lista, devolve o schema montado
    com `{"summary": "Resposta simulada."}`. `has_key=False` simula provedor sem chave e
    `name` permite fingir outro provedor nos testes da cadeia.
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

    def has_credentials(self) -> bool:
        return self.has_key

    def _create_client(self) -> Any:  # pragma: no cover — o cliente é sempre injetado
        return object()

    def thinking_setting(self, model_id: str, level: ThinkingLevel) -> tuple[str, Any]:
        return "thinking", level

    def _request(self, prompt: str, schema: type[BaseModel], config: LLMCallConfig, level: Optional[ThinkingLevel]) -> Any:
        self.calls.append((prompt, config))
        if self.responses is None:
            return {"summary": "Resposta simulada."}
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    def _usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        return None, None

    def _parse(self, response: Any, schema: type[T]) -> T:
        return response if isinstance(response, schema) else schema.model_validate(response)

    def map_error(self, error: Exception, timeout_sec: float) -> AIProviderError:
        return AIProviderError(str(error), provider=self.provider_name)
