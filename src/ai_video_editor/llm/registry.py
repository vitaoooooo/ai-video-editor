"""Registro de provedores de IA: nome → fábrica do cliente.

Os provedores embutidos são registrados com import tardio, para que o SDK de cada um só
seja exigido quando ele for usado.
"""

import importlib
from typing import Any, Callable

from ai_video_editor.llm.client import StructuredLLMClient

ClientFactory = Callable[..., StructuredLLMClient]

_FACTORIES: dict[str, ClientFactory] = {}
_LAZY: dict[str, str] = {}


def register_provider(name: str, factory: ClientFactory | str) -> None:
    """Registra um provedor.

    Args:
        name: Nome usado na CLI e na cadeia (`provedor:modelo`).
        factory: Classe/fábrica que aceita `api_key=` e `client=`, ou o caminho
            `"modulo:Classe"` para import tardio.
    """
    _FACTORIES.pop(name, None)
    _LAZY.pop(name, None)
    if isinstance(factory, str):
        _LAZY[name] = factory
    else:
        _FACTORIES[name] = factory


def provider_names() -> list[str]:
    return sorted({*_FACTORIES, *_LAZY})


def get_factory(name: str) -> ClientFactory:
    if name not in _FACTORIES:
        if name not in _LAZY:
            raise ValueError(f"provedor de IA desconhecido: {name!r} (disponíveis: {', '.join(provider_names())})")
        module, attr = _LAZY[name].split(":")
        _FACTORIES[name] = getattr(importlib.import_module(module), attr)
    return _FACTORIES[name]


def create_client(name: str, **kwargs: Any) -> StructuredLLMClient:
    """Instancia o cliente do provedor (sem chamar a rede)."""
    return get_factory(name)(**kwargs)


def default_model(name: str) -> str:
    return get_factory(name).default_model


def default_fallback(name: str) -> tuple[str, ...]:
    return tuple(get_factory(name).default_fallback)


def default_timeout(name: str, fallback: float) -> float:
    """Tempo limite padrão do provedor (o Ollama local é mais lento), ou `fallback`.

    `none` e provedores desconhecidos devolvem `fallback` (a CLI valida o nome depois).
    """
    if name not in provider_names():
        return fallback
    return get_factory(name).default_timeout_sec or fallback


register_provider("gemini", "ai_video_editor.llm.providers.gemini:GeminiLLMClient")
register_provider("openai", "ai_video_editor.llm.providers.openai:OpenAILLMClient")
register_provider("ollama", "ai_video_editor.llm.providers.ollama:OllamaLLMClient")
register_provider("mock", "ai_video_editor.llm.providers.mock:MockLLMClient")
