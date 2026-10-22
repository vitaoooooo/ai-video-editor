"""Camada genérica de IA com saída estruturada: interface, provedores plugáveis e cadeia de modelos."""

# `intelligence` precisa estar carregado antes: o provedor visual do Gemini importa desta camada.
import ai_video_editor.intelligence  # noqa: F401, E402
from ai_video_editor.llm.chain import ModelChain, parse_chain, resolve_chain  # noqa: E402
from ai_video_editor.llm.client import StructuredLLMClient  # noqa: E402
from ai_video_editor.llm.registry import create_client, provider_names, register_provider  # noqa: E402
from ai_video_editor.llm.schema import (  # noqa: E402
    ImageDetail,
    ImageInput,
    LLMCallConfig,
    LLMCallMetrics,
    ModelSpec,
    ModelSwitch,
    StructuredResult,
    ThinkingLevel,
)

__all__ = [
    "ImageDetail",
    "ImageInput",
    "LLMCallConfig",
    "LLMCallMetrics",
    "ModelChain",
    "ModelSpec",
    "ModelSwitch",
    "StructuredLLMClient",
    "StructuredResult",
    "ThinkingLevel",
    "create_client",
    "parse_chain",
    "provider_names",
    "register_provider",
    "resolve_chain",
]
