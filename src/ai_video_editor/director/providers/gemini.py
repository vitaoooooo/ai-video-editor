"""Diretor de IA com o Google Gemini: atalho para `LLMDirector` com um único modelo Gemini.

A lógica do Gemini (saída estruturada, novas tentativas, raciocínio, timeout) está em
`ai_video_editor.llm.providers.gemini`.
"""

from typing import Optional

from ai_video_editor.director.prompt import build_director_prompt  # noqa: F401 — reexportado
from ai_video_editor.director.providers.llm import LLMDirector
from ai_video_editor.director.schema import DirectorThinking
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.providers.gemini import (  # noqa: F401 — reexportados
    DEFAULT_GEMINI_MODEL as DEFAULT_DIRECTOR_MODEL,
    GeminiLLMClient,
    generate_structured,
)
from ai_video_editor.llm.schema import ModelSpec


def gemini_chain(
    model_id: str,
    api_key: Optional[str],
    temperature: float,
    timeout_sec: float,
    thinking: DirectorThinking,
    client,
    retry: Optional[RetryPolicy],
) -> ModelChain:
    """Cadeia de um só modelo Gemini, com cliente do SDK opcionalmente injetado."""
    return ModelChain(
        [ModelSpec(provider="gemini", model_id=model_id)],
        thinking=thinking, temperature=temperature, timeout_sec=timeout_sec, retry=retry,
        clients={"gemini": GeminiLLMClient(api_key=api_key, client=client)},
    )


class GeminiDirector(LLMDirector):
    """Diretor usando um único modelo Gemini com `response_schema` = `DirectorDecision`."""

    def __init__(
        self,
        model_id: str = DEFAULT_DIRECTOR_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout_sec: float = 120.0,
        thinking: DirectorThinking = "low",
        client=None,
        retry: Optional[RetryPolicy] = None,
    ):
        super().__init__(gemini_chain(model_id, api_key, temperature, timeout_sec, thinking, client, retry))
