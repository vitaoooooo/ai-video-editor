"""Revisor de IA com o Google Gemini: atalho para `LLMReviser` com um único modelo Gemini.

Reaproveita a configuração do Diretor: modelo, nível de raciocínio com fallback e timeout.
"""

from typing import Optional

from ai_video_editor.director.providers.gemini import DEFAULT_DIRECTOR_MODEL, gemini_chain
from ai_video_editor.director.schema import DirectorThinking
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.revision.prompt import build_reviser_prompt  # noqa: F401 — reexportado
from ai_video_editor.revision.providers.llm import LLMReviser


class GeminiReviser(LLMReviser):
    """Revisor usando um único modelo Gemini com `response_schema` = `RevisionDecision`."""

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
