"""Provedor Google Gemini (SDK `google-genai`) com saída JSON estruturada."""

import json
from typing import Any, Optional, Sequence

from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm.client import StructuredLLMClient, as_int
from ai_video_editor.llm.errors import is_connection_error, redact, retry_after_sec, status_code
from ai_video_editor.llm.keys import resolve_api_key
from ai_video_editor.llm.schema import ImageInput, LLMCallConfig, T, ThinkingLevel

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Família 3.x: `thinking_level`. Família 2.5 (legado): `thinking_budget` em tokens (-1 = dinâmico).
_THINKING_LEVELS = {"off": "MINIMAL", "low": "LOW", "high": "HIGH"}
_THINKING_BUDGETS = {"off": 0, "low": 1024, "high": -1}
# Detalhe da imagem → `media_resolution` (docs "Media resolution": Gemini 3 cobra 280 tokens por
# imagem em LOW e 1120 em HIGH, que é o padrão).
_MEDIA_RESOLUTIONS = {"low": "MEDIA_RESOLUTION_LOW", "high": "MEDIA_RESOLUTION_HIGH"}


def resolve_gemini_api_key() -> str:
    """Chave do Gemini: variável `GEMINI_API_KEY` ou `.env` local."""
    return resolve_api_key("GEMINI_API_KEY", "gemini")


def build_http_options(timeout_sec: float):
    """HttpOptions do SDK com o tempo limite (o SDK espera milissegundos)."""
    from google.genai import types

    return types.HttpOptions(timeout=int(timeout_sec * 1000))


def map_gemini_error(error: Exception, timeout_sec: float) -> AIProviderError:
    """Converte uma exceção do SDK em exceção de domínio, sem expor credenciais.

    A mensagem inclui o detalhe original da API; `transient` indica se vale repetir
    (503/500/504, 429 por minuto, conexão e timeout) ou não (autenticação, 400, 404).
    Cota diária esgotada não é repetida, mas fica marcada em `quota_exhausted`.
    """
    text = str(error)
    lower = text.lower()
    detail = redact(text)
    code = status_code(error)

    def build(exc: AIProviderError, transient: bool) -> AIProviderError:
        exc.args = (f"{exc.args[0]} Detalhe da API: {detail}",)
        exc.details = detail
        exc.transient = transient
        exc.status_code = code
        exc.retry_after_sec = retry_after_sec(error) if transient else None
        return exc

    if code in (401, 403) or "unauthorized" in lower or "permission" in lower or "api_key" in lower:
        return build(AIAuthenticationError("gemini"), False)
    if code in (400, 404):
        return build(AIProviderError("Pedido recusado pelo Gemini.", provider="gemini"), False)
    if code == 429 or "quota" in lower or "rate limit" in lower or "resource_exhausted" in lower:
        # Cota diária esgotada não volta em segundos: repetir só atrasaria o erro.
        daily = "perday" in lower.replace(" ", "").replace("_", "")
        err = build(AIRateLimitError("gemini"), not daily)
        err.quota_exhausted = daily
        return err
    if code in (500, 502, 503, 504) or "unavailable" in lower or "deadline" in lower or "overloaded" in lower:
        return build(AIUnavailableError("gemini"), True)
    if "timeout" in lower or "timed out" in lower:
        return build(AITimeoutError("gemini", timeout_sec=timeout_sec), True)
    if is_connection_error(error):
        err = AIUnavailableError("gemini")
        err.args = ("[GEMINI] Falha de conexão com o serviço 'gemini'.",)
        return build(err, True)
    return build(AIProviderError("Erro na chamada do Gemini.", provider="gemini"), False)


def build_contents(prompt: str, images: Sequence[ImageInput]):
    """Uma mensagem do usuário: o prompt e, para cada imagem, o rótulo seguido dos bytes inline.

    Formato da documentação oficial ("Image understanding", imagens inline com
    `types.Part.from_bytes`); cada pedido inline tem limite de 20 MB no total.
    """
    from google.genai import types

    parts = [types.Part.from_text(text=prompt)]
    for image in images:
        parts.append(types.Part.from_text(text=image.caption))
        parts.append(types.Part.from_bytes(data=image.read_bytes(), mime_type=image.mime_type))
    return [types.Content(role="user", parts=parts)]


def media_resolution(images: Sequence[ImageInput]) -> Optional[str]:
    """`media_resolution` da chamada: o maior detalhe pedido entre as imagens (None = padrão do modelo)."""
    details = {image.detail for image in images} - {"auto"}
    if not details:
        return None
    return _MEDIA_RESOLUTIONS["high" if "high" in details else "low"]


def is_thinking_level_unsupported(error: Exception) -> bool:
    """True se o erro do SDK é a recusa 400 do nível de raciocínio pelo modelo."""
    text = str(error).lower()
    return "thinking level" in text and "not supported" in text and (
        "400" in text or "invalid_argument" in text
    )


class GeminiLLMClient(StructuredLLMClient):
    """Gemini com `response_schema` = schema Pydantic e `http_options` com o tempo limite."""

    provider_name = "gemini"
    api_key_env = "GEMINI_API_KEY"
    default_model = DEFAULT_GEMINI_MODEL
    default_fallback = ("gemini-3.6-flash", "gemini-3.5-flash-lite")

    def _create_client(self) -> Any:
        from google import genai

        return genai.Client(api_key=self._api_key or resolve_gemini_api_key())

    def thinking_setting(self, model_id: str, level: ThinkingLevel) -> tuple[str, Any]:
        if "gemini-2.5" in model_id:
            return "thinking_budget", _THINKING_BUDGETS[level]
        return "thinking_level", _THINKING_LEVELS[level]

    def thinking_ladder(self, model_id: str, thinking: ThinkingLevel) -> list[Optional[ThinkingLevel]]:
        # Só a família 3.x tem níveis; a 2.5 usa orçamento e não sobe de nível.
        if "gemini-2.5" in model_id:
            return [thinking]
        return super().thinking_ladder(model_id, thinking)

    def is_thinking_refused(self, error: Exception) -> bool:
        return is_thinking_level_unsupported(error)

    def build_thinking_config(self, model_id: str, level: ThinkingLevel):
        from google.genai import types

        name, value = self.thinking_setting(model_id, level)
        return types.ThinkingConfig(**{name: value})

    def _request(
        self,
        prompt: str,
        schema: type[BaseModel],
        config: LLMCallConfig,
        level: Optional[ThinkingLevel],
        images: Sequence[ImageInput] = (),
    ) -> Any:
        from google.genai import types

        extra: dict[str, Any] = {}
        resolution = media_resolution(images)
        if resolution is not None:
            extra["media_resolution"] = resolution
        return self.get_client().models.generate_content(
            model=config.model_id,
            contents=build_contents(prompt, images) if images else prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=config.temperature,
                thinking_config=self.build_thinking_config(config.model_id, level or config.thinking),
                http_options=build_http_options(config.timeout_sec),
                **extra,
            ),
        )

    def _usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        usage = getattr(response, "usage_metadata", None)
        return (
            as_int(getattr(usage, "prompt_token_count", None)),
            as_int(getattr(usage, "candidates_token_count", None)),
        )

    def _parse(self, response: Any, schema: type[T]) -> T:
        try:
            return schema.model_validate(json.loads(response.text))
        except Exception as e:
            raise AIResponseValidationError("gemini", details=f"JSON inválido do modelo: {e}") from e

    def map_error(self, error: Exception, timeout_sec: float) -> AIProviderError:
        return map_gemini_error(error, timeout_sec)


def generate_structured(
    client,
    model_id: str,
    thinking: ThinkingLevel,
    prompt: str,
    schema: type[T],
    temperature: float,
    timeout_sec: float,
    retry: Optional[RetryPolicy] = None,
) -> tuple[T, Optional[str]]:
    """Atalho de uma chamada com um cliente do SDK já criado.

    Returns:
        (objeto validado, rótulo do nível efetivo ou None se o nível pedido foi aceito).
    """
    gemini = GeminiLLMClient(client=client)
    config = LLMCallConfig(model_id=model_id, thinking=thinking, temperature=temperature, timeout_sec=timeout_sec)
    result = gemini.generate(prompt, schema, config, retry)
    label = result.metrics.thinking
    return result.value, (label if label != gemini.thinking_label(model_id, thinking) else None)
