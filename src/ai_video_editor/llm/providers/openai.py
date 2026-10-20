"""Provedor OpenAI (SDK oficial `openai`, Responses API) com saída estruturada por JSON schema.

Usa `client.responses.parse(text_format=<schema Pydantic>)`, que envia o schema como
`text.format` do tipo `json_schema` (modo estrito) e devolve o objeto em `output_parsed`.
Instale com `pip install -e ".[openai]"`.
"""

import re
from typing import Any, Optional

from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.llm.client import StructuredLLMClient, as_int
from ai_video_editor.llm.errors import is_connection_error, redact, retry_after_sec, status_code
from ai_video_editor.llm.keys import resolve_api_key
from ai_video_editor.llm.schema import LLMCallConfig, T, ThinkingLevel

# Modelo de texto mais econômico indicado na documentação oficial (developers.openai.com/api/docs/models).
DEFAULT_OPENAI_MODEL = "gpt-6-luna"

# `reasoning.effort`. "none" desliga o raciocínio (nem todo modelo aceita; aí sobe de nível).
_EFFORTS = {"off": "none", "low": "low", "high": "high"}
# Erros 429 de cobrança/cota: repetir não devolve o acesso (docs: "Error codes").
_QUOTA_CODES = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}
# Modelos com raciocínio configurável: série o (o1, o3, o4-mini...) e gpt-5 em diante.
_REASONING_MODEL = re.compile(r"^(o\d|gpt-(\d+))")


def resolve_openai_api_key() -> str:
    """Chave da OpenAI: variável `OPENAI_API_KEY` ou `.env` local."""
    return resolve_api_key("OPENAI_API_KEY", "openai")


def supports_reasoning(model_id: str) -> bool:
    match = _REASONING_MODEL.match(model_id.lower())
    if match is None:
        return False
    return match.group(2) is None or int(match.group(2)) >= 5


def map_openai_error(error: Exception, timeout_sec: float) -> AIProviderError:
    """Converte uma exceção do SDK `openai` em exceção de domínio, sem expor credenciais.

    Temporários (repetidos): 429 por limite de requisições, 5xx, timeout e conexão.
    Permanentes: autenticação (401/403), requisição inválida (400/422), modelo
    inexistente (404) e cota/cobrança esgotada (429 com código de cota, marcado em
    `quota_exhausted`).
    """
    text = str(error)
    detail = redact(text)
    code = status_code(error)
    name = type(error).__name__
    error_code = str(getattr(error, "code", "") or "").lower()
    error_type = str(getattr(error, "type", "") or "").lower()

    def build(exc: AIProviderError, transient: bool) -> AIProviderError:
        exc.args = (f"{exc.args[0]} Detalhe da API: {detail}",)
        exc.details = detail
        exc.transient = transient
        exc.status_code = code
        exc.retry_after_sec = retry_after_sec(error) if transient else None
        return exc

    # O SDK valida a saída estruturada ao montar a resposta: falha aqui é resposta inválida.
    if name in ("ValidationError", "JSONDecodeError", "LengthFinishReasonError", "ContentFilterFinishReasonError"):
        return AIResponseValidationError("openai", details=f"JSON inválido do modelo: {detail}")
    if "Timeout" in name:
        return build(AITimeoutError("openai", timeout_sec=timeout_sec), True)
    if code in (401, 403) or name in ("AuthenticationError", "PermissionDeniedError"):
        return build(AIAuthenticationError("openai"), False)
    if code == 429 or name == "RateLimitError":
        quota = error_code in _QUOTA_CODES or error_type in _QUOTA_CODES
        err = build(AIRateLimitError("openai"), not quota)
        err.quota_exhausted = quota
        return err
    if code == 404:
        return build(AIProviderError("Modelo ou recurso inexistente na OpenAI.", provider="openai"), False)
    if code in (400, 409, 422):
        return build(AIProviderError("Pedido recusado pela OpenAI.", provider="openai"), False)
    if (code is not None and code >= 500) or name == "InternalServerError":
        return build(AIUnavailableError("openai"), True)
    if code is None and is_connection_error(error):
        err = AIUnavailableError("openai")
        err.args = ("[OPENAI] Falha de conexão com o serviço 'openai'.",)
        return build(err, True)
    return build(AIProviderError("Erro na chamada da OpenAI.", provider="openai"), False)


def is_reasoning_refused(error: Exception) -> bool:
    """True se a API recusou o esforço de raciocínio (ou o parâmetro) para o modelo."""
    text = str(error).lower()
    return (
        status_code(error) == 400
        and "reasoning" in text
        and any(hint in text for hint in ("not supported", "unsupported", "does not support", "invalid value"))
    )


class OpenAILLMClient(StructuredLLMClient):
    """OpenAI pela Responses API com `text_format` = schema Pydantic."""

    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    default_model = DEFAULT_OPENAI_MODEL

    def _create_client(self) -> Any:
        from openai import OpenAI

        # As novas tentativas são feitas por `RetryPolicy`, não pelo SDK.
        return OpenAI(api_key=self._api_key or resolve_openai_api_key(), max_retries=0)

    def thinking_setting(self, model_id: str, level: ThinkingLevel) -> Optional[tuple[str, Any]]:
        if not supports_reasoning(model_id):
            return None
        return "reasoning.effort", _EFFORTS[level]

    def thinking_ladder(self, model_id: str, thinking: ThinkingLevel) -> list[Optional[ThinkingLevel]]:
        if not supports_reasoning(model_id):
            return [None]
        # Se nenhum esforço for aceito, a última tentativa vai sem o parâmetro.
        return [*super().thinking_ladder(model_id, thinking), None]

    def is_thinking_refused(self, error: Exception) -> bool:
        return is_reasoning_refused(error)

    def _request(self, prompt: str, schema: type[BaseModel], config: LLMCallConfig, level: Optional[ThinkingLevel]) -> Any:
        kwargs: dict[str, Any] = {
            "model": config.model_id,
            "input": prompt,
            "text_format": schema,
            "timeout": config.timeout_sec,
        }
        setting = self.thinking_setting(config.model_id, level) if level is not None else None
        if setting is not None:
            kwargs["reasoning"] = {"effort": setting[1]}
        # Modelos de raciocínio só aceitam `temperature` com o raciocínio desligado.
        if setting is None or setting[1] == "none":
            kwargs["temperature"] = config.temperature
        return self.get_client().responses.parse(**kwargs)

    def _usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        usage = getattr(response, "usage", None)
        return as_int(getattr(usage, "input_tokens", None)), as_int(getattr(usage, "output_tokens", None))

    def _parse(self, response: Any, schema: type[T]) -> T:
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) == "refusal":
                    raise AIResponseValidationError(
                        "openai", details=f"o modelo recusou responder: {getattr(content, 'refusal', '')}"
                    )
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None) or "desconhecido"
            raise AIResponseValidationError("openai", details=f"resposta incompleta ({reason})")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise AIResponseValidationError("openai", details="resposta sem conteúdo estruturado")
        try:
            return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
        except Exception as e:
            raise AIResponseValidationError("openai", details=f"JSON inválido do modelo: {e}") from e

    def map_error(self, error: Exception, timeout_sec: float) -> AIProviderError:
        return map_openai_error(error, timeout_sec)
