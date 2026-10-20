"""Utilitários comuns para mapear erros dos SDKs em exceções de domínio."""

import re
from typing import Optional

# Chaves do Gemini (AIza...), da OpenAI (sk-...) e parâmetros `key=` em URLs.
_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{16,}|(?i:key=)[^&\s'\"]+")
_STATUS_PATTERN = re.compile(r"\b([45]\d\d)\b")
_RETRY_AFTER_PATTERN = re.compile(r"retry(?:delay)?\W{1,6}(?:in\s+)?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_CONNECTION_TYPES = (
    "connecterror", "connecttimeout", "remoteprotocolerror", "readerror", "networkerror", "apiconnectionerror",
)
_CONNECTION_HINTS = (
    "connection", "server disconnected", "reset by peer", "temporary failure in name resolution",
    "name or service not known", "getaddrinfo", "eof occurred", "broken pipe",
)


def redact(text: str, limit: int = 600) -> str:
    """Detalhe original da API, sem chaves e em uma linha curta."""
    text = _KEY_PATTERN.sub("[chave omitida]", " ".join(text.split()))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def status_code(error: Exception) -> Optional[int]:
    """Código HTTP do erro: atributo `code`/`status_code` do SDK ou número no texto."""
    for attr in ("status_code", "code"):
        code = getattr(error, attr, None)
        if isinstance(code, int) and not isinstance(code, bool) and 100 <= code < 600:
            return code
    match = _STATUS_PATTERN.search(str(error))
    return int(match.group(1)) if match else None


def retry_after_sec(error: Exception) -> Optional[float]:
    """Retry-After do cabeçalho HTTP ou o `retryDelay` embutido no corpo do 429."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError, AttributeError):
            pass
    match = _RETRY_AFTER_PATTERN.search(str(error))
    return float(match.group(1)) if match else None


def is_connection_error(error: Exception) -> bool:
    """Falha de rede (sem resposta HTTP), pelo tipo da exceção ou pelo texto."""
    lower = str(error).lower()
    return (
        type(error).__name__.lower() in _CONNECTION_TYPES
        or isinstance(error, (ConnectionError, TimeoutError))
        or any(hint in lower for hint in _CONNECTION_HINTS)
    )
