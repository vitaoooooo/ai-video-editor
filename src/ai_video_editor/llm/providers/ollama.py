"""Provedor Ollama (modelos locais) pela API de chat oficial, com saída estruturada por JSON schema.

`POST /api/chat` com `stream=false`, o JSON schema do modelo Pydantic em `format` e
`options.num_ctx` sempre explícito. O cliente HTTP é o `urllib` da biblioteca padrão: são
três rotas JSON (`/api/chat`, `/api/tags`, `/api/version`), e o pacote `ollama` só traria
mais uma dependência (com o `httpx`) sem ganho aqui. Não há novas tentativas internas: elas
são feitas por `RetryPolicy`.

Configuração (a CLI tem `--ollama-host`, `--ollama-ctx` e `--ollama-keep-alive`, que gravam
estas variáveis para o processo):
- `OLLAMA_HOST`: endereço do servidor (padrão `http://localhost:11434`; a mesma variável do Ollama).
- `OLLAMA_CONTEXT_LENGTH`: `num_ctx` enviado em toda chamada (padrão 8192; a mesma variável
  com que o servidor define o contexto padrão).
- `OLLAMA_KEEP_ALIVE`: quanto tempo o modelo fica na memória depois da chamada (`10m`, `3600`,
  `-1`); sem ela, o parâmetro não é enviado e vale o padrão do servidor.
"""

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence, Union
from urllib.parse import urlsplit

from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import (
    AIContextLimitError,
    AIProviderError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.llm.client import StructuredLLMClient, as_int
from ai_video_editor.llm.errors import is_connection_error, redact
from ai_video_editor.llm.schema import ImageInput, LLMCallConfig, T, ThinkingLevel

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Estimativa para 6 GB de VRAM (RTX 2060): qwen2.5:7b Q4_K_M ocupa ~4,7 GB e o cache KV de
# 8192 tokens, ~0,45 GB. A medir com `ollama-check` e `ollama ps`.
DEFAULT_NUM_CTX = 8192
# Modelo local numa GPU de 6 GB é bem mais lento que a nuvem (carga do modelo + imagens).
DEFAULT_OLLAMA_TIMEOUT_SEC = 600.0
# Entrada a menos de tantos tokens do `num_ctx` é tratada como cortada pelo Ollama.
CONTEXT_MARGIN_TOKENS = 32

HOST_ENV = "OLLAMA_HOST"
NUM_CTX_ENV = "OLLAMA_CONTEXT_LENGTH"
KEEP_ALIVE_ENV = "OLLAMA_KEEP_ALIVE"

KeepAlive = Union[str, int, float]
# Texto do 500 quando o processo do modelo (llama-server) cai no meio da chamada.
_RUNNER_CRASH = "process has terminated"


def normalize_host(host: str) -> str:
    """`localhost:11434`, `0.0.0.0` ou `http://x:1234/` → URL com esquema e porta, sem barra no fim.

    `0.0.0.0` é o endereço em que o servidor escuta, não um destino válido (no Windows a
    conexão falha): vira `127.0.0.1`.
    """
    host = host.strip() or DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    parts = urlsplit(host)
    hostname = parts.hostname or "localhost"
    if hostname == "0.0.0.0":
        hostname = "127.0.0.1"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parts.port or (443 if parts.scheme == "https" else 11434)
    return f"{parts.scheme}://{hostname}:{port}{parts.path.rstrip('/')}"


def resolve_host(host: Optional[str] = None) -> str:
    return normalize_host(host or os.environ.get(HOST_ENV, "") or DEFAULT_OLLAMA_HOST)


def resolve_num_ctx(num_ctx: Optional[int] = None) -> int:
    """`num_ctx` explícito, o de `OLLAMA_CONTEXT_LENGTH` ou o padrão."""
    if num_ctx is not None:
        value = num_ctx
    else:
        text = os.environ.get(NUM_CTX_ENV, "").strip()
        if not text:
            return DEFAULT_NUM_CTX
        try:
            value = int(text)
        except ValueError:
            raise ValueError(f"{NUM_CTX_ENV} deve ser um número inteiro de tokens, recebido {text!r}")
    if value < 256:
        raise ValueError(f"contexto do Ollama muito pequeno: {value} tokens (mínimo 256)")
    return value


def resolve_keep_alive(keep_alive: Optional[KeepAlive] = None) -> Optional[KeepAlive]:
    """`keep_alive` explícito ou de `OLLAMA_KEEP_ALIVE`; número em texto vira número (segundos)."""
    value = keep_alive if keep_alive is not None else os.environ.get(KEEP_ALIVE_ENV, "").strip() or None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    return value


class OllamaHTTPError(Exception):
    """Resposta HTTP de erro do servidor Ollama (`{"error": "..."}`), com o modelo pedido."""

    def __init__(self, status_code: int, message: str, model: str = ""):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.model = model


class OllamaTransport:
    """Chamadas JSON ao servidor Ollama pelo `urllib` (sem novas tentativas internas)."""

    def __init__(self, host: str):
        self.host = host

    def request(self, method: str, path: str, body: Optional[dict] = None, timeout: float = 30.0) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self.host}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read().decode("utf-8")).get("error") or e.reason
            except Exception:
                message = str(e.reason)
            raise OllamaHTTPError(e.code, str(message), model=(body or {}).get("model", "")) from None
        except urllib.error.URLError as e:
            # Sem resposta HTTP: sobe a causa (conexão recusada, tempo esgotado, DNS...).
            reason = e.reason
            raise (reason if isinstance(reason, Exception) else ConnectionError(str(reason))) from None


def _is_refused(error: BaseException) -> bool:
    text = str(error).lower()
    return (
        isinstance(error, ConnectionRefusedError)
        or getattr(error, "winerror", None) == 10061
        or "connection refused" in text
        or "actively refused" in text
        or "recusou" in text
    )


def _is_timeout(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, TimeoutError) or "timed out" in text or "timeout" in type(error).__name__.lower()


def _is_missing_model(error: OllamaHTTPError) -> bool:
    text = error.message.lower()
    return error.status_code == 404 or ("not found" in text and "model" in text) or "try pulling" in text


def map_ollama_error(error: Exception, timeout_sec: float, host: str = DEFAULT_OLLAMA_HOST) -> AIProviderError:
    """Converte a falha da chamada ao Ollama em exceção de domínio, com a saída em português.

    - Conexão recusada: o Ollama não está rodando (não transitório; não troca de modelo).
    - Modelo não baixado (404): `ollama pull <modelo>` (não transitório).
    - Tempo esgotado e servidor ocupado (503) ou queda de conexão: transitório.
    - Outros erros HTTP (400, 500 por falta de memória...): permanentes, com o detalhe do servidor.
    """
    detail = redact(str(error))

    def build(exc: AIProviderError, transient: bool, code: Optional[int] = None) -> AIProviderError:
        exc.details = detail
        exc.transient = transient
        exc.status_code = code
        return exc

    if isinstance(error, AIProviderError):
        return error
    if isinstance(error, OllamaHTTPError):
        code = error.status_code
        if _is_missing_model(error):
            model = error.model or "<modelo>"
            return build(AIProviderError(
                f"O modelo '{model}' não está baixado no Ollama: rode `ollama pull {model}`. "
                f"Detalhe do servidor: {redact(error.message)}", provider="ollama",
            ), False, code)
        if code == 503:
            err = AIUnavailableError("ollama")
            err.args = (f"[OLLAMA] Servidor Ollama ocupado (503). Detalhe do servidor: {redact(error.message)}",)
            return build(err, True, code)
        if code >= 500 and _RUNNER_CRASH in error.message.lower():
            # O processo do modelo caiu (visto num erro de CUDA ao carregar); a próxima chamada o reinicia.
            err = AIUnavailableError("ollama")
            err.args = (f"[OLLAMA] O processo do modelo no Ollama caiu ({code}). Detalhe do servidor: "
                        f"{redact(error.message)}",)
            return build(err, True, code)
        return build(AIProviderError(
            f"O Ollama recusou o pedido ({code}). Detalhe do servidor: {redact(error.message)}", provider="ollama",
        ), False, code)
    if _is_refused(error):
        err = AIUnavailableError("ollama")
        err.args = (
            f"[OLLAMA] Ollama não está rodando em {host}: abra o Ollama ou rode `ollama serve`. Detalhe: {detail}",
        )
        return build(err, False)
    if _is_timeout(error):
        return build(AITimeoutError("ollama", timeout_sec=timeout_sec), True)
    if is_connection_error(error):
        err = AIUnavailableError("ollama")
        err.args = (f"[OLLAMA] Falha de conexão com o Ollama em {host}. Detalhe: {detail}",)
        return build(err, True)
    return build(AIProviderError(f"Erro na chamada do Ollama. Detalhe: {detail}", provider="ollama"), False)


def build_messages(prompt: str, images: Sequence[ImageInput]) -> list[dict[str, Any]]:
    """Mensagens do usuário: o prompt e, para cada imagem, o rótulo com a imagem em base64.

    Na API do Ollama as imagens são uma lista na mensagem (`images`), sem posição no texto;
    para cada rótulo ficar junto da sua imagem (a regra dos outros provedores), cada imagem
    vai numa mensagem do usuário própria, cujo texto é o `caption`.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for image in images:
        messages.append({
            "role": "user",
            "content": image.caption,
            "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        })
    return messages


def json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema do modelo Pydantic, como o `format` do Ollama espera."""
    return schema.model_json_schema()


class OllamaLLMClient(StructuredLLMClient):
    """Ollama local: `/api/chat` com `format` = JSON schema e `num_ctx` explícito.

    `client` (testes) substitui o transporte HTTP: um objeto com
    `request(method, path, body, timeout) -> dict`.
    """

    provider_name = "ollama"
    api_key_env = ""
    default_model = DEFAULT_OLLAMA_MODEL
    default_timeout_sec = DEFAULT_OLLAMA_TIMEOUT_SEC
    model_ids_have_colon = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Any = None,
        host: Optional[str] = None,
        num_ctx: Optional[int] = None,
        keep_alive: Optional[KeepAlive] = None,
    ):
        super().__init__(api_key=api_key, client=client)
        self.host = resolve_host(host)
        self.num_ctx = resolve_num_ctx(num_ctx)
        self.keep_alive = resolve_keep_alive(keep_alive)

    def has_credentials(self) -> bool:
        return True

    def _create_client(self) -> Any:
        return OllamaTransport(self.host)

    def thinking_setting(self, model_id: str, level: ThinkingLevel) -> Optional[tuple[str, Any]]:
        return None

    def thinking_ladder(self, model_id: str, thinking: ThinkingLevel) -> list[Optional[ThinkingLevel]]:
        return [None]

    def _note(self, text: str) -> None:
        metrics = self.last_metrics
        if metrics is not None and text not in metrics.notes:
            metrics.notes.append(text)

    def build_payload(
        self, prompt: str, schema: type[BaseModel], config: LLMCallConfig, images: Sequence[ImageInput] = ()
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model_id,
            "messages": build_messages(prompt, images) if images else [{"role": "user", "content": prompt}],
            "format": json_schema(schema),
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": config.temperature},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _request(
        self,
        prompt: str,
        schema: type[BaseModel],
        config: LLMCallConfig,
        level: Optional[ThinkingLevel],
        images: Sequence[ImageInput] = (),
    ) -> Any:
        self._note(f"num_ctx={self.num_ctx}")
        details = sorted({image.detail for image in images} - {"auto"})
        if details:
            self._note(f"detalhe da imagem {'/'.join(details)} ignorado: o Ollama não tem esse parâmetro")
        payload = self.build_payload(prompt, schema, config, images)
        response = self.get_client().request("POST", "/api/chat", payload, timeout=config.timeout_sec)
        if not isinstance(response, dict):
            raise AIResponseValidationError("ollama", details="resposta do servidor não é um objeto JSON")
        return response

    def _usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        return as_int(response.get("prompt_eval_count")), as_int(response.get("eval_count"))

    def check_context(self, response: dict) -> None:
        """Erro claro se a entrada (ou a entrada + a resposta) encostou no `num_ctx`.

        O Ollama corta o começo da entrada que passa do contexto sem devolver erro: a conta
        dos tokens informados é o único sinal. (Com o prefixo em cache, `prompt_eval_count`
        conta só a parte nova, então um corte pode passar despercebido; nunca o contrário.)

        Raises:
            AIContextLimitError: não transitório.
        """
        tokens_in, tokens_out = self._usage(response)
        if tokens_in is None:
            return
        limit = self.num_ctx - CONTEXT_MARGIN_TOKENS
        advice = (
            f"Aumente o contexto (--ollama-ctx ou {NUM_CTX_ENV}; cabe na memória da GPU?) "
            "ou reduza a entrada (menos frases ou imagens por chamada)."
        )
        if tokens_in >= limit:
            raise AIContextLimitError(
                "ollama",
                details=f"Entrada de {tokens_in} tokens com num_ctx={self.num_ctx}: o Ollama cortou o começo. {advice}",
            )
        if tokens_out is not None and tokens_in + tokens_out >= limit:
            raise AIContextLimitError(
                "ollama",
                details=(
                    f"Entrada de {tokens_in} + resposta de {tokens_out} tokens encheram o num_ctx={self.num_ctx}: "
                    f"a resposta pode ter sido cortada. {advice}"
                ),
            )

    def _parse(self, response: Any, schema: type[T]) -> T:
        self.check_context(response)
        load_ns = response.get("load_duration")
        if isinstance(load_ns, (int, float)) and load_ns >= 1e8:
            self._note(f"carga do modelo {load_ns / 1e9:.1f} s".replace(".", ","))
        content =(response.get("message") or {}).get("content")
        if not content:
            raise AIResponseValidationError("ollama", details="resposta sem conteúdo")
        try:
            return schema.model_validate(json.loads(content))
        except Exception as e:
            raise AIResponseValidationError("ollama", details=f"JSON inválido do modelo: {redact(str(e))}") from e

    def map_error(self, error: Exception, timeout_sec: float) -> AIProviderError:
        return map_ollama_error(error, timeout_sec, self.host)

    # --- diagnóstico (`ollama-check`) ---

    def version(self, timeout: float = 10.0) -> str:
        return str(self.get_client().request("GET", "/api/version", timeout=timeout).get("version", "?"))

    def installed_models(self, timeout: float = 10.0) -> list[str]:
        data = self.get_client().request("GET", "/api/tags", timeout=timeout)
        return [str(m.get("name") or m.get("model")) for m in data.get("models", [])]
