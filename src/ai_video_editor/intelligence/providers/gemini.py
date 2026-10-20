"""Adaptador para a API do Google Gemini utilizando o SDK oficial google-genai."""

import json
import os
import re
from pathlib import Path
from typing import Optional
from PIL import Image

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.provider import MultimodalAIProvider
from ai_video_editor.intelligence.retry import RetryPolicy, call_with_retry
from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    VisualAnalysisResponse,
)


def resolve_gemini_api_key() -> str:
    """Obtém a chave da API do Gemini prioritariamente de variável de ambiente.

    Suporta opcionalmente leitura de arquivo .env local sem expor chaves em logs.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # Suporte opcional a arquivo .env local (ignorado pelo Git)
    env_file = Path(".env")
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass

    raise AIAuthenticationError(
        "gemini",
        details="Variável de ambiente GEMINI_API_KEY não encontrada.",
    )


def build_http_options(timeout_sec: float):
    """HttpOptions do SDK com o tempo limite (o SDK espera milissegundos)."""
    from google.genai import types

    return types.HttpOptions(timeout=int(timeout_sec * 1000))


_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|(?i:key=)[^&\s'\"]+")
_STATUS_PATTERN = re.compile(r"\b([45]\d\d)\b")
_RETRY_AFTER_PATTERN = re.compile(r"retry(?:delay)?\W{1,6}(?:in\s+)?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_CONNECTION_TYPES = ("connecterror", "connecttimeout", "remoteprotocolerror", "readerror", "networkerror")
_CONNECTION_HINTS = (
    "connection", "server disconnected", "reset by peer", "temporary failure in name resolution",
    "name or service not known", "getaddrinfo", "eof occurred", "broken pipe",
)


def _redact(text: str, limit: int = 600) -> str:
    """Detalhe original da API, sem chaves e em uma linha curta."""
    text = _KEY_PATTERN.sub("[chave omitida]", " ".join(text.split()))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _status_code(error: Exception) -> Optional[int]:
    code = getattr(error, "code", None)
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code < 600:
        return code
    match = _STATUS_PATTERN.search(str(error))
    return int(match.group(1)) if match else None


def _retry_after_sec(error: Exception) -> Optional[float]:
    """Retry-After do cabeçalho HTTP ou o `retryDelay` que o Gemini embute no corpo do 429."""
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


def map_gemini_error(error: Exception, timeout_sec: float) -> AIProviderError:
    """Converte uma exceção do SDK em exceção de domínio, sem expor credenciais.

    A mensagem inclui o detalhe original da API; `transient` indica se vale repetir
    (503/500/504, 429 por minuto, conexão e timeout) ou não (autenticação, 400, 404).
    """
    text = str(error)
    lower = text.lower()
    detail = _redact(text)
    code = _status_code(error)

    def build(exc: AIProviderError, transient: bool) -> AIProviderError:
        exc.args = (f"{exc.args[0]} Detalhe da API: {detail}",)
        exc.details = detail
        exc.transient = transient
        exc.status_code = code
        exc.retry_after_sec = _retry_after_sec(error) if transient else None
        return exc

    if code in (401, 403) or "unauthorized" in lower or "permission" in lower or "api_key" in lower:
        return build(AIAuthenticationError("gemini"), False)
    if code in (400, 404):
        return build(AIProviderError("Pedido recusado pelo Gemini.", provider="gemini"), False)
    if code == 429 or "quota" in lower or "rate limit" in lower or "resource_exhausted" in lower:
        # Cota diária esgotada não volta em segundos: repetir só atrasaria o erro.
        daily = "perday" in lower.replace(" ", "").replace("_", "")
        return build(AIRateLimitError("gemini"), not daily)
    if code in (500, 502, 503, 504) or "unavailable" in lower or "deadline" in lower or "overloaded" in lower:
        return build(AIUnavailableError("gemini"), True)
    if "timeout" in lower or "timed out" in lower:
        return build(AITimeoutError("gemini", timeout_sec=timeout_sec), True)
    if (
        type(error).__name__.lower() in _CONNECTION_TYPES
        or isinstance(error, (ConnectionError, TimeoutError))
        or any(hint in lower for hint in _CONNECTION_HINTS)
    ):
        err = AIUnavailableError("gemini")
        err.args = ("[GEMINI] Falha de conexão com o serviço 'gemini'.",)
        return build(err, True)
    return build(AIProviderError("Erro na chamada do Gemini.", provider="gemini"), False)


def _as_int(value) -> Optional[int]:
    """Converte contagem de tokens do SDK para int, ignorando valores ausentes ou inesperados."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class GeminiMultimodalProvider(MultimodalAIProvider):
    """Adaptador de IA multimodal conectando ao Google Gemini via SDK oficial."""

    def __init__(self, api_key: Optional[str] = None, retry: Optional[RetryPolicy] = None):
        # A chave é resolvida sob demanda para permitir mockar o ambiente facilmente
        self._api_key = api_key
        self.retry = retry if retry is not None else RetryPolicy()

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        """Instancia o cliente do SDK google-genai de forma tardia (lazy)."""
        from google import genai

        key = self._api_key or resolve_gemini_api_key()
        return genai.Client(api_key=key)

    def analyze_frames(
        self,
        metadata: VideoMetadata,
        frames: list[ExtractedFrame],
        config: AIAnalysisConfig,
    ) -> VisualAnalysisResponse:
        """Envia imagens e prompt estruturado ao Gemini, solicitando resposta em JSON Pydantic."""
        from google import genai
        from google.genai import types

        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_api_attempts = 1
        client = self._get_client()

        # Constrói instruções anti-alucinação estritas
        frame_manifest = "\n".join(
            f"- Frame #{f.frame_index}: timestamp {f.target_timestamp_sec:.2f}s (arquivo: {f.image_path.name})"
            for f in frames
        )

        prompt_text = (
            f"Você é um assistente especialista em análise técnica e visual de vídeo.\n"
            f"Analise a sequência de {len(frames)} frames amostrados do arquivo '{metadata.source_path.name}'.\n"
            f"Duração total do vídeo original: {metadata.duration_sec:.2f} segundos.\n\n"
            f"MANIFESTO DOS FRAMES ENVIADOS (use estritamente estes índices e timestamps):\n"
            f"{frame_manifest}\n\n"
            f"DIRETRIZES OBRIGATÓRIAS:\n"
            f"1. Para CADA frame da lista, preencha 'chronological_timeline' com 'frame_index' e 'target_timestamp_sec' exatos.\n"
            f"2. 'observable_content': descreva apenas o que é comprovadamente visível na imagem, sem suposições.\n"
            f"3. 'inferences': separe em hipóteses qualquer dedução de intenção, contexto narrativo ou acontecimentos não visíveis.\n"
            f"4. 'scene_transitions': se notar mudanças bruscas de conteúdo entre dois frames vizinhos, aponte a hipótese. "
            f"Como a amostragem é esparsa, 'is_confirmed_exact' DEVE ser false.\n"
            f"5. 'limitations_and_blindspots': liste os pontos cegos decorrentes dos intervalos não amostrados entre os frames.\n"
        )

        # Prepara as partes do conteúdo multimodal (texto + imagens PIL)
        contents = [prompt_text]
        for f in frames:
            img = Image.open(f.image_path)
            contents.append(f"\n[Imagem do Frame {f.frame_index} em {f.target_timestamp_sec:.2f}s]:")
            contents.append(img)

        self.retry.reset()
        try:
            response = call_with_retry(
                lambda: client.models.generate_content(
                    model=config.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VisualAnalysisResponse,
                        temperature=config.temperature,
                        http_options=build_http_options(config.timeout_sec),
                    ),
                ),
                self.retry,
                lambda e: map_gemini_error(e, config.timeout_sec),
            )
        finally:
            self.last_api_attempts = max(1, self.retry.attempts_made)

        usage = getattr(response, "usage_metadata", None)
        self.last_input_tokens = _as_int(getattr(usage, "prompt_token_count", None))
        self.last_output_tokens = _as_int(getattr(usage, "candidates_token_count", None))

        # Converte a resposta estruturada para o modelo Pydantic
        try:
            raw_text = response.text
            data = json.loads(raw_text)
            validated = VisualAnalysisResponse.model_validate(data)
            # Assegura que provider_used e model_used correspondam aos parâmetros executados
            validated.provider_used = "gemini"
            validated.model_used = config.model_id
            return validated
        except Exception as e:
            raise AIResponseValidationError("gemini", details=f"Falha ao validar JSON do modelo: {e}")
