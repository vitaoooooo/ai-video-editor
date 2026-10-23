"""Interface comum dos provedores de IA com saída estruturada (Diretor, revisor e futuros usos)."""

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import AIProviderError
from ai_video_editor.intelligence.retry import RetryPolicy, call_with_retry
from ai_video_editor.llm.keys import find_api_key
from ai_video_editor.llm.per_thread import PerThread
from ai_video_editor.llm.schema import ImageInput, LLMCallConfig, LLMCallMetrics, StructuredResult, T, ThinkingLevel

_THINKING_ORDER: tuple[ThinkingLevel, ...] = ("off", "low", "high")


class StructuredLLMClient(ABC):
    """Contrato de um provedor: prompt + schema Pydantic → objeto validado + métricas.

    `generate` já implementa o fluxo comum: novas tentativas em falhas temporárias
    (`RetryPolicy`), subida do nível de raciocínio quando a API recusa o pedido e medição
    de latência, tokens e chamadas. Cada provedor implementa só as partes específicas:
    a chamada ao SDK, a leitura da resposta, o mapeamento de erros e a tradução do
    raciocínio.

    Para um provedor novo: defina `provider_name`, `api_key_env`, `default_model`
    (e opcionalmente `default_fallback`), implemente os métodos abstratos e registre a
    classe em `ai_video_editor.llm.registry`.
    """

    provider_name: str = ""
    # Variável de ambiente da chave ("" = o provedor não precisa de chave).
    api_key_env: str = ""
    default_model: str = ""
    # Modelos de reserva do mesmo provedor quando o usuário não informa a cadeia.
    default_fallback: tuple[str, ...] = ()
    # Tempo limite padrão das chamadas quando o usuário não informa (None = o da etapa).
    default_timeout_sec: Optional[float] = None
    # True se os IDs dos modelos têm `:` (`qwen2.5:7b`), para a cadeia não ler a tag como provedor.
    model_ids_have_colon: bool = False
    # Métricas da última chamada desta thread (lotes do `visual` chamam em paralelo).
    last_metrics = PerThread(None)

    def __init__(self, api_key: Optional[str] = None, client: Any = None):
        self._api_key = api_key
        self._client = client
        self._client_lock = threading.Lock()

    # --- credenciais e cliente do SDK ---

    def has_credentials(self) -> bool:
        """True se há chave (ou cliente injetado); usado para pular o provedor na cadeia."""
        if self._client is not None or self._api_key or not self.api_key_env:
            return True
        return find_api_key(self.api_key_env) is not None

    def get_client(self) -> Any:
        """Cliente do SDK, criado sob demanda (import tardio) uma única vez, mesmo com threads.

        Sem a trava, threads paralelas (lotes do `visual`) criavam um cliente cada e
        sobrescreviam `_client`; o SDK (Gemini e OpenAI) fecha a conexão no `__del__` do
        cliente descartado, e a chamada da thread que ainda o usava falhava com
        "client has been closed".
        """
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = self._create_client()
            return self._client

    @abstractmethod
    def _create_client(self) -> Any:
        """Instancia o cliente do SDK com a chave resolvida."""
        ...

    # --- raciocínio ---

    @abstractmethod
    def thinking_setting(self, model_id: str, level: ThinkingLevel) -> Optional[tuple[str, Any]]:
        """(nome do parâmetro, valor) para o nível pedido, ou None se o modelo não tem raciocínio."""
        ...

    def thinking_ladder(self, model_id: str, thinking: ThinkingLevel) -> list[Optional[ThinkingLevel]]:
        """Níveis a tentar se a API recusar o pedido, do pedido para cima.

        None no fim significa "tentar sem o parâmetro de raciocínio".
        """
        return list(_THINKING_ORDER[_THINKING_ORDER.index(thinking):])

    def thinking_label(self, model_id: str, thinking: ThinkingLevel) -> str:
        """Descrição da configuração de raciocínio, para o relatório (ex: `low (thinking_level=LOW)`)."""
        setting = self.thinking_setting(model_id, thinking)
        if setting is None:
            return f"{thinking} (ignorado: o modelo não tem raciocínio configurável)"
        return f"{thinking} ({setting[0]}={setting[1]})"

    def is_thinking_refused(self, error: Exception) -> bool:
        """True se o erro do SDK é a recusa do nível de raciocínio pelo modelo."""
        return False

    # --- chamada ---

    @abstractmethod
    def _request(
        self,
        prompt: str,
        schema: type[BaseModel],
        config: LLMCallConfig,
        level: Optional[ThinkingLevel],
        images: Sequence[ImageInput] = (),
    ) -> Any:
        """Faz uma chamada ao SDK e devolve a resposta crua (exceções do SDK sobem como estão).

        Sem `images`, a chamada é exatamente a de só texto; com elas, o prompt vai primeiro e
        cada imagem vai inline, precedida do seu `caption` (formato oficial de cada SDK).
        """
        ...

    @abstractmethod
    def _usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        """Tokens (entrada, saída) da resposta; None se o provedor não informar."""
        ...

    @abstractmethod
    def _parse(self, response: Any, schema: type[T]) -> T:
        """Objeto validado no schema. Raises: AIResponseValidationError."""
        ...

    @abstractmethod
    def map_error(self, error: Exception, timeout_sec: float) -> AIProviderError:
        """Exceção de domínio com `transient`/`quota_exhausted`, sem expor credenciais."""
        ...

    def generate(
        self,
        prompt: str,
        schema: type[T],
        config: LLMCallConfig,
        retry: Optional[RetryPolicy] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> StructuredResult[T]:
        """Chama o modelo com saída no `schema` (e, opcionalmente, imagens junto com o prompt).

        Falhas temporárias (503, 504, 429 por minuto, conexão) são repetidas conforme `retry`;
        erros permanentes sobem na hora. Se a API recusar o nível de raciocínio, sobe de nível
        sem contar como nova tentativa. `last_metrics` fica preenchido mesmo em caso de erro
        (inclusive tokens, se a resposta chegou mas não pôde ser validada).

        Raises:
            FileNotFoundError: uma das imagens não existe (antes de qualquer chamada).
        """
        images = list(images or ())
        for image in images:
            if not image.path.is_file():
                raise FileNotFoundError(f"imagem não encontrada: {image.path}")
        retry = retry if retry is not None else RetryPolicy()
        retry.reset()
        metrics = LLMCallMetrics(
            provider=self.provider_name,
            model_id=config.model_id,
            thinking=self.thinking_label(config.model_id, config.thinking),
            images=len(images),
        )
        self.last_metrics = metrics
        started = time.perf_counter()
        refused: list[str] = []
        # Contadas aqui, não em `retry.attempts_made`: a política é compartilhada entre threads.
        attempts = 0
        try:
            for level in self.thinking_ladder(config.model_id, config.thinking):

                def call(level=level):
                    nonlocal attempts
                    attempts += 1
                    if images:
                        return self._request(prompt, schema, config, level, images)
                    return self._request(prompt, schema, config, level)

                try:
                    response = call_with_retry(call, retry, lambda e: self.map_error(e, config.timeout_sec))
                except AIProviderError as e:
                    if not self.is_thinking_refused(e.__cause__ or e):
                        raise
                    refused.append(self._level_value(config.model_id, level))
                    continue
                if refused:
                    used = self._level_value(config.model_id, level)
                    metrics.thinking = f"{config.thinking} → {used} ({', '.join(refused)} não suportado)"
                metrics.input_tokens, metrics.output_tokens = self._usage(response)
                return StructuredResult(value=self._parse(response, schema), metrics=metrics)
            raise AIProviderError(
                f"O modelo '{config.model_id}' recusou todos os níveis de raciocínio tentados "
                f"({', '.join(refused)}).",
                provider=self.provider_name,
            )
        finally:
            metrics.api_calls = max(1, attempts)
            metrics.latency_sec = round(time.perf_counter() - started, 3)

    def _level_value(self, model_id: str, level: Optional[ThinkingLevel]) -> str:
        if level is None:
            return "sem raciocínio"
        setting = self.thinking_setting(model_id, level)
        return str(setting[1]) if setting else "sem raciocínio"


def as_int(value) -> Optional[int]:
    """Inteiro informado pelo SDK, ou None (ignora mocks e tipos inesperados)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
