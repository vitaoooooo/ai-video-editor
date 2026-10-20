"""Cadeia ordenada de modelos (`provedor:modelo`) com troca automática quando um fica indisponível."""

from typing import Callable, Optional

from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm.client import StructuredLLMClient
from ai_video_editor.llm.registry import create_client, default_fallback, default_model, provider_names
from ai_video_editor.llm.schema import (
    LLMCallConfig,
    LLMCallMetrics,
    ModelSpec,
    ModelSwitch,
    StructuredResult,
    T,
    ThinkingLevel,
)

NO_FALLBACK = "none"


def parse_model_spec(text: str, default_provider: Optional[str] = None) -> ModelSpec:
    """`provedor:modelo` (ou só `modelo`, se `default_provider` for dado)."""
    text = text.strip()
    provider, sep, model = text.partition(":")
    if not sep:
        if default_provider is None:
            raise ValueError(f"esperado `provedor:modelo`, recebido {text!r}")
        provider, model = default_provider, text
    provider, model = provider.strip().lower(), model.strip()
    if provider not in provider_names():
        raise ValueError(f"provedor de IA desconhecido: {provider!r} (disponíveis: {', '.join(provider_names())})")
    if not model:
        raise ValueError(f"modelo vazio em {text!r}")
    return ModelSpec(provider=provider, model_id=model)


def parse_chain(text: str, default_provider: Optional[str] = None) -> list[ModelSpec]:
    """Lista `provedor:modelo,...`; `none` (ou vazio) devolve lista vazia."""
    if text.strip().lower() in ("", NO_FALLBACK):
        return []
    return [parse_model_spec(item, default_provider) for item in text.split(",") if item.strip()]


def resolve_chain(provider: str, model: Optional[str] = None, fallback: Optional[str] = None) -> list[ModelSpec]:
    """Cadeia completa: o modelo principal seguido das reservas, sem repetições.

    Args:
        provider: Provedor principal.
        model: Modelo principal (padrão: o do provedor).
        fallback: `provedor:modelo,...`, `none`, ou None para a reserva padrão do provedor.
    """
    primary = ModelSpec(provider=provider, model_id=model or default_model(provider))
    if fallback is None:
        rest = [ModelSpec(provider=provider, model_id=m) for m in default_fallback(provider)]
    else:
        rest = parse_chain(fallback, default_provider=provider)
    chain: list[ModelSpec] = []
    for spec in [primary, *rest]:
        if spec not in chain:
            chain.append(spec)
    return chain


def chain_label(chain: list[ModelSpec]) -> str:
    return " → ".join(spec.label for spec in chain)


def is_switchable(error: AIProviderError) -> bool:
    """Vale trocar de modelo: falha temporária que persistiu ou cota esgotada daquele modelo.

    Autenticação e requisição inválida (400/404) não trocam: o problema não é do modelo.
    """
    return error.transient or error.quota_exhausted


def describe_failure(error: AIProviderError) -> str:
    """Motivo curto da troca, ex: `indisponível (503) após 5 tentativas`."""
    code = error.status_code
    if error.quota_exhausted:
        reason = f"com a cota esgotada ({code})" if code else "com a cota esgotada"
    elif isinstance(error, AIRateLimitError):
        reason = f"com limite de requisições ({code})" if code else "com limite de requisições"
    elif isinstance(error, AITimeoutError) or code == 504:
        reason = f"demorou demais para responder ({code})" if code else "demorou demais para responder"
    elif code is None:
        reason = "sem conexão"
    else:
        reason = f"indisponível ({code})"
    if error.api_attempts > 1:
        reason += f" após {error.api_attempts} tentativas"
    return reason


class ModelChain:
    """Executa uma chamada estruturada na cadeia de modelos.

    Usa o modelo atual; se ele falhar com erro que permite troca (`is_switchable`), avisa
    por `on_event` e passa ao próximo. A troca é persistente: chamadas seguintes (ex: a
    correção de uma resposta inválida) continuam no modelo que funcionou. Provedores sem
    chave são pulados, com um aviso por provedor.
    """

    def __init__(
        self,
        specs: list[ModelSpec],
        thinking: ThinkingLevel = "low",
        temperature: float = 0.2,
        timeout_sec: float = 120.0,
        retry: Optional[RetryPolicy] = None,
        on_event: Optional[Callable[[str], None]] = None,
        clients: Optional[dict[str, StructuredLLMClient]] = None,
    ):
        if not specs:
            raise ValueError("a cadeia de modelos está vazia")
        self.specs = list(specs)
        self.thinking = thinking
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.retry = retry if retry is not None else RetryPolicy()
        self.on_event = on_event
        self._clients: dict[str, StructuredLLMClient] = dict(clients or {})
        self._index = 0
        self._warned_keys: set[str] = set()
        self.switches: list[ModelSwitch] = []
        self.last_metrics: Optional[LLMCallMetrics] = None

    @property
    def requested(self) -> ModelSpec:
        return self.specs[0]

    @property
    def current(self) -> ModelSpec:
        return self.specs[self._index]

    @property
    def labels(self) -> list[str]:
        return [spec.label for spec in self.specs]

    def client_for(self, provider: str) -> StructuredLLMClient:
        if provider not in self._clients:
            self._clients[provider] = create_client(provider)
        return self._clients[provider]

    def thinking_label(self, spec: Optional[ModelSpec] = None) -> str:
        spec = spec or self.requested
        return self.client_for(spec.provider).thinking_label(spec.model_id, self.thinking)

    def _notify(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)

    def _usable(self, index: int) -> bool:
        """True se o provedor do elo tem chave; avisa uma vez por provedor se não tiver."""
        spec = self.specs[index]
        client = self.client_for(spec.provider)
        if client.has_credentials():
            return True
        if spec.provider not in self._warned_keys:
            self._warned_keys.add(spec.provider)
            self._notify(f"{spec.provider} sem chave configurada ({client.api_key_env}) → pulando {spec.label}")
        return False

    def _next_usable(self, start: int) -> Optional[int]:
        for index in range(start, len(self.specs)):
            if self._usable(index):
                return index
        return None

    def _missing_key_error(self) -> AIAuthenticationError:
        spec = self.specs[self._index]
        client = self.client_for(spec.provider)
        detail = f"Variável de ambiente {client.api_key_env} não encontrada."
        if len({s.provider for s in self.specs}) > 1:
            envs = sorted({self.client_for(s.provider).api_key_env for s in self.specs[self._index:]})
            detail = f"Nenhum provedor da cadeia tem chave configurada ({', '.join(envs)})."
        return AIAuthenticationError(spec.provider, details=detail)

    def _switch_label(self, frm: ModelSpec, to: ModelSpec) -> str:
        return to.model_id if to.provider == frm.provider else to.label

    def generate(self, prompt: str, schema: type[T]) -> StructuredResult[T]:
        """Chama o modelo atual da cadeia, trocando de modelo quando necessário.

        `last_metrics` resume a chamada (modelo efetivo, chamadas somadas de todos os
        modelos tentados, tokens da última resposta), mesmo em caso de erro.

        Raises:
            AIAuthenticationError: nenhum provedor restante da cadeia tem chave.
            AIProviderError: erro que não permite troca, ou o último modelo da cadeia falhou.
        """
        index = self._next_usable(self._index)
        if index is None:
            raise self._missing_key_error()
        self._index = index
        total_calls = 0
        latency = 0.0
        while True:
            spec = self.current
            client = self.client_for(spec.provider)
            config = LLMCallConfig(
                model_id=spec.model_id, thinking=self.thinking,
                temperature=self.temperature, timeout_sec=self.timeout_sec,
            )
            try:
                return client.generate(prompt, schema, config, self.retry)
            except AIProviderError as error:
                if not is_switchable(error):
                    raise
                following = self._next_usable(self._index + 1)
                if following is None:
                    raise
                to = self.specs[following]
                reason = describe_failure(error)
                self.switches.append(
                    ModelSwitch(
                        from_model=spec.label, to_model=to.label, reason=reason,
                        status_code=error.status_code, api_attempts=error.api_attempts,
                    )
                )
                self._notify(f"{spec.model_id} {reason} → trocando para {self._switch_label(spec, to)}")
                self._index = following
            finally:
                metrics = client.last_metrics
                if metrics is not None:
                    total_calls += metrics.api_calls
                    latency += metrics.latency_sec
                    self.last_metrics = metrics.model_copy(
                        update={"api_calls": total_calls, "latency_sec": round(latency, 3)}
                    )
