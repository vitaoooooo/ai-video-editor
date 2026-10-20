"""Nova tentativa automática para falhas temporárias de provedores de IA."""

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

from ai_video_editor.intelligence.exceptions import AIProviderError

T = TypeVar("T")

DEFAULT_API_RETRIES = 4
# Teto para o Retry-After informado pela API, para não travar o terminal por minutos.
MAX_RETRY_AFTER_SEC = 120.0


@dataclass
class RetryEvent:
    """Uma nova tentativa prestes a acontecer (para mostrar no terminal)."""

    error: AIProviderError
    delay_sec: float
    next_attempt: int
    total_attempts: int

    @property
    def message(self) -> str:
        """Ex: `Gemini indisponível (503), nova tentativa em 4s (2/5)`."""
        code = f" ({self.error.status_code})" if self.error.status_code else ""
        provider = (self.error.provider or "provedor").capitalize()
        if self.error.status_code == 429:
            reason = f"{provider} com limite de requisições{code}"
        elif self.error.status_code is None:
            reason = f"falha de conexão com o {provider}"
        elif self.error.status_code == 504:
            reason = f"{provider} demorou demais para responder{code}"
        else:
            reason = f"{provider} indisponível{code}"
        return f"{reason}, nova tentativa em {self.delay_sec:.0f}s ({self.next_attempt}/{self.total_attempts})"


@dataclass
class RetryPolicy:
    """Espera exponencial com jitter. `max_retries=0` desliga as novas tentativas.

    `sleep` e `rand` são injetáveis para os testes não dormirem de verdade.
    `attempts_made` conta as chamadas feitas desde o último `reset()`.
    """

    max_retries: int = DEFAULT_API_RETRIES
    base_delay_sec: float = 2.0
    max_delay_sec: float = 60.0
    jitter: float = 0.25
    sleep: Callable[[float], None] = time.sleep
    rand: Callable[[], float] = random.random
    on_retry: Optional[Callable[[RetryEvent], None]] = None
    attempts_made: int = field(default=0, init=False)

    def reset(self) -> None:
        self.attempts_made = 0

    def delay_for(self, retry_index: int, retry_after_sec: Optional[float] = None) -> float:
        """Espera antes da nova tentativa nº `retry_index` (0 = primeira): 2s, 4s, 8s, 16s + jitter."""
        backoff = min(self.max_delay_sec, self.base_delay_sec * (2 ** retry_index))
        backoff *= 1.0 + self.jitter * self.rand()
        if retry_after_sec is not None:
            return max(backoff, min(retry_after_sec, MAX_RETRY_AFTER_SEC))
        return backoff


def call_with_retry(
    fn: Callable[[], T],
    policy: Optional[RetryPolicy],
    map_error: Callable[[Exception], AIProviderError],
) -> T:
    """Executa `fn`, repetindo só enquanto o erro mapeado for temporário (`transient`).

    Erros permanentes sobem na hora. Ao esgotar as tentativas, sobe o último erro mapeado
    (com o detalhe original da API), com `api_attempts` preenchido. O erro original fica em
    `__cause__`.
    """
    policy = policy or RetryPolicy(max_retries=0)
    max_retries = max(0, policy.max_retries)
    total = max_retries + 1
    for attempt in range(1, total + 1):
        policy.attempts_made += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — o SDK lança tipos variados
            mapped = exc if isinstance(exc, AIProviderError) else map_error(exc)
            if not mapped.transient or attempt >= total:
                raise mapped.with_attempts(attempt) from (exc if mapped is not exc else exc.__cause__)
            delay = policy.delay_for(attempt - 1, mapped.retry_after_sec)
            if policy.on_retry is not None:
                policy.on_retry(RetryEvent(mapped, delay, attempt + 1, total))
            policy.sleep(delay)
    raise AssertionError("inalcançável")  # pragma: no cover
