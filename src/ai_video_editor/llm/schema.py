"""Schemas da camada genérica de IA: configuração da chamada, métricas e cadeia de modelos."""

from dataclasses import dataclass
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

# Nível de raciocínio genérico; cada provedor traduz para o seu parâmetro (ou ignora).
ThinkingLevel = Literal["off", "low", "high"]

T = TypeVar("T", bound=BaseModel)


class LLMCallConfig(BaseModel):
    """Configuração de uma chamada com saída estruturada."""

    model_id: str
    thinking: ThinkingLevel = "low"
    temperature: float = 0.2
    timeout_sec: float = Field(default=120.0, gt=0.0)


class LLMCallMetrics(BaseModel):
    """Métricas de uma chamada (ou da cadeia inteira, somando as chamadas)."""

    provider: str
    model_id: str = Field(..., description="Modelo efetivamente usado")
    thinking: Optional[str] = Field(default=None, description="Raciocínio efetivo (com a tradução do provedor)")
    latency_sec: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    api_calls: int = Field(default=0, ge=0, description="Chamadas à API, incluindo novas tentativas")


@dataclass
class StructuredResult(Generic[T]):
    """Objeto validado no schema pedido + métricas da chamada."""

    value: T
    metrics: LLMCallMetrics


class ModelSpec(BaseModel):
    """Um elo da cadeia: `provedor:modelo`."""

    provider: str
    model_id: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_id}"


class ModelSwitch(BaseModel):
    """Troca automática para o próximo modelo da cadeia."""

    from_model: str = Field(..., description="`provedor:modelo` que falhou")
    to_model: str = Field(..., description="`provedor:modelo` usado em seguida")
    reason: str = Field(..., description="Motivo, ex: `indisponível (503) após 5 tentativas`")
    status_code: Optional[int] = None
    api_attempts: int = 1
