"""Schemas Pydantic do ciclo de revisão: estado das frases, pedido e decisão do revisor."""

from typing import Optional

from pydantic import BaseModel, Field

from ai_video_editor.director.schema import Sentence, SentenceRemoval
from ai_video_editor.intelligence.schema import AIRunMetrics
from ai_video_editor.planning.schema import PausePreset


class SentenceState(BaseModel):
    """Estado de uma frase na versão base: mantida ou removida (com categoria e motivo)."""

    sentence: Sentence
    removed: bool = False
    category: Optional[str] = None
    reason: Optional[str] = None


class RequestHistoryItem(BaseModel):
    """Um pedido anterior, para o revisor entender o rumo da edição."""

    version: int
    request: str


class RevisionContext(BaseModel):
    """Informações do vídeo e do pedido passadas ao revisor junto com as frases."""

    video_name: str
    duration_sec: float = Field(..., ge=0.0)
    language: Optional[str] = None
    base_version: int
    base_pacing: PausePreset = "normal"
    request: str
    history: list[RequestHistoryItem] = Field(default_factory=list)
    previous_error: Optional[str] = Field(
        default=None, description="Erro da resposta anterior, enviado ao modelo na nova tentativa"
    )


class SentenceRestoration(BaseModel):
    """Frase removida na versão base que volta ao vídeo. Só IDs, nunca tempos."""

    sentence_id: int = Field(..., description="ID da frase restaurada")
    reason: str = Field(..., description="Motivo da restauração, em português")


class RevisionDecision(BaseModel):
    """Resposta estruturada do revisor.

    `removals` é o NOVO conjunto completo de frases removidas (não um diff): toda frase
    ausente dele fica no vídeo.
    """

    feasible: bool = Field(
        default=True, description="False se o pedido é impossível ou ambíguo demais para atender"
    )
    problem: Optional[str] = Field(
        default=None, description="Quando feasible=False, o que impede ou o que precisa ser esclarecido"
    )
    removals: list[SentenceRemoval] = Field(
        default_factory=list, description="Conjunto COMPLETO de frases removidas na nova versão"
    )
    restorations: list[SentenceRestoration] = Field(
        default_factory=list, description="Frases removidas na versão base que voltam, com motivo"
    )
    pacing: Optional[PausePreset] = Field(
        default=None, description="Ritmo das pausas; omita para manter o da versão base"
    )
    summary: str = Field(default="", description="Resumo do que mudou, em português")


class RevisionRun(BaseModel):
    """Resultado de `run_reviser`: decisão aceita e métricas das tentativas."""

    provider: str
    model_id: str
    thinking: Optional[str] = None
    decision: Optional[RevisionDecision] = None
    attempts: list[AIRunMetrics] = Field(default_factory=list)
