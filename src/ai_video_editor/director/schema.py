"""Schemas Pydantic do Diretor de IA: frases, briefing do usuário e decisões de corte."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ai_video_editor.intelligence.schema import AIRunMetrics

Intensity = Literal["light", "medium", "aggressive"]
RemovalCategory = Literal["repetition", "false_start", "filler", "off_topic", "other"]


class Sentence(BaseModel):
    """Unidade de decisão: segmentos consecutivos da transcrição unidos até o fim da frase."""

    id: int = Field(..., ge=0, description="ID sequencial da frase (0, 1, 2, ...)")
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    text: str
    segment_ids: list[int] = Field(..., min_length=1, description="IDs dos segmentos de origem")

    @model_validator(mode="after")
    def _check_range(self) -> "Sentence":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec deve ser maior que start_sec")
        return self

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


class UserBrief(BaseModel):
    """O que o usuário pediu para a edição."""

    intensity: Intensity = Field(default="light", description="Quanto cortar: light, medium ou aggressive")
    target_duration_sec: Optional[float] = Field(
        default=None, gt=0.0, description="Duração final desejada do vídeo, se houver"
    )
    remove_fillers: bool = Field(default=True, description="Permite remover frases de enrolação")
    instructions: Optional[str] = Field(default=None, description="Pedido livre do usuário")


class DirectorContext(BaseModel):
    """Informações do vídeo passadas ao Diretor junto com as frases."""

    video_name: str
    duration_sec: float = Field(..., ge=0.0)
    language: Optional[str] = None
    previous_error: Optional[str] = Field(
        default=None, description="Erro da resposta anterior, enviado ao modelo na nova tentativa"
    )


class SentenceRemoval(BaseModel):
    """Uma frase escolhida para sair do vídeo. A IA só referencia IDs, nunca tempos."""

    sentence_id: int = Field(..., description="ID da frase a remover")
    category: RemovalCategory = Field(..., description="Tipo de problema da frase")
    reason: str = Field(..., description="Motivo da remoção, em português")


class DirectorDecision(BaseModel):
    """Resposta estruturada do Diretor."""

    removals: list[SentenceRemoval] = Field(default_factory=list, description="Frases a remover")
    summary: str = Field(..., description="Resumo das decisões, em português")


class DirectorRun(BaseModel):
    """Conteúdo do `director_decision.json`: frases, decisão aceita (se houver) e métricas."""

    provider: str
    model_id: str
    brief: UserBrief
    sentences: list[Sentence]
    decision: Optional[DirectorDecision] = Field(
        default=None, description="Decisão validada; None quando caiu no corte só de silêncio"
    )
    attempts: list[AIRunMetrics] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def removed_sentences(self) -> list[tuple[Sentence, SentenceRemoval]]:
        """Pares (frase, remoção) em ordem cronológica."""
        if self.decision is None:
            return []
        by_id = {s.id: s for s in self.sentences}
        pairs = [(by_id[r.sentence_id], r) for r in self.decision.removals]
        return sorted(pairs, key=lambda pair: pair[0].start_sec)
