"""Schemas Pydantic do planejamento de cortes."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SilenceCutParams(BaseModel):
    """Parâmetros do corte de silêncio (aplicados sobre a análise de áudio)."""

    padding_before_sec: float = Field(
        default=0.15, ge=0.0, description="Folga mantida antes de cada trecho de fala"
    )
    padding_after_sec: float = Field(
        default=0.20, ge=0.0, description="Folga mantida depois de cada trecho de fala"
    )
    min_gap_to_cut_sec: float = Field(
        default=0.30,
        ge=0.0,
        description="Pausas (já descontado o padding) menores que isso não são cortadas",
    )
    min_keep_sec: float = Field(
        default=0.10,
        ge=0.0,
        description="Trechos de fala mais curtos que isso (antes do padding) são tratados como ruído",
    )


class TimeRange(BaseModel):
    """Intervalo [start_sec, end_sec) da mídia de origem."""

    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> "TimeRange":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec deve ser maior que start_sec")
        return self

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


# Destino de um corte de frase(s): aplicado (ajustado aos silêncios), estendido até o limite
# da frase para não deixar pedaço de palavra, anulado pela folga das frases vizinhas
# ("protected") ou desfeito por deixar pedaço de palavra que não podia ser removido ("undone").
SentenceCutOutcome = Literal["applied", "extended", "protected", "undone"]


class SentenceCutResult(BaseModel):
    """O que aconteceu com um corte de frase(s) pedido pelo Diretor ou pela revisão."""

    requested: TimeRange = Field(..., description="Intervalo da(s) frase(s) na transcrição")
    applied: Optional[TimeRange] = Field(default=None, description="Corte efetivo; None se nada saiu")
    outcome: SentenceCutOutcome


PausePreset = Literal["natural", "normal", "tight"]

# (fator do padding, fator do min_gap): natural deixa mais folga, tight deixa menos.
_PAUSE_SCALES: dict[str, tuple[float, float]] = {
    "natural": (1.75, 1.5),
    "normal": (1.0, 1.0),
    "tight": (0.5, 0.6),
}


def apply_pause_preset(cut: SilenceCutParams, preset: PausePreset) -> SilenceCutParams:
    """Ajusta o ritmo das pausas: "normal" mantém `cut`, os demais escalam padding e `min_gap`."""
    padding, gap = _PAUSE_SCALES[preset]
    return cut.model_copy(
        update={
            "padding_before_sec": round(cut.padding_before_sec * padding, 3),
            "padding_after_sec": round(cut.padding_after_sec * padding, 3),
            "min_gap_to_cut_sec": round(cut.min_gap_to_cut_sec * gap, 3),
        }
    )
