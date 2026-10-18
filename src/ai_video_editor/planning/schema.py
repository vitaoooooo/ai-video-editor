"""Schemas Pydantic do planejamento de cortes."""

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
