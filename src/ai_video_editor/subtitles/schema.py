"""Schemas Pydantic das legendas."""

from pydantic import BaseModel, Field, model_validator


class SubtitleParams(BaseModel):
    """Parâmetros de agrupamento de palavras em blocos de legenda."""

    max_chars_per_line: int = Field(default=42, ge=5, description="Máximo de caracteres por linha")
    max_lines: int = Field(default=2, ge=1, description="Máximo de linhas por bloco")
    min_duration_sec: float = Field(default=1.0, gt=0.0, description="Duração mínima de um bloco")
    max_duration_sec: float = Field(default=6.0, gt=0.0, description="Duração máxima de um bloco")
    max_gap_sec: float = Field(
        default=1.0, ge=0.0, description="Pausa entre palavras que força um novo bloco"
    )

    min_kept_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fração mínima da palavra em trechos mantidos para ela ser preservada",
    )
    min_kept_sec: float = Field(
        default=0.15,
        ge=0.0,
        description="Duração mínima da palavra em trechos mantidos para ela ser preservada",
    )

    @model_validator(mode="after")
    def _check_durations(self) -> "SubtitleParams":
        if self.min_duration_sec > self.max_duration_sec:
            raise ValueError("min_duration_sec não pode ser maior que max_duration_sec")
        return self


class SubtitleBlock(BaseModel):
    """Bloco de legenda; tempos referem-se ao VÍDEO EDITADO."""

    index: int = Field(..., ge=1, description="Numeração sequencial (começa em 1)")
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    lines: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_range(self) -> "SubtitleBlock":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec deve ser maior que start_sec")
        return self

    @property
    def text(self) -> str:
        return "\n".join(self.lines)
