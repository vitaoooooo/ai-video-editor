"""Pydantic schemas for edit plans and video segments."""

from pathlib import Path
from typing import Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

LoudnessPreset = Literal["voz", "youtube", "broadcast", "off"]


class ClipSegment(BaseModel):
    """Represents a sliced portion of an input video."""

    source_path: Path = Field(..., description="Path to source video file")
    start_sec: float = Field(default=0.0, ge=0.0, description="Start timestamp in seconds")
    end_sec: Optional[float] = Field(default=None, description="End timestamp in seconds (None = clip end)")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Audio volume multiplier")

    @model_validator(mode="after")
    def validate_time_range(self) -> "ClipSegment":
        if self.end_sec is not None and self.end_sec <= self.start_sec:
            raise ValueError(
                f"end_sec ({self.end_sec}) must be greater than start_sec ({self.start_sec})"
            )
        return self

    @property
    def duration(self) -> Optional[float]:
        """Estimated duration if end_sec is specified."""
        if self.end_sec is not None:
            return round(self.end_sec - self.start_sec, 3)
        return None


class LoudnessTarget(BaseModel):
    """Alvo de normalização de loudness (EBU R128) aplicado ao áudio final com o filtro loudnorm."""

    preset: str = Field(default="custom", description="Nome do preset que originou o alvo")
    integrated_lufs: float = Field(..., ge=-70.0, le=-5.0, description="Loudness integrado alvo (I), em LUFS")
    true_peak_dbtp: float = Field(..., ge=-9.0, le=0.0, description="True peak máximo (TP), em dBTP")
    lra: float = Field(..., ge=1.0, le=50.0, description="Faixa de loudness alvo (LRA), em LU")

    @classmethod
    def from_preset(cls, preset: LoudnessPreset) -> Optional["LoudnessTarget"]:
        """Alvo do preset; None para `off` (sem normalização)."""
        if preset == "off":
            return None
        return cls(preset=preset, **LOUDNESS_PRESETS[preset])

    def describe(self) -> str:
        """`voz (-16 LUFS, TP -1.5 dBTP, LRA 11)`."""
        return (
            f"{self.preset} ({self.integrated_lufs:g} LUFS, TP {self.true_peak_dbtp:g} dBTP, "
            f"LRA {self.lra:g})"
        )


# voz: fala em plataformas online (padrão do `edit`); youtube: referência de -14 LUFS das
# plataformas de streaming; broadcast: EBU R128 para TV.
LOUDNESS_PRESETS: dict[str, dict[str, float]] = {
    "voz": {"integrated_lufs": -16.0, "true_peak_dbtp": -1.5, "lra": 11.0},
    "youtube": {"integrated_lufs": -14.0, "true_peak_dbtp": -1.5, "lra": 11.0},
    "broadcast": {"integrated_lufs": -23.0, "true_peak_dbtp": -1.5, "lra": 11.0},
}


class EditPlan(BaseModel):
    """Declarative specification of a complete video edit."""

    title: str = Field(default="Untitled Edit", description="Title or description of the edit")
    output_path: Path = Field(..., description="Destination path for rendered output MP4")
    target_resolution: Tuple[int, int] = Field(
        default=(1280, 720),
        description="Target (width, height) for standardized output canvas",
    )
    target_fps: int = Field(default=30, gt=0, description="Target frames per second")
    audio_fade_sec: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Micro fade de áudio (in e out) aplicado em cada segmento, em segundos (0 desativa)",
    )
    loudness: Optional[LoudnessTarget] = Field(
        default=None,
        description="Normalização de loudness do áudio final em duas passadas (None mantém o volume original)",
    )
    segments: list[ClipSegment] = Field(
        ..., min_length=1, description="Ordered list of video segments to trim and concatenate"
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "EditPlan":
        if self.target_resolution[0] <= 0 or self.target_resolution[1] <= 0:
            raise ValueError(f"Invalid target resolution: {self.target_resolution}")
        return self
