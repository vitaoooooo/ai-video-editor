"""Schemas Pydantic da análise de áudio."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SilenceInterval(BaseModel):
    """Intervalo de silêncio (ou de fala, no complemento), em segundos."""

    start_sec: float = Field(..., ge=0.0, description="Início do intervalo")
    end_sec: float = Field(..., ge=0.0, description="Fim do intervalo")
    duration: float = Field(..., ge=0.0, description="Duração (end_sec - start_sec)")

    @model_validator(mode="after")
    def _check_order(self) -> "SilenceInterval":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec deve ser maior ou igual a start_sec")
        return self

    @classmethod
    def from_bounds(cls, start_sec: float, end_sec: float) -> "SilenceInterval":
        return cls(
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration=round(end_sec - start_sec, 3),
        )


class AudioAnalysisParams(BaseModel):
    """Parâmetros usados na detecção de silêncio."""

    threshold_db: float = Field(..., description="Limiar de silêncio em dB")
    min_silence_sec: float = Field(..., gt=0.0, description="Duração mínima do silêncio")
    threshold_mode: Literal["auto", "fixed"] = Field(
        default="fixed", description="`auto` mede o ruído do vídeo; `fixed` usa o número informado"
    )


class AudioLevelStats(BaseModel):
    """Medidas de nível (pico por janela de 50 ms) e limite escolhido no modo automático."""

    noise_floor_db: float = Field(..., description="Piso de ruído (percentil baixo dos níveis)")
    speech_level_db: float = Field(..., description="Nível típico da fala (percentil alto dos níveis)")
    threshold_db: float = Field(..., description="Limite de silêncio escolhido")
    reliable: bool = Field(default=True, description="False quando a detecção é pouco confiável")
    warnings: list[str] = Field(default_factory=list)


class AudioAnalysisResult(BaseModel):
    """Resultado da análise de áudio: silêncios e fala (complemento)."""

    source_path: Path = Field(..., description="Arquivo analisado")
    duration_sec: float = Field(..., ge=0.0, description="Duração total da mídia")
    params: AudioAnalysisParams
    levels: Optional[AudioLevelStats] = Field(
        default=None, description="Medidas do modo automático (ausente no modo fixo)"
    )
    silences: list[SilenceInterval] = Field(default_factory=list)
    speech: list[SilenceInterval] = Field(
        default_factory=list, description="Intervalos de fala: complemento dos silêncios"
    )

    def save_json(self, path: Path) -> Path:
        """Salva o resultado em JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Path) -> "AudioAnalysisResult":
        """Carrega um resultado salvo com save_json."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
