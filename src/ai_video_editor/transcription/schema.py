"""Schemas Pydantic da transcrição com timestamps por palavra."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# Modelos que "auto" escolhe pelo dispositivo efetivo.
AUTO_MODEL_GPU = "large-v3-turbo"
AUTO_MODEL_CPU = "small"


def resolve_model_size(model_size: str, device: str) -> str:
    """Resolve "auto": `large-v3-turbo` na GPU (cuda) e `small` na CPU; valores explícitos passam direto."""
    if model_size != "auto":
        return model_size
    return AUTO_MODEL_GPU if device == "cuda" else AUTO_MODEL_CPU


class TranscriptionConfig(BaseModel):
    """Parâmetros de execução da transcrição."""

    model_size: str = Field(
        default="auto",
        description="Nome do modelo (ex: 'small', 'medium'); 'auto' = large-v3-turbo na GPU e small na CPU",
    )
    language: Optional[str] = Field(default=None, description="Código do idioma (ex: 'pt'); None = detecção automática")
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", description="Dispositivo de inferência: 'auto' usa a GPU NVIDIA quando disponível"
    )
    compute_type: str = Field(
        default="auto",
        description="Tipo de computação: 'auto' = float16 na GPU e int8 na CPU; aceita valores explícitos (ex: 'int8')",
    )
    vad_filter: bool = Field(default=True, description="Filtra trechos sem fala (VAD) para reduzir texto inventado em silêncio")


class Word(BaseModel):
    """Palavra transcrita com seu intervalo no áudio."""

    text: str
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_range(self) -> "Word":
        if self.start_sec >= self.end_sec:
            raise ValueError("start_sec deve ser menor que end_sec")
        return self


class TranscriptSegment(BaseModel):
    """Trecho contínuo de fala. O `id` é a referência estável usada pela IA para escolher cortes."""

    id: int = Field(..., ge=0, description="ID sequencial do segmento (0, 1, 2, ...)")
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    text: str
    words: list[Word] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "TranscriptSegment":
        if self.start_sec >= self.end_sec:
            raise ValueError("start_sec deve ser menor que end_sec")
        previous_start = 0.0
        for word in self.words:
            if word.start_sec < previous_start:
                raise ValueError("palavras devem estar em ordem cronológica")
            previous_start = word.start_sec
        return self


class Transcript(BaseModel):
    """Transcrição completa de uma mídia."""

    source_path: Path
    language: str
    duration_sec: float = Field(..., ge=0.0)
    model_used: str
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_segments(self) -> "Transcript":
        ids = [segment.id for segment in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("IDs de segmentos devem ser únicos")
        previous_start = 0.0
        for segment in self.segments:
            if segment.start_sec < previous_start:
                raise ValueError("segmentos devem estar em ordem cronológica")
            previous_start = segment.start_sec
        return self

    def save(self, path: Path) -> Path:
        """Grava a transcrição em JSON (UTF-8) e retorna o caminho."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        """Carrega e valida uma transcrição gravada com `save`."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
