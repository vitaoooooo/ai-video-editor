"""Schemas Pydantic do benchmark de transcrição."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

DEFAULT_PROVIDER = "faster_whisper"


class BenchmarkConfig(BaseModel):
    """Uma configuração a medir: `[provedor/]modelo:dispositivo:compute_type`."""

    provider: str = DEFAULT_PROVIDER
    model: str
    device: Literal["auto", "cpu", "cuda"]
    compute_type: str

    @property
    def label(self) -> str:
        prefix = "" if self.provider == DEFAULT_PROVIDER else f"{self.provider}/"
        return f"{prefix}{self.model}:{self.device}:{self.compute_type}"

    @classmethod
    def parse(cls, spec: str) -> "BenchmarkConfig":
        """Lê `small:cuda:float16` (ou `mock/x:cpu:int8` para outro provedor)."""
        provider = DEFAULT_PROVIDER
        rest = spec.strip()
        if "/" in rest:
            provider, rest = rest.split("/", 1)
        parts = rest.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"Configuração inválida '{spec}': use modelo:dispositivo:compute_type "
                "(ex: small:cuda:float16)"
            )
        model, device, compute_type = parts
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Dispositivo inválido '{device}' em '{spec}': use auto, cpu ou cuda")
        return cls(provider=provider, model=model, device=device, compute_type=compute_type)


class BenchmarkItem(BaseModel):
    """Um áudio a transcrever, com referência opcional."""

    name: str
    audio: Path
    reference: Optional[Path] = None
    language: Optional[str] = Field(default=None, description="Código do idioma; None = detecção automática")


class BenchmarkManifest(BaseModel):
    """Manifesto JSON com vários itens. Caminhos relativos valem a partir da pasta do manifesto."""

    items: list[BenchmarkItem] = Field(..., min_length=1)

    @field_validator("items")
    @classmethod
    def _unique_names(cls, items: list[BenchmarkItem]) -> list[BenchmarkItem]:
        names = [item.name for item in items]
        if len(set(names)) != len(names):
            raise ValueError("os nomes dos itens do manifesto devem ser únicos")
        return items

    @classmethod
    def load(cls, path: Path) -> "BenchmarkManifest":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls.model_validate({"items": data} if isinstance(data, list) else data)
        base = path.resolve().parent
        for item in manifest.items:
            if not item.audio.is_absolute():
                item.audio = base / item.audio
            if item.reference is not None and not item.reference.is_absolute():
                item.reference = base / item.reference
        return manifest


class SubstitutionCount(BaseModel):
    reference: str
    hypothesis: str
    count: int = Field(..., ge=1)


class WerResult(BaseModel):
    """Resultado do alinhamento entre referência e hipótese."""

    wer: float = Field(..., ge=0.0)
    substitutions: int = Field(..., ge=0)
    deletions: int = Field(..., ge=0)
    insertions: int = Field(..., ge=0)
    reference_words: int = Field(..., ge=0)
    hypothesis_words: int = Field(..., ge=0)
    tail_words: int = Field(
        default=0,
        ge=0,
        description="palavras da hipótese depois do fim da referência; NÃO entram em `insertions` nem no WER",
    )
    tail_preview: Optional[str] = Field(default=None, description="início da cauda, para conferência")
    top_substitutions: list[SubstitutionCount] = Field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


class ItemResult(BaseModel):
    """Medição de uma configuração em um item."""

    item: str
    status: Literal["ok", "error"]
    error: Optional[str] = None
    audio_duration_sec: Optional[float] = None
    transcription_time_sec: Optional[float] = None
    real_time_factor: Optional[float] = Field(default=None, description="tempo de transcrição / duração do áudio")
    wer: Optional[WerResult] = Field(default=None, description="WER bruto; None quando o item não tem referência")
    wer_normalized: Optional[WerResult] = Field(
        default=None, description="WER com a normalização do idioma; None sem referência ou sem regras para o idioma"
    )
    hypothesis_file: Optional[str] = Field(
        default=None, description="texto gerado, relativo à pasta da execução (usado pelo `rescore`)"
    )


class ConfigRun(BaseModel):
    """Resultado de uma configuração: carga do modelo (uma vez) e os itens."""

    config: BenchmarkConfig
    status: Literal["ok", "error"]
    error: Optional[str] = None
    effective_device: Optional[str] = None
    effective_compute_type: Optional[str] = None
    effective_model: Optional[str] = None
    load_time_sec: Optional[float] = None
    items: list[ItemResult] = Field(default_factory=list)


class ConfigAverage(BaseModel):
    """Média geral de uma configuração sobre os itens que deram certo.

    O RTF é o tempo total sobre o áudio total; o WER é agregado (erros somados sobre palavras da
    referência somadas), só com os itens que têm referência.
    """

    config: str
    items_ok: int
    items_total: int
    audio_duration_sec: float
    transcription_time_sec: float
    real_time_factor: Optional[float] = None
    wer: Optional[WerResult] = None
    wer_normalized: Optional[WerResult] = None


class EnvironmentInfo(BaseModel):
    project_version: str
    python: str
    platform: str
    cpu: str
    gpu: Optional[str] = None
    faster_whisper: Optional[str] = None


class BenchmarkRun(BaseModel):
    """Arquivo `results.json`."""

    schema_version: int = 2
    name: str
    created_at: datetime
    environment: EnvironmentInfo
    items: list[BenchmarkItem]
    configs: list[ConfigRun]
    averages: list[ConfigAverage] = Field(default_factory=list)
    normalization: str = (
        "minúsculas, apóstrofos unificados, pontuação/hífens removidos, espaços colapsados"
    )
    normalization_extra: str = (
        "por idioma (en): números por extenso → dígitos (0–999 e milhares simples), okay → ok, "
        "x's → xs, separador de milhar removido"
    )
