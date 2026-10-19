"""Schemas Pydantic do pipeline de edição e do manifesto do projeto."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ai_video_editor.director.schema import DirectorThinking, UserBrief
from ai_video_editor.planning.schema import SilenceCutParams
from ai_video_editor.subtitles.schema import SubtitleParams
from ai_video_editor.transcription.schema import TranscriptionConfig

StageName = Literal["probe", "silences", "transcription", "director", "plan", "render", "subtitles", "report"]
STAGES: tuple[StageName, ...] = (
    "probe",
    "silences",
    "transcription",
    "director",
    "plan",
    "render",
    "subtitles",
    "report",
)

StageStatus = Literal["pending", "done", "skipped", "failed"]
# Resultado de uma etapa nesta execução: executada, reaproveitada do cache ou não pedida.
StageOutcome = Literal["ran", "cached", "skipped"]

MetricValue = Union[int, float, str]
DirectorName = Literal["none", "mock", "gemini"]


class EditPipelineParams(BaseModel):
    """Todos os parâmetros que influenciam o resultado do comando `edit`."""

    threshold_db: float = Field(default=-35.0, description="Limiar de silêncio em dB")
    min_silence_sec: float = Field(default=0.5, gt=0.0, description="Duração mínima do silêncio")
    cut: SilenceCutParams = Field(default_factory=SilenceCutParams)
    transcribe: bool = Field(default=False, description="Gera transcript.json (necessário para o Diretor)")
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    director: DirectorName = Field(default="none", description="Diretor de IA que remove frases")
    director_model: str = Field(default="gemini-3.8-flash", description="Modelo do Diretor Gemini")
    director_thinking: DirectorThinking = Field(
        default="off", description="Nível de raciocínio do Diretor Gemini (off/low/high)"
    )
    director_timeout_sec: float = Field(default=120.0, gt=5.0, description="Tempo limite da chamada ao Diretor")
    brief: UserBrief = Field(default_factory=UserBrief)
    subtitles: bool = Field(default=True, description="Gera o .srt quando há transcrição")
    subtitle: SubtitleParams = Field(default_factory=SubtitleParams)


class StageRecord(BaseModel):
    """Estado persistido de uma etapa no manifesto."""

    status: StageStatus = "pending"
    fingerprint: Optional[str] = Field(
        default=None, description="Hash das entradas e parâmetros que afetam a etapa"
    )
    artifact: Optional[str] = Field(default=None, description="Artefato relativo à pasta do projeto")
    updated_at: Optional[datetime] = None
    elapsed_sec: Optional[float] = None
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    error: Optional[str] = None


class ProjectManifest(BaseModel):
    """Conteúdo do `project.json`."""

    video_path: Path
    video_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    params: EditPipelineParams
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {name: StageRecord() for name in STAGES}
    )

    @field_validator("stages")
    @classmethod
    def _fill_missing_stages(cls, stages: dict[StageName, StageRecord]) -> dict[StageName, StageRecord]:
        """Manifestos antigos não têm as etapas novas: entram como pendentes."""
        return {name: stages.get(name, StageRecord()) for name in STAGES}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ProjectManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class EditPipelineResult(BaseModel):
    """Resumo de uma execução do pipeline."""

    project_dir: Path
    manifest_path: Path
    output_path: Path
    report_path: Path
    subtitles_path: Optional[Path] = Field(
        default=None, description="Legenda .srt gerada ou reaproveitada nesta execução"
    )
    original_duration_sec: float
    final_duration_sec: float
    outcomes: dict[StageName, StageOutcome]
    warnings: list[str] = Field(default_factory=list)
