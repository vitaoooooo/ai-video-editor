"""Schemas Pydantic do pipeline de edição e do manifesto do projeto."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ai_video_editor.director.schema import DirectorThinking, SentenceParams, UserBrief
from ai_video_editor.planning.schema import PausePreset, SilenceCutParams
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

    threshold_db: Union[Literal["auto"], float] = Field(
        default="auto", description="Limiar de silêncio em dB, ou `auto` para medir o ruído de cada vídeo"
    )
    min_silence_sec: float = Field(default=0.5, gt=0.0, description="Duração mínima do silêncio")
    cut: SilenceCutParams = Field(default_factory=SilenceCutParams)
    transcribe: bool = Field(default=False, description="Gera transcript.json (necessário para o Diretor)")
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    director: DirectorName = Field(default="none", description="Diretor de IA que remove frases")
    director_model: str = Field(default="gemini-3.8-flash", description="Modelo do Diretor Gemini")
    director_thinking: DirectorThinking = Field(
        default="low", description="Nível de raciocínio do Diretor Gemini (off/low/high)"
    )
    director_timeout_sec: float = Field(default=120.0, gt=5.0, description="Tempo limite da chamada ao Diretor")
    api_retries: int = Field(
        default=4, ge=0, description="Novas tentativas da API em falhas temporárias (0 desliga)"
    )
    sentences: SentenceParams = Field(default_factory=SentenceParams)
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


class StageTiming(BaseModel):
    """Tempo de relógio de uma etapa em uma execução (sem tempo se veio do cache ou foi pulada)."""

    name: str = Field(..., description="Nome da etapa (`probe`, `render`, `reviser`...)")
    status: StageOutcome
    elapsed_sec: Optional[float] = Field(default=None, description="Só em etapas executadas")


class RunTimings(BaseModel):
    """Tempos de uma execução do `edit` ou do `revise`."""

    stages: list[StageTiming] = Field(default_factory=list)
    total_sec: float = Field(..., description="Tempo de relógio da execução inteira")
    video_duration_sec: Optional[float] = Field(default=None, description="Duração do vídeo original")


class VersionRecord(BaseModel):
    """Uma versão do projeto (`versions/vN/`): o resultado do `edit` (v1) ou de uma revisão."""

    number: int = Field(..., ge=1)
    kind: Literal["edit", "revision"]
    request: str = Field(..., description="Pedido que originou a versão")
    parent: Optional[int] = Field(default=None, description="Versão em que a revisão se baseou")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = Field(default=None, description="Resumo do revisor (só em revisões)")
    cut: SilenceCutParams = Field(
        default_factory=SilenceCutParams, description="Parâmetros de corte do `edit` (antes do preset)"
    )
    pacing: PausePreset = "normal"
    duration_sec: float
    cuts: int = Field(..., description="Número de trechos removidos do vídeo original")
    segments: int
    removed_sentences: int = 0
    video: str
    subtitles: Optional[str] = None
    edit_fp: Optional[str] = Field(
        default=None, description="Fingerprint do render/legendas do `edit` que gerou a versão"
    )
    timings: Optional[RunTimings] = Field(
        default=None, description="Tempos da execução que gerou a versão"
    )


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
    versions: list[VersionRecord] = Field(default_factory=list)
    current_version: Optional[int] = Field(default=None, description="Número da versão atual")
    last_run: Optional[RunTimings] = Field(
        default=None, description="Tempos da última execução de `edit` ou `revise`"
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
    version: Optional[int] = Field(default=None, description="Versão do projeto que corresponde ao resultado")
    original_duration_sec: float
    final_duration_sec: float
    outcomes: dict[StageName, StageOutcome]
    timings: Optional[RunTimings] = None
    warnings: list[str] = Field(default_factory=list)
