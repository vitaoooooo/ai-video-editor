"""Schemas Pydantic da espinha falada (`edit <pasta>`, receita `vlog`): clipes escolhidos, plano por
clipe, transcrição comparativa por clipe e manifesto do projeto."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_video_editor.coverage.schema import CoverageRun
from ai_video_editor.pipeline.schema import (
    EditDiff,
    EditPipelineParams,
    EditPipelineResult,
    RunTimings,
    StageRecord,
    VersionRecord,
)
from ai_video_editor.planning.schema import SentenceCutResult, TimeRange

MULTICLIP_KIND = "multiclip"
MULTICLIP_REVISE_MESSAGE = (
    "revisão de projetos com vários clipes chega na rodada 4; por enquanto, rode `edit <pasta>` de novo "
    "com outro --instructions/--intensity (silêncios e transcrição vêm do cache)"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SpineClip(BaseModel):
    """Clipe principal com fala, na ordem do catálogo: entra na espinha."""

    id: str = Field(..., description="ID do catálogo (P006)")
    path: Path = Field(..., description="Caminho absoluto do arquivo")
    relative_path: str
    duration_sec: float = Field(..., ge=0.0)
    fps: float
    width: int
    height: int
    speech_ratio: Optional[float] = None
    speech_sec: Optional[float] = None
    speech_status: str = Field(..., description="analisada, desconhecido ou sem_audio (do catálogo)")

    @property
    def name(self) -> str:
        return Path(self.relative_path).name


class UnusedClip(BaseModel):
    """Clipe do catálogo que não entra na espinha, com o motivo."""

    id: Optional[str] = None
    relative_path: str
    role: str
    duration_sec: Optional[float] = None
    reason: str
    coverage: bool = Field(default=False, description="Disponível para cobertura (rodada 3)")


class SpineSelection(BaseModel):
    """Conteúdo do `spine_clips.json`: o que o catálogo ofereceu e o que entrou na espinha."""

    source_dir: Path
    catalog_path: Path
    clips: list[SpineClip] = Field(default_factory=list)
    unused: list[UnusedClip] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def clip(self, clip_id: str) -> SpineClip:
        return next(c for c in self.clips if c.id == clip_id)


class ClipPlan(BaseModel):
    """Trechos mantidos de um clipe e o destino de cada corte de frase (tempos do clipe)."""

    clip_id: str
    keep: list[TimeRange] = Field(default_factory=list, description="Trechos que viraram itens da Timeline")
    cut_results: list[SentenceCutResult] = Field(default_factory=list)
    threshold_db: Optional[float] = Field(default=None, description="Limite de silêncio efetivo do clipe")
    dropped: Optional[str] = Field(default=None, description="Por que o clipe ficou fora (None = está na espinha)")


class SpinePlan(BaseModel):
    """Conteúdo do `spine_plan.json`: plano de cada clipe, na ordem da espinha."""

    fps: int
    resolution: tuple[int, int]
    source_fps: list[float] = Field(default_factory=list, description="FPS dos clipes (antes de arredondar)")
    clips: list[ClipPlan] = Field(default_factory=list)

    def plan_for(self, clip_id: str) -> ClipPlan:
        return next(c for c in self.clips if c.clip_id == clip_id)


class ClipDiff(BaseModel):
    """Transcrição comparativa de um clipe; os tempos "editados" já estão no vídeo final."""

    clip_id: str
    file: str
    offset_sec: float = Field(..., ge=0.0, description="Onde o clipe começa no vídeo final")
    dropped: Optional[str] = None
    diff: EditDiff


class SpineDiff(BaseModel):
    """Conteúdo do `edit_diff.json` de um projeto com vários clipes."""

    project_name: str
    original_duration_sec: float = Field(..., ge=0.0, description="Soma das durações dos clipes da espinha")
    final_duration_sec: float = Field(..., ge=0.0)
    clips: list[ClipDiff] = Field(default_factory=list)
    unused: list[UnusedClip] = Field(default_factory=list)

    def merged(self) -> EditDiff:
        """Todas as frases num só `EditDiff` (os IDs de frase são globais), para contagens e consultas."""
        entries = [e for c in self.clips for e in c.diff.entries]
        return EditDiff(
            video_name=self.project_name,
            original_duration_sec=self.original_duration_sec,
            final_duration_sec=self.final_duration_sec,
            removed_pause_sec=round(sum(c.diff.removed_pause_sec for c in self.clips), 3),
            min_pause_marker_sec=self.clips[0].diff.min_pause_marker_sec if self.clips else 1.0,
            entries=entries,
        )


class SpineManifest(BaseModel):
    """Conteúdo do `project.json` de um projeto com vários clipes (`kind="multiclip"`)."""

    kind: Literal["multiclip"] = MULTICLIP_KIND
    source_dir: Path
    recipe: str
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    params: EditPipelineParams
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    versions: list[VersionRecord] = Field(default_factory=list)
    current_version: Optional[int] = None
    last_run: Optional[RunTimings] = None

    def ensure_stages(self, steps: list[str]) -> None:
        for step in steps:
            self.stages.setdefault(step, StageRecord())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SpineManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def is_multiclip_manifest(path: Path) -> bool:
    """O `project.json` é de um projeto com vários clipes?"""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("kind") == MULTICLIP_KIND
    except (OSError, ValueError, AttributeError):
        return False


class SpineResult(EditPipelineResult):
    """Resumo de uma execução do `edit <pasta>`."""

    clips_used: list[str] = Field(default_factory=list, description="IDs dos clipes na espinha, em ordem")
    clips_unused: int = 0
    timeline_path: Optional[Path] = Field(default=None, description="Timeline renderizada (com cobertura, se houve)")
    coverage: Optional[CoverageRun] = None
