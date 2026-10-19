"""Pipeline de edição com pasta de projeto e cache por etapa."""

from ai_video_editor.pipeline.revise import ChangedSentence, ProjectReviser, RevisionResult
from ai_video_editor.pipeline.runner import EditPipeline, ProgressCallback
from ai_video_editor.pipeline.schema import (
    STAGES,
    EditPipelineParams,
    EditPipelineResult,
    ProjectManifest,
    StageRecord,
    VersionRecord,
)

__all__ = [
    "STAGES",
    "ChangedSentence",
    "EditPipeline",
    "EditPipelineParams",
    "EditPipelineResult",
    "ProgressCallback",
    "ProjectManifest",
    "ProjectReviser",
    "RevisionResult",
    "StageRecord",
    "VersionRecord",
]
