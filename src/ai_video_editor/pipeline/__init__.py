"""Pipeline de edição com pasta de projeto e cache por etapa."""

from ai_video_editor.pipeline.revise import ChangedSentence, ProjectReviser, RevisionResult
from ai_video_editor.pipeline.runner import EditPipeline, ProgressCallback, StageProgressCallback
from ai_video_editor.pipeline.spine import SpinePipeline
from ai_video_editor.pipeline.spine_schema import SpineResult
from ai_video_editor.pipeline.schema import (
    STAGES,
    EditPipelineParams,
    EditPipelineResult,
    ProjectManifest,
    StageProgress,
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
    "SpinePipeline",
    "SpineResult",
    "StageProgress",
    "StageProgressCallback",
    "StageRecord",
    "VersionRecord",
]
