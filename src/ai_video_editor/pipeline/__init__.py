"""Pipeline de edição com pasta de projeto e cache por etapa."""

from ai_video_editor.pipeline.runner import EditPipeline, ProgressCallback
from ai_video_editor.pipeline.schema import (
    STAGES,
    EditPipelineParams,
    EditPipelineResult,
    ProjectManifest,
    StageRecord,
)

__all__ = [
    "STAGES",
    "EditPipeline",
    "EditPipelineParams",
    "EditPipelineResult",
    "ProgressCallback",
    "ProjectManifest",
    "StageRecord",
]
