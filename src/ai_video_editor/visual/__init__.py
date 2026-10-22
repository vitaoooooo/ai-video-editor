"""Análise visual dos clipes de um projeto: tomadas, miniaturas, medidas (FFmpeg) e descrição por IA."""

from ai_video_editor.visual.builder import (
    VISUAL_STAGE_LABELS,
    VISUAL_STAGES,
    VisualBuilder,
    VisualEstimate,
    VisualResult,
)
from ai_video_editor.visual.describer import VisualDescriber, mock_visual_responder
from ai_video_editor.visual.schema import (
    VISUAL_SCHEMA_VERSION,
    ClipVisual,
    ProjectVisual,
    Shot,
    ShotAnalysis,
    ShotFrame,
    ShotMeasures,
)

__all__ = [
    "VISUAL_SCHEMA_VERSION",
    "VISUAL_STAGES",
    "VISUAL_STAGE_LABELS",
    "ClipVisual",
    "ProjectVisual",
    "Shot",
    "ShotAnalysis",
    "ShotFrame",
    "ShotMeasures",
    "VisualBuilder",
    "VisualDescriber",
    "VisualEstimate",
    "VisualResult",
    "mock_visual_responder",
]
