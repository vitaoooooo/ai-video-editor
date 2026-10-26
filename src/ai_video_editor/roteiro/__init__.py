"""Roteiro unificado (fala + imagem + ritmo) de qualquer material, com IDs para o Editor de IA."""

from ai_video_editor.roteiro.builder import (
    ROTEIRO_STAGE_LABELS,
    ROTEIRO_STAGES,
    RoteiroBuilder,
    RoteiroParams,
    RoteiroResult,
)
from ai_video_editor.roteiro.describer import PhotoDescriber, mock_vision_responder
from ai_video_editor.roteiro.schema import (
    ROTEIRO_SCHEMA_VERSION,
    Moment,
    PhotoDescription,
    ProjectRoteiro,
    RoteiroClip,
    RoteiroEstimate,
    RoteiroPhoto,
    RoteiroSentence,
    SentenceRhythm,
)

__all__ = [
    "ROTEIRO_SCHEMA_VERSION",
    "ROTEIRO_STAGES",
    "ROTEIRO_STAGE_LABELS",
    "Moment",
    "PhotoDescriber",
    "PhotoDescription",
    "ProjectRoteiro",
    "RoteiroBuilder",
    "RoteiroClip",
    "RoteiroEstimate",
    "RoteiroParams",
    "RoteiroPhoto",
    "RoteiroResult",
    "RoteiroSentence",
    "SentenceRhythm",
    "mock_vision_responder",
]
