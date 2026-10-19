"""Revisor de IA: ajusta uma edição existente a partir de um pedido, referenciando só IDs de frases."""

from ai_video_editor.revision.exceptions import ProjectStateError, RevisionError, RevisionRejectedError
from ai_video_editor.revision.provider import EditReviser
from ai_video_editor.revision.schema import (
    RequestHistoryItem,
    RevisionContext,
    RevisionDecision,
    RevisionRun,
    SentenceRestoration,
    SentenceState,
)
from ai_video_editor.revision.service import removed_ids_of, run_reviser, validate_revision

__all__ = [
    "EditReviser",
    "ProjectStateError",
    "RequestHistoryItem",
    "RevisionContext",
    "RevisionDecision",
    "RevisionError",
    "RevisionRejectedError",
    "RevisionRun",
    "SentenceRestoration",
    "SentenceState",
    "removed_ids_of",
    "run_reviser",
    "validate_revision",
]
