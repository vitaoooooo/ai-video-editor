"""Planejamento determinístico de cortes (EditPlan a partir de análises)."""

from ai_video_editor.planning.exceptions import NoSpeechDetectedError, PlanningError
from ai_video_editor.planning.schema import (
    PausePreset,
    SentenceCutOutcome,
    SentenceCutResult,
    SilenceCutParams,
    TimeRange,
    apply_pause_preset,
)
from ai_video_editor.planning.sentence_cut import (
    MIN_FRAGMENT_SEC,
    apply_sentence_cuts,
    kept_overlap,
    snap_cut,
    subtract_ranges,
)
from ai_video_editor.planning.silence_cut import build_edit_plan, plan_silence_cuts, removed_ranges

__all__ = [
    "MIN_FRAGMENT_SEC",
    "NoSpeechDetectedError",
    "PausePreset",
    "PlanningError",
    "SentenceCutOutcome",
    "SentenceCutResult",
    "SilenceCutParams",
    "TimeRange",
    "apply_pause_preset",
    "apply_sentence_cuts",
    "build_edit_plan",
    "kept_overlap",
    "plan_silence_cuts",
    "removed_ranges",
    "snap_cut",
    "subtract_ranges",
]
