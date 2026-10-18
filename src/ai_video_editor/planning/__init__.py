"""Planejamento determinístico de cortes (EditPlan a partir de análises)."""

from ai_video_editor.planning.exceptions import NoSpeechDetectedError, PlanningError
from ai_video_editor.planning.schema import SilenceCutParams, TimeRange
from ai_video_editor.planning.silence_cut import build_edit_plan, plan_silence_cuts, removed_ranges

__all__ = [
    "NoSpeechDetectedError",
    "PlanningError",
    "SilenceCutParams",
    "TimeRange",
    "build_edit_plan",
    "plan_silence_cuts",
    "removed_ranges",
]
