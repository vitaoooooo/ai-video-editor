"""Diretor de cobertura: B-roll/drone/POV por cima da fala e respiros, referenciando só IDs."""

from ai_video_editor.coverage.assemble import AssembledCoverage, assemble_coverage
from ai_video_editor.coverage.material import (
    CoverageError,
    build_visual_material,
    coverage_candidates,
    coverage_roles,
    coverage_sentences,
    find_mandatory_windows,
)
from ai_video_editor.coverage.prompt import COVERAGE_PROMPT_VERSION, build_coverage_prompt
from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import (
    BreathChoice,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    CoverageParams,
    CoverageRun,
    MandatoryWindow,
    VisualMaterial,
    VisualShot,
)
from ai_video_editor.coverage.service import coverage_problems, fill_windows, run_coverage

__all__ = [
    "COVERAGE_PROMPT_VERSION",
    "AssembledCoverage",
    "BreathChoice",
    "CoverageChoice",
    "CoverageContext",
    "CoverageDecision",
    "CoverageDirector",
    "CoverageError",
    "CoverageParams",
    "CoverageRun",
    "MandatoryWindow",
    "VisualMaterial",
    "VisualShot",
    "assemble_coverage",
    "build_coverage_prompt",
    "build_visual_material",
    "coverage_candidates",
    "coverage_problems",
    "coverage_roles",
    "coverage_sentences",
    "fill_windows",
    "find_mandatory_windows",
    "run_coverage",
]
