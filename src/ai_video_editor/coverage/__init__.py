"""Diretor de cobertura: B-roll/drone/POV por cima da fala onde a imagem mostra o que é falado, e respiros
entre os assuntos, referenciando só IDs e evidências conferidas pelo código."""

from ai_video_editor.coverage.assemble import AssembledCoverage, assemble_coverage
from ai_video_editor.coverage.material import (
    CoverageError,
    annotate_screen,
    build_visual_material,
    coverage_candidates,
    coverage_roles,
    coverage_sentences,
    find_bad_spans,
)
from ai_video_editor.coverage.plan import PLAN_FILE, plan_lines, plan_markdown
from ai_video_editor.coverage.prompt import COVERAGE_PROMPT_VERSION, build_coverage_prompt, build_topic_prompt
from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import (
    BadSpan,
    BreathChoice,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    CoverageParams,
    CoverageRun,
    Topic,
    TopicBlock,
    TopicMap,
    VisualMaterial,
    VisualShot,
)
from ai_video_editor.coverage.service import CoverageState, apply_limit, fill_uniform, review_decision, run_coverage
from ai_video_editor.coverage.topics import topics_by_clip, validate_topics

__all__ = [
    "COVERAGE_PROMPT_VERSION",
    "PLAN_FILE",
    "AssembledCoverage",
    "BadSpan",
    "BreathChoice",
    "CoverageChoice",
    "CoverageContext",
    "CoverageDecision",
    "CoverageDirector",
    "CoverageError",
    "CoverageParams",
    "CoverageRun",
    "CoverageState",
    "Topic",
    "TopicBlock",
    "TopicMap",
    "VisualMaterial",
    "VisualShot",
    "annotate_screen",
    "apply_limit",
    "assemble_coverage",
    "build_coverage_prompt",
    "build_topic_prompt",
    "build_visual_material",
    "coverage_candidates",
    "coverage_roles",
    "coverage_sentences",
    "fill_uniform",
    "find_bad_spans",
    "plan_lines",
    "plan_markdown",
    "review_decision",
    "run_coverage",
    "topics_by_clip",
    "validate_topics",
]
