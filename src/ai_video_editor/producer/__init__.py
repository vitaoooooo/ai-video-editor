"""Produtor: lê o pedido e o resumo do material e escreve o briefing (receita, metas e instruções)."""

from ai_video_editor.producer.provider import BriefingProducer
from ai_video_editor.producer.schema import (
    BRIEFING_SCHEMA_VERSION,
    Briefing,
    BriefingContent,
    MaterialSummary,
    ProducerContext,
    RecipeInfo,
    RoleSummary,
    WorkEstimate,
)
from ai_video_editor.producer.service import (
    estimate_work,
    fallback_content,
    recipe_problems,
    run_producer,
    validate_content,
)
from ai_video_editor.producer.summary import recipe_info, registered_recipes, summarize_clips

BRIEFING_FILE = "briefing.json"

__all__ = [
    "BRIEFING_FILE",
    "BRIEFING_SCHEMA_VERSION",
    "Briefing",
    "BriefingContent",
    "BriefingProducer",
    "MaterialSummary",
    "ProducerContext",
    "RecipeInfo",
    "RoleSummary",
    "WorkEstimate",
    "estimate_work",
    "fallback_content",
    "recipe_info",
    "recipe_problems",
    "registered_recipes",
    "run_producer",
    "summarize_clips",
    "validate_content",
]
