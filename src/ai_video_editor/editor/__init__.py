"""Editor de IA: resumos por clipe, estrutura em blocos, três perguntas por bloco (cortes, inclusões e
respiros, coberturas) e montagem, só por IDs, letras e números.

O programa mede, descreve, confere e executa; a IA decide (ordem dos blocos, cortes, inclusões,
coberturas) e responde só com IDs do roteiro unificado (`roteiro/`) ou com as letras e números das listas
que o código montou. O código valida segurança e converte tudo em IDs e tempos.
"""

from ai_video_editor.editor.assemble import Assembled, ClipSource, assemble_plan, subtitle_sources
from ai_video_editor.editor.budget import Budget, resolve_budget
from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.prompt import (
    COVERAGE_PROMPT_VERSION,
    CUTS_PROMPT_VERSION,
    INSERTS_PROMPT_VERSION,
    STRUCTURE_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    EditorContext,
)
from ai_video_editor.editor.provider import EditorAI
from ai_video_editor.editor.schema import EditorPlan, EditorRun
from ai_video_editor.editor.service import EditorService, editor_cache

__all__ = [
    "COVERAGE_PROMPT_VERSION",
    "CUTS_PROMPT_VERSION",
    "INSERTS_PROMPT_VERSION",
    "STRUCTURE_PROMPT_VERSION",
    "SUMMARY_PROMPT_VERSION",
    "Assembled",
    "Budget",
    "ClipSource",
    "EditorAI",
    "EditorContext",
    "EditorMaterial",
    "EditorPlan",
    "EditorRun",
    "EditorService",
    "assemble_plan",
    "editor_cache",
    "resolve_budget",
    "subtitle_sources",
]
