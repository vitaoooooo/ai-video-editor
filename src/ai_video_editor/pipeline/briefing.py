"""Briefing do Produtor no `edit`: reaproveitar ou refazer o `briefing.json` e escolher os valores efetivos.

Regra de cada valor: opção explícita da CLI > briefing > padrão do código; a origem fica registrada
(`EditSettings`) no `project.json` e no relatório.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from ai_video_editor.pipeline.project import file_hash, project_dir_for
from ai_video_editor.pipeline.schema import BriefingUse, EditSetting, EditSettings
from ai_video_editor.producer import BRIEFING_FILE
from ai_video_editor.producer.schema import BRIEFING_SCHEMA_VERSION, Briefing

DEFAULT_INTENSITY = "light"
DEFAULT_PACE = "normal"


@dataclass
class EditBriefing:
    """O briefing que o `edit` vai usar (ou nenhum), como foi obtido e avisos para mostrar."""

    briefing: Optional[Briefing]
    status: BriefingUse
    path: Optional[Path] = None
    notes: list[str] = field(default_factory=list)


def briefing_location(target: Path, projects_root: Path) -> Path:
    """`briefing.json` na pasta do projeto: a do catálogo (pasta de clipes) ou a do `edit` (um vídeo)."""
    target = Path(target).resolve()
    if target.is_dir():
        from ai_video_editor.pipeline.clips import catalog_location

        project_dir, _ = catalog_location(target, Path(projects_root))
    else:
        project_dir = project_dir_for(target, file_hash(target), Path(projects_root)).resolve()
    return project_dir / BRIEFING_FILE


def _load(path: Path) -> tuple[Optional[Briefing], Optional[str]]:
    """(briefing, motivo para ignorá-lo). Arquivo ausente: (None, None)."""
    if not path.is_file():
        return None, None
    try:
        briefing = Briefing.load(path)
    except ValueError as exc:
        return None, f"briefing.json ignorado (formato inválido: {exc.__class__.__name__})"
    if briefing.schema_version != BRIEFING_SCHEMA_VERSION:
        return None, f"briefing.json ignorado (formato antigo: versão {briefing.schema_version})"
    return briefing, None


def prepare_briefing(
    target: Path,
    request: Optional[str],
    projects_root: Path,
    producer_factory: Callable[[], object],
    forced_recipe: Optional[str] = None,
    speech_detector=None,
    on_progress=None,
    on_stage_progress=None,
) -> EditBriefing:
    """Reaproveita o `briefing.json` atual (mesmo pedido e mesmo material) ou roda o Produtor e o salva.

    - Sem pedido e sem `briefing.json` atual: nenhum briefing (o `edit` de antes).
    - Sem pedido e com `briefing.json` atual (ex.: do comando `briefing`): reaproveitado.
    - Com pedido: reaproveitado se o pedido e o material são os do arquivo; senão o Produtor roda de novo.

    Material: num vídeo, a pasta do projeto já é a do conteúdo do arquivo (hash); numa pasta, o resumo do
    catálogo atual (papéis, durações, fala) tem de ser o do briefing.

    Raises:
        RecipeError: `forced_recipe` não está registrada.
        VideoIngestError / FileNotFoundError: material ilegível (ao gerar o resumo).
    """
    from ai_video_editor.producer import run_producer
    from ai_video_editor.producer.material import load_material

    target = Path(target).resolve()
    request = (request or "").strip()
    path = briefing_location(target, projects_root)
    existing, problem = _load(path)
    notes = [problem] if problem else []
    if not request and existing is None:
        return EditBriefing(None, "sem_briefing", path, notes)

    reason: Optional[str] = None
    if existing is not None and request and existing.request.strip() != request:
        reason = "o pedido mudou"
    material = None
    if target.is_dir() or existing is None or reason:
        material = load_material(
            target, Path(projects_root), speech_detector=speech_detector,
            on_progress=on_progress, on_stage_progress=on_stage_progress,
        )
        path = material.project_dir / BRIEFING_FILE
        if existing is not None and reason is None and existing.material != material.summary:
            reason = "o material mudou (clipes, papéis ou fala)"
    if existing is not None and reason is None:
        return EditBriefing(existing, "reaproveitado", path, notes)
    if not request:
        notes.append(f"briefing.json ignorado: {reason}; rode `briefing` de novo ou passe --pedido")
        return EditBriefing(None, "sem_briefing", path, notes)

    briefing = run_producer(producer_factory(), request, material.summary, forced_recipe)
    briefing.save(path)
    if reason:
        notes.append(f"briefing refeito: {reason}")
    return EditBriefing(briefing, "gerado", path, notes)


def _pick(cli: Optional[Union[str, float]], from_briefing: Optional[Union[str, float]], default) -> EditSetting:
    if cli is not None:
        return EditSetting(value=cli, origin="cli")
    if from_briefing is not None:
        return EditSetting(value=from_briefing, origin="briefing")
    return EditSetting(value=default, origin="padrao")


def resolve_settings(
    edit_briefing: Optional[EditBriefing],
    default_recipe: str,
    recipe: Optional[str] = None,
    intensity: Optional[str] = None,
    pace: Optional[str] = None,
    instructions: Optional[str] = None,
    target_duration_sec: Optional[float] = None,
) -> EditSettings:
    """Valores efetivos: opção da CLI (não None) > briefing > padrão, com a origem de cada um.

    A receita de um briefing de fallback sem `--receita` não conta: a regra do fallback não sabe da
    análise visual nem do `--transcribe`, e o padrão do `edit` sabe (`vlog_cobertura`).
    """
    briefing = edit_briefing.briefing if edit_briefing is not None else None
    status: BriefingUse = edit_briefing.status if edit_briefing is not None else "sem_briefing"
    b_recipe = b_intensity = b_pace = b_instructions = b_target = None
    if briefing is not None:
        if briefing.origin == "ia" or briefing.forced_recipe is not None:
            b_recipe = briefing.recipe
        b_intensity, b_pace = briefing.intensity, briefing.pace
        b_instructions = briefing.director_instructions or None
        b_target = briefing.target_duration_sec
    return EditSettings(
        briefing=status,
        briefing_origin=briefing.origin if briefing is not None else None,
        request=briefing.request if briefing is not None else None,
        understood=briefing.understood if briefing is not None else None,
        recipe=_pick(recipe, b_recipe, default_recipe),
        intensity=_pick(intensity, b_intensity, DEFAULT_INTENSITY),
        pace=_pick(pace, b_pace, DEFAULT_PACE),
        instructions=_pick(instructions, b_instructions, None),
        target_duration_sec=_pick(target_duration_sec, b_target, None),
        notes=list(edit_briefing.notes) if edit_briefing is not None else [],
    )
