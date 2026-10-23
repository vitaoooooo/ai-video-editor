"""Versões do projeto: cada resultado (do `edit` ou de uma revisão) vive em `versions/vN/`.

O `edit` continua gravando na raiz do projeto (é o que o cache usa); ao terminar, o
resultado é copiado para uma versão nova, a menos que uma versão já o represente.
Projetos antigos, sem versões, viram v1 de forma transparente (`ensure_versions`).
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ai_video_editor.core.schema import EditPlan
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.pipeline.project import (
    COVERAGE_DECISION_FILE,
    COVERAGE_TIMELINE_FILE,
    DIRECTOR_DECISION_FILE,
    EDIT_DIFF_FILE,
    EDIT_PLAN_FILE,
    EDIT_REPORT_FILE,
    EDIT_TIMELINE_FILE,
    EDIT_TRANSCRIPT_FILE,
    SPINE_PLAN_FILE,
    TIMELINE_FILE,
    fingerprint,
)
from ai_video_editor.pipeline.schema import ProjectManifest, RunTimings, VersionRecord
from ai_video_editor.revision.exceptions import ProjectStateError

if TYPE_CHECKING:
    from ai_video_editor.pipeline.spine_schema import SpineManifest

VERSIONS_DIR = "versions"
VERSION_FILE = "version.json"
DEFAULT_EDIT_REQUEST = "Edição inicial"


def version_dir(project_dir: Path, number: int) -> Path:
    return project_dir / VERSIONS_DIR / f"v{number}"


def parse_version_ref(ref: str) -> int:
    """`"v2"` ou `"2"` → 2."""
    text = ref.strip().lower().removeprefix("v")
    if not text.isdigit() or int(text) < 1:
        raise ProjectStateError(f"versão inválida: '{ref}' (use algo como v2)")
    return int(text)


def get_version(manifest: ProjectManifest, number: int) -> VersionRecord:
    for record in manifest.versions:
        if record.number == number:
            return record
    existing = ", ".join(f"v{r.number}" for r in manifest.versions) or "nenhuma"
    raise ProjectStateError(f"versão v{number} não existe neste projeto (existentes: {existing})")


def current_version(manifest: ProjectManifest) -> VersionRecord:
    if manifest.current_version is None:
        raise ProjectStateError("o projeto não tem versão atual")
    return get_version(manifest, manifest.current_version)


def edit_fingerprint(manifest: ProjectManifest) -> Optional[str]:
    """Identifica o resultado do `edit` na raiz (render + legendas); None se incompleto."""
    render, report = manifest.stages["render"], manifest.stages["report"]
    if render.status != "done" or report.status != "done" or render.fingerprint is None:
        return None
    return fingerprint("edit-version", render.fingerprint, manifest.stages["subtitles"].fingerprint)


def _backfill_comparison(project_dir: Path, target: Path) -> None:
    """Versão criada antes da transcrição comparativa ou da linha do tempo: recebe os arquivos novos e o relatório refeito."""
    if not target.is_dir() or not (project_dir / EDIT_TRANSCRIPT_FILE).is_file():
        return
    if (target / EDIT_TRANSCRIPT_FILE).is_file() and (
        (target / EDIT_TIMELINE_FILE).is_file() or not (project_dir / EDIT_TIMELINE_FILE).is_file()
    ):
        return
    for name in (EDIT_TRANSCRIPT_FILE, EDIT_DIFF_FILE, EDIT_TIMELINE_FILE, EDIT_REPORT_FILE):
        if (project_dir / name).is_file():
            shutil.copy2(project_dir / name, target / name)


def register_edit_version(
    project_dir: Path,
    manifest: ProjectManifest,
    manifest_path: Path,
    timings: Optional[RunTimings] = None,
) -> Optional[VersionRecord]:
    """Copia o resultado atual do `edit` para uma versão nova, se nenhuma versão já o representa.

    Devolve a versão que representa o resultado (existente ou nova), ou None se o
    projeto não tem uma edição concluída. Salva o manifesto quando cria a versão.
    """
    fp = edit_fingerprint(manifest)
    plan_path = project_dir / EDIT_PLAN_FILE
    video_name = manifest.stages["render"].artifact
    if fp is None or video_name is None or not plan_path.is_file() or not (project_dir / video_name).is_file():
        return None
    for record in manifest.versions:
        if record.edit_fp == fp:
            _backfill_comparison(project_dir, version_dir(project_dir, record.number))
            return record

    number = max((r.number for r in manifest.versions), default=0) + 1
    target = version_dir(project_dir, number)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    try:
        shutil.copy2(project_dir / video_name, target / video_name)
        plan = EditPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        plan = plan.model_copy(update={"output_path": target / video_name})
        (target / EDIT_PLAN_FILE).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        shutil.copy2(project_dir / EDIT_REPORT_FILE, target / EDIT_REPORT_FILE)

        subtitles = None
        srt = manifest.stages["subtitles"]
        if srt.status == "done" and srt.artifact and (project_dir / srt.artifact).is_file():
            subtitles = srt.artifact
            shutil.copy2(project_dir / subtitles, target / subtitles)
        if manifest.stages["director"].status == "done" and (project_dir / DIRECTOR_DECISION_FILE).is_file():
            shutil.copy2(project_dir / DIRECTOR_DECISION_FILE, target / DIRECTOR_DECISION_FILE)
        for name in (EDIT_TRANSCRIPT_FILE, EDIT_DIFF_FILE, EDIT_TIMELINE_FILE):
            if (project_dir / name).is_file():
                shutil.copy2(project_dir / name, target / name)

        director_metrics = manifest.stages["director"].metrics
        record = VersionRecord(
            number=number,
            kind="edit",
            request=manifest.params.brief.instructions or DEFAULT_EDIT_REQUEST,
            parent=manifest.current_version,
            cut=manifest.params.cut,
            loudness=manifest.params.loudness,
            duration_sec=float(manifest.stages["render"].metrics["duration_sec"]),
            cuts=int(manifest.stages["report"].metrics.get("cuts", 0)),
            segments=len(plan.segments),
            removed_sentences=int(director_metrics.get("removed", 0)) if director_metrics else 0,
            video=video_name,
            subtitles=subtitles,
            edit_fp=fp,
            timings=timings,
        )
        (target / VERSION_FILE).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    manifest.versions.append(record)
    manifest.current_version = number
    manifest.save(manifest_path)
    return record


def ensure_versions(project_dir: Path, manifest: ProjectManifest, manifest_path: Path) -> ProjectManifest:
    """Migra um projeto antigo: sem versões, o resultado atual do `edit` vira v1.

    Raises:
        ProjectStateError: sem versões e sem edição concluída para migrar.
    """
    if not manifest.versions:
        if register_edit_version(project_dir, manifest, manifest_path) is None:
            raise ProjectStateError(
                "o projeto não tem edição concluída (rode `edit` antes de revisar ou listar versões)"
            )
    return manifest


def register_spine_version(
    project_dir: Path,
    manifest: "SpineManifest",
    manifest_path: Path,
    timings: Optional[RunTimings] = None,
) -> Optional[VersionRecord]:
    """Projeto com vários clipes: copia o resultado do `edit <pasta>` para uma versão nova, como no
    `register_edit_version` (vídeo, timeline.json, relatórios, legendas, decisão do Diretor).

    Com cobertura, a Timeline renderizada é a `coverage_timeline.json` (é ela que aponta para o vídeo da
    versão); a espinha (`timeline.json`) e a `coverage_decision.json` vão junto, como referência."""
    render, report = manifest.stages.get("render"), manifest.stages.get("report")
    timeline_path = project_dir / TIMELINE_FILE
    if (
        render is None or report is None or render.status != "done" or report.status != "done"
        or render.artifact is None or not (project_dir / render.artifact).is_file() or not timeline_path.is_file()
    ):
        return None
    subtitles_record = manifest.stages.get("subtitles")
    fp = fingerprint("edit-version", render.fingerprint, subtitles_record.fingerprint if subtitles_record else None)
    for record in manifest.versions:
        if record.edit_fp == fp:
            return record

    number = max((r.number for r in manifest.versions), default=0) + 1
    target = version_dir(project_dir, number)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    video_name = render.artifact
    try:
        shutil.copy2(project_dir / video_name, target / video_name)
        coverage = manifest.stages.get("coverage")
        with_coverage = (
            coverage is not None and coverage.status == "done" and (project_dir / COVERAGE_TIMELINE_FILE).is_file()
        )
        rendered_file = COVERAGE_TIMELINE_FILE if with_coverage else TIMELINE_FILE
        if with_coverage:
            shutil.copy2(timeline_path, target / TIMELINE_FILE)
        timeline = Timeline.model_validate_json((project_dir / rendered_file).read_text(encoding="utf-8"))
        timeline = timeline.model_copy(update={"output_path": target / video_name})
        (target / rendered_file).write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        subtitles = None
        if subtitles_record is not None and subtitles_record.status == "done" and subtitles_record.artifact:
            if (project_dir / subtitles_record.artifact).is_file():
                subtitles = subtitles_record.artifact
                shutil.copy2(project_dir / subtitles, target / subtitles)
        director = manifest.stages.get("director")
        names = [EDIT_REPORT_FILE, SPINE_PLAN_FILE, EDIT_TRANSCRIPT_FILE, EDIT_DIFF_FILE, EDIT_TIMELINE_FILE]
        if director is not None and director.status == "done":
            names.append(DIRECTOR_DECISION_FILE)
        if with_coverage:
            names.append(COVERAGE_DECISION_FILE)
        for name in names:
            if (project_dir / name).is_file():
                shutil.copy2(project_dir / name, target / name)
        record = VersionRecord(
            number=number,
            kind="edit",
            request=manifest.params.brief.instructions or DEFAULT_EDIT_REQUEST,
            parent=manifest.current_version,
            cut=manifest.params.cut,
            loudness=manifest.params.loudness,
            duration_sec=float(render.metrics["duration_sec"]),
            cuts=int(report.metrics.get("cuts", 0)),
            segments=len(timeline.video_main),
            removed_sentences=int(director.metrics.get("removed", 0)) if director is not None and director.metrics else 0,
            video=video_name,
            subtitles=subtitles,
            edit_fp=fp,
            timings=timings,
        )
        (target / VERSION_FILE).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    manifest.versions.append(record)
    manifest.current_version = number
    manifest.save(manifest_path)
    return record
