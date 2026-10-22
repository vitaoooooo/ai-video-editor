"""Material do briefing: catálogo da pasta (reaproveitado ou refeito) ou resumo de um arquivo único."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai_video_editor.catalog.builder import (
    CatalogBuilder,
    CatalogProgress,
    CatalogStageProgress,
    current_catalog,
    describe_clip,
)
from ai_video_editor.catalog.schema import ProjectCatalog
from ai_video_editor.catalog.speech import FasterWhisperVAD, SpeechDetector
from ai_video_editor.pipeline.project import DEFAULT_PROJECTS_ROOT, file_hash, project_dir_for
from ai_video_editor.producer.schema import MaterialSummary
from ai_video_editor.producer.summary import summarize_clips


@dataclass
class Material:
    summary: MaterialSummary
    project_dir: Path
    catalog: Optional[ProjectCatalog] = None
    # "reaproveitado" (catálogo atual), "gerado" (catalog rodou agora) ou "arquivo" (resumo mínimo)
    source: str = "arquivo"


def load_material(
    target: Path,
    projects_root: Path = DEFAULT_PROJECTS_ROOT,
    speech_detector: Optional[SpeechDetector] = None,
    on_progress: Optional[CatalogProgress] = None,
    on_stage_progress: Optional[CatalogStageProgress] = None,
) -> Material:
    """Pasta: catálogo atual ou um novo (cache por clipe). Arquivo: ffprobe + fala, na pasta de projeto do `edit`.

    Raises:
        FileNotFoundError: caminho inexistente.
        VideoIngestError: arquivo ilegível ou sem vídeo.
    """
    target = Path(target).resolve()
    projects_root = Path(projects_root)
    if target.is_dir():
        catalog = current_catalog(target, projects_root)
        source = "reaproveitado"
        if catalog is None:
            builder = CatalogBuilder(
                projects_root=projects_root, speech_detector=speech_detector,
                on_progress=on_progress, on_stage_progress=on_stage_progress,
            )
            catalog = builder.build(target)
            source = "gerado"
        summary = summarize_clips(catalog.clips, "pasta", target.name)
        return Material(summary=summary, project_dir=catalog.project_dir, catalog=catalog, source=source)
    if not target.is_file():
        raise FileNotFoundError(f"arquivo ou pasta não encontrado: {target}")
    clip = describe_clip(target, speech_detector or FasterWhisperVAD())
    project_dir = project_dir_for(target, file_hash(target), projects_root).resolve()
    return Material(summary=summarize_clips([clip], "arquivo", target.name), project_dir=project_dir)
