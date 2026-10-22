"""Clipes de um projeto de pasta: catálogo, escolha dos clipes da espinha e cache por clipe.

O catálogo é só lido (`catalog/` não muda aqui); se o `catalog.json` ainda não existe, ele é gerado
pela API pública do módulo (`CatalogBuilder`), no mesmo diretório de projeto do comando `catalog`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ai_video_editor.pipeline.project import CLIP_STAGE_CACHE_DIR
from ai_video_editor.pipeline.spine_schema import SpineClip, SpineSelection, UnusedClip

if TYPE_CHECKING:
    # `catalog.builder` importa `pipeline`; os imports de execução ficam dentro das funções.
    from ai_video_editor.catalog import CatalogBuilder, ProjectCatalog

# Um clipe principal entra na espinha com fala em pelo menos 20% do tempo OU pelo menos 3 s de fala
# (um clipe longo com uma fala curta ainda conta a história). Abaixo disso, fica para a cobertura.
MIN_SPEECH_RATIO = 0.20
MIN_SPEECH_SEC = 3.0
# Sobe quando a regra de escolha muda (refaz a etapa do catálogo).
SELECTION_VERSION = 1

COVERAGE_NOTE = "disponível para cobertura (rodada 3)"


def catalog_location(source_dir: Path, projects_root: Path) -> tuple[Path, str]:
    """(pasta do projeto, hash da listagem) de uma pasta de clipes, como o comando `catalog` calcula."""
    from ai_video_editor.catalog.builder import catalog_project_dir, listing_hash, scan_folder

    sources, _ = scan_folder(source_dir)
    digest = listing_hash(sources)
    return catalog_project_dir(source_dir, digest, projects_root).resolve(), digest


def catalog_path(project_dir: Path) -> Path:
    from ai_video_editor.catalog.builder import CATALOG_DIR, CATALOG_JSON

    return project_dir / CATALOG_DIR / CATALOG_JSON


def load_or_build_catalog(
    source_dir: Path, projects_root: Path, builder_factory: Callable[[], "CatalogBuilder"]
) -> tuple["ProjectCatalog", bool]:
    """(catálogo, se foi gerado agora). Um `catalog.json` atual (mesma listagem e `schema_version`, a
    mesma regra do Produtor) é reaproveitado como está, preservando os papéis (`--papel`) e ajustes de
    relógio de quem rodou o `catalog`; faltando ou em formato antigo, é gerado de novo."""
    from ai_video_editor.catalog.builder import current_catalog

    catalog = current_catalog(source_dir, projects_root)
    if catalog is not None:
        return catalog, False
    return builder_factory().build(source_dir), True


def _has_enough_speech(ratio: Optional[float], speech_sec: Optional[float]) -> bool:
    return (ratio or 0.0) >= MIN_SPEECH_RATIO or (speech_sec or 0.0) >= MIN_SPEECH_SEC


def _id_order(clip_id: str) -> tuple[str, int]:
    """`P012` -> ("P", 12): ordena os IDs pelo número (P1000 depois de P999)."""
    digits = clip_id[1:]
    return clip_id[:1], int(digits) if digits.isdigit() else 0


def select_spine_clips(catalog: "ProjectCatalog") -> SpineSelection:
    """Clipes da espinha: papel principal, sem erro, com fala acima do mínimo, na ordem do catálogo.

    Fala `desconhecido` (VAD indisponível) entra, com aviso: a transcrição decide depois. Os
    principais sem fala e os clipes de outros papéis ficam disponíveis para a cobertura.
    """
    source_dir = Path(catalog.source_dir)
    clips: list[SpineClip] = []
    unused: list[UnusedClip] = []
    warnings: list[str] = []
    unknown: list[str] = []
    principal = sorted(
        (c for c in catalog.clips if c.role == "principal" and c.id is not None), key=lambda c: _id_order(c.id)
    )
    for clip in principal:
        media, speech = clip.media, clip.speech
        duration = media.duration_sec if media is not None else None
        reason: Optional[str] = None
        coverage = False
        if clip.error is not None or media is None:
            reason = f"arquivo com problema: {clip.error or 'sem metadados'}"
        elif not media.has_audio or (speech is not None and speech.status == "sem_audio"):
            reason, coverage = "sem áudio", True
        elif speech is not None and speech.status == "analisada" and not _has_enough_speech(
            speech.ratio, speech.speech_sec
        ):
            pct = f"{(speech.ratio or 0) * 100:.0f}%"
            reason = f"pouca fala ({pct}, {speech.speech_sec or 0:.1f}s)"
            coverage = True
        if reason is not None:
            unused.append(
                UnusedClip(
                    id=clip.id, relative_path=clip.relative_path, role=clip.role,
                    duration_sec=duration, reason=reason, coverage=coverage,
                )
            )
            continue
        status = speech.status if speech is not None else "desconhecido"
        if status != "analisada":
            unknown.append(clip.id)
        clips.append(
            SpineClip(
                id=clip.id,
                path=(source_dir / clip.relative_path).resolve(),
                relative_path=clip.relative_path,
                duration_sec=media.duration_sec,
                fps=media.fps,
                width=media.width,
                height=media.height,
                speech_ratio=speech.ratio if speech is not None else None,
                speech_sec=speech.speech_sec if speech is not None else None,
                speech_status=status,
            )
        )
    for clip in catalog.clips:
        if clip.role == "principal" and clip.id is not None:
            continue
        reason = f"arquivo com problema: {clip.error}" if clip.error is not None else f"papel {clip.role}"
        unused.append(
            UnusedClip(
                id=clip.id, relative_path=clip.relative_path, role=clip.role,
                duration_sec=clip.media.duration_sec if clip.media is not None else None,
                reason=reason, coverage=clip.error is None,
            )
        )
    if unknown:
        warnings.append(
            f"Fala não verificada pelo catálogo em {', '.join(unknown)} (detecção de fala indisponível); "
            "esses clipes entram na espinha e a transcrição decide."
        )
    return SpineSelection(
        source_dir=source_dir,
        catalog_path=catalog_path(Path(catalog.project_dir)),
        clips=clips,
        unused=unused,
        warnings=warnings,
    )


def clip_cache_file(projects_root: Path, file_hash: str, stage: str, fp: str) -> Path:
    """Arquivo do cache por clipe: chave = hash do arquivo + fingerprint dos parâmetros da etapa."""
    return projects_root.resolve() / CLIP_STAGE_CACHE_DIR / file_hash / f"{stage}-{fp}.json"
