"""Clipes de um projeto de pasta: catálogo, escolha dos clipes da espinha e cache por clipe.

O catálogo é só lido (`catalog/` não muda aqui); se o `catalog.json` ainda não existe, ele é gerado
pela API pública do módulo (`CatalogBuilder`), no mesmo diretório de projeto do comando `catalog`.

Silêncios e transcrição por clipe (fingerprints e cache em `_cache/edit_clips/`) ficam aqui para
serem os mesmos em quem os usa (`edit <pasta>` e `roteiro`): um clipe já processado não é refeito.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

from ai_video_editor.audio import AudioAnalysisResult, detect_silences
from ai_video_editor.audio.levels import AUTO_RULE_VERSION
from ai_video_editor.pipeline.project import CLIP_STAGE_CACHE_DIR, fingerprint
from ai_video_editor.pipeline.spine_schema import SpineClip, SpineSelection, UnusedClip
from ai_video_editor.transcription.exceptions import TranscriptionError
from ai_video_editor.transcription.normalize import SegmentAdjustments
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig

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


# --- silêncios e transcrição por clipe (mesmas chaves do `edit <vídeo>`) ---


def silences_fingerprint(file_hash: str, threshold_db: Union[float, str], min_silence_sec: float) -> str:
    """Chave dos silêncios de um arquivo. No modo `auto` vale o modo + a versão da regra."""
    threshold_key = ("auto", AUTO_RULE_VERSION) if threshold_db == "auto" else ("fixed", threshold_db)
    return fingerprint("silences", file_hash, threshold_key, min_silence_sec)


def load_or_detect_silences(
    path: Path, cache_path: Path, threshold_db: Union[float, str], min_silence_sec: float, force: bool = False
) -> tuple[AudioAnalysisResult, bool]:
    """(silêncios do clipe, veio do cache por clipe)."""
    if cache_path.is_file() and not force:
        return AudioAnalysisResult.load_json(cache_path), True
    result = detect_silences(path, threshold_db, min_silence_sec)
    result.save_json(cache_path)
    return result, False


def effective_transcription_config(provider: TranscriptionProvider, config: TranscriptionConfig) -> TranscriptionConfig:
    """Configuração com modelo, dispositivo e compute_type efetivos (resolvidos de "auto"): é o que entra no cache."""
    effective = provider.effective_settings(config)
    return config.model_copy(
        update={"model_size": provider.effective_model(config), "device": effective[0], "compute_type": effective[1]}
    )


def transcription_fingerprint(file_hash: str, provider: TranscriptionProvider, config: TranscriptionConfig) -> str:
    return fingerprint("transcription", file_hash, provider.provider_name, effective_transcription_config(provider, config))


def transcribe_to_cache(
    provider: TranscriptionProvider,
    label: str,
    path: Path,
    cache_path: Path,
    config: TranscriptionConfig,
    on_progress=None,
) -> Transcript:
    """Transcreve um clipe e grava no cache por clipe, com os ajustes da normalização ao lado.

    O erro diz o clipe (`label`): com 22 clipes, "segmentos fora de ordem" sozinho não diz onde olhar.
    """
    try:
        transcript = provider.transcribe(path, config, on_progress=on_progress)
    except Exception as exc:
        raise TranscriptionError(f"{label} ({path.name}): {exc}") from exc
    transcript.save(cache_path)
    sidecar = adjustments_path(cache_path)
    adjustments = provider.last_adjustments
    if adjustments is not None and adjustments.changed:
        sidecar.write_text(adjustments.model_dump_json(indent=2), encoding="utf-8")
    elif sidecar.is_file():
        sidecar.unlink()
    return transcript


def adjustments_path(transcript_path: Path) -> Path:
    """Ajustes da normalização ao lado da transcrição do cache por clipe (só existe se houve ajuste)."""
    return transcript_path.with_name(f"{transcript_path.stem}.adjustments.json")


def load_adjustments(transcript_path: Path) -> Optional[SegmentAdjustments]:
    sidecar = adjustments_path(transcript_path)
    if not sidecar.is_file():
        return None
    return SegmentAdjustments.model_validate_json(sidecar.read_text(encoding="utf-8"))
