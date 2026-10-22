"""Comando `catalog`: lê uma pasta de clipes (sem modificá-la) e gera o catálogo do projeto.

Etapas: varredura → metadados e miniaturas (ffprobe, signalstats; em paralelo) → detecção de fala
(VAD; em paralelo) → catálogo (datas, ajuste de relógio, ordem, IDs, folhas de contato, avisos).

O projeto fica em `<projects>/<pasta>-<hash>/catalog/`; o hash vem da lista de arquivos (caminho
relativo + tamanho + data de modificação), sem ler o conteúdo dos vídeos. A análise de cada clipe
fica em cache em `<projects>/_cache/clips/<chave>/` (chave = caminho + tamanho + mtime), fora do
projeto: acrescentar um clipe muda o hash do projeto, mas não reprocessa os outros.
"""

import hashlib
import json
import os
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor.analyzer.exceptions import NoVideoStreamError, VideoIngestError
from ai_video_editor.catalog.dates import is_future, recording_date
from ai_video_editor.catalog.media import probe_clip, thumbnails_and_color
from ai_video_editor.catalog.report import build_catalog_markdown
from ai_video_editor.catalog.roles import format_offset, match_folder_role, normalize_name
from ai_video_editor.catalog.schema import (
    ROLE_PREFIX,
    ROLES,
    CatalogClip,
    ClipAnalysis,
    ProjectCatalog,
    Role,
    RoleAssignment,
    SpeechDetection,
    StageTime,
)
from ai_video_editor.catalog.sheets import build_contact_sheet
from ai_video_editor.catalog.speech import SPEECH_MIN_RATIO, FasterWhisperVAD, SpeechDetector
from ai_video_editor.pipeline.project import DEFAULT_PROJECTS_ROOT, slugify
from ai_video_editor.pipeline.schema import StageProgress
from ai_video_editor.pipeline.timing import format_duration

# Sobe quando muda o que é calculado por clipe (refaz o cache).
CLIP_ANALYSIS_VERSION = 1

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm", ".3gp", ".wmv", ".mpg", ".mpeg",
    ".insv", ".360",
}

CATALOG_DIR = "catalog"
CATALOG_JSON = "catalog.json"
CATALOG_MD = "catalog.md"
THUMBS_DIR = "thumbs"
CLIP_CACHE_DIR = Path("_cache") / "clips"
ANALYSIS_FILE = "analysis.json"

CATALOG_STAGES = ("scan", "media", "speech", "catalog")
CATALOG_STAGE_LABELS = {
    "scan": "Varredura",
    "media": "Metadados e miniaturas",
    "speech": "Detecção de fala",
    "catalog": "Catálogo",
}

# Um papel cujas datas ficam a mais que isso do intervalo do principal tem relógio suspeito.
CLOCK_TOLERANCE = timedelta(minutes=30)

# Um clipe que "começa" antes de o anterior da mesma câmera terminar (data + duração), com esta
# folga para arredondamentos, indica datas de exportação em vez de gravação.
OVERLAP_TOLERANCE_SEC = 1.0

# (etapa, evento, detalhe); eventos: "start", "done", "cached", "notice".
CatalogProgress = Callable[[str, str, str], None]
CatalogStageProgress = Callable[[str, StageProgress], None]


class _Source:
    """Arquivo de vídeo encontrado na pasta."""

    def __init__(self, root: Path, path: Path):
        stat = path.stat()
        self.path = path.resolve()
        self.relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        self.folder = parts[0] if len(parts) > 1 else ""
        self.size = stat.st_size
        self.mtime_ns = stat.st_mtime_ns

    @property
    def cache_key(self) -> str:
        payload = json.dumps([str(self.path), self.size, self.mtime_ns, CLIP_ANALYSIS_VERSION])
        return hashlib.sha256(payload.encode()).hexdigest()[:20]


def scan_folder(source_dir: Path) -> tuple[list[_Source], int]:
    """Vídeos da pasta (recursivo, ordem alfabética) e quantos arquivos foram ignorados."""
    found: list[_Source] = []
    ignored = 0
    for path in sorted(source_dir.rglob("*"), key=lambda p: p.as_posix().lower()):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(source_dir).parts
        if any(part.startswith(".") for part in rel_parts) or path.name.startswith("._"):
            continue  # ocultos e lixo do macOS
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            found.append(_Source(source_dir, path))
        else:
            ignored += 1
    return found, ignored


def listing_hash(sources: list[_Source], length: int = 10) -> str:
    payload = json.dumps([[s.relative, s.size, s.mtime_ns] for s in sources])
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def catalog_project_dir(source_dir: Path, digest: str, root: Path = DEFAULT_PROJECTS_ROOT) -> Path:
    return root / f"{slugify(source_dir.name)}-{digest}"


def assign_roles(
    folders: list[str], overrides: dict[str, Role]
) -> tuple[list[RoleAssignment], list[str]]:
    """Papel de cada subpasta de primeiro nível; `overrides` (nome da pasta → papel) tem prioridade."""
    by_key = {normalize_name(name): role for name, role in overrides.items()}
    assignments: list[RoleAssignment] = []
    warnings: list[str] = []
    used: set[str] = set()
    for folder in folders:
        key = normalize_name(folder)
        if folder == "":
            role = by_key.get("", "principal")
            assignments.append(RoleAssignment(folder="", role=role, source="raiz"))
        elif key in by_key:
            used.add(key)
            assignments.append(RoleAssignment(folder=folder, role=by_key[key], source="--papel"))
        else:
            role, source, found = match_folder_role(folder)
            assignments.append(RoleAssignment(folder=folder, role=role, source=source))
            if len(found) > 1:
                warnings.append(
                    f"Pasta \"{folder}\" cita mais de um papel ({', '.join(found)}); ficou como outro. "
                    f"Use --papel \"{folder}=papel\"."
                )
    warnings += [
        f"--papel {name}={role}: nenhuma subpasta com esse nome."
        for name, role in overrides.items()
        if normalize_name(name) not in used and normalize_name(name) != ""
    ]
    return assignments, warnings


def natural_key(text: str) -> list:
    """Ordenação natural: `clip (4)` < `clip (15)` < `clip (27)`; sem diferenciar maiúsculas."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _name_key(clip: CatalogClip) -> tuple:
    return natural_key(Path(clip.relative_path).name), natural_key(clip.relative_path)


def date_conflicts(clips: list[CatalogClip]) -> list[tuple[CatalogClip, CatalogClip]]:
    """Pares (anterior, seguinte) da mesma pasta em que o seguinte "começa" antes de o anterior terminar.

    A comparação é por pasta (uma câmera): câmeras diferentes no mesmo papel podem gravar ao mesmo tempo.
    """
    conflicts: list[tuple[CatalogClip, CatalogClip]] = []
    for folder in sorted({c.folder for c in clips}):
        dated = sorted(
            (c for c in clips if c.folder == folder and c.adjusted_at is not None and c.media is not None),
            key=lambda c: (c.adjusted_at, _name_key(c)),
        )
        for before, after in zip(dated, dated[1:]):
            end = before.adjusted_at + timedelta(seconds=before.media.duration_sec - OVERLAP_TOLERANCE_SEC)
            if after.adjusted_at < end:
                conflicts.append((before, after))
    return conflicts


def order_role(members: list[CatalogClip]) -> list[CatalogClip]:
    """Ordem dos clipes de um papel: pela data; se as datas parecerem de exportação, pelo nome.

    Com datas sobrepostas, as datas do papel são marcadas como não confiáveis (`reliable=False`)
    e a origem da ordem fica em `order_source = "nome"`.
    """
    if not date_conflicts(members):
        return sorted(members, key=lambda c: (c.adjusted_at, _name_key(c)))
    for clip in members:
        clip.order_source = "nome"
        if clip.recorded_at is not None:
            clip.recorded_at = clip.recorded_at.model_copy(update={"reliable": False})
    return sorted(members, key=_name_key)


class CatalogBuilder:
    """Monta o catálogo de uma pasta de clipes, com cache por clipe."""

    def __init__(
        self,
        projects_root: Path = DEFAULT_PROJECTS_ROOT,
        speech_detector: Optional[SpeechDetector] = None,
        on_progress: Optional[CatalogProgress] = None,
        on_stage_progress: Optional[CatalogStageProgress] = None,
        workers: Optional[int] = None,
    ):
        self.projects_root = Path(projects_root)
        self.speech_detector = speech_detector or FasterWhisperVAD()
        self.on_progress = on_progress
        self.on_stage_progress = on_stage_progress
        self.workers = workers or min(8, os.cpu_count() or 2)

    def _emit(self, stage: str, event: str, detail: str = "") -> None:
        if self.on_progress is not None:
            self.on_progress(stage, event, detail)

    def _progress(self, stage: str, done: int, total: int) -> None:
        if self.on_stage_progress is not None:
            self.on_stage_progress(stage, StageProgress(done=done, total=total, unit="clips"))

    def build(
        self,
        source_dir: Path,
        role_overrides: Optional[dict[str, Role]] = None,
        clock_adjustments: Optional[dict[Role, float]] = None,
        force: bool = False,
    ) -> ProjectCatalog:
        started = time.perf_counter()
        source_dir = Path(source_dir).resolve()
        if not source_dir.is_dir():
            raise NotADirectoryError(f"pasta não encontrada: {source_dir}")
        clock_adjustments = dict(clock_adjustments or {})
        stage_times: list[StageTime] = []

        # 1. varredura
        t0 = self._start("scan")
        sources, ignored = scan_folder(source_dir)
        digest = listing_hash(sources)
        project_dir = catalog_project_dir(source_dir, digest, self.projects_root).resolve()
        folders = sorted({s.folder for s in sources}, key=str.lower)
        assignments, warnings = assign_roles(folders, role_overrides or {})
        role_of = {a.folder: a.role for a in assignments}
        detail = f"{len(sources)} vídeos em {len([f for f in folders if f]) or 1} pasta(s)"
        if ignored:
            detail += f", {ignored} arquivo(s) que não são vídeo ignorado(s)"
        stage_times.append(self._done("scan", t0, detail))

        # 2. metadados e miniaturas
        cache_root = self.projects_root.resolve() / CLIP_CACHE_DIR
        t0 = self._start("media")
        analyses, cached_keys = self._analyze_media(sources, cache_root, force)
        valid = [a for a in analyses if a.media is not None]
        cached = len(cached_keys)
        if cached == len(sources):
            stage_times.append(self._cached("media", t0))
        else:
            stage_times.append(self._done(
                "media", t0, f"{len(valid)} clipes lidos, {len(sources) - len(valid)} com problema, {cached} do cache"
            ))

        # 3. fala
        t0 = self._start("speech")
        if not self.speech_detector.available:
            warnings.append(
                "Detecção de fala indisponível: faster-whisper não instalado "
                "(pip install -e \".[transcription]\"); fala marcada como desconhecida."
            )
            self._emit("speech", "notice", "faster-whisper não instalado; fala marcada como desconhecida")
        ran = self._analyze_speech(sources, analyses, cache_root)
        if ran == 0:
            stage_times.append(self._cached("speech", t0))
        elif not self.speech_detector.available:
            stage_times.append(self._done("speech", t0, f"indisponível; {ran} clipes com fala desconhecida"))
        else:
            with_speech = sum(1 for a in analyses if a.speech and (a.speech.ratio or 0) >= SPEECH_MIN_RATIO)
            stage_times.append(self._done("speech", t0, f"{ran} clipes analisados, {with_speech} com fala"))

        # 4. catálogo
        t0 = self._start("catalog")
        catalog_dir = project_dir / CATALOG_DIR
        clips = self._catalog_clips(sources, analyses, role_of, clock_adjustments, cached_keys)
        warnings += project_warnings(clips, clock_adjustments)
        self._write_thumbnails(clips, sources, analyses, cache_root, catalog_dir)
        sheets = {}
        for role in ROLES:
            members = [c for c in clips if c.role == role and c.id and c.thumbnails]
            if members:
                name = f"contact_sheet_{role}.jpg"
                build_contact_sheet(members, catalog_dir, catalog_dir / name)
                sheets[role] = name
        for stale in catalog_dir.glob("contact_sheet_*.jpg"):
            if stale.name not in sheets.values():
                stale.unlink()
        catalog = ProjectCatalog(
            source_dir=source_dir,
            project_dir=project_dir,
            listing_hash=digest,
            roles=assignments,
            clock_adjustments={role: offset for role, offset in clock_adjustments.items()},
            clips=clips,
            ignored_files=ignored,
            warnings=warnings,
            contact_sheets=sheets,
        )
        stage_times.append(StageTime(name="catalog", elapsed_sec=round(time.perf_counter() - t0, 3)))
        catalog.stage_times = stage_times
        catalog.total_sec = round(time.perf_counter() - started, 3)
        catalog.save(catalog_dir / CATALOG_JSON)
        (catalog_dir / CATALOG_MD).write_text(build_catalog_markdown(catalog), encoding="utf-8")
        self._emit(
            "catalog", "done",
            f"{len([c for c in clips if c.id])} clipes, {len(warnings)} aviso(s) ({format_duration(stage_times[-1].elapsed_sec)})",
        )
        return catalog

    # --- etapas ---

    def _start(self, stage: str) -> float:
        self._emit(stage, "start")
        return time.perf_counter()

    def _done(self, stage: str, started: float, detail: str) -> StageTime:
        elapsed = time.perf_counter() - started
        self._emit(stage, "done", f"{detail} ({format_duration(elapsed)})")
        return StageTime(name=stage, elapsed_sec=round(elapsed, 3))

    def _cached(self, stage: str, started: float) -> StageTime:
        self._emit(stage, "cached")
        return StageTime(name=stage, elapsed_sec=round(time.perf_counter() - started, 3))

    def _load_cached(self, source: _Source, cache_dir: Path) -> Optional[ClipAnalysis]:
        path = cache_dir / ANALYSIS_FILE
        if not path.is_file():
            return None
        try:
            analysis = ClipAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if analysis.format_version != CLIP_ANALYSIS_VERSION or any(
            not (cache_dir / name).is_file() for name in analysis.thumbnails
        ):
            return None
        return analysis

    def _analyze_media(
        self, sources: list[_Source], cache_root: Path, force: bool
    ) -> tuple[list[ClipAnalysis], set[str]]:
        """Análise de mídia de cada clipe (do cache ou nova) e as chaves que vieram do cache."""
        cached_keys: set[str] = set()
        results: list[Optional[ClipAnalysis]] = [None] * len(sources)
        pending: list[int] = []
        for i, source in enumerate(sources):
            cached = None if force else self._load_cached(source, cache_root / source.cache_key)
            if cached is not None:
                results[i] = cached
                cached_keys.add(source.cache_key)
            else:
                pending.append(i)
        done = len(sources) - len(pending)
        self._progress("media", done, len(sources))

        def work(i: int) -> ClipAnalysis:
            source = sources[i]
            cache_dir = cache_root / source.cache_key
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            analysis = ClipAnalysis(
                format_version=CLIP_ANALYSIS_VERSION, path=source.path, size_bytes=source.size,
                mtime_ns=source.mtime_ns,
            )
            try:
                analysis.media, analysis.tag_date = probe_clip(source.path)
            except VideoIngestError as e:
                analysis.error = _media_error(e)
            else:
                names, color, warns = thumbnails_and_color(source.path, analysis.media, cache_dir)
                analysis.thumbnails, analysis.color = names, color
                analysis.warnings += warns
            _save_analysis(analysis, cache_dir)
            return analysis

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i, analysis in zip(pending, pool.map(work, pending)):
                results[i] = analysis
                done += 1
                self._progress("media", done, len(sources))
        return [r for r in results if r is not None], cached_keys

    def _analyze_speech(self, sources: list[_Source], analyses: list[ClipAnalysis], cache_root: Path) -> int:
        """Fala dos clipes com áudio que ainda não têm resultado deste detector; devolve quantos rodaram.

        Com `force`, a etapa de mídia já recriou as análises (sem fala), então tudo roda de novo.
        """
        detector = self.speech_detector
        pending: list[int] = []
        for i, analysis in enumerate(analyses):
            if analysis.media is None:
                continue
            if not analysis.media.has_audio:
                if analysis.speech is None or analysis.speech.status != "sem_audio":
                    analysis.speech = SpeechDetection(status="sem_audio")
                    _save_analysis(analysis, cache_root / sources[i].cache_key)
                continue
            speech = analysis.speech
            # Resultado de outro detector (ou "desconhecido") é refeito quando há um detector disponível.
            if speech is None or detector.available and (
                speech.detector != detector.name or speech.status == "desconhecido"
            ):
                pending.append(i)
        if not pending:
            return 0
        total = len(pending)
        self._progress("speech", 0, total)
        workers = max(1, min(4, self.workers))

        def work(i: int) -> None:
            analysis = analyses[i]
            analysis.speech = detector.detect(analysis.path, analysis.media.duration_sec)
            _save_analysis(analysis, cache_root / sources[i].cache_key)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, _ in enumerate(pool.map(work, pending), start=1):
                self._progress("speech", n, total)
        return total

    def _catalog_clips(
        self,
        sources: list[_Source],
        analyses: list[ClipAnalysis],
        role_of: dict[str, Role],
        clock_adjustments: dict[Role, float],
        cached_keys: set[str],
    ) -> list[CatalogClip]:
        clips: list[CatalogClip] = []
        for source, analysis in zip(sources, analyses):
            role = role_of[source.folder]
            clip = CatalogClip(
                role=role,
                folder=source.folder,
                relative_path=source.relative,
                size_bytes=source.size,
                media=analysis.media,
                speech=analysis.speech,
                color=analysis.color,
                warnings=list(analysis.warnings),
                error=analysis.error,
                cached=source.cache_key in cached_keys,
            )
            if analysis.media is not None:
                clip.recorded_at = recording_date(source.path, analysis.tag_date, source.mtime_ns)
                clip.clock_offset_sec = clock_adjustments.get(role, 0.0)
                clip.adjusted_at = clip.recorded_at.value + timedelta(seconds=clip.clock_offset_sec)
            clips.append(clip)

        ordered: list[CatalogClip] = []
        for role in ROLES:
            members = order_role([c for c in clips if c.role == role and c.error is None])
            for n, clip in enumerate(members, start=1):
                clip.id = f"{ROLE_PREFIX[role]}{n:03d}"
            ordered += members
        ordered += [c for c in clips if c.error is not None]
        return ordered

    def _write_thumbnails(
        self,
        clips: list[CatalogClip],
        sources: list[_Source],
        analyses: list[ClipAnalysis],
        cache_root: Path,
        catalog_dir: Path,
    ) -> None:
        """Copia as miniaturas do cache para `catalog/thumbs/<ID>_<pct>.jpg` (IDs mudam se a pasta mudar)."""
        thumbs_dir = catalog_dir / THUMBS_DIR
        if thumbs_dir.exists():
            shutil.rmtree(thumbs_dir)
        thumbs_dir.mkdir(parents=True)
        by_path = {s.relative: (s, a) for s, a in zip(sources, analyses)}
        for clip in clips:
            if clip.id is None:
                continue
            source, analysis = by_path[clip.relative_path]
            for name in analysis.thumbnails:
                target = f"{clip.id}_{name[1:]}"  # t50.jpg → P001_50.jpg
                shutil.copyfile(cache_root / source.cache_key / name, thumbs_dir / target)
                clip.thumbnails.append(f"{THUMBS_DIR}/{target}")


def _media_error(error: VideoIngestError) -> str:
    if isinstance(error, NoVideoStreamError):
        return "sem trilha de vídeo"
    return "arquivo ilegível ou corrompido"


def _save_analysis(analysis: ClipAnalysis, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / ANALYSIS_FILE).write_text(analysis.model_dump_json(indent=2), encoding="utf-8")


def _ids(clips: list[CatalogClip], limit: int = 12) -> str:
    ids = [c.id or c.relative_path for c in clips]
    return ", ".join(ids[:limit]) + (f" e mais {len(ids) - limit}" if len(ids) > limit else "")


def project_warnings(clips: list[CatalogClip], clock_adjustments: dict[Role, float]) -> list[str]:
    """Avisos do catálogo: problemas, relógios suspeitos, datas de mtime, formatos misturados, sem áudio."""
    warnings: list[str] = []
    broken = [c for c in clips if c.error is not None]
    for clip in broken:
        warnings.append(f"Arquivo com problema (fora do catálogo): {clip.relative_path} — {clip.error}.")
    valid = [c for c in clips if c.error is None]

    now = datetime.now(timezone.utc)
    future = [c for c in valid if c.adjusted_at and is_future(c.adjusted_at, now)]
    if future:
        warnings.append(f"Relógio suspeito: data de gravação no futuro em {_ids(future)}.")
    by_role = {role: [c for c in valid if c.role == role] for role in ROLES}
    principal = [c.adjusted_at for c in by_role["principal"] if c.recorded_at and c.recorded_at.reliable]
    if principal:
        low, high = min(principal) - CLOCK_TOLERANCE, max(principal) + CLOCK_TOLERANCE
        for role in ROLES[1:]:
            dates = [c.adjusted_at for c in by_role[role] if c.recorded_at and c.recorded_at.reliable]
            if not dates or any(low <= d <= high for d in dates):
                continue
            hours = (statistics.median(d.timestamp() for d in principal) - statistics.median(d.timestamp() for d in dates)) / 3600
            hint = f"--ajuste-relogio {role}={format_offset(round(hours) * 3600)}" if round(hours) else "--ajuste-relogio"
            note = " (já com ajuste)" if role in clock_adjustments else ""
            warnings.append(
                f"Relógio suspeito no papel {role}{note}: nenhuma data dentro do intervalo do principal "
                f"(diferença de ~{abs(hours):.1f}h); confira e, se for o caso, use {hint}."
            )
    for role in ROLES:
        conflicts = date_conflicts(by_role[role])
        if conflicts:
            pairs = [f"{after.id} começa antes de {before.id} terminar" for before, after in conflicts]
            shown = "; ".join(pairs[:3]) + (f"; e mais {len(pairs) - 3}" if len(pairs) > 3 else "")
            warnings.append(
                f"Datas não confiáveis ({role}): {shown}. As datas parecem de exportação, não de gravação; "
                "o papel foi ordenado pelo número no nome do arquivo (ordem natural)."
            )
    for role in ROLES:
        from_mtime = [c for c in by_role[role] if c.recorded_at and c.recorded_at.source == "mtime"]
        if from_mtime:
            warnings.append(
                f"Datas pouco confiáveis ({role}): {_ids(from_mtime)} sem data de gravação no arquivo nem no "
                "nome; usada a data de modificação, que muda ao copiar."
            )
    for role in ROLES:
        members = by_role[role]
        for label, values in (
            ("resoluções", {c.media.resolution for c in members}),
            ("fps", {f"{c.media.fps:g}" for c in members}),
            ("codecs", {c.media.codec for c in members}),
        ):
            if len(values) > 1:
                warnings.append(f"Formatos misturados ({role}): {label} {', '.join(sorted(values))}.")
    all_fps = {f"{c.media.fps:g}" for c in valid}
    if len(all_fps) > 1:
        warnings.append(f"Taxas de quadros diferentes no projeto: {', '.join(sorted(all_fps, key=float))} fps.")
    silent = [c for c in valid if not c.media.has_audio]
    if silent:
        warnings.append(f"Clipes sem áudio: {_ids(silent)}.")
    log = [c for c in valid if c.color and c.color.log_suspect]
    if log:
        warnings.append(
            f"Suspeita de perfil de cor log (baixo contraste e saturação): {_ids(log)}; confira antes de usar."
        )
    return warnings
