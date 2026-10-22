"""Comando `visual`: tomadas, miniaturas, medidas e descrição por IA dos clipes de um projeto.

Etapas: catálogo (lê o existente ou gera) → tomadas e medidas (FFmpeg, por clipe, em paralelo e
com cache) → descrição por IA (lotes de miniaturas rotuladas) → relatório (`visual/visual.json`
e `visual/visual.md` na pasta do projeto).

Cache por clipe em `<projects>/_cache/clips/visual-<chave>/` (chave = caminho + tamanho + mtime +
parâmetros das tomadas): `shots.json`, as miniaturas e as análises da IA (`ai-<fingerprint>.json`,
fingerprint = cadeia pedida + raciocínio + detalhe da imagem + versão do prompt). Uma tomada
`sem_analise` é pedida de novo na execução seguinte; as analisadas não.
"""

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ai_video_editor.catalog import CatalogBuilder, CatalogClip, ProjectCatalog
from ai_video_editor.catalog.builder import (
    CATALOG_DIR,
    CATALOG_JSON,
    CLIP_CACHE_DIR,
    catalog_project_dir,
    listing_hash,
    scan_folder,
)
from ai_video_editor.catalog.schema import ROLES, Role, StageTime
from ai_video_editor.llm.schema import ImageDetail
from ai_video_editor.pipeline.project import DEFAULT_PROJECTS_ROOT
from ai_video_editor.pipeline.schema import StageProgress
from ai_video_editor.pipeline.timing import format_duration
from ai_video_editor.visual.describer import (
    PROMPT_VERSION,
    ShotRequest,
    VisualDescriber,
    estimate_tokens,
    make_batches,
)
from ai_video_editor.visual.report import build_visual_markdown
from ai_video_editor.visual.schema import (
    ClipVisual,
    ProjectVisual,
    Shot,
    ShotAnalysis,
    ShotBoundary,
    ShotFrame,
    ShotMeasures,
    VisualAIRun,
)
from ai_video_editor.visual.shots import (
    ShotSpan,
    extract_frames,
    frame_times,
    measure_shot,
    scan_clip,
    shot_params,
    split_shots,
)

# Sobe quando muda o que é guardado no cache de tomadas.
VISUAL_CACHE_VERSION = 1

VISUAL_DIR = "visual"
VISUAL_JSON = "visual.json"
VISUAL_MD = "visual.md"
FRAMES_DIR = "frames"
SHOTS_FILE = "shots.json"

VISUAL_STAGES = ("catalog", "shots", "ai", "report")
VISUAL_STAGE_LABELS = {
    "catalog": "Catálogo",
    "shots": "Tomadas e medidas",
    "ai": "Descrição por IA",
    "report": "Relatório",
}

# Sobreposição mínima com a fala detectada para a tomada contar "com fala".
SPEECH_OVERLAP_SEC = 1.0
SPEECH_OVERLAP_FRACTION = 0.3

# (etapa, evento, detalhe); eventos: "start", "done", "cached", "skipped", "notice".
VisualProgress = Callable[[str, str, str], None]
VisualStageProgress = Callable[[str, StageProgress], None]


class CachedShot(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    boundary: ShotBoundary
    frame_times: list[float]
    frame_files: list[str] = Field(description="Na pasta do cache do clipe")
    measures: ShotMeasures


class ClipShotsCache(BaseModel):
    format_version: int
    path: Path
    size_bytes: int
    mtime_ns: int
    params: dict[str, float]
    duration_sec: float
    shots: list[CachedShot] = Field(default_factory=list)


class CachedAnalyses(BaseModel):
    """Análises da IA de um clipe, por índice da tomada; `best_frames` guardado como F1..F3."""

    fingerprint: str
    analyses: dict[int, ShotAnalysis] = Field(default_factory=dict)


class VisualEstimate(BaseModel):
    """O que `--so-estimar` mostra: volume de trabalho da IA, sem chamá-la."""

    shots: int
    images: int
    cached_shots: int
    calls: int
    input_tokens: int
    output_tokens: int
    image_detail: ImageDetail
    shots_by_role: dict[str, int] = Field(default_factory=dict)


class VisualResult(BaseModel):
    project_dir: Path
    visual: Optional[ProjectVisual] = None
    estimate: Optional[VisualEstimate] = None


def clip_cache_key(path: Path, size: int, mtime_ns: int) -> str:
    payload = json.dumps([str(path), size, mtime_ns, VISUAL_CACHE_VERSION, shot_params()], sort_keys=True)
    return "visual-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def ai_fingerprint(describer: VisualDescriber) -> str:
    payload = json.dumps([PROMPT_VERSION, describer.model_chain, describer.chain.thinking, describer.detail])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def find_catalog(folder: Path, projects_root: Path = DEFAULT_PROJECTS_ROOT) -> Optional[ProjectCatalog]:
    """Catálogo existente: `folder` é a pasta do projeto ou a pasta de clipes já catalogada."""
    own = folder / CATALOG_DIR / CATALOG_JSON
    if own.is_file():
        return ProjectCatalog.load(own)
    sources, _ = scan_folder(folder.resolve())
    project = catalog_project_dir(folder.resolve(), listing_hash(sources), projects_root)
    path = project / CATALOG_DIR / CATALOG_JSON
    return ProjectCatalog.load(path) if path.is_file() else None


def speech_overlap(clip: CatalogClip, start: float, end: float) -> Optional[bool]:
    """True se a tomada sobrepõe fala detectada; None se a fala do clipe é desconhecida."""
    speech = clip.speech
    if speech is None or speech.status == "desconhecido":
        return None
    if speech.status == "sem_audio":
        return False
    overlap = sum(max(0.0, min(end, s.end_sec) - max(start, s.start_sec)) for s in speech.segments)
    return overlap >= min(SPEECH_OVERLAP_SEC, SPEECH_OVERLAP_FRACTION * (end - start))


def _local_frames(frames: list[str]) -> list[str]:
    return [f.rsplit(".", 1)[-1] for f in frames]  # P007.T03.F1 → F1


class VisualBuilder:
    """Monta a análise visual de um projeto, com cache por clipe."""

    def __init__(
        self,
        projects_root: Path = DEFAULT_PROJECTS_ROOT,
        describer: Optional[VisualDescriber] = None,
        detail: ImageDetail = "auto",
        on_progress: Optional[VisualProgress] = None,
        on_stage_progress: Optional[VisualStageProgress] = None,
        catalog_builder: Optional[CatalogBuilder] = None,
        workers: Optional[int] = None,
    ):
        self.projects_root = Path(projects_root)
        self.describer = describer
        self.detail = describer.detail if describer is not None else detail
        self.on_progress = on_progress
        self.on_stage_progress = on_stage_progress
        self.catalog_builder = catalog_builder
        self.workers = workers or max(1, min(4, os.cpu_count() or 2))

    # --- eventos ---

    def _emit(self, stage: str, event: str, detail: str = "") -> None:
        if self.on_progress is not None:
            self.on_progress(stage, event, detail)

    def _progress(self, stage: str, progress: StageProgress) -> None:
        if self.on_stage_progress is not None:
            self.on_stage_progress(stage, progress)

    def _start(self, stage: str) -> float:
        self._emit(stage, "start")
        return time.perf_counter()

    def _done(self, stage: str, started: float, detail: str) -> StageTime:
        elapsed = time.perf_counter() - started
        self._emit(stage, "done", f"{detail} ({format_duration(elapsed)})")
        return StageTime(name=stage, elapsed_sec=round(elapsed, 3))

    # --- execução ---

    def build(
        self,
        folder: Path,
        roles: Optional[list[Role]] = None,
        estimate_only: bool = False,
        force: bool = False,
    ) -> VisualResult:
        started = time.perf_counter()
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"pasta não encontrada: {folder}")
        roles = list(roles or ROLES)
        stage_times: list[StageTime] = []

        # 1. catálogo
        t0 = self._start("catalog")
        catalog = find_catalog(folder, self.projects_root)
        if catalog is None:
            self._emit("catalog", "notice", "catálogo não encontrado; catalogando a pasta")
            builder = self.catalog_builder or CatalogBuilder(projects_root=self.projects_root)
            catalog = builder.build(folder)
            how = "gerado"
        else:
            how = "lido"
        clips = [c for c in catalog.clips if c.id and c.error is None and c.role in roles]
        stage_times.append(self._done("catalog", t0, f"{how}; {len(clips)} clipe(s) nos papéis {', '.join(roles)}"))
        project_dir = Path(catalog.project_dir)
        visual_dir = project_dir / VISUAL_DIR

        # 2. tomadas e medidas
        t0 = self._start("shots")
        cache_root = self.projects_root.resolve() / CLIP_CACHE_DIR
        caches, cached_ids, errors = self._shots(clips, Path(catalog.source_dir), cache_root, force)
        visuals = [self._clip_visual(clip, caches.get(clip.id), clip.id in cached_ids, errors.get(clip.id))
                   for clip in clips]
        total_shots = sum(len(v.shots) for v in visuals)
        detail = f"{total_shots} tomadas em {len(clips)} clipe(s), {len(cached_ids)} do cache"
        if errors:
            detail += f", {len(errors)} com erro"
        stage_times.append(self._done("shots", t0, detail))

        fingerprint = ai_fingerprint(self.describer) if self.describer is not None else None
        cached_ai = {
            clip.id: self._load_analyses(cache_root / caches[clip.id][0], fingerprint)
            for clip in clips if clip.id in caches and fingerprint is not None
        }
        for visual in visuals:
            self._apply_cached(visual, cached_ai.get(visual.clip_id, {}))

        if estimate_only:
            estimate = self._estimate(visuals)
            return VisualResult(project_dir=project_dir, estimate=estimate)

        # 3. descrição por IA
        ai_run = None
        if self.describer is None:
            self._emit("ai", "skipped", "sem provedor de IA (--visual-provider none)")
        else:
            t0 = self._start("ai")
            ai_run = self._describe(visuals, clips, caches, cache_root, fingerprint)
            done = sum(1 for s in (s for v in visuals for s in v.shots) if s.status == "analisada")
            failed = sum(1 for s in (s for v in visuals for s in v.shots) if s.status == "sem_analise")
            detail = f"{done} tomadas analisadas ({ai_run.cached_shots} do cache), {ai_run.batches} lote(s)"
            if failed:
                detail += f", {failed} sem análise"
            stage_times.append(self._done("ai", t0, detail))

        # 4. relatório
        t0 = self._start("report")
        self._write_frames(visuals, caches, cache_root, visual_dir)
        warnings = [f"{clip_id}: {error}" for clip_id, error in errors.items()]
        visual = ProjectVisual(
            project_dir=project_dir,
            source_dir=catalog.source_dir,
            roles=roles,
            clips=visuals,
            ai=ai_run,
            warnings=warnings,
        )
        stage_times.append(StageTime(name="report", elapsed_sec=round(time.perf_counter() - t0, 3)))
        visual.stage_times = stage_times
        visual.total_sec = round(time.perf_counter() - started, 3)
        visual.save(visual_dir / VISUAL_JSON)
        (visual_dir / VISUAL_MD).write_text(build_visual_markdown(visual), encoding="utf-8")
        self._emit("report", "done", f"{visual_dir / VISUAL_MD} ({format_duration(stage_times[-1].elapsed_sec)})")
        return VisualResult(project_dir=project_dir, visual=visual)

    # --- tomadas ---

    def _shots(
        self, clips: list[CatalogClip], source_dir: Path, cache_root: Path, force: bool
    ) ->tuple[dict[str, tuple[str, ClipShotsCache]], set[str], dict[str, str]]:
        """Tomadas de cada clipe (cache ou FFmpeg): {ID: (chave, cache)}, IDs do cache e erros."""
        results: dict[str, tuple[str, ClipShotsCache]] = {}
        cached_ids: set[str] = set()
        errors: dict[str, str] = {}
        pending: list[tuple[CatalogClip, Path, str]] = []
        for clip in clips:
            path = (source_dir / clip.relative_path).resolve()
            try:
                stat = path.stat()
            except OSError:
                errors[clip.id] = f"arquivo não encontrado: {path}"
                continue
            key = clip_cache_key(path, stat.st_size, stat.st_mtime_ns)
            cached = None if force else self._load_shots(cache_root / key)
            if cached is not None:
                results[clip.id] = (key, cached)
                cached_ids.add(clip.id)
            else:
                pending.append((clip, path, key))
        total = len(clips)
        done = len(results) + len(errors)
        self._progress("shots", StageProgress(done=done, total=total, unit="clips"))

        def work(item: tuple[CatalogClip, Path, str]) -> tuple[str, str, Optional[ClipShotsCache], Optional[str]]:
            clip, path, key = item
            try:
                return clip.id, key, self._analyze_clip(path, clip.media.duration_sec, cache_root / key), None
            except Exception as e:  # noqa: BLE001 — um clipe com problema não derruba os outros
                return clip.id, key, None, str(e)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for clip_id, key, cache, error in pool.map(work, pending):
                if cache is not None:
                    results[clip_id] = (key, cache)
                else:
                    errors[clip_id] = error or "falha desconhecida"
                done += 1
                self._progress("shots", StageProgress(done=done, total=total, unit="clips"))
        return results, cached_ids, errors

    def _analyze_clip(self, path: Path, duration: float, cache_dir: Path) -> ClipShotsCache:
        # As análises da IA (`ai-*.json`) ficam: a mesma chave gera as mesmas tomadas.
        cache_dir.mkdir(parents=True, exist_ok=True)
        for old in cache_dir.iterdir():
            if old.is_file() and not old.name.startswith("ai-"):
                old.unlink()
        scan = scan_clip(path)
        stat = path.stat()
        shots: list[CachedShot] = []
        for index, span in enumerate(split_shots(duration, scan.cuts), start=1):
            times = frame_times(span, duration)
            files = [f"T{index:02d}_F{position}.jpg" for position in range(1, len(times) + 1)]
            if not extract_frames(path, times, [cache_dir / name for name in files]):
                raise RuntimeError(f"falha ao extrair miniaturas da tomada {index}")
            shots.append(CachedShot(
                index=index, start_sec=span.start_sec, end_sec=span.end_sec, boundary=span.boundary,
                frame_times=times, frame_files=files, measures=measure_shot(span, scan.samples),
            ))
        cache = ClipShotsCache(
            format_version=VISUAL_CACHE_VERSION, path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            params=shot_params(), duration_sec=duration, shots=shots,
        )
        (cache_dir / SHOTS_FILE).write_text(cache.model_dump_json(indent=2), encoding="utf-8")
        return cache

    @staticmethod
    def _load_shots(cache_dir: Path) -> Optional[ClipShotsCache]:
        path = cache_dir / SHOTS_FILE
        if not path.is_file():
            return None
        try:
            cache = ClipShotsCache.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if cache.format_version != VISUAL_CACHE_VERSION or any(
            not (cache_dir / name).is_file() for shot in cache.shots for name in shot.frame_files
        ):
            return None
        return cache

    def _clip_visual(
        self,
        clip: CatalogClip,
        entry: Optional[tuple[str, ClipShotsCache]],
        cached: bool,
        error: Optional[str],
    ) -> ClipVisual:
        visual = ClipVisual(
            clip_id=clip.id, role=clip.role, relative_path=clip.relative_path,
            duration_sec=clip.media.duration_sec if clip.media else 0.0,
            speech_status=clip.speech.status if clip.speech else None,
            cached=cached, error=error,
        )
        if entry is None:
            return visual
        for item in entry[1].shots:
            shot_id = f"{clip.id}.T{item.index:02d}"
            frames = [
                ShotFrame(id=f"{shot_id}.F{n}", position=n, time_sec=t, file=f"{FRAMES_DIR}/{shot_id}.F{n}.jpg")
                for n, t in enumerate(item.frame_times, start=1)
            ]
            visual.shots.append(Shot(
                id=shot_id, index=item.index, start_sec=item.start_sec, end_sec=item.end_sec,
                boundary=item.boundary, frames=frames, measures=item.measures,
                has_speech=speech_overlap(clip, item.start_sec, item.end_sec),
            ))
        return visual

    def _write_frames(
        self,
        visuals: list[ClipVisual],
        caches: dict[str, tuple[str, ClipShotsCache]],
        cache_root: Path,
        visual_dir: Path,
    ) -> None:
        """Copia as miniaturas do cache para `visual/frames/<ID>.jpg` (IDs mudam se a pasta mudar)."""
        frames_dir = visual_dir / FRAMES_DIR
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)
        for visual in visuals:
            if visual.clip_id not in caches:
                continue
            key, cache = caches[visual.clip_id]
            for shot, item in zip(visual.shots, cache.shots):
                for frame, name in zip(shot.frames, item.frame_files):
                    shutil.copyfile(cache_root / key / name, visual_dir / frame.file)

    # --- IA ---

    @staticmethod
    def _load_analyses(cache_dir: Path, fingerprint: Optional[str]) -> dict[int, ShotAnalysis]:
        if fingerprint is None:
            return {}
        path = cache_dir / f"ai-{fingerprint}.json"
        if not path.is_file():
            return {}
        try:
            cached = CachedAnalyses.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return cached.analyses if cached.fingerprint == fingerprint else {}

    @staticmethod
    def _apply_cached(visual: ClipVisual, analyses: dict[int, ShotAnalysis]) -> None:
        for shot in visual.shots:
            analysis = analyses.get(shot.index)
            if analysis is not None:
                full = [f"{shot.id}.{local}" for local in analysis.best_frames]
                shot.analysis = analysis.model_copy(update={"best_frames": full})
                shot.status = "analisada"

    def _estimate(self, visuals: list[ClipVisual]) -> VisualEstimate:
        todo = [s for v in visuals for s in v.shots if s.status != "analisada"]
        images = sum(len(s.frames) for s in todo)
        calls, tokens_in, tokens_out = estimate_tokens(len(todo), images, self.detail)
        by_role: dict[str, int] = {}
        for visual in visuals:
            by_role[visual.role] = by_role.get(visual.role, 0) + len(visual.shots)
        return VisualEstimate(
            shots=len(todo), images=images, cached_shots=sum(len(v.shots) for v in visuals) - len(todo),
            calls=calls, input_tokens=tokens_in, output_tokens=tokens_out, image_detail=self.detail,
            shots_by_role=by_role,
        )

    def _describe(
        self,
        visuals: list[ClipVisual],
        clips: list[CatalogClip],
        caches: dict[str, tuple[str, ClipShotsCache]],
        cache_root: Path,
        fingerprint: str,
    ) -> VisualAIRun:
        describer = self.describer
        run = VisualAIRun(
            provider=describer.provider_name, model_chain=describer.model_chain,
            thinking=describer.thinking_label, image_detail=describer.detail,
            cached_shots=sum(1 for v in visuals for s in v.shots if s.status == "analisada"),
        )
        by_shot: dict[str, tuple[ClipVisual, Shot]] = {}
        requests: list[ShotRequest] = []
        for visual in visuals:
            if visual.clip_id not in caches:
                continue
            key, cache = caches[visual.clip_id]
            for shot, item in zip(visual.shots, cache.shots):
                by_shot[shot.id] = (visual, shot)
                if shot.status != "analisada":
                    paths = [cache_root / key / name for name in item.frame_files]
                    requests.append(ShotRequest(shot=shot, clip_role=visual.role, image_paths=paths))
        batches = make_batches(requests)
        if not batches:
            self._emit("ai", "notice", "todas as tomadas já analisadas (cache)")
        for number, batch in enumerate(batches, start=1):
            model = describer.chain.current.label
            self._progress("ai", StageProgress(note=f"lote {number}/{len(batches)} · {model}"))
            outcome = describer.describe_batch(batch)
            run.batches += 1
            run.api_calls += outcome.api_calls
            run.images_sent += outcome.images_sent
            if outcome.input_tokens is not None:
                run.input_tokens = (run.input_tokens or 0) + outcome.input_tokens
            if outcome.output_tokens is not None:
                run.output_tokens = (run.output_tokens or 0) + outcome.output_tokens
            touched: set[str] = set()
            for shot_id, analysis in outcome.analyses.items():
                visual, shot = by_shot[shot_id]
                shot.analysis, shot.status, shot.note = analysis, "analisada", ""
                touched.add(visual.clip_id)
            for shot_id, reason in outcome.failed.items():
                _, shot = by_shot[shot_id]
                shot.status, shot.note = "sem_analise", reason
            for clip_id in touched:
                self._save_analyses(next(v for v in visuals if v.clip_id == clip_id),
                                    cache_root / caches[clip_id][0], fingerprint)
            if outcome.failed:
                self._emit("ai", "notice", f"lote {number}: {len(outcome.failed)} tomada(s) sem análise")
        if describer.effective_provider is not None:
            run.effective_model = f"{describer.effective_provider}:{describer.effective_model_id}"
            run.thinking = describer.effective_thinking_label
        run.model_switches = list(describer.model_switches)
        return run

    @staticmethod
    def _save_analyses(visual: ClipVisual, cache_dir: Path, fingerprint: str) -> None:
        analyses = {
            shot.index: shot.analysis.model_copy(update={"best_frames": _local_frames(shot.analysis.best_frames)})
            for shot in visual.shots if shot.status == "analisada" and shot.analysis is not None
        }
        path = cache_dir / f"ai-{fingerprint}.json"
        path.write_text(CachedAnalyses(fingerprint=fingerprint, analyses=analyses).model_dump_json(indent=2),
                        encoding="utf-8")
