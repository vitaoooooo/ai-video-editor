"""Orquestrador da receita `editor_ia` (`edit <pasta-ou-vídeo> --receita editor_ia`): o Editor de IA.

catálogo → roteiro (RoteiroBuilder, com os caches dele; refeito se faltar ou se o material mudou) →
editor (resumos por clipe, estrutura, detalhe por bloco: só IDs) → timeline (IDs → tempos) → render →
legendas → relatório. Com `--so-plano`, para depois da timeline (plano e relatório do plano, sem render).

Funciona com qualquer material: uma pasta (todos os clipes legíveis, de qualquer papel) ou um vídeo só
(projeto de 1 clipe). O roteiro termina (IA de visão) antes do editor chamar a IA de texto: as duas
nunca rodam ao mesmo tempo (GPU de 6 GB), e as chamadas do editor vão uma de cada vez.

Tudo do editor fica em `<projeto>/editor_ia/` (manifesto, plano, Timeline, vídeo, relatório), sem tocar no
`project.json` nem nos arquivos do `edit <pasta>`/`edit <vídeo>` do mesmo projeto (comparação lado a lado).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from ai_video_editor.core.schema import LoudnessTarget
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.editor import (
    DETAIL_PROMPT_VERSION,
    STRUCTURE_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    ClipSource,
    EditorAI,
    EditorContext,
    EditorMaterial,
    EditorPlan,
    EditorRun,
    EditorService,
    assemble_plan,
    editor_cache,
    resolve_budget,
    subtitle_sources,
)
from ai_video_editor.editor import budget as editor_budget
from ai_video_editor.editor import validate as editor_validate
from ai_video_editor.editor.assemble import COVERAGE_SNAP_SEC, EDGE_SEC
from ai_video_editor.editor.report import build_editor_report, build_plan_markdown
from ai_video_editor.intelligence.retry import DEFAULT_API_RETRIES, RetryEvent, RetryPolicy
from ai_video_editor.llm.chain import ModelChain, resolve_chain
from ai_video_editor.llm.schema import ModelSpec, ThinkingLevel
from ai_video_editor.pipeline.clips import (
    catalog_location,
    catalog_path,
    clip_cache_file,
    load_or_build_catalog,
    load_or_detect_silences,
    silences_fingerprint,
    transcription_fingerprint,
)
from ai_video_editor.pipeline.project import DEFAULT_PROJECTS_ROOT, MANIFEST_FILE, file_hash, fingerprint, project_dir_for, slugify
from ai_video_editor.pipeline.runner import EditPipeline, ProgressCallback, StageProgressCallback, _Run
from ai_video_editor.pipeline.schema import (
    EditPipelineParams,
    EditSettings,
    RunTimings,
    StageOutcome,
    StageProgress,
    StageRecord,
    settings_fingerprint,
)
from ai_video_editor.pipeline.timing import TimingRecorder, format_duration
from ai_video_editor.recipes import Recipe, RecipeError
from ai_video_editor.renderer.timeline_executor import TimelineRenderExecutor, TimelineRenderResult
from ai_video_editor.subtitles import build_clip_subtitle_blocks, write_srt
from ai_video_editor.transcription.schema import Transcript

EDITOR_RECIPE = "editor_ia"
EDITOR_STEPS = ("catalog", "roteiro", "editor", "timeline", "render", "subtitles", "report")
EDITOR_DIR = "editor_ia"
EDITOR_KIND = "editor_ia"
EDITOR_RUN_FILE = "editor_run.json"
EDITOR_PLAN_JSON = "editor_plano.json"
EDITOR_PLAN_MD = "editor_plano.md"
EDITOR_TIMELINE_FILE = "timeline.json"
EDITOR_REPORT_FILE = "editor_relatorio.md"

# Modelo padrão do editor por provedor (os outros usam o padrão do provedor).
DEFAULT_EDITOR_MODELS = {"ollama": "qwen2.5:7b"}
DEFAULT_EDITOR_PROVIDER = "ollama"

# Versões de formato que entram nos fingerprints (sobem quando a saída da etapa muda).
EDITOR_FORMAT = "editor-v1"
TIMELINE_FORMAT = "editor-timeline-v1"
REPORT_FORMAT = "editor-report-v1"
# Regras do código que mudam o resultado do editor (entram no fingerprint da etapa).
EDITOR_RULES = (
    editor_validate.OVERLAP_TOLERANCE_SEC, editor_validate.DURATION_TOLERANCE_SEC, editor_validate.RETRY_INVALID_RATIO,
    editor_budget.TOKEN_FACTOR, editor_budget.SUMMARY_OUTPUT_TOKENS, editor_budget.STRUCTURE_OUTPUT_TOKENS,
    editor_budget.DETAIL_OUTPUT_TOKENS, editor_budget.MARGIN_TOKENS, editor_budget.LIBRARY_MIN_TOKENS,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EditorParams(BaseModel):
    """IA do editor e o modo da execução (as opções de silêncio/transcrição/render são as do `edit`)."""

    provider: str = DEFAULT_EDITOR_PROVIDER
    model: Optional[str] = None
    fallback: Optional[str] = Field(default=None, description="`provedor:modelo,...`, `none` ou None (reserva padrão)")
    thinking: ThinkingLevel = "low"
    timeout_sec: float = Field(default=600.0, gt=0.0)
    api_retries: int = Field(default=DEFAULT_API_RETRIES, ge=0)
    context_tokens: Optional[int] = Field(default=None, ge=256, description="--editor-ctx (None = o do provedor)")
    plan_only: bool = Field(default=False, description="--so-plano: para depois da Timeline, sem render")
    interval_sec: float = Field(default=2.0, gt=0.0, description="Grade de fotos do roteiro (--intervalo)")

    def chain(self) -> list[ModelSpec]:
        if self.provider == "mock":
            return [ModelSpec(provider="mock", model_id="mock")]
        return resolve_chain(self.provider, self.model or DEFAULT_EDITOR_MODELS.get(self.provider), self.fallback)


class EditorManifest(BaseModel):
    """Conteúdo do `editor_ia/project.json`."""

    kind: Literal["editor_ia"] = EDITOR_KIND
    source: Path
    recipe: str = EDITOR_RECIPE
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    params: EditPipelineParams
    editor: EditorParams
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    last_run: Optional[RunTimings] = None
    settings: Optional[EditSettings] = None

    def ensure_stages(self, steps: list[str]) -> None:
        for step in steps:
            self.stages.setdefault(step, StageRecord())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EditorManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class EditorResult(BaseModel):
    project_dir: Path
    work_dir: Path
    manifest_path: Path
    output_path: Optional[Path] = None
    subtitles_path: Optional[Path] = None
    report_path: Path
    plan_path: Path
    plan_md_path: Path
    timeline_path: Path
    plan_only: bool = False
    material_sec: float = 0.0
    plan_duration_sec: float = 0.0
    final_duration_sec: Optional[float] = None
    blocks: int = 0
    coverages: int = 0
    breaths: int = 0
    fallbacks: list[str] = Field(default_factory=list, description="Passadas/blocos que foram pelo fallback neutro")
    calls: int = 0
    cached_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    outcomes: dict[str, StageOutcome] = Field(default_factory=dict)
    timings: Optional[RunTimings] = None
    warnings: list[str] = Field(default_factory=list)


@dataclass
class _State:
    run: _Run
    params: EditPipelineParams
    editor: EditorParams
    context: EditorContext
    target: Path
    kind: Literal["arquivo", "pasta"]
    project_dir: Path
    work_dir: Path
    output_path: Path
    srt_path: Path
    settings: Optional[EditSettings] = None
    outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    material: Optional[EditorMaterial] = None
    sources: dict[str, ClipSource] = field(default_factory=dict)
    fp_roteiro: Optional[str] = None
    fp_editor: Optional[str] = None
    fp_timeline: Optional[str] = None
    fp_render: Optional[str] = None
    fp_subtitles: Optional[str] = None
    editor_run: Optional[EditorRun] = None
    plan: Optional[EditorPlan] = None
    timeline: Optional[Timeline] = None
    final_duration: Optional[float] = None
    encoder_label: Optional[str] = None
    subtitle_blocks: Optional[int] = None
    render_warnings: list[str] = field(default_factory=list)
    roteiro_warnings: list[str] = field(default_factory=list)


def output_names(target: Path) -> tuple[str, str]:
    stem = f"{slugify(target.stem if target.is_file() else target.name)}_editor"
    return f"{stem}.mp4", f"{stem}.srt"


def project_location(target: Path, projects_root: Path) -> Path:
    """Pasta do projeto: a do catálogo (pasta de clipes) ou a do `edit <vídeo>` (um arquivo)."""
    target = Path(target).resolve()
    if target.is_dir():
        return catalog_location(target, projects_root)[0]
    return project_dir_for(target, file_hash(target), projects_root).resolve()


class EditorPipeline(EditPipeline):
    """Executa a receita `editor_ia`, com cache por etapa (e por chamada da IA, dentro do editor)."""

    def __init__(
        self,
        projects_root: Path = DEFAULT_PROJECTS_ROOT,
        transcription_provider=None,
        render_executor: Optional[TimelineRenderExecutor] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_stage_progress: Optional[StageProgressCallback] = None,
        recipe: Recipe | str = EDITOR_RECIPE,
        speech_detector=None,
        editor_ai: Optional[EditorAI] = None,
        describer=None,
        vision_parallel: int = 1,
    ):
        super().__init__(
            projects_root=projects_root, transcription_provider=transcription_provider, on_progress=on_progress,
            on_stage_progress=on_stage_progress, recipe=recipe,
        )
        unknown = [step for step in self.recipe.steps if step not in EDITOR_STEPS]
        if unknown or "editor" not in self.recipe.steps:
            raise RecipeError(f"Receita '{self.recipe.name}' não é do Editor de IA (etapas: {', '.join(self.recipe.steps)}).")
        self._timeline_executor = render_executor
        self._speech_detector = speech_detector
        self._editor_ai = editor_ai
        self.describer = describer
        self.vision_parallel = vision_parallel

    @property
    def timeline_executor(self) -> TimelineRenderExecutor:
        if self._timeline_executor is None:
            self._timeline_executor = TimelineRenderExecutor()
        return self._timeline_executor

    # --- IA do editor ---

    def editor_ai_for(self, editor: EditorParams) -> tuple[EditorAI, list[str], dict]:
        """(IA, provedores da cadeia, clientes já criados) — a injetada, o mock ou a cadeia pedida."""
        if self._editor_ai is not None:
            return self._editor_ai, [self._editor_ai.provider_name], {}
        if editor.provider == "mock":
            from ai_video_editor.editor.providers.mock import MockEditorAI

            return MockEditorAI(), ["mock"], {}
        from ai_video_editor.editor.providers.llm import LLMEditorAI

        chain = ModelChain(
            editor.chain(),
            thinking=editor.thinking,
            timeout_sec=editor.timeout_sec,
            retry=RetryPolicy(max_retries=editor.api_retries, on_retry=lambda event: self._editor_retry(chain, event)),
            on_event=lambda message: self._emit("editor", "notice", message),
        )
        providers = list(dict.fromkeys(spec.provider for spec in chain.specs))
        clients = {p: chain.client_for(p) for p in providers if p == "ollama"}
        return LLMEditorAI(chain), providers, clients

    def _editor_retry(self, chain: ModelChain, event: RetryEvent) -> None:
        self._emit("editor", "retry", event.message)
        self._stage_progress(
            "editor", StageProgress(note=f"{chain.current.label}, tentativa {event.next_attempt}/{event.total_attempts}")
        )

    # --- execução ---

    def run(  # type: ignore[override]
        self,
        target: Path,
        params: Optional[EditPipelineParams] = None,
        editor: Optional[EditorParams] = None,
        context: Optional[EditorContext] = None,
        force: bool = False,
        settings: Optional[EditSettings] = None,
    ) -> EditorResult:
        target = Path(target).resolve()
        if not target.exists():
            raise FileNotFoundError(f"arquivo ou pasta não encontrado: {target}")
        params = params or EditPipelineParams()
        editor = editor or EditorParams()
        recorder = TimingRecorder()
        project_dir = project_location(target, self.projects_root)
        work_dir = project_dir / EDITOR_DIR
        manifest_path = work_dir / MANIFEST_FILE
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = EditorManifest.load(manifest_path)
                manifest.source, manifest.params, manifest.editor = target, params, editor
            except ValueError:
                manifest = None
        if manifest is None:
            manifest = EditorManifest(source=target, params=params, editor=editor)
        manifest.recipe, manifest.settings = self.recipe.name, settings
        manifest.ensure_stages(list(self.recipe.steps))
        manifest.save(manifest_path)

        video_name, srt_name = output_names(target)
        s = _State(
            run=_Run(self, manifest, manifest_path, force, recorder), params=params, editor=editor,
            context=context or EditorContext(), target=target, kind="pasta" if target.is_dir() else "arquivo",
            project_dir=project_dir, work_dir=work_dir, output_path=work_dir / video_name, srt_path=work_dir / srt_name,
            settings=settings,
        )
        steps: dict[str, Callable[[_State], StageOutcome]] = {
            "catalog": self._catalog, "roteiro": self._roteiro, "editor": self._editor, "timeline": self._timeline,
            "render": self._render, "subtitles": self._subtitles, "report": self._report,
        }
        for step in self.recipe.steps:
            s.outcomes[step] = steps[step](s)
        timings = recorder.snapshot()
        manifest.last_run = timings
        manifest.save(manifest_path)
        return self._result(s, timings)

    def _result(self, s: _State, timings: RunTimings) -> EditorResult:
        run = s.editor_run
        total = run.total_stats if run is not None else None
        fallbacks = []
        if run is not None:
            if run.structure is not None and run.structure.origin == "fallback":
                fallbacks.append("estrutura")
            fallbacks += [d.unit_id for d in run.details if d.origin == "fallback"]
        material_sec = sum(c.duration_sec for c in s.material.clips.values()) if s.material else 0.0
        return EditorResult(
            project_dir=s.project_dir, work_dir=s.work_dir, manifest_path=s.run.manifest_path,
            output_path=s.output_path if s.final_duration is not None else None,
            subtitles_path=s.srt_path if s.subtitle_blocks is not None else None,
            report_path=s.work_dir / EDITOR_REPORT_FILE, plan_path=s.work_dir / EDITOR_PLAN_JSON,
            plan_md_path=s.work_dir / EDITOR_PLAN_MD, timeline_path=s.work_dir / EDITOR_TIMELINE_FILE,
            plan_only=s.editor.plan_only, material_sec=round(material_sec, 3),
            plan_duration_sec=s.plan.duration_sec if s.plan else 0.0, final_duration_sec=s.final_duration,
            blocks=len(s.plan.blocks) if s.plan else 0,
            coverages=sum(len(b.coverages) for b in s.plan.blocks) if s.plan else 0,
            breaths=sum(1 for b in s.plan.blocks for p in b.pieces if p.kind == "respiro") if s.plan else 0,
            fallbacks=fallbacks, calls=total.calls if total else 0, cached_calls=total.cached if total else 0,
            input_tokens=total.input_tokens if total else None, output_tokens=total.output_tokens if total else None,
            outcomes=s.outcomes, timings=timings,
            warnings=[*s.roteiro_warnings, *(run.warnings if run else []), *(s.plan.warnings if s.plan else []), *s.render_warnings],
        )

    # --- etapas ---

    def _catalog(self, s: _State) -> StageOutcome:
        if s.kind == "arquivo":
            return s.run.skip("catalog", "arquivo único: projeto de 1 clipe (P001)")
        catalog_file = catalog_path(s.project_dir)

        def catalog_fp() -> str:
            return fingerprint("editor-catalog", file_hash(catalog_file) if catalog_file.is_file() else "missing")

        def do_catalog() -> tuple[dict, str]:
            from ai_video_editor.catalog import CATALOG_STAGE_LABELS, CatalogBuilder

            def on_progress(stage: str, event: str, detail: str) -> None:
                if event == "start":
                    self._stage_progress("catalog", StageProgress(note=CATALOG_STAGE_LABELS.get(stage, stage)))
                elif event == "notice":
                    self._emit("catalog", "notice", detail)

            catalog, built = load_or_build_catalog(
                s.target, self.projects_root,
                lambda: CatalogBuilder(projects_root=self.projects_root, speech_detector=self._speech_detector, on_progress=on_progress),
            )
            valid = sum(1 for c in catalog.clips if c.id is not None and c.error is None)
            return {"clips": valid, "catalog_built": int(built)}, f"{valid} clipe(s)" + (" (catálogo gerado agora)" if built else "")

        relative = Path("..") / catalog_file.relative_to(s.project_dir)
        outcome = s.run.stage("catalog", catalog_fp(), str(relative), do_catalog)
        record = s.run.manifest.stages["catalog"]
        if record.fingerprint != catalog_fp():
            record.fingerprint = catalog_fp()
            s.run.manifest.save(s.run.manifest_path)
        return outcome

    def _roteiro_builder(self):
        from ai_video_editor.roteiro import ROTEIRO_STAGE_LABELS, RoteiroBuilder

        def on_progress(stage: str, event: str, detail: str) -> None:
            label = ROTEIRO_STAGE_LABELS.get(stage, stage)
            if event == "start":
                self._stage_progress("roteiro", StageProgress(note=label))
            elif event in ("notice", "skipped"):
                self._emit("roteiro", "notice", f"{label}: {detail}")

        return RoteiroBuilder(
            projects_root=self.projects_root, describer=self.describer, transcription_provider=self.transcription_provider,
            speech_detector=self._speech_detector, on_progress=on_progress,
            on_stage_progress=lambda stage, progress: self._stage_progress(
                "roteiro", progress.model_copy(update={"note": f"{ROTEIRO_STAGE_LABELS.get(stage, stage)} {progress.note}".strip()})
            ),
            vision_parallel=self.vision_parallel,
        )

    def _roteiro(self, s: _State) -> StageOutcome:
        """Sempre roda o RoteiroBuilder: ele refaz só o que mudou (caches por clipe e por foto)."""
        from ai_video_editor.roteiro import RoteiroParams

        params = s.params
        rparams = RoteiroParams(
            interval_sec=s.editor.interval_sec, threshold_db=params.threshold_db,
            min_silence_sec=params.min_silence_sec, transcription=params.transcription,
        )
        self._emit("roteiro", "start")
        started = time.perf_counter()
        try:
            result = self._roteiro_builder().build(s.target, rparams, force=s.run.force)
        except Exception as exc:
            s.run.manifest.stages["roteiro"] = StageRecord(status="failed", updated_at=_now(), error=str(exc))
            s.run.manifest.save(s.run.manifest_path)
            self._emit("roteiro", "failed", str(exc))
            raise
        roteiro = result.roteiro
        elapsed = time.perf_counter() - started
        s.material = EditorMaterial.from_roteiro(roteiro)
        s.roteiro_warnings = list(roteiro.warnings)
        s.fp_roteiro = fingerprint(
            "roteiro", roteiro.model_dump(mode="json", include={"clips", "moments", "sentences", "kind", "interval_sec"})
        )
        s.sources = self._sources(s, roteiro)
        described = sum(1 for p in roteiro.photos if p.status == "descrita")
        metrics = {
            "clips": len(roteiro.clips), "sentences": len(roteiro.sentences), "moments": len(roteiro.moments),
            "photos": len(roteiro.photos), "described": described, "tokens": roteiro.estimated_tokens,
        }
        s.run.manifest.stages["roteiro"] = StageRecord(
            status="done", fingerprint=s.fp_roteiro, artifact=str(Path("..") / "roteiro" / "roteiro.json"),
            updated_at=_now(), elapsed_sec=round(elapsed, 3), metrics=metrics,
        )
        s.run.manifest.save(s.run.manifest_path)
        s.run.recorder.record("roteiro", "ran", elapsed)
        s.run.recorder.video_duration_sec = sum(c.duration_sec for c in roteiro.clips)
        detail = (
            f"{len(roteiro.clips)} clipe(s), {len(roteiro.moments)} momentos, {len(roteiro.sentences)} frases, "
            f"{described}/{len(roteiro.photos)} fotos descritas, ~{roteiro.estimated_tokens} tokens"
        )
        self._emit("roteiro", "done", f"{detail} ({format_duration(elapsed)})")
        return "ran"

    def _sources(self, s: _State, roteiro) -> dict[str, ClipSource]:
        """Arquivo, fps, silêncios e transcrição de cada clipe (dos caches por clipe que o roteiro usou)."""
        medias = self._medias(s)
        provider = self.transcription_provider
        params = s.params
        sources: dict[str, ClipSource] = {}
        for clip in roteiro.clips:
            path = (s.target if s.kind == "arquivo" else s.target / clip.relative_path).resolve()
            media = medias.get(clip.id)
            source = ClipSource(
                clip_id=clip.id, path=path, fps=media.fps if media else 30.0, duration_sec=clip.duration_sec,
                has_audio=clip.has_audio,
            )
            if clip.has_speech:
                digest = file_hash(path)
                fp = silences_fingerprint(digest, params.threshold_db, params.min_silence_sec)
                source.audio, _ = load_or_detect_silences(
                    path, clip_cache_file(self.projects_root, digest, "silences", fp), params.threshold_db, params.min_silence_sec
                )
                transcript_file = clip_cache_file(
                    self.projects_root, digest, "transcript", transcription_fingerprint(digest, provider, params.transcription)
                )
                if transcript_file.is_file():
                    source.transcript = Transcript.load(transcript_file)
            sources[clip.id] = source
        return sources

    def _medias(self, s: _State) -> dict:
        from ai_video_editor.catalog.schema import CatalogClip, ProjectCatalog
        from ai_video_editor.roteiro.builder import ROTEIRO_DIR, SINGLE_CLIP_FILE

        if s.kind == "pasta":
            catalog = ProjectCatalog.load(catalog_path(s.project_dir))
            return {c.id: c.media for c in catalog.clips if c.id is not None and c.media is not None}
        data = json.loads((s.project_dir / ROTEIRO_DIR / SINGLE_CLIP_FILE).read_text(encoding="utf-8"))
        clip = CatalogClip.model_validate(data["clip"])
        return {"P001": clip.media}

    def _editor(self, s: _State) -> StageOutcome:
        ai, providers, clients = self.editor_ai_for(s.editor)
        budget = resolve_budget(providers, s.editor.context_tokens, clients)
        context = s.context
        s.fp_editor = fingerprint(
            "editor", EDITOR_FORMAT, s.fp_roteiro, context.request, context.target_sec, context.director_instructions,
            context.coverage_instructions, ai.provider_name, list(ai.model_chain), ai.thinking_label, budget.num_ctx,
            SUMMARY_PROMPT_VERSION, STRUCTURE_PROMPT_VERSION, DETAIL_PROMPT_VERSION, EDITOR_RULES,
        )

        def do_editor() -> tuple[dict, str]:
            self._stage_progress("editor", StageProgress(note=ai.model_chain[0] if ai.model_chain else ai.model_id))
            service = EditorService(
                ai, s.material, context, budget, editor_cache(self.projects_root),
                on_notice=lambda message: self._emit("editor", "notice", message),
                on_progress=lambda progress: self._stage_progress("editor", progress),
            )
            run = service.run()
            run.save(s.work_dir / EDITOR_RUN_FILE)
            total = run.total_stats
            fallbacks = sum(1 for d in run.details if d.origin == "fallback") + int(run.structure.origin == "fallback")
            metrics = {
                "provider": ai.provider_name, "model": run.effective_model or (ai.model_chain[0] if ai.model_chain else ""),
                "num_ctx": budget.num_ctx, "blocks": len(run.structure.blocks), "units": len(run.details),
                "coverages": sum(len(d.coverages) for d in run.details), "breaths": sum(len(d.breaths) for d in run.details),
                "fallbacks": fallbacks, "calls": total.calls, "cached": total.cached, "api_calls": total.api_calls,
                "input_tokens": total.input_tokens or 0, "output_tokens": total.output_tokens or 0,
            }
            detail = (
                f"{len(run.structure.blocks)} bloco(s) em {len(run.details)} pedido(s) de detalhe, "
                f"{metrics['coverages']} cobertura(s), {metrics['breaths']} respiro(s); {total.calls} pedido(s) à IA "
                f"({total.cached} do cache)"
            )
            if fallbacks:
                detail += f"; {fallbacks} FALLBACK(S)"
            return metrics, detail

        outcome = s.run.stage("editor", s.fp_editor, EDITOR_RUN_FILE, do_editor)
        s.editor_run = EditorRun.load(s.work_dir / EDITOR_RUN_FILE)
        return outcome

    def _timeline(self, s: _State) -> StageOutcome:
        params = s.params
        audio_fps = [
            (c, src.audio.params.threshold_db if src.audio else None, str(src.path)) for c, src in s.sources.items()
        ]
        s.fp_timeline = fingerprint(
            "editor-timeline", TIMELINE_FORMAT, file_hash(s.work_dir / EDITOR_RUN_FILE), s.fp_roteiro, audio_fps,
            params.effective_cut, params.loudness, EDGE_SEC, COVERAGE_SNAP_SEC,
        )

        def do_timeline() -> tuple[dict, str]:
            assembled = assemble_plan(
                s.editor_run, s.material, s.sources, params.effective_cut, s.output_path,
                title=f"Editor de IA — {s.target.name}", loudness=LoudnessTarget.from_preset(params.loudness),
                project_name=s.target.name,
            )
            for warning in assembled.warnings:
                self._emit("timeline", "notice", warning)
            (s.work_dir / EDITOR_TIMELINE_FILE).write_text(assembled.timeline.model_dump_json(indent=2), encoding="utf-8")
            assembled.plan.save(s.work_dir / EDITOR_PLAN_JSON)
            (s.work_dir / EDITOR_PLAN_MD).write_text(build_plan_markdown(assembled.plan, s.material), encoding="utf-8")
            timeline = assembled.timeline
            metrics = {
                "items": len(timeline.video_main), "overlays": len(timeline.video_overlay),
                "ambient": len(timeline.audio_ambient), "duration_sec": round(timeline.duration_sec, 3), "fps": timeline.fps,
            }
            return metrics, (
                f"{len(timeline.video_main)} itens na base, {len(timeline.video_overlay)} de cobertura "
                f"({timeline.duration_sec:.1f}s) @ {timeline.fps}fps; plano em {EDITOR_PLAN_MD}"
            )

        outcome = s.run.stage("timeline", s.fp_timeline, EDITOR_PLAN_JSON, do_timeline)
        s.plan = EditorPlan.load(s.work_dir / EDITOR_PLAN_JSON)
        s.timeline = Timeline.model_validate_json((s.work_dir / EDITOR_TIMELINE_FILE).read_text(encoding="utf-8"))
        return outcome

    def _render(self, s: _State) -> StageOutcome:
        if s.editor.plan_only:
            return s.run.skip("render", "--so-plano")
        encoder = self.timeline_executor.resolve_encoder()

        def render_fp(enc) -> str:
            return fingerprint("render", file_hash(s.work_dir / EDITOR_TIMELINE_FILE), *enc.fingerprint_parts)

        s.fp_render = render_fp(encoder)
        used: list[TimelineRenderResult] = []

        def do_render() -> tuple[dict, str]:
            result = self.timeline_executor.execute(
                s.timeline,
                on_progress=lambda done, total: self._stage_progress("render", StageProgress(done=done, total=total, unit="items")),
            )
            used.append(result)
            metrics: dict = {
                "duration_sec": round(result.duration_sec, 3), "frames": result.frame_count,
                "size_bytes": result.size_bytes, "encoder": result.encoder.label,
            }
            if result.loudness is not None:
                metrics.update(result.loudness.to_metrics())
            if result.warnings:
                metrics["warnings"] = " | ".join(result.warnings)
            return metrics, f"{result.duration_sec:.2f}s, {result.size_bytes / 1_048_576:.1f} MB, {result.encoder.label}"

        outcome = s.run.stage("render", s.fp_render, s.output_path.name, do_render)
        if used and used[0].encoder.name != encoder.name:
            s.fp_render = s.run.rekey("render", render_fp(used[0].encoder))
        metrics = s.run.manifest.stages["render"].metrics
        s.final_duration = float(metrics["duration_sec"])
        s.encoder_label = str(metrics.get("encoder", "cpu"))
        s.render_warnings = [w for w in str(metrics.get("warnings", "")).split(" | ") if w]
        return outcome

    def _subtitles(self, s: _State) -> StageOutcome:
        srt_name = s.srt_path.name
        if s.editor.plan_only:
            return s.run.skip("subtitles", "--so-plano")
        if not s.params.subtitles:
            return s.run.skip("subtitles", "desligada por --no-subtitles", artifact=srt_name)
        transcripts = [
            (c, file_hash(src.path)) for c, src in s.sources.items() if src.transcript is not None
        ]
        s.fp_subtitles = fingerprint(
            "subtitles", file_hash(s.work_dir / EDITOR_TIMELINE_FILE), transcripts, s.fp_roteiro, s.params.subtitle
        )

        def do_subtitles() -> tuple[dict, str]:
            blocks = build_clip_subtitle_blocks(subtitle_sources(s.timeline, s.sources), s.timeline.fps, s.params.subtitle)
            write_srt(blocks, s.srt_path)
            return {"blocks": len(blocks)}, f"{srt_name}, {len(blocks)} blocos"

        outcome = s.run.stage("subtitles", s.fp_subtitles, srt_name, do_subtitles)
        s.subtitle_blocks = int(s.run.manifest.stages["subtitles"].metrics["blocks"])
        return outcome

    def _report(self, s: _State) -> StageOutcome:
        fp = fingerprint(
            "report", REPORT_FORMAT, s.fp_editor, s.fp_timeline, s.fp_render, s.fp_subtitles, s.editor.plan_only,
            self.recipe.name, s.encoder_label, *settings_fingerprint(s.settings),
        )

        def do_report() -> tuple[dict, str]:
            started = time.perf_counter()
            markdown = build_editor_report(
                s.target.name, s.editor_run, s.plan, s.material, self.recipe.name,
                output_path=s.output_path if s.final_duration is not None else None,
                final_duration_sec=s.final_duration,
                subtitles_path=s.srt_path if s.subtitle_blocks is not None else None,
                subtitle_blocks=s.subtitle_blocks, encoder=s.encoder_label,
                timings=s.run.recorder.snapshot(in_progress=("report", started)),
                extra_warnings=[*s.roteiro_warnings, *s.render_warnings],
            )
            (s.work_dir / EDITOR_REPORT_FILE).write_text(markdown, encoding="utf-8")
            return {"blocks": len(s.plan.blocks)}, EDITOR_REPORT_FILE

        return s.run.stage("report", fp, EDITOR_REPORT_FILE, do_report)
