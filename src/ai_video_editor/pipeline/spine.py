"""Orquestrador do `edit <pasta>`: a espinha falada de um projeto com vários clipes (receitas `vlog` e
`vlog_cobertura`).

catálogo → silêncios por clipe → transcrição por clipe → Diretor → Timeline → [análise visual →
cobertura] → render → legendas → relatório. Os clipes de papel `principal` com fala entram em ordem; cada um perde as pausas e as
frases ruins (mesmo corte do caminho de um vídeo, `plan_keep_ranges`) e os trechos mantidos viram
itens de `video_main` numa Timeline, renderizada por `renderer/timeline_*`. Na `vlog_cobertura`, o
Diretor de cobertura acrescenta B-roll por cima da fala (`video_overlay`) e respiros na base, e o
resto (render, legendas, relatórios) usa essa Timeline final.

O projeto é a pasta do catálogo (`<projects>/<pasta>-<hash da listagem>/`). Cada etapa grava um
artefato e um fingerprint no `project.json` (`kind="multiclip"`), como no `edit` de um vídeo.
Silêncios e transcrição têm também cache por clipe, fora do projeto (chave: hash do arquivo +
parâmetros), então trocar um clipe da pasta só reprocessa esse clipe.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor.audio import AudioAnalysisResult, detect_silences
from ai_video_editor.audio.levels import AUTO_RULE_VERSION
from ai_video_editor.core.schema import LoudnessTarget
from ai_video_editor.catalog.schema import ProjectCatalog
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.coverage import (
    COVERAGE_PROMPT_VERSION,
    CoverageContext,
    CoverageDirector,
    CoverageError,
    CoverageParams,
    CoverageRun,
    VisualMaterial,
    assemble_coverage,
    build_visual_material,
    coverage_candidates,
    coverage_roles,
    coverage_sentences,
    find_mandatory_windows,
    run_coverage,
)
from ai_video_editor.coverage import assemble as coverage_assemble
from ai_video_editor.coverage import material as coverage_material
from ai_video_editor.coverage import service as coverage_service
from ai_video_editor.intelligence.retry import RetryEvent, RetryPolicy
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.director import DirectorClip, DirectorContext, DirectorRun, EditDirector, build_clip_sentences, run_director
from ai_video_editor.pipeline.clips import (
    MIN_SPEECH_RATIO,
    MIN_SPEECH_SEC,
    SELECTION_VERSION,
    catalog_location,
    catalog_path,
    clip_cache_file,
    load_or_build_catalog,
    select_spine_clips,
)
from ai_video_editor.pipeline.cutting import plan_keep_ranges
from ai_video_editor.pipeline.project import (
    COVERAGE_DECISION_FILE,
    COVERAGE_SHOTS_FILE,
    COVERAGE_TIMELINE_FILE,
    DIRECTOR_DECISION_FILE,
    EDIT_REPORT_FILE,
    EDIT_TIMELINE_FILE,
    EDIT_TRANSCRIPT_FILE,
    MANIFEST_FILE,
    SILENCES_FILE,
    SPINE_CLIPS_FILE,
    SPINE_PLAN_FILE,
    TIMELINE_FILE,
    TRANSCRIPTION_FILE,
    DEFAULT_PROJECTS_ROOT,
    file_hash,
    fingerprint,
    slugify,
)
from ai_video_editor.pipeline.runner import EditPipeline, ProgressCallback, StageProgressCallback, _Run
from ai_video_editor.pipeline.schema import (
    EditPipelineParams,
    EditSettings,
    StageOutcome,
    StageProgress,
    settings_fingerprint,
)
from ai_video_editor.pipeline.spine_report import (
    build_spine_diff,
    build_spine_report,
    removed_range_count,
    write_spine_diff,
)
from ai_video_editor.pipeline.spine_schema import (
    ClipPlan,
    SpineClip,
    SpineManifest,
    SpinePlan,
    SpineResult,
    SpineSelection,
    is_multiclip_manifest,
)
from ai_video_editor.pipeline.timing import TimingRecorder
from ai_video_editor.pipeline.versions import register_spine_version
from ai_video_editor.planning import NoSpeechDetectedError, PlanningError
from ai_video_editor.planning.timeline_plan import (
    SPINE_RESOLUTION,
    ClipCut,
    build_spine_timeline,
    clip_placements,
    clip_spans,
    spine_fps,
    usable_ranges,
)
from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.recipes import DEFAULT_FOLDER_RECIPE, Recipe, RecipeError
from ai_video_editor.renderer.loudness import LoudnessResult
from ai_video_editor.renderer.timeline_executor import TimelineRenderExecutor, TimelineRenderResult
from ai_video_editor.subtitles import ClipSubtitleSource, build_clip_subtitle_blocks, write_srt
from ai_video_editor.transcription.exceptions import TranscriptionError
from ai_video_editor.transcription.normalize import SegmentAdjustments, adjustments_note
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig

# Etapas que o `edit <pasta>` sabe executar.
SPINE_STEPS = (
    "catalog", "silences", "transcription", "director", "timeline", "visual", "coverage", "render", "subtitles", "report",
)

# Versões de formato que entram nos fingerprints (sobem quando a saída da etapa muda).
DIRECTOR_PROMPT_FORMAT = "spine-prompt-v1"
TIMELINE_FORMAT = "spine-timeline-v1"
REPORT_FORMAT = "spine-report-v1"
VISUAL_MATERIAL_FORMAT = "coverage-shots-v1"
COVERAGE_FORMAT = "coverage-v1"
# Regras do código que mudam o resultado da cobertura (entram no fingerprint da etapa).
COVERAGE_RULES = (
    coverage_service.MIN_COVERAGE_SEC, coverage_service.MAX_COVERAGE_SEC, coverage_service.BREATH_SECONDS,
    coverage_service.MIN_BREATH_SHOT_SEC, coverage_service.BREATH_EVERY_SEC, coverage_service.SOURCE_END_MARGIN_SEC,
    coverage_material.MIN_BAD_OVERLAP_SEC, coverage_material.COVERAGE_ROLES, coverage_material.BAD_ISSUES,
    coverage_assemble.BREATH_VOLUME, coverage_assemble.COVERAGE_AMBIENT_VOLUME, coverage_assemble.AMBIENT_FADE_SEC,
    coverage_assemble.MIN_CHAIN_SEC,
)
VISUAL_DIR = "visual"
VISUAL_JSON = "visual.json"
BRIEFING_FILE = "briefing.json"

# Clipes com a detecção de silêncio rodando ao mesmo tempo (um FFmpeg cada).
SILENCE_WORKERS = min(4, os.cpu_count() or 2)


@dataclass
class _SpineState:
    """O que as etapas de uma execução passam umas às outras."""

    run: _Run
    params: EditPipelineParams
    source_dir: Path
    project_dir: Path
    output_path: Path
    srt_path: Path
    outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    selection: Optional[SpineSelection] = None
    hashes: dict[str, str] = field(default_factory=dict)
    audio: dict[str, AudioAnalysisResult] = field(default_factory=dict)
    transcripts: dict[str, Transcript] = field(default_factory=dict)
    fp_silences: Optional[str] = None
    fp_transcript: Optional[str] = None
    fp_director: Optional[str] = None
    fp_timeline: Optional[str] = None
    fp_render: Optional[str] = None
    fp_subtitles: Optional[str] = None
    fp_visual: Optional[str] = None
    fp_coverage: Optional[str] = None
    # Do que a cobertura produziu (Timeline final e decisão): uma nova execução da etapa com as mesmas
    # entradas pode decidir diferente (IA), e o que vem depois precisa acompanhar.
    fp_coverage_result: Optional[str] = None
    director_run: Optional[DirectorRun] = None
    plan: Optional[SpinePlan] = None
    timeline: Optional[Timeline] = None  # a espinha (etapa `timeline`)
    final_timeline: Optional[Timeline] = None  # o que é renderizado: a espinha ou a Timeline com cobertura
    coverage_params: CoverageParams = field(default_factory=CoverageParams)
    visual: Optional[VisualMaterial] = None
    coverage_run: Optional[CoverageRun] = None
    final_duration: float = 0.0
    loudness: Optional[LoudnessResult] = None
    render_warnings: list[str] = field(default_factory=list)
    transcription_notes: list[str] = field(default_factory=list)
    encoder_label: str = "cpu"
    subtitle_blocks: Optional[int] = None
    settings: Optional[EditSettings] = None
    use_briefing: bool = True

    @property
    def clips(self) -> list[SpineClip]:
        return self.selection.clips if self.selection is not None else []


def output_names(source_dir: Path) -> tuple[str, str]:
    """(`<pasta>_edited.mp4`, `<pasta>_edited.srt`)."""
    stem = f"{slugify(source_dir.name)}_edited"
    return f"{stem}.mp4", f"{stem}.srt"


class SpinePipeline(EditPipeline):
    """Executa a receita `vlog` sobre uma pasta de clipes, com cache por etapa e por clipe.

    Reaproveita do `EditPipeline` o Diretor (`director_for`), o provedor de transcrição e os
    avisos de progresso; o render é o da Timeline.
    """

    def __init__(
        self,
        projects_root: Path = DEFAULT_PROJECTS_ROOT,
        transcription_provider: Optional[TranscriptionProvider] = None,
        render_executor: Optional[TimelineRenderExecutor] = None,
        on_progress: Optional[ProgressCallback] = None,
        edit_director: Optional[EditDirector] = None,
        on_stage_progress: Optional[StageProgressCallback] = None,
        recipe: Recipe | str = DEFAULT_FOLDER_RECIPE,
        speech_detector=None,
        coverage_director: Optional[CoverageDirector] = None,
    ):
        super().__init__(
            projects_root=projects_root,
            transcription_provider=transcription_provider,
            on_progress=on_progress,
            edit_director=edit_director,
            on_stage_progress=on_stage_progress,
            recipe=recipe,
        )
        unknown = [step for step in self.recipe.steps if step not in SPINE_STEPS]
        if unknown or not self.recipe.multi_clip or "catalog" not in self.recipe.steps:
            detail = f"etapas sem suporte em pasta: {', '.join(unknown)}" if unknown else "não é multi_clip"
            raise RecipeError(f"Receita '{self.recipe.name}' não edita pastas de clipes ({detail}).")
        self._timeline_executor = render_executor
        self._speech_detector = speech_detector
        self._coverage_director = coverage_director

    @property
    def timeline_executor(self) -> TimelineRenderExecutor:
        if self._timeline_executor is None:
            self._timeline_executor = TimelineRenderExecutor()
        return self._timeline_executor

    def _catalog_builder(self):
        from ai_video_editor.catalog import CATALOG_STAGE_LABELS, CatalogBuilder

        def on_progress(stage: str, event: str, detail: str) -> None:
            if event == "start":
                self._stage_progress("catalog", StageProgress(note=CATALOG_STAGE_LABELS.get(stage, stage)))
            elif event == "notice":
                self._emit("catalog", "notice", detail)

        return CatalogBuilder(
            projects_root=self.projects_root,
            speech_detector=self._speech_detector,
            on_progress=on_progress,
            on_stage_progress=lambda stage, progress: self._stage_progress(
                "catalog", progress.model_copy(update={"note": CATALOG_STAGE_LABELS.get(stage, stage)})
            ),
        )

    def coverage_director_for(self, cparams: CoverageParams) -> Optional[CoverageDirector]:
        """Diretor de cobertura injetado ou o pedido em `cparams` (None = `none`: só as janelas obrigatórias)."""
        if self._coverage_director is not None:
            return self._coverage_director
        if cparams.provider == "none":
            return None
        if cparams.provider == "mock":
            from ai_video_editor.coverage.providers.mock import MockCoverageDirector

            return MockCoverageDirector()
        from ai_video_editor.coverage.providers.llm import LLMCoverageDirector

        chain = ModelChain(
            cparams.chain(),
            thinking=cparams.thinking,
            timeout_sec=cparams.timeout_sec,
            retry=RetryPolicy(max_retries=cparams.api_retries, on_retry=lambda event: self._coverage_retry(chain, event)),
            on_event=lambda message: self._coverage_notice(chain, message),
        )
        return LLMCoverageDirector(chain)

    def _coverage_retry(self, chain: ModelChain, event: RetryEvent) -> None:
        self._emit("coverage", "retry", event.message)
        self._stage_progress(
            "coverage", StageProgress(note=f"{chain.current.label}, tentativa {event.next_attempt}/{event.total_attempts}")
        )

    def _coverage_notice(self, chain: ModelChain, message: str) -> None:
        self._emit("coverage", "notice", message)
        if chain.switches:
            self._stage_progress("coverage", StageProgress(note=f"{chain.switches[-1].to_model}, tentativa 1"))

    def run(  # type: ignore[override]
        self,
        source_dir: Path,
        params: Optional[EditPipelineParams] = None,
        force: bool = False,
        coverage: Optional[CoverageParams] = None,
        settings: Optional[EditSettings] = None,
        use_briefing: bool = True,
    ) -> SpineResult:
        """Executa (ou retoma do cache) as etapas da receita sobre a pasta e devolve o resumo.

        `coverage`: IA da cobertura (receita `vlog_cobertura`); sem ela, a do Diretor (`CoverageParams.inherit`).
        `settings`: valores efetivos e origens (só `project.json` e relatório). `use_briefing=False`
        (`--sem-briefing` ou briefing de outro material): a cobertura não lê o `briefing.json`.
        """
        source_dir = Path(source_dir).resolve()
        if not source_dir.is_dir():
            raise NotADirectoryError(f"pasta não encontrada: {source_dir}")
        params = params or EditPipelineParams()
        recorder = TimingRecorder()

        project_dir, _ = catalog_location(source_dir, self.projects_root)
        manifest_path = project_dir / MANIFEST_FILE
        if manifest_path.is_file() and is_multiclip_manifest(manifest_path):
            manifest = SpineManifest.load(manifest_path)
            manifest.source_dir, manifest.params, manifest.recipe = source_dir, params, self.recipe.name
        else:
            manifest = SpineManifest(source_dir=source_dir, recipe=self.recipe.name, params=params)
        manifest.settings = settings
        manifest.ensure_stages(list(self.recipe.steps))
        manifest.save(manifest_path)

        video_name, srt_name = output_names(source_dir)
        state = _SpineState(
            run=_Run(self, manifest, manifest_path, force, recorder),
            params=params,
            source_dir=source_dir,
            project_dir=project_dir,
            output_path=project_dir / video_name,
            srt_path=project_dir / srt_name,
            coverage_params=coverage or CoverageParams.inherit(
                params.director, params.director_model, params.director_fallback, params.director_thinking,
                params.director_timeout_sec, params.api_retries,
            ),
            settings=settings,
            use_briefing=use_briefing,
        )
        steps: dict[str, Callable[[_SpineState], StageOutcome]] = {
            "catalog": self._catalog,
            "silences": self._clip_silences,
            "transcription": self._clip_transcription,
            "director": self._clip_director,
            "timeline": self._timeline,
            "visual": self._visual,
            "coverage": self._coverage,
            "render": self._timeline_render,
            "subtitles": self._clip_subtitles,
            "report": self._spine_report,
        }
        for step in self.recipe.steps:
            state.outcomes[step] = steps[step](state)

        timings = recorder.snapshot()
        manifest.last_run = timings
        manifest.save(manifest_path)
        version = register_spine_version(project_dir, manifest, manifest_path, timings)

        transcript_path = project_dir / EDIT_TRANSCRIPT_FILE
        director_run = state.director_run
        plan = state.plan
        return SpineResult(
            project_dir=project_dir,
            version=version.number if version is not None else None,
            manifest_path=manifest_path,
            output_path=state.output_path,
            report_path=project_dir / EDIT_REPORT_FILE,
            subtitles_path=state.srt_path if state.subtitle_blocks is not None else None,
            transcript_path=transcript_path if transcript_path.is_file() else None,
            timeline_path=project_dir / (COVERAGE_TIMELINE_FILE if state.coverage_run is not None else TIMELINE_FILE),
            coverage=state.coverage_run,
            original_duration_sec=sum(c.duration_sec for c in state.clips),
            final_duration_sec=state.final_duration,
            outcomes=state.outcomes,
            timings=timings,
            clips_used=[c.clip_id for c in plan.clips if c.dropped is None] if plan is not None else [],
            clips_unused=len(state.selection.unused) if state.selection is not None else 0,
            warnings=[
                *(state.selection.warnings if state.selection is not None else []),
                *state.transcription_notes,
                *(f"{c.clip_id} ficou fora da espinha: {c.dropped}" for c in (plan.clips if plan else []) if c.dropped),
                *(director_run.warnings if director_run is not None else []),
                *(state.visual.warnings if state.visual is not None else []),
                *(state.coverage_run.warnings if state.coverage_run is not None else []),
                *state.render_warnings,
                *(state.loudness.warnings if state.loudness is not None else []),
            ],
        )

    # --- etapas (peças da receita) ---

    def _catalog(self, s: _SpineState) -> StageOutcome:
        """Lê (ou gera) o catálogo e escolhe os clipes da espinha."""
        catalog_file = catalog_path(s.project_dir)

        def catalog_fp() -> str:
            current = file_hash(catalog_file) if catalog_file.is_file() else "missing"
            return fingerprint("catalog", current, SELECTION_VERSION, MIN_SPEECH_RATIO, MIN_SPEECH_SEC)

        fp = catalog_fp()

        def do_catalog() -> tuple[dict, str]:
            catalog, built = load_or_build_catalog(s.source_dir, self.projects_root, self._catalog_builder)
            selection = select_spine_clips(catalog)
            (s.project_dir / SPINE_CLIPS_FILE).write_text(selection.model_dump_json(indent=2), encoding="utf-8")
            for warning in selection.warnings:
                self._emit("catalog", "notice", warning)
            principal = sum(1 for c in catalog.clips if c.role == "principal" and c.id is not None)
            detail = f"{len(selection.clips)} de {principal} clipes principais na espinha"
            coverage = sum(1 for u in selection.unused if u.coverage)
            if coverage:
                detail += f", {coverage} para cobertura"
            if built:
                detail += " (catálogo gerado agora)"
            metrics = {"clips": len(selection.clips), "unused": len(selection.unused), "catalog_built": int(built)}
            return metrics, detail

        outcome = s.run.stage("catalog", fp, SPINE_CLIPS_FILE, do_catalog)
        record = s.run.manifest.stages["catalog"]
        if record.fingerprint != catalog_fp():  # catálogo gerado agora: o fingerprint passa a ser o dele
            record.fingerprint = catalog_fp()
            s.run.manifest.save(s.run.manifest_path)
        s.selection = SpineSelection.model_validate_json((s.project_dir / SPINE_CLIPS_FILE).read_text(encoding="utf-8"))
        if not s.clips:
            raise PlanningError(
                "nenhum clipe principal com fala no catálogo (confira os papéis das subpastas com `catalog --papel`)"
            )
        s.hashes = {clip.id: file_hash(clip.path) for clip in s.clips}
        s.run.recorder.video_duration_sec = sum(c.duration_sec for c in s.clips)
        return outcome

    def _silence_fps(self, s: _SpineState) -> dict[str, str]:
        params = s.params
        threshold_key = (
            ("auto", AUTO_RULE_VERSION) if params.threshold_db == "auto" else ("fixed", params.threshold_db)
        )
        return {
            clip.id: fingerprint("silences", s.hashes[clip.id], threshold_key, params.min_silence_sec)
            for clip in s.clips
        }

    def _clip_audio(self, s: _SpineState, clip: SpineClip, fp: str, force: bool = False) -> tuple[AudioAnalysisResult, bool]:
        """(silêncios do clipe, veio do cache por clipe)."""
        path = clip_cache_file(self.projects_root, s.hashes[clip.id], "silences", fp)
        if path.is_file() and not force:
            return AudioAnalysisResult.load_json(path), True
        result = detect_silences(clip.path, s.params.threshold_db, s.params.min_silence_sec)
        result.save_json(path)
        return result, False

    def _clip_silences(self, s: _SpineState) -> StageOutcome:
        per_clip = self._silence_fps(s)
        s.fp_silences = fingerprint("silences", [(clip.id, per_clip[clip.id]) for clip in s.clips])

        def do_silences() -> tuple[dict, str]:
            # Um FFmpeg por clipe, independentes: em paralelo (a transcrição, na GPU, fica sequencial).
            total_clips = len(s.clips)
            self._stage_progress("silences", StageProgress(done=0, total=total_clips, unit="clips"))
            results: dict[str, tuple[AudioAnalysisResult, bool]] = {}
            with ThreadPoolExecutor(max_workers=SILENCE_WORKERS) as pool:
                futures = {pool.submit(self._clip_audio, s, clip, per_clip[clip.id], s.run.force): clip for clip in s.clips}
                for done, future in enumerate(as_completed(futures), start=1):
                    results[futures[future].id] = future.result()
                    self._stage_progress(
                        "silences", StageProgress(done=done, total=total_clips, unit="clips", note=futures[future].id)
                    )
            cached, summary = 0, {}
            for clip in s.clips:
                result, from_cache = results[clip.id]
                cached += from_cache
                summary[clip.id] = {
                    "silences": len(result.silences),
                    "silence_total_sec": round(sum(x.duration for x in result.silences), 3),
                    "threshold_db": result.params.threshold_db,
                }
            (s.project_dir / SILENCES_FILE).write_text(_json(summary), encoding="utf-8")
            total = sum(v["silence_total_sec"] for v in summary.values())
            metrics = {"clips": len(s.clips), "clips_cached": cached, "silence_total_sec": round(total, 3)}
            return metrics, f"{len(s.clips)} clipes ({cached} do cache), {total:.1f}s de silêncio"

        outcome = s.run.stage("silences", s.fp_silences, SILENCES_FILE, do_silences)
        s.audio = {clip.id: self._clip_audio(s, clip, per_clip[clip.id])[0] for clip in s.clips}
        return outcome

    def _clip_transcription(self, s: _SpineState) -> StageOutcome:
        """Opcional (`--transcribe`), um clipe por vez, com barra de progresso por clipe."""
        params = s.params
        if not params.transcribe:
            return s.run.skip("transcription", "use --transcribe para gerar")
        provider = self.transcription_provider
        effective = provider.effective_settings(params.transcription)
        config = params.transcription.model_copy(
            update={"model_size": provider.effective_model(params.transcription), "device": effective[0], "compute_type": effective[1]}
        )
        per_clip = {
            clip.id: fingerprint("transcription", s.hashes[clip.id], provider.provider_name, config) for clip in s.clips
        }
        s.fp_transcript = fingerprint("transcription", [(clip.id, per_clip[clip.id]) for clip in s.clips])
        total_sec = sum(c.duration_sec for c in s.clips)

        def cache_of(clip: SpineClip) -> Path:
            return clip_cache_file(self.projects_root, s.hashes[clip.id], "transcript", per_clip[clip.id])

        def do_transcription() -> tuple[dict, str]:
            cached, done_sec, summary = 0, 0.0, {}
            for k, clip in enumerate(s.clips):
                note = f"{clip.id} ({k + 1}/{len(s.clips)})"
                self._stage_progress("transcription", StageProgress(done=done_sec, total=total_sec, note=note))
                path = cache_of(clip)
                if path.is_file() and not s.run.force:
                    transcript = Transcript.load(path)
                    cached += 1
                else:
                    offset = done_sec
                    transcript = self._transcribe_clip(
                        provider,
                        clip,
                        path,
                        params.transcription,
                        on_progress=lambda done, _total, offset=offset, note=note: self._stage_progress(
                            "transcription", StageProgress(done=offset + done, total=total_sec, note=note)
                        ),
                    )
                    adjusted = _load_adjustments(path)
                    if adjusted is not None:
                        self._emit("transcription", "notice", adjustments_note(adjusted.summary(), clip.id))
                done_sec += clip.duration_sec
                summary[clip.id] = {"language": transcript.language, "segments": len(transcript.segments)}
            (s.project_dir / TRANSCRIPTION_FILE).write_text(_json(summary), encoding="utf-8")
            segments = sum(v["segments"] for v in summary.values())
            metrics = {"provider": provider.provider_name, "clips": len(s.clips), "clips_cached": cached, "segments": segments}
            return metrics, f"{len(s.clips)} clipes ({cached} do cache), {segments} segmentos"

        outcome = s.run.stage("transcription", s.fp_transcript, TRANSCRIPTION_FILE, do_transcription)
        for clip in s.clips:
            path = cache_of(clip)
            if not path.is_file():  # cache por clipe apagado depois da execução anterior
                self._transcribe_clip(provider, clip, path, params.transcription)
            s.transcripts[clip.id] = Transcript.load(path)
            adjusted = _load_adjustments(path)
            if adjusted is not None:
                s.transcription_notes.append(adjustments_note(adjusted.summary(), clip.id))
        return outcome

    def _transcribe_clip(
        self, provider: TranscriptionProvider, clip: SpineClip, path: Path, config: TranscriptionConfig, on_progress=None
    ) -> Transcript:
        """Transcreve um clipe e grava no cache por clipe, com os ajustes da normalização ao lado.

        O erro diz o clipe: com 22 clipes, "segmentos fora de ordem" sozinho não diz onde olhar.
        """
        try:
            transcript = provider.transcribe(clip.path, config, on_progress=on_progress)
        except Exception as exc:
            raise TranscriptionError(f"{clip.id} ({clip.path.name}): {exc}") from exc
        transcript.save(path)
        sidecar = _adjustments_path(path)
        adjustments = provider.last_adjustments
        if adjustments is not None and adjustments.changed:
            sidecar.write_text(adjustments.model_dump_json(indent=2), encoding="utf-8")
        elif sidecar.is_file():
            sidecar.unlink()
        return transcript

    def _clip_director(self, s: _SpineState) -> StageOutcome:
        """Opcional; escolhe frases (com IDs globais) a remover, com os clipes como seções do prompt."""
        params = s.params
        if params.director == "none":
            return s.run.skip("director", "use --director para ativar")
        if s.fp_transcript is None:
            return s.run.skip("director", "requer --transcribe")
        director = self.director_for(params)
        s.fp_director = fingerprint(
            "director", DIRECTOR_PROMPT_FORMAT, s.fp_transcript, director.provider_name, director.model_id,
            director.thinking_label, params.brief, params.sentences, list(director.model_chain),
        )

        def do_director() -> tuple[dict, str]:
            model = director.model_chain[0] if director.model_chain else director.model_id
            self._stage_progress("director", StageProgress(note=model))
            sentences = build_clip_sentences([(c.id, s.transcripts[c.id]) for c in s.clips], params.sentences)
            languages = [s.transcripts[c.id].language for c in s.clips]
            context = DirectorContext(
                video_name=s.source_dir.name,
                duration_sec=sum(c.duration_sec for c in s.clips),
                language=max(set(languages), key=languages.count) if languages else None,
                clips=[DirectorClip(id=c.id, name=c.name, duration_sec=c.duration_sec) for c in s.clips],
            )
            result = run_director(director, sentences, params.brief, context)
            (s.project_dir / DIRECTOR_DECISION_FILE).write_text(result.model_dump_json(indent=2), encoding="utf-8")
            removed = result.removed_sentences
            removed_sec = sum(x.duration for x, _ in removed)
            retakes = sum(1 for _, r in removed if r.category == "retake")
            metrics = {
                "provider": director.provider_name,
                "model": director.model_id,
                "effective_model": result.effective_model_id or director.model_id,
                "model_switches": len(result.model_switches),
                "sentences": len(sentences),
                "removed": len(removed),
                "retakes": retakes,
                "removed_sec": round(removed_sec, 3),
                "attempts": len(result.attempts),
                "fallback": int(result.decision is None),
            }
            if result.decision is None:
                return metrics, "resposta inválida; seguindo só com o corte de silêncio"
            summary = f"{len(removed)} de {len(sentences)} frases removidas ({removed_sec:.2f}s, {retakes} retake)"
            if result.model_switches:
                summary += f" com {result.effective_provider}:{result.effective_model_id}"
            return metrics, summary

        outcome = s.run.stage("director", s.fp_director, DIRECTOR_DECISION_FILE, do_director)
        s.director_run = DirectorRun.model_validate_json(
            (s.project_dir / DIRECTOR_DECISION_FILE).read_text(encoding="utf-8")
        )
        return outcome

    def _timeline(self, s: _SpineState) -> StageOutcome:
        """Por clipe, os trechos mantidos (silêncios + remoções do Diretor); em sequência na Timeline."""
        params, director_run = s.params, s.director_run
        fps = spine_fps([c.fps for c in s.clips])
        thresholds = [s.audio[c.id].params.threshold_db for c in s.clips]
        s.fp_timeline = fingerprint(
            "timeline", TIMELINE_FORMAT, s.fp_silences, thresholds, s.fp_director, params.effective_cut,
            [str(c.path) for c in s.clips], fps, SPINE_RESOLUTION, params.loudness,
        )

        def do_timeline() -> tuple[dict, str]:
            sentences, removed_ids = [], None
            if director_run is not None and director_run.decision is not None:
                sentences = director_run.sentences
                removed_ids = {r.sentence_id for r in director_run.decision.removals}
            plans: list[ClipPlan] = []
            cuts: list[ClipCut] = []
            for clip in s.clips:
                audio = s.audio[clip.id]
                clip_sentences = [x for x in sentences if x.clip_id == clip.id]
                clip_removed = None if removed_ids is None else {x.id for x in clip_sentences} & removed_ids
                base = ClipPlan(clip_id=clip.id, threshold_db=audio.params.threshold_db)
                try:
                    keep, results = plan_keep_ranges(audio, clip_sentences, clip_removed, params.effective_cut)
                except NoSpeechDetectedError:
                    everything = bool(clip_sentences) and clip_removed is not None and len(clip_removed) == len(clip_sentences)
                    reason = "todas as frases removidas pelo Diretor" if everything else "nenhuma fala nos silêncios detectados"
                    plans.append(base.model_copy(update={"dropped": reason}))
                    continue
                keep = usable_ranges(keep, fps)
                if not keep:
                    plans.append(base.model_copy(update={"dropped": "nenhum trecho com pelo menos 1 frame"}))
                    continue
                plans.append(base.model_copy(update={"keep": keep, "cut_results": results}))
                cuts.append(ClipCut(clip_id=clip.id, source_path=clip.path, keep=keep))
            timeline = build_spine_timeline(
                cuts, s.output_path, fps,
                title=f"Espinha falada — {s.source_dir.name}",
                loudness=LoudnessTarget.from_preset(params.loudness),
            )
            plan = SpinePlan(fps=fps, resolution=SPINE_RESOLUTION, source_fps=[c.fps for c in s.clips], clips=plans)
            (s.project_dir / SPINE_PLAN_FILE).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            (s.project_dir / TIMELINE_FILE).write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
            dropped = sum(1 for p in plans if p.dropped)
            metrics = {
                "items": len(timeline.video_main),
                "clips": len(cuts),
                "dropped_clips": dropped,
                "duration_sec": round(timeline.duration_sec, 3),
                "fps": fps,
                "director_cuts": sum(len(p.cut_results) for p in plans),
            }
            detail = f"{len(cuts)} clipes, {len(timeline.video_main)} trechos ({timeline.duration_sec:.2f}s) @ {fps}fps"
            if dropped:
                detail += f", {dropped} clipe(s) fora"
            return metrics, detail

        outcome = s.run.stage("timeline", s.fp_timeline, TIMELINE_FILE, do_timeline)
        s.plan = SpinePlan.model_validate_json((s.project_dir / SPINE_PLAN_FILE).read_text(encoding="utf-8"))
        s.timeline = Timeline.model_validate_json((s.project_dir / TIMELINE_FILE).read_text(encoding="utf-8"))
        return outcome

    def _visual(self, s: _SpineState) -> StageOutcome:
        """Lê o `visual.json` do projeto (comando `visual`), sem chamar a IA; só as tomadas que a cobertura usa."""
        visual_file = s.project_dir / VISUAL_DIR / VISUAL_JSON
        catalog_file = catalog_path(s.project_dir)
        fp = fingerprint(
            "visual", VISUAL_MATERIAL_FORMAT,
            file_hash(visual_file) if visual_file.is_file() else "missing",
            file_hash(catalog_file) if catalog_file.is_file() else "missing",
            [c.id for c in s.clips],
        )

        def do_visual() -> tuple[dict, str]:
            if not visual_file.is_file():
                raise CoverageError(coverage_material.missing_visual_message(s.source_dir, visual_file))
            from ai_video_editor.visual.schema import ProjectVisual

            material = build_visual_material(
                ProjectVisual.load(visual_file), ProjectCatalog.load(catalog_file), visual_file, [c.id for c in s.clips]
            )
            (s.project_dir / COVERAGE_SHOTS_FILE).write_text(material.model_dump_json(indent=2), encoding="utf-8")
            analyzed = sum(1 for x in material.shots if x.status == "analisada")
            for warning in material.warnings:
                self._emit("visual", "notice", warning)
            metrics = {"shots": len(material.shots), "analyzed": analyzed}
            return metrics, f"{len(material.shots)} tomadas ({analyzed} analisadas pela IA)"

        outcome = s.run.stage("visual", fp, COVERAGE_SHOTS_FILE, do_visual)
        shots_file = s.project_dir / COVERAGE_SHOTS_FILE
        s.visual = VisualMaterial.model_validate_json(shots_file.read_text(encoding="utf-8"))
        s.fp_visual = fingerprint("visual-material", file_hash(shots_file))
        return outcome

    def _briefing(self, s: _SpineState) -> tuple[list[str], list[str]]:
        """(instruções de cobertura, papéis a usar) do `briefing.json` do projeto, se existir."""
        path = s.project_dir / BRIEFING_FILE
        if not s.use_briefing or not path.is_file():
            return [], []
        from ai_video_editor.producer.schema import Briefing

        try:
            briefing = Briefing.load(path)
        except ValueError as exc:
            self._emit("coverage", "notice", f"briefing.json ignorado (formato inválido: {exc.__class__.__name__})")
            return [], []
        instructions = [briefing.coverage_instructions.strip()] if briefing.coverage_instructions.strip() else []
        return instructions, list(briefing.roles_to_use)

    def _coverage(self, s: _SpineState) -> StageOutcome:
        """Diretor de cobertura: janelas obrigatórias (código), escolha por IDs (IA), validação e Timeline final."""
        s.final_timeline = s.timeline
        params = s.params
        if s.fp_transcript is None:
            # Uma Timeline com cobertura de uma execução anterior não vale mais (o render usa a espinha).
            return s.run.skip(
                "coverage", "requer --transcribe (a cobertura escolhe frases por ID)", artifact=COVERAGE_TIMELINE_FILE
            )
        director = self.coverage_director_for(s.coverage_params)
        briefing_instructions, briefing_roles = self._briefing(s)
        instructions = list(briefing_instructions)
        if params.brief.instructions:
            instructions.append(f"Pedido do usuário: {params.brief.instructions.strip()}")
        roles = coverage_roles(briefing_roles)
        identity = (
            [director.provider_name, director.model_id, director.thinking_label, list(director.model_chain)]
            if director is not None else ["none"]
        )
        s.fp_coverage = fingerprint(
            "coverage", COVERAGE_FORMAT, COVERAGE_PROMPT_VERSION, s.fp_timeline, s.fp_visual, instructions, roles,
            identity, COVERAGE_RULES,
        )

        def do_coverage() -> tuple[dict, str]:
            if director is not None:
                self._stage_progress("coverage", StageProgress(note=director.model_chain[0] if director.model_chain else director.model_id))
            sentences, removed = self._kept_sentences(s)
            spine_ids = [c.id for c in s.clips]
            kept = coverage_sentences(sentences, removed, s.timeline)
            context = CoverageContext(
                project_name=s.source_dir.name, fps=s.timeline.fps, spine_duration_sec=s.timeline.duration_sec,
                sentences=kept, windows=find_mandatory_windows(kept, s.visual, spine_ids),
                candidates=coverage_candidates(s.visual, roles), instructions=instructions, roles=roles,
            )
            run = run_coverage(director, context, s.visual.shots)
            assembled = assemble_coverage(
                s.timeline, context, run, s.visual.by_id(), title=f"Vlog com cobertura — {s.source_dir.name}"
            )
            run.placed_coverages, run.placed_breaths = assembled.coverages, assembled.breaths
            run.warnings += assembled.warnings
            run.final_duration_sec = round(assembled.timeline.duration_sec, 3)
            for warning in run.warnings:
                self._emit("coverage", "notice", warning)
            (s.project_dir / COVERAGE_DECISION_FILE).write_text(run.model_dump_json(indent=2), encoding="utf-8")
            (s.project_dir / COVERAGE_TIMELINE_FILE).write_text(assembled.timeline.model_dump_json(indent=2), encoding="utf-8")
            breath_sec = sum(b.duration_sec for b in run.placed_breaths)
            metrics = {
                "provider": run.provider,
                "effective_model": run.effective_model_id or run.model_id,
                "model_switches": len(run.model_switches),
                "origin": run.origin,
                "windows": len(run.windows),
                "uncovered_windows": len(run.uncovered_windows),
                "coverages": len(run.placed_coverages),
                "breaths": len(run.placed_breaths),
                "breath_sec": round(breath_sec, 3),
                "candidates": run.candidates,
                "attempts": len(run.attempts),
                "duration_sec": run.final_duration_sec,
            }
            covered = len(run.windows) - len(run.uncovered_windows)
            detail = (
                f"{len(run.placed_coverages)} coberturas, {len(run.placed_breaths)} respiros (+{breath_sec:.1f}s), "
                f"{covered}/{len(run.windows)} janelas obrigatórias ({run.origin})"
            )
            return metrics, detail

        outcome = s.run.stage("coverage", s.fp_coverage, COVERAGE_TIMELINE_FILE, do_coverage)
        s.fp_coverage_result = fingerprint(
            "coverage-result", file_hash(s.project_dir / COVERAGE_TIMELINE_FILE), file_hash(s.project_dir / COVERAGE_DECISION_FILE)
        )
        s.coverage_run = CoverageRun.model_validate_json((s.project_dir / COVERAGE_DECISION_FILE).read_text(encoding="utf-8"))
        s.final_timeline = Timeline.model_validate_json((s.project_dir / COVERAGE_TIMELINE_FILE).read_text(encoding="utf-8"))
        return outcome

    def _kept_sentences(self, s: _SpineState):
        """(frases com IDs globais, IDs removidos pelo Diretor): as mesmas frases do diff e da Timeline."""
        run = s.director_run
        if run is not None:
            removed = {r.sentence_id for r in run.decision.removals} if run.decision is not None else set()
            return run.sentences, removed
        return build_clip_sentences([(c.id, s.transcripts[c.id]) for c in s.clips], s.params.sentences), set()

    def _timeline_render(self, s: _SpineState) -> StageOutcome:
        if s.final_timeline is None:
            s.final_timeline = s.timeline
        # O codificador entra só aqui (CPU = o fingerprint de antes); as etapas anteriores não mudam.
        encoder = self.timeline_executor.resolve_encoder()

        def render_fp(enc) -> str:
            return (
                fingerprint("render", s.fp_timeline, *enc.fingerprint_parts)
                if s.fp_coverage is None
                else fingerprint(
                    "render",
                    s.fp_timeline,
                    file_hash(s.project_dir / COVERAGE_TIMELINE_FILE),
                    *enc.fingerprint_parts,
                )
            )

        s.fp_render = render_fp(encoder)
        used: list[TimelineRenderResult] = []

        def do_render() -> tuple[dict, str]:
            result = self.timeline_executor.execute(
                s.final_timeline,
                on_progress=lambda done, total: self._stage_progress(
                    "render", StageProgress(done=done, total=total, unit="items")
                ),
            )
            used.append(result)
            metrics: dict = {
                "duration_sec": round(result.duration_sec, 3),
                "frames": result.frame_count,
                "size_bytes": result.size_bytes,
                "encoder": result.encoder.label,
            }
            if result.loudness is not None:
                metrics.update(result.loudness.to_metrics())
            source_warnings = [w for w in result.warnings if not result.loudness or w not in result.loudness.warnings]
            if source_warnings:
                metrics["warnings"] = " | ".join(source_warnings)
            return metrics, f"{result.duration_sec:.2f}s, {result.size_bytes / 1_048_576:.1f} MB, {result.encoder.label}"

        outcome = s.run.stage("render", s.fp_render, s.output_path.name, do_render)
        if used and used[0].encoder.name != encoder.name:
            # Voltou para a CPU no meio: o vídeo é de CPU e fica registrado como tal.
            s.fp_render = s.run.rekey("render", render_fp(used[0].encoder))
        metrics = s.run.manifest.stages["render"].metrics
        s.final_duration = float(metrics["duration_sec"])
        s.loudness = LoudnessResult.from_metrics(metrics)
        s.encoder_label = str(metrics.get("encoder", "cpu"))
        s.render_warnings = [w for w in str(metrics.get("warnings", "")).split(" | ") if w]
        return outcome

    def _clip_subtitles(self, s: _SpineState) -> StageOutcome:
        """Só com transcrição; a fala de cada clipe remapeada para o tempo final pela Timeline."""
        params = s.params
        srt_name = s.srt_path.name
        if not params.subtitles:
            return s.run.skip("subtitles", "desligada por --no-subtitles", artifact=srt_name)
        if s.fp_transcript is None:
            return s.run.skip("subtitles", "requer --transcribe", artifact=srt_name)
        s.fp_subtitles = fingerprint(
            "subtitles", s.fp_timeline, s.fp_transcript, params.subtitle,
            *([s.fp_coverage_result] if s.fp_coverage else []),
        )

        def do_subtitles() -> tuple[dict, str]:
            fps = s.timeline.fps
            if s.fp_coverage is None:
                spans = clip_spans(s.timeline)
                sources = [
                    ClipSubtitleSource(
                        transcript=s.transcripts[p.clip_id], kept_ranges=p.keep, offset_sec=spans[p.clip_id][0] / fps
                    )
                    for p in s.plan.clips
                    if p.dropped is None
                ]
            else:
                # Timeline final: os trechos de um clipe podem estar separados por respiros.
                placements = clip_placements(s.final_timeline)
                sources = [
                    ClipSubtitleSource(
                        transcript=s.transcripts[p.clip_id],
                        kept_ranges=[TimeRange(start_sec=x.source_in, end_sec=x.source_out) for x in placements[p.clip_id]],
                        offset_sec=placements[p.clip_id][0].start_frame / fps,
                        range_starts_sec=[x.start_frame / fps for x in placements[p.clip_id]],
                    )
                    for p in s.plan.clips
                    if p.dropped is None and p.clip_id in placements
                ]
            blocks = build_clip_subtitle_blocks(sources, fps, params.subtitle)
            write_srt(blocks, s.srt_path)
            return {"blocks": len(blocks)}, f"{srt_name}, {len(blocks)} blocos"

        outcome = s.run.stage("subtitles", s.fp_subtitles, srt_name, do_subtitles)
        s.subtitle_blocks = int(s.run.manifest.stages["subtitles"].metrics["blocks"])
        return outcome

    def _spine_report(self, s: _SpineState) -> StageOutcome:
        """Relatório por clipe (+ transcrição comparativa e linha do tempo, que exigem transcrição)."""
        params = s.params
        fp_report = fingerprint(
            "report", REPORT_FORMAT, s.fp_render, s.fp_director, s.fp_subtitles, s.fp_transcript,
            params.fingerprint_view(), s.selection, s.encoder_label, *([s.fp_coverage_result] if s.fp_coverage else []),
            *settings_fingerprint(s.settings),
        )
        coverage = s.coverage_run if s.fp_coverage is not None else None
        final = s.final_timeline if coverage is not None else None

        def do_report() -> tuple[dict, str]:
            started = time.perf_counter()
            diff = build_spine_diff(
                s.source_dir.name, s.selection, s.plan, s.timeline, s.director_run,
                s.transcripts if s.fp_transcript is not None else None, params, final=final,
            )
            write_spine_diff(
                diff, s.timeline, s.project_dir, with_transcript=s.fp_transcript is not None, final=final, coverage=coverage
            )
            markdown = build_spine_report(
                s.source_dir,
                s.selection,
                s.plan,
                s.timeline,
                diff,
                params,
                s.output_path,
                s.final_duration,
                director=s.director_run,
                subtitles_path=s.srt_path if s.subtitle_blocks is not None else None,
                subtitle_blocks=s.subtitle_blocks,
                transcription_model=(
                    next(iter(s.transcripts.values())).model_used if s.transcripts else None
                ),
                transcription_notes=s.transcription_notes,
                timings=s.run.recorder.snapshot(in_progress=("report", started)),
                loudness=s.loudness,
                encoder=s.encoder_label,
                recipe_name=self.recipe.name,
                coverage=coverage,
                settings=s.settings,
            )
            (s.project_dir / EDIT_REPORT_FILE).write_text(markdown, encoding="utf-8")
            merged = diff.merged()
            metrics: dict = {"cuts": removed_range_count(s.plan, diff), "clips": len(clip_spans(s.timeline))}
            detail = f"{len(diff.clips)} clipes no relatório"
            if s.fp_transcript is not None:
                metrics.update(
                    applied=merged.count("applied"), partial=merged.count("partial"),
                    not_applied=merged.count("not_applied"),
                )
                detail += f", transcrição comparativa em {EDIT_TRANSCRIPT_FILE} e linha do tempo em {EDIT_TIMELINE_FILE}"
            else:
                detail += f", linha do tempo em {EDIT_TIMELINE_FILE}"
            return metrics, detail

        return s.run.stage("report", fp_report, EDIT_REPORT_FILE, do_report)


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _adjustments_path(transcript_path: Path) -> Path:
    """Ajustes da normalização ao lado da transcrição do cache por clipe (só existe se houve ajuste)."""
    return transcript_path.with_name(f"{transcript_path.stem}.adjustments.json")


def _load_adjustments(transcript_path: Path) -> Optional[SegmentAdjustments]:
    sidecar = _adjustments_path(transcript_path)
    if not sidecar.is_file():
        return None
    return SegmentAdjustments.model_validate_json(sidecar.read_text(encoding="utf-8"))
