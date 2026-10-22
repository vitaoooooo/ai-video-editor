"""Orquestrador do `edit <pasta>`: a espinha falada de um projeto com vários clipes (receita `vlog`).

catálogo → silêncios por clipe → transcrição por clipe → Diretor → Timeline → render → legendas →
relatório. Os clipes de papel `principal` com fala entram em ordem; cada um perde as pausas e as
frases ruins (mesmo corte do caminho de um vídeo, `plan_keep_ranges`) e os trechos mantidos viram
itens de `video_main` numa Timeline, renderizada por `renderer/timeline_*`.

O projeto é a pasta do catálogo (`<projects>/<pasta>-<hash da listagem>/`). Cada etapa grava um
artefato e um fingerprint no `project.json` (`kind="multiclip"`), como no `edit` de um vídeo.
Silêncios e transcrição têm também cache por clipe, fora do projeto (chave: hash do arquivo +
parâmetros), então trocar um clipe da pasta só reprocessa esse clipe.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor.audio import AudioAnalysisResult, detect_silences
from ai_video_editor.audio.levels import AUTO_RULE_VERSION
from ai_video_editor.core.schema import LoudnessTarget
from ai_video_editor.core.timeline import Timeline
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
from ai_video_editor.pipeline.schema import EditPipelineParams, StageOutcome, StageProgress
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
    clip_spans,
    spine_fps,
    usable_ranges,
)
from ai_video_editor.recipes import DEFAULT_FOLDER_RECIPE, Recipe, RecipeError
from ai_video_editor.renderer.loudness import LoudnessResult
from ai_video_editor.renderer.timeline_executor import TimelineRenderExecutor
from ai_video_editor.subtitles import ClipSubtitleSource, build_clip_subtitle_blocks, write_srt
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript

# Etapas que o `edit <pasta>` sabe executar.
SPINE_STEPS = ("catalog", "silences", "transcription", "director", "timeline", "render", "subtitles", "report")

# Versões de formato que entram nos fingerprints (sobem quando a saída da etapa muda).
DIRECTOR_PROMPT_FORMAT = "spine-prompt-v1"
TIMELINE_FORMAT = "spine-timeline-v1"
REPORT_FORMAT = "spine-report-v1"


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
    director_run: Optional[DirectorRun] = None
    plan: Optional[SpinePlan] = None
    timeline: Optional[Timeline] = None
    final_duration: float = 0.0
    loudness: Optional[LoudnessResult] = None
    render_warnings: list[str] = field(default_factory=list)
    subtitle_blocks: Optional[int] = None

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

    def run(  # type: ignore[override]
        self, source_dir: Path, params: Optional[EditPipelineParams] = None, force: bool = False
    ) -> SpineResult:
        """Executa (ou retoma do cache) as etapas da receita sobre a pasta e devolve o resumo."""
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
        )
        steps: dict[str, Callable[[_SpineState], StageOutcome]] = {
            "catalog": self._catalog,
            "silences": self._clip_silences,
            "transcription": self._clip_transcription,
            "director": self._clip_director,
            "timeline": self._timeline,
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
            timeline_path=project_dir / TIMELINE_FILE,
            original_duration_sec=sum(c.duration_sec for c in state.clips),
            final_duration_sec=state.final_duration,
            outcomes=state.outcomes,
            timings=timings,
            clips_used=[c.clip_id for c in plan.clips if c.dropped is None] if plan is not None else [],
            clips_unused=len(state.selection.unused) if state.selection is not None else 0,
            warnings=[
                *(state.selection.warnings if state.selection is not None else []),
                *(f"{c.clip_id} ficou fora da espinha: {c.dropped}" for c in (plan.clips if plan else []) if c.dropped),
                *(director_run.warnings if director_run is not None else []),
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
            cached, summary = 0, {}
            for k, clip in enumerate(s.clips):
                self._stage_progress(
                    "silences", StageProgress(done=k, total=len(s.clips), unit="clips", note=f"{clip.id} ({k + 1}/{len(s.clips)})")
                )
                result, from_cache = self._clip_audio(s, clip, per_clip[clip.id], s.run.force)
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
                    transcript = provider.transcribe(
                        clip.path,
                        params.transcription,
                        on_progress=lambda done, _total, offset=offset, note=note: self._stage_progress(
                            "transcription", StageProgress(done=offset + done, total=total_sec, note=note)
                        ),
                    )
                    transcript.save(path)
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
                provider.transcribe(clip.path, params.transcription).save(path)
            s.transcripts[clip.id] = Transcript.load(path)
        return outcome

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
            "timeline", TIMELINE_FORMAT, s.fp_silences, thresholds, s.fp_director, params.cut,
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
                    keep, results = plan_keep_ranges(audio, clip_sentences, clip_removed, params.cut)
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

    def _timeline_render(self, s: _SpineState) -> StageOutcome:
        s.fp_render = fingerprint("render", s.fp_timeline)

        def do_render() -> tuple[dict, str]:
            result = self.timeline_executor.execute(
                s.timeline,
                on_progress=lambda done, total: self._stage_progress(
                    "render", StageProgress(done=done, total=total, unit="items")
                ),
            )
            metrics: dict = {
                "duration_sec": round(result.duration_sec, 3),
                "frames": result.frame_count,
                "size_bytes": result.size_bytes,
            }
            if result.loudness is not None:
                metrics.update(result.loudness.to_metrics())
            source_warnings = [w for w in result.warnings if not result.loudness or w not in result.loudness.warnings]
            if source_warnings:
                metrics["warnings"] = " | ".join(source_warnings)
            return metrics, f"{result.duration_sec:.2f}s, {result.size_bytes / 1_048_576:.1f} MB"

        outcome = s.run.stage("render", s.fp_render, s.output_path.name, do_render)
        metrics = s.run.manifest.stages["render"].metrics
        s.final_duration = float(metrics["duration_sec"])
        s.loudness = LoudnessResult.from_metrics(metrics)
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
        s.fp_subtitles = fingerprint("subtitles", s.fp_timeline, s.fp_transcript, params.subtitle)

        def do_subtitles() -> tuple[dict, str]:
            spans = clip_spans(s.timeline)
            fps = s.timeline.fps
            sources = [
                ClipSubtitleSource(
                    transcript=s.transcripts[p.clip_id], kept_ranges=p.keep, offset_sec=spans[p.clip_id][0] / fps
                )
                for p in s.plan.clips
                if p.dropped is None
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
            "report", REPORT_FORMAT, s.fp_render, s.fp_director, s.fp_subtitles, s.fp_transcript, params,
            s.selection,
        )

        def do_report() -> tuple[dict, str]:
            started = time.perf_counter()
            diff = build_spine_diff(
                s.source_dir.name, s.selection, s.plan, s.timeline, s.director_run,
                s.transcripts if s.fp_transcript is not None else None, params,
            )
            write_spine_diff(diff, s.timeline, s.project_dir, with_transcript=s.fp_transcript is not None)
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
                timings=s.run.recorder.snapshot(in_progress=("report", started)),
                loudness=s.loudness,
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
