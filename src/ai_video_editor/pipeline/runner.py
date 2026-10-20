"""Orquestrador do comando `edit`: probe → silêncios → transcrição → Diretor → plano → render → legendas → relatório.

O relatório vem com a transcrição comparativa (`edit_transcript.md` + `edit_diff.json`) quando há transcrição.

Cada etapa grava um artefato na pasta do projeto e um fingerprint no `project.json`.
Na execução seguinte, a etapa é pulada se o artefato existe e o fingerprint (hash das
entradas e parâmetros que a afetam, encadeado com o das etapas anteriores) não mudou.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor.analyzer.exceptions import VideoNotFoundError
from ai_video_editor.analyzer.probe import VideoProbe
from ai_video_editor.analyzer.schema import VideoMetadata
from ai_video_editor.audio import AudioAnalysisResult, detect_silences
from ai_video_editor.audio.levels import AUTO_RULE_VERSION
from ai_video_editor.core.schema import EditPlan, LoudnessTarget
from ai_video_editor.director import (
    DirectorContext,
    DirectorRun,
    EditDirector,
    build_sentences,
    run_director,
)
from ai_video_editor.pipeline.project import (
    AUDIO_ANALYSIS_FILE,
    DEFAULT_PROJECTS_ROOT,
    DIRECTOR_DECISION_FILE,
    EDIT_PLAN_FILE,
    EDIT_REPORT_FILE,
    EDIT_TRANSCRIPT_FILE,
    MANIFEST_FILE,
    PROBE_FILE,
    TRANSCRIPT_FILE,
    file_hash,
    fingerprint,
    output_video_name,
    project_dir_for,
    subtitles_file_name,
)
from ai_video_editor.pipeline.cutting import plan_keep_ranges
from ai_video_editor.pipeline.edit_diff import build_edit_diff, remove_edit_diff, write_edit_diff
from ai_video_editor.pipeline.report import build_edit_report
from ai_video_editor.pipeline.timing import TimingRecorder, format_duration
from ai_video_editor.pipeline.versions import register_edit_version
from ai_video_editor.pipeline.schema import (
    EditDiff,
    EditPipelineParams,
    EditPipelineResult,
    ProjectManifest,
    StageName,
    StageOutcome,
    StageProgress,
    StageRecord,
)
from ai_video_editor.planning import build_edit_plan, removed_ranges
from ai_video_editor.intelligence.retry import RetryEvent, RetryPolicy
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.renderer.loudness import LoudnessResult
from ai_video_editor.renderer.ffmpeg_locator import get_ffprobe_path
from ai_video_editor.subtitles import build_subtitle_blocks, write_srt
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript

# Versão do formato do relatório; entra no fingerprint para refazer relatórios antigos.
REPORT_FORMAT = "report-v3-loudness"

# (etapa, evento, detalhe). Eventos: "start", "done", "cached", "skipped", "failed", "retry" e
# "notice" (aviso, ex: troca automática de modelo).
ProgressCallback = Callable[[StageName, str, str], None]

# Andamento da etapa em curso (fração concluída, modelo e tentativa...), para exibição ao vivo.
StageProgressCallback = Callable[[StageName, StageProgress], None]

# Resultado de uma etapa: métricas gravadas no manifesto e uma linha de resumo para o terminal.
StageWork = Callable[[], tuple[dict, str]]


def probe_rotation(video_path: Path, ffprobe_bin: Optional[Path] = None) -> int:
    """Rotação de exibição do vídeo em graus (metadado `rotate` ou display matrix)."""
    args = [
        str(ffprobe_bin or get_ffprobe_path()),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        return 0
    try:
        streams = json.loads(proc.stdout or "{}").get("streams", [])
    except json.JSONDecodeError:
        return 0
    if not streams:
        return 0
    stream = streams[0]
    candidates = [stream.get("tags", {}).get("rotate")]
    candidates += [sd.get("rotation") for sd in stream.get("side_data_list", [])]
    for value in candidates:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EditPipeline:
    """Executa o fluxo de corte de silêncio de um vídeo com cache por etapa."""

    def __init__(
        self,
        projects_root: Path = DEFAULT_PROJECTS_ROOT,
        transcription_provider: Optional[TranscriptionProvider] = None,
        render_executor: Optional[RenderExecutor] = None,
        on_progress: Optional[ProgressCallback] = None,
        edit_director: Optional[EditDirector] = None,
        on_stage_progress: Optional[StageProgressCallback] = None,
    ):
        self.projects_root = Path(projects_root)
        self._transcription_provider = transcription_provider
        self._edit_director = edit_director
        self._render_executor = render_executor
        self.on_progress = on_progress
        self.on_stage_progress = on_stage_progress

    @property
    def transcription_provider(self) -> TranscriptionProvider:
        if self._transcription_provider is None:
            from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider

            # Com quem acompanhe o progresso, avisos (ex: GPU indisponível) vão pelo callback.
            on_warning = None
            if self.on_progress is not None:
                on_warning = lambda message: self._emit("transcription", "notice", message)  # noqa: E731
            self._transcription_provider = FasterWhisperProvider(on_warning=on_warning)
        return self._transcription_provider

    def director_for(self, params: EditPipelineParams) -> EditDirector:
        """Diretor injetado no construtor ou o escolhido em `params.director`."""
        if self._edit_director is not None:
            return self._edit_director
        if params.director == "mock":
            from ai_video_editor.director.providers.mock import MockDirector

            return MockDirector()
        from ai_video_editor.director.providers.llm import LLMDirector

        chain = ModelChain(
            params.director_chain(),
            thinking=params.director_thinking,
            timeout_sec=params.director_timeout_sec,
            retry=RetryPolicy(
                max_retries=params.api_retries,
                on_retry=lambda event: self._director_retry(chain, event),
            ),
            on_event=lambda message: self._director_notice(chain, message),
        )
        return LLMDirector(chain)

    def _director_retry(self, chain: ModelChain, event: RetryEvent) -> None:
        self._emit("director", "retry", event.message)
        self._stage_progress(
            "director",
            StageProgress(note=f"{chain.current.label}, tentativa {event.next_attempt}/{event.total_attempts}"),
        )

    def _director_notice(self, chain: ModelChain, message: str) -> None:
        self._emit("director", "notice", message)
        if chain.switches:
            # O aviso sai antes da troca; o próximo elo é o destino da última troca registrada.
            self._stage_progress("director", StageProgress(note=f"{chain.switches[-1].to_model}, tentativa 1"))

    @property
    def render_executor(self) -> RenderExecutor:
        if self._render_executor is None:
            self._render_executor = RenderExecutor()
        return self._render_executor

    def _emit(self, stage: StageName, event: str, detail: str = "") -> None:
        if self.on_progress is not None:
            self.on_progress(stage, event, detail)

    def _stage_progress(self, stage: StageName, progress: StageProgress) -> None:
        if self.on_stage_progress is not None:
            self.on_stage_progress(stage, progress)

    def run(
        self,
        video_path: Path,
        params: Optional[EditPipelineParams] = None,
        force: bool = False,
    ) -> EditPipelineResult:
        """Executa (ou retoma do cache) todas as etapas e devolve o resumo."""
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            raise VideoNotFoundError(video_path)
        params = params or EditPipelineParams()
        recorder = TimingRecorder()

        video_hash = file_hash(video_path)
        project_dir = project_dir_for(video_path, video_hash, self.projects_root).resolve()
        manifest_path = project_dir / MANIFEST_FILE
        if manifest_path.is_file():
            manifest = ProjectManifest.load(manifest_path)
            manifest.video_path = video_path
            manifest.params = params
        else:
            manifest = ProjectManifest(video_path=video_path, video_hash=video_hash, params=params)
        manifest.save(manifest_path)

        run = _Run(self, manifest, manifest_path, force, recorder)
        outcomes: dict[StageName, StageOutcome] = {}
        output_path = project_dir / output_video_name(video_path)

        # 1. probe
        fp_probe = fingerprint("probe", video_hash)

        def do_probe() -> tuple[dict, str]:
            metadata = VideoProbe().probe(video_path)
            (project_dir / PROBE_FILE).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
            v = metadata.video_stream
            width, height = v.width, v.height
            rotation = probe_rotation(video_path)
            if abs(rotation) % 180 == 90:
                width, height = height, width
            metrics = {
                "duration_sec": metadata.duration_sec,
                "width": width,
                "height": height,
                "fps": v.fps,
                "rotation": rotation,
            }
            return metrics, f"{metadata.duration_sec:.2f}s, {width}x{height} @ {v.fps}fps"

        outcomes["probe"] = run.stage("probe", fp_probe, PROBE_FILE, do_probe)
        metadata = VideoMetadata.model_validate_json((project_dir / PROBE_FILE).read_text(encoding="utf-8"))
        probe_metrics = manifest.stages["probe"].metrics
        recorder.video_duration_sec = metadata.duration_sec
        resolution = (int(probe_metrics["width"]), int(probe_metrics["height"]))
        source_fps = float(probe_metrics["fps"])

        # 2. silêncios
        # No modo auto o limite efetivo é função determinística do áudio; ele entra no fingerprint do
        # plano (abaixo) depois de medido. Aqui vale o modo + a versão da regra.
        threshold_key = (
            ("auto", AUTO_RULE_VERSION) if params.threshold_db == "auto" else ("fixed", params.threshold_db)
        )
        fp_silences = fingerprint("silences", video_hash, threshold_key, params.min_silence_sec)

        def do_silences() -> tuple[dict, str]:
            result = detect_silences(video_path, params.threshold_db, params.min_silence_sec)
            result.save_json(project_dir / AUDIO_ANALYSIS_FILE)
            total = sum(s.duration for s in result.silences)
            metrics: dict = {
                "silences": len(result.silences),
                "silence_total_sec": round(total, 3),
                "threshold_mode": result.params.threshold_mode,
                "threshold_db": result.params.threshold_db,
            }
            detail = f"{len(result.silences)} silêncios ({total:.2f}s)"
            if result.levels is not None:
                metrics["noise_floor_db"] = result.levels.noise_floor_db
                metrics["speech_level_db"] = result.levels.speech_level_db
                detail += f", limite {result.params.threshold_db} dB (auto)"
                if not result.levels.reliable:
                    detail += " — detecção pouco confiável"
            return metrics, detail

        outcomes["silences"] = run.stage("silences", fp_silences, AUDIO_ANALYSIS_FILE, do_silences)

        # 3. transcrição (opcional; só influencia os cortes através do Diretor)
        fp_transcript: Optional[str] = None
        if params.transcribe:
            provider = self.transcription_provider
            # Modelo, dispositivo e compute_type efetivos (resolvidos de "auto") entram no cache.
            effective = provider.effective_settings(params.transcription)
            effective_model = provider.effective_model(params.transcription)
            fp_transcript = fingerprint(
                "transcription", video_hash, provider.provider_name,
                params.transcription.model_copy(update={"model_size": effective_model, "device": effective[0], "compute_type": effective[1]}),
            )

            def do_transcription() -> tuple[dict, str]:
                transcript = provider.transcribe(
                    video_path,
                    params.transcription,
                    on_progress=lambda done, total: self._stage_progress(
                        "transcription", StageProgress(done=done, total=total, unit="sec")
                    ),
                )
                transcript.save(project_dir / TRANSCRIPT_FILE)
                metrics = {
                    "provider": provider.provider_name,
                    "language": transcript.language,
                    "segments": len(transcript.segments),
                }
                return metrics, f"{len(transcript.segments)} segmentos ({transcript.language})"

            outcomes["transcription"] = run.stage(
                "transcription", fp_transcript, TRANSCRIPT_FILE, do_transcription
            )
        else:
            outcomes["transcription"] = run.skip("transcription", "use --transcribe para gerar")

        # 4. Diretor (opcional; escolhe frases a remover pelos IDs)
        fp_director: Optional[str] = None
        director_run: Optional[DirectorRun] = None
        if params.director == "none":
            outcomes["director"] = run.skip("director", "use --director para ativar")
        elif fp_transcript is None:
            outcomes["director"] = run.skip("director", "requer --transcribe")
        else:
            director = self.director_for(params)
            fp_director = fingerprint(
                "director", fp_transcript, director.provider_name, director.model_id,
                director.thinking_label, params.brief, params.sentences, list(director.model_chain)
            )

            def do_director() -> tuple[dict, str]:
                model = director.model_chain[0] if director.model_chain else director.model_id
                self._stage_progress("director", StageProgress(note=model))
                transcript = Transcript.load(project_dir / TRANSCRIPT_FILE)
                sentences = build_sentences(transcript, params.sentences)
                context = DirectorContext(
                    video_name=video_path.name,
                    duration_sec=metadata.duration_sec,
                    language=transcript.language,
                )
                result = run_director(director, sentences, params.brief, context)
                (project_dir / DIRECTOR_DECISION_FILE).write_text(
                    result.model_dump_json(indent=2), encoding="utf-8"
                )
                removed = result.removed_sentences
                removed_sec = sum(s.duration for s, _ in removed)
                metrics = {
                    "provider": director.provider_name,
                    "model": director.model_id,
                    "effective_model": result.effective_model_id or director.model_id,
                    "model_switches": len(result.model_switches),
                    "sentences": len(sentences),
                    "removed": len(removed),
                    "removed_sec": round(removed_sec, 3),
                    "attempts": len(result.attempts),
                    "fallback": int(result.decision is None),
                }
                if result.decision is None:
                    return metrics, "resposta inválida; seguindo só com o corte de silêncio"
                summary = f"{len(removed)} de {len(sentences)} frases removidas ({removed_sec:.2f}s)"
                if result.model_switches:
                    summary += f" com {result.effective_provider}:{result.effective_model_id}"
                return metrics, summary

            outcomes["director"] = run.stage("director", fp_director, DIRECTOR_DECISION_FILE, do_director)
            director_run = DirectorRun.model_validate_json(
                (project_dir / DIRECTOR_DECISION_FILE).read_text(encoding="utf-8")
            )

        # 5. plano
        effective_threshold = AudioAnalysisResult.load_json(
            project_dir / AUDIO_ANALYSIS_FILE
        ).params.threshold_db
        fp_plan = fingerprint(
            "plan", fp_silences, effective_threshold, fp_director, params.cut, video_path, resolution, source_fps,
            params.loudness,
        )

        def do_plan() -> tuple[dict, str]:
            audio = AudioAnalysisResult.load_json(project_dir / AUDIO_ANALYSIS_FILE)
            removed_ids = None
            sentences = []
            if director_run is not None and director_run.decision is not None:
                removed_ids = {r.sentence_id for r in director_run.decision.removals}
                sentences = director_run.sentences
            keep, director_cuts = plan_keep_ranges(audio, sentences, removed_ids, params.cut)
            plan = build_edit_plan(
                video_path,
                keep,
                output_path,
                resolution,
                source_fps,
                title=f"Corte de silêncio — {video_path.name}",
                loudness=LoudnessTarget.from_preset(params.loudness),
            )
            (project_dir / EDIT_PLAN_FILE).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            kept = sum(r.duration for r in keep)
            metrics = {
                "segments": len(keep),
                "kept_sec": round(kept, 3),
                "director_cuts": len(director_cuts),
            }
            return metrics, f"{len(keep)} trechos mantidos ({kept:.2f}s)"

        outcomes["plan"] = run.stage("plan", fp_plan, EDIT_PLAN_FILE, do_plan)
        plan = EditPlan.model_validate_json((project_dir / EDIT_PLAN_FILE).read_text(encoding="utf-8"))

        # 6. render
        fp_render = fingerprint("render", fp_plan, params.loudness)

        def do_render() -> tuple[dict, str]:
            result = self.render_executor.execute(
                plan,
                on_progress=lambda done, total: self._stage_progress(
                    "render", StageProgress(done=done, total=total, unit="items")
                ),
            )
            metrics = {
                "duration_sec": round(result.duration_sec, 3),
                "size_bytes": result.size_bytes,
            }
            if result.loudness is not None:
                metrics.update(result.loudness.to_metrics())
            return metrics, f"{result.duration_sec:.2f}s, {result.size_bytes / 1_048_576:.1f} MB"

        outcomes["render"] = run.stage("render", fp_render, output_path.name, do_render)
        final_duration = float(manifest.stages["render"].metrics["duration_sec"])
        loudness = LoudnessResult.from_metrics(manifest.stages["render"].metrics)

        # 7. legendas (só com transcrição); tempos mapeados pelos mesmos trechos do plano final
        srt_name = subtitles_file_name(video_path)
        srt_path = project_dir / srt_name
        fp_subtitles: Optional[str] = None
        if not params.subtitles:
            outcomes["subtitles"] = run.skip("subtitles", "desligada por --no-subtitles", artifact=srt_name)
        elif fp_transcript is None:
            outcomes["subtitles"] = run.skip("subtitles", "requer --transcribe", artifact=srt_name)
        else:
            fp_subtitles = fingerprint("subtitles", fp_plan, fp_transcript, params.subtitle)

            def do_subtitles() -> tuple[dict, str]:
                transcript = Transcript.load(project_dir / TRANSCRIPT_FILE)
                kept_ranges = [
                    TimeRange(start_sec=s.start_sec, end_sec=s.end_sec) for s in plan.segments
                ]
                blocks = build_subtitle_blocks(transcript, kept_ranges, plan.target_fps, params.subtitle)
                write_srt(blocks, srt_path)
                return {"blocks": len(blocks)}, f"{srt_name}, {len(blocks)} blocos"

            outcomes["subtitles"] = run.stage("subtitles", fp_subtitles, srt_name, do_subtitles)
        subtitle_blocks: Optional[int] = None
        if fp_subtitles is not None:
            subtitle_blocks = int(manifest.stages["subtitles"].metrics["blocks"])

        # 8. relatório (+ transcrição comparativa, que só existe com transcrição)
        fp_report = fingerprint(
            "report", REPORT_FORMAT, fp_render, fp_director, fp_subtitles, fp_transcript, params
        )
        removed = removed_ranges(
            [TimeRange(start_sec=s.start_sec, end_sec=s.end_sec) for s in plan.segments],
            metadata.duration_sec,
        )

        def do_report() -> tuple[dict, str]:
            report_started = time.perf_counter()
            audio = AudioAnalysisResult.load_json(project_dir / AUDIO_ANALYSIS_FILE)
            comparison: Optional[EditDiff] = None
            if fp_transcript is not None:
                comparison = _edit_comparison(
                    video_path, Transcript.load(project_dir / TRANSCRIPT_FILE), director_run, plan,
                    audio, params, metadata.duration_sec, final_duration,
                )
                write_edit_diff(comparison, project_dir)
            else:
                remove_edit_diff(project_dir)
            has_decision = director_run is not None and director_run.decision is not None
            markdown = build_edit_report(
                video_path,
                metadata.duration_sec,
                final_duration,
                plan,
                removed,
                params,
                output_path,
                director=director_run,
                audio=audio,
                subtitles_path=srt_path if subtitle_blocks is not None else None,
                subtitle_blocks=subtitle_blocks,
                transcription_model=(
                    Transcript.load(project_dir / TRANSCRIPT_FILE).model_used if fp_transcript else None
                ),
                timings=recorder.snapshot(in_progress=("report", report_started)),
                comparison=comparison,
                removal_label="Diretor" if has_decision else None,
                loudness=loudness,
            )
            (project_dir / EDIT_REPORT_FILE).write_text(markdown, encoding="utf-8")
            metrics: dict = {"cuts": len(removed)}
            detail = f"{len(removed)} cortes"
            if comparison is not None:
                metrics.update(
                    applied=comparison.count("applied"),
                    partial=comparison.count("partial"),
                    not_applied=comparison.count("not_applied"),
                )
                detail += f", transcrição comparativa em {EDIT_TRANSCRIPT_FILE}"
            return metrics, detail

        outcomes["report"] = run.stage("report", fp_report, EDIT_REPORT_FILE, do_report)

        transcript_path = project_dir / EDIT_TRANSCRIPT_FILE
        timings = recorder.snapshot()
        manifest.last_run = timings
        manifest.save(manifest_path)
        version = register_edit_version(project_dir, manifest, manifest_path, timings)

        return EditPipelineResult(
            project_dir=project_dir,
            version=version.number if version is not None else None,
            manifest_path=manifest_path,
            output_path=output_path,
            report_path=project_dir / EDIT_REPORT_FILE,
            subtitles_path=srt_path if subtitle_blocks is not None else None,
            transcript_path=transcript_path if transcript_path.is_file() else None,
            original_duration_sec=metadata.duration_sec,
            final_duration_sec=final_duration,
            outcomes=outcomes,
            timings=timings,
            warnings=[
                *(director_run.warnings if director_run is not None else []),
                *(loudness.warnings if loudness is not None else []),
            ],
        )


def _edit_comparison(
    video_path: Path,
    transcript: Transcript,
    director_run: Optional[DirectorRun],
    plan: EditPlan,
    audio: AudioAnalysisResult,
    params: EditPipelineParams,
    original_duration_sec: float,
    final_duration_sec: float,
) -> EditDiff:
    """Transcrição comparativa do `edit`: frases do Diretor (ou da transcrição) contra o plano final."""
    removals = {}
    cut_results = []
    if director_run is not None:
        sentences = director_run.sentences
        if director_run.decision is not None:
            removals = {r.sentence_id: r for r in director_run.decision.removals}
            # Mesmo cálculo da etapa do plano: explica remoções não aplicadas ou parciais.
            _, cut_results = plan_keep_ranges(audio, sentences, set(removals), params.cut)
    else:
        sentences = build_sentences(transcript, params.sentences)
    return build_edit_diff(
        video_path.name,
        sentences,
        removals,
        [TimeRange(start_sec=s.start_sec, end_sec=s.end_sec) for s in plan.segments],
        cut_results,
        plan.target_fps,
        original_duration_sec,
        final_duration_sec,
        min_pause_sec=params.transcript_min_pause_sec,
    )


class _Run:
    """Estado de uma execução: aplica a regra de cache e persiste o manifesto a cada etapa."""

    def __init__(
        self,
        pipeline: EditPipeline,
        manifest: ProjectManifest,
        manifest_path: Path,
        force: bool,
        recorder: TimingRecorder,
    ):
        self.pipeline = pipeline
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.project_dir = manifest_path.parent
        self.force = force
        self.recorder = recorder

    def _save(self) -> None:
        self.manifest.updated_at = _now()
        self.manifest.save(self.manifest_path)

    def stage(self, name: StageName, fp: str, artifact: str, work: StageWork) -> StageOutcome:
        record = self.manifest.stages[name]
        if (
            not self.force
            and record.status == "done"
            and record.fingerprint == fp
            and (self.project_dir / artifact).is_file()
        ):
            self.recorder.record(name, "cached")
            self.pipeline._emit(name, "cached", "")
            return "cached"

        self.pipeline._emit(name, "start", "")
        started = time.perf_counter()
        try:
            metrics, detail = work()
        except Exception as exc:
            self.manifest.stages[name] = StageRecord(
                status="failed", artifact=artifact, updated_at=_now(), error=str(exc)
            )
            self._save()
            self.pipeline._emit(name, "failed", str(exc))
            raise

        elapsed = time.perf_counter() - started
        self.manifest.stages[name] = StageRecord(
            status="done",
            fingerprint=fp,
            artifact=artifact,
            updated_at=_now(),
            elapsed_sec=round(elapsed, 3),
            metrics=metrics,
        )
        self._save()
        self.recorder.record(name, "ran", elapsed)
        self.pipeline._emit(name, "done", f"{detail} ({format_duration(elapsed)})")
        return "ran"

    def skip(self, name: StageName, detail: str, artifact: Optional[str] = None) -> StageOutcome:
        """Etapa não pedida. Um artefato válido de execução anterior é preservado.

        Com `artifact`, o arquivo fica obsoleto quando a etapa não vale mais (ex.: legenda de
        um plano anterior ao lado do vídeo): é removido e a etapa volta a "skipped".
        """
        if artifact is not None:
            stale = self.project_dir / artifact
            if stale.is_file():
                stale.unlink()
            if self.manifest.stages[name].status != "skipped":
                self.manifest.stages[name] = StageRecord(status="skipped", updated_at=_now())
                self._save()
        elif self.manifest.stages[name].status != "done":
            self.manifest.stages[name] = StageRecord(status="skipped", updated_at=_now())
            self._save()
        self.recorder.record(name, "skipped")
        self.pipeline._emit(name, "skipped", detail)
        return "skipped"
