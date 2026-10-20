"""Comando `revise`: gera uma versão nova do projeto a partir de um pedido em linguagem natural.

Reaproveita probe, silêncios e transcrição já gravados na pasta (nunca refaz). O revisor
devolve só IDs de frases; plano, render, legendas e relatório são determinísticos.
"""

import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ai_video_editor.audio import AudioAnalysisResult
from ai_video_editor.director import DirectorDecision, DirectorRun, Sentence, build_sentences
from ai_video_editor.director.schema import SentenceRemoval
from ai_video_editor.pipeline.cutting import plan_keep_ranges
from ai_video_editor.pipeline.edit_diff import build_edit_diff, write_edit_diff
from ai_video_editor.pipeline.project import (
    AUDIO_ANALYSIS_FILE,
    DIRECTOR_DECISION_FILE,
    EDIT_DIFF_FILE,
    EDIT_PLAN_FILE,
    EDIT_REPORT_FILE,
    EDIT_TRANSCRIPT_FILE,
    MANIFEST_FILE,
    PROBE_FILE,
    TRANSCRIPT_FILE,
    subtitles_file_name,
)
from ai_video_editor.pipeline.report import application_cell, build_edit_report, build_revision_section
from ai_video_editor.pipeline.schema import (
    EditDiff,
    ProjectManifest,
    RemovalOrigin,
    RunTimings,
    VersionRecord,
)
from ai_video_editor.pipeline.timing import TimingRecorder, build_timings_section, format_duration
from ai_video_editor.pipeline.versions import (
    VERSION_FILE,
    ensure_versions,
    get_version,
    version_dir,
)
from ai_video_editor.planning import (
    NoSpeechDetectedError,
    apply_pause_preset,
    build_edit_plan,
    removed_ranges,
)
from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.revision import (
    EditReviser,
    ProjectStateError,
    RequestHistoryItem,
    RevisionContext,
    RevisionRejectedError,
    SentenceState,
    run_reviser,
)
from ai_video_editor.subtitles import build_subtitle_blocks, write_srt
from ai_video_editor.transcription.schema import Transcript

# Quantos pedidos anteriores o revisor recebe.
HISTORY_LIMIT = 5

# (etapa, detalhe) para o terminal.
RevisionProgress = Callable[[str, str], None]


class ChangedSentence(BaseModel):
    id: int
    text: str
    reason: str


class RevisionResult(BaseModel):
    """Resumo de uma revisão concluída."""

    project_dir: Path
    version: VersionRecord
    base_version: int
    video_path: Path
    report_path: Path
    subtitles_path: Optional[Path] = None
    transcript_path: Optional[Path] = Field(default=None, description="Transcrição comparativa da versão")
    removed: list[ChangedSentence] = Field(default_factory=list, description="Frases removidas nesta versão")
    restored: list[ChangedSentence] = Field(default_factory=list, description="Frases que voltaram ao vídeo")
    duration_before_sec: float
    duration_after_sec: float
    pacing_before: str
    pacing_after: str
    summary: str
    timings: Optional[RunTimings] = None


class ProjectReviser:
    """Orquestra uma revisão sobre uma pasta de projeto."""

    def __init__(
        self,
        reviser: EditReviser,
        render_executor: Optional[RenderExecutor] = None,
        on_progress: Optional[RevisionProgress] = None,
    ):
        self.reviser = reviser
        self._render_executor = render_executor
        self.on_progress = on_progress

    @property
    def render_executor(self) -> RenderExecutor:
        if self._render_executor is None:
            self._render_executor = RenderExecutor()
        return self._render_executor

    def _emit(self, step: str, detail: str = "", elapsed: Optional[float] = None) -> None:
        if self.on_progress is not None:
            if elapsed is not None:
                detail = f"{detail} ({format_duration(elapsed)})"
            self.on_progress(step, detail)

    def revise(
        self, project_dir: Path, request: str, from_version: Optional[int] = None
    ) -> RevisionResult:
        """Cria a próxima versão. Qualquer falha antes do fim deixa o projeto como estava.

        Raises:
            ProjectStateError: pasta sem projeto, sem edição concluída/transcrição, ou versão inexistente.
            RevisionRejectedError: pedido vazio, impossível, ambíguo, sem efeito ou respondido de
                forma inválida; nenhuma versão é criada.
        """
        recorder = TimingRecorder()
        request = request.strip()
        if not request:
            raise RevisionRejectedError("o pedido está vazio")
        project_dir = Path(project_dir).resolve()
        manifest_path = project_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            raise ProjectStateError(f"{project_dir} não é uma pasta de projeto (falta {MANIFEST_FILE})")
        manifest = ensure_versions(project_dir, ProjectManifest.load(manifest_path), manifest_path)

        base = get_version(manifest, from_version if from_version is not None else manifest.current_version)
        for name in (PROBE_FILE, AUDIO_ANALYSIS_FILE, TRANSCRIPT_FILE):
            if not (project_dir / name).is_file():
                hint = " (rode `edit --transcribe` antes)" if name == TRANSCRIPT_FILE else ""
                raise ProjectStateError(f"falta {name} no projeto{hint}; a revisão não refaz etapas")

        # Reaproveitados do cache: a revisão nunca refaz estas etapas.
        for name in ("probe", "silences", "transcription"):
            recorder.record(name, "cached")
        recorder.video_duration_sec = float(manifest.stages["probe"].metrics["duration_sec"])

        transcript = Transcript.load(project_dir / TRANSCRIPT_FILE)
        audio = AudioAnalysisResult.load_json(project_dir / AUDIO_ANALYSIS_FILE)
        sentences = self._base_sentences(project_dir, base)
        if sentences is None:
            sentences = build_sentences(transcript, manifest.params.sentences)
        if not sentences:
            raise ProjectStateError("a transcrição não tem frases para revisar")
        by_id = {s.id: s for s in sentences}

        base_removals = self._base_removals(project_dir, base, set(by_id))
        states = [
            SentenceState(
                sentence=s,
                removed=s.id in base_removals,
                category=base_removals[s.id].category if s.id in base_removals else None,
                reason=base_removals[s.id].reason if s.id in base_removals else None,
            )
            for s in sentences
        ]
        brief = manifest.params.brief
        context = RevisionContext(
            video_name=manifest.video_path.name,
            duration_sec=float(manifest.stages["probe"].metrics["duration_sec"]),
            language=transcript.language,
            base_version=base.number,
            base_pacing=base.pacing,
            request=request,
            history=[
                RequestHistoryItem(version=v.number, request=v.request)
                for v in manifest.versions[-HISTORY_LIMIT:]
            ],
        )

        self._emit("revisor", f"v{base.number} + pedido")
        reviser_started = time.perf_counter()
        run = run_reviser(self.reviser, states, brief, context)
        reviser_elapsed = time.perf_counter() - reviser_started
        recorder.record("reviser", "ran", reviser_elapsed)
        self._emit("revisor", "pedido analisado", reviser_elapsed)
        decision = run.decision
        assert decision is not None

        base_ids = set(base_removals)
        new_removals = {r.sentence_id: r for r in decision.removals}
        new_ids = set(new_removals)
        pacing = decision.pacing or base.pacing
        if new_ids == base_ids and pacing == base.pacing:
            raise RevisionRejectedError(
                "o pedido não muda nada na edição"
                + (f": {decision.summary.strip()}" if decision.summary.strip() else "")
            )

        cut = apply_pause_preset(base.cut, pacing)
        try:
            keep, cut_results = plan_keep_ranges(audio, sentences, new_ids, cut)
        except NoSpeechDetectedError as exc:
            raise RevisionRejectedError(f"o pedido removeria todo o conteúdo ({exc})") from exc

        number = max(v.number for v in manifest.versions) + 1
        target = version_dir(project_dir, number)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        try:
            record = self._build_version(
                project_dir, manifest, base, number, target, request, decision.summary, pacing,
                cut, keep, cut_results, sentences, transcript, audio, new_removals, run, decision,
                base_removals, recorder,
            )
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

        manifest.versions.append(record)
        manifest.current_version = number
        manifest.last_run = record.timings
        manifest.save(manifest_path)

        restored_reasons = {r.sentence_id: r.reason for r in decision.restorations}
        return RevisionResult(
            project_dir=project_dir,
            version=record,
            base_version=base.number,
            video_path=target / record.video,
            report_path=target / EDIT_REPORT_FILE,
            subtitles_path=target / record.subtitles if record.subtitles else None,
            transcript_path=target / EDIT_TRANSCRIPT_FILE,
            removed=[
                ChangedSentence(id=i, text=by_id[i].text, reason=new_removals[i].reason)
                for i in sorted(new_ids - base_ids, key=lambda i: by_id[i].start_sec)
            ],
            restored=[
                ChangedSentence(id=i, text=by_id[i].text, reason=restored_reasons.get(i, ""))
                for i in sorted(base_ids - new_ids, key=lambda i: by_id[i].start_sec)
            ],
            duration_before_sec=base.duration_sec,
            duration_after_sec=record.duration_sec,
            pacing_before=base.pacing,
            pacing_after=pacing,
            summary=decision.summary,
            timings=record.timings,
        )

    @staticmethod
    def _base_sentences(project_dir: Path, base: VersionRecord) -> Optional[list[Sentence]]:
        """Frases gravadas na versão base: os IDs das remoções valem para elas, não para uma remontagem.

        None se a versão não tem Diretor (então as frases são montadas da transcrição).
        """
        path = version_dir(project_dir, base.number) / DIRECTOR_DECISION_FILE
        if not path.is_file():
            return None
        return DirectorRun.model_validate_json(path.read_text(encoding="utf-8")).sentences or None

    @staticmethod
    def _base_removals(
        project_dir: Path, base: VersionRecord, valid_ids: set[int]
    ) -> dict[int, SentenceRemoval]:
        """Remoções da versão base (vazio se a versão não teve Diretor ou ele caiu no fallback)."""
        path = version_dir(project_dir, base.number) / DIRECTOR_DECISION_FILE
        if not path.is_file():
            return {}
        run = DirectorRun.model_validate_json(path.read_text(encoding="utf-8"))
        if run.decision is None:
            return {}
        return {r.sentence_id: r for r in run.decision.removals if r.sentence_id in valid_ids}

    @staticmethod
    def _base_origins(project_dir: Path, base: VersionRecord) -> dict[int, RemovalOrigin]:
        """Quem removeu cada frase na versão base (sem diff gravado: Diretor no `edit`, senão revisão)."""
        path = version_dir(project_dir, base.number) / EDIT_DIFF_FILE
        if not path.is_file():
            return {}
        diff = EditDiff.model_validate_json(path.read_text(encoding="utf-8"))
        return {e.sentence_id: e.origin for e in diff.sentences if e.removal_requested and e.origin}

    def _build_version(
        self, project_dir, manifest, base, number, target, request, summary, pacing,
        cut, keep, cut_results, sentences, transcript, audio, new_removals, run, decision,
        base_removals, recorder,
    ):
        probe = manifest.stages["probe"].metrics
        output_path = target / base.video
        plan_started = time.perf_counter()
        plan = build_edit_plan(
            manifest.video_path,
            keep,
            output_path,
            (int(probe["width"]), int(probe["height"])),
            float(probe["fps"]),
            title=f"Revisão v{number} — {manifest.video_path.name}",
        )
        (target / EDIT_PLAN_FILE).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        plan_elapsed = time.perf_counter() - plan_started
        recorder.record("plan", "ran", plan_elapsed)
        self._emit("plano", f"{len(keep)} trechos mantidos", plan_elapsed)

        render_started = time.perf_counter()
        rendered = self.render_executor.execute(plan)
        render_elapsed = time.perf_counter() - render_started
        recorder.record("render", "ran", render_elapsed)
        self._emit("render", f"{rendered.duration_sec:.2f}s", render_elapsed)

        params = manifest.params
        subtitles_name: Optional[str] = None
        blocks = 0
        if params.subtitles:
            subtitles_started = time.perf_counter()
            subtitles_name = subtitles_file_name(manifest.video_path)
            kept_ranges = [TimeRange(start_sec=s.start_sec, end_sec=s.end_sec) for s in plan.segments]
            built = build_subtitle_blocks(transcript, kept_ranges, plan.target_fps, params.subtitle)
            write_srt(built, target / subtitles_name)
            blocks = len(built)
            subtitles_elapsed = time.perf_counter() - subtitles_started
            recorder.record("subtitles", "ran", subtitles_elapsed)
            self._emit("legendas", f"{blocks} blocos", subtitles_elapsed)
        else:
            recorder.record("subtitles", "skipped")

        director_run = DirectorRun(
            provider=run.provider,
            model_id=run.model_id,
            thinking=run.thinking,
            brief=params.brief,
            sentences=sentences,
            decision=DirectorDecision(removals=list(new_removals.values()), summary=decision.summary),
            attempts=run.attempts,
            model_chain=run.model_chain,
            effective_provider=run.effective_provider,
            effective_model_id=run.effective_model_id,
            model_switches=run.model_switches,
        )
        (target / DIRECTOR_DECISION_FILE).write_text(director_run.model_dump_json(indent=2), encoding="utf-8")

        report_started = time.perf_counter()
        duration = float(manifest.stages["probe"].metrics["duration_sec"])
        removed = removed_ranges(keep, duration)
        by_id = {s.id: s for s in sentences}
        base_ids, new_ids = set(base_removals), set(new_removals)
        restored_reasons = {r.sentence_id: r.reason for r in decision.restorations}

        base_origins = self._base_origins(project_dir, base)
        default_origin: RemovalOrigin = "director" if base.kind == "edit" else "revision"
        origins: dict[int, RemovalOrigin] = {
            i: "revision" if i not in base_ids else base_origins.get(i, default_origin) for i in new_ids
        }
        comparison = build_edit_diff(
            manifest.video_path.name,
            sentences,
            new_removals,
            keep,
            cut_results,
            plan.target_fps,
            duration,
            rendered.duration_sec,
            min_pause_sec=params.transcript_min_pause_sec,
            origins=origins,
            version=number,
            base_version=base.number,
            request=request,
            restored={i: restored_reasons.get(i, "") for i in base_ids - new_ids},
            removed_now=new_ids - base_ids,
        )
        write_edit_diff(comparison, target)

        changes = [
            (
                i,
                by_id[i].text,
                "removida" if i in base_ids else "mantida",
                "removida" if i in new_ids else "mantida",
                application_cell(comparison, i),
                new_removals[i].reason if i in new_ids else restored_reasons.get(i, ""),
            )
            for i in sorted(base_ids ^ new_ids, key=lambda i: by_id[i].start_sec)
        ]
        report = build_edit_report(
            manifest.video_path,
            duration,
            rendered.duration_sec,
            plan,
            removed,
            params.model_copy(update={"cut": cut}),
            output_path,
            audio=audio,
            subtitles_path=target / subtitles_name if subtitles_name else None,
            subtitle_blocks=blocks if subtitles_name else None,
            comparison=comparison,
            removal_label=f"Revisão v{number}",
        ) + build_revision_section(
            request, base.number, number, summary, base.pacing, pacing, changes, reviser=run
        )
        timings = recorder.snapshot(in_progress=("report", report_started))
        report += build_timings_section(timings)
        (target / EDIT_REPORT_FILE).write_text(report, encoding="utf-8")
        self._emit("relatório", f"{len(removed)} cortes", time.perf_counter() - report_started)

        record = VersionRecord(
            number=number,
            kind="revision",
            request=request,
            parent=base.number,
            summary=summary,
            cut=base.cut,
            pacing=pacing,
            duration_sec=round(rendered.duration_sec, 3),
            cuts=len(removed),
            segments=len(plan.segments),
            removed_sentences=len(new_removals),
            video=base.video,
            subtitles=subtitles_name,
            timings=timings,
        )
        (target / VERSION_FILE).write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record
