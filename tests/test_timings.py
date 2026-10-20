"""Tempos de processamento: formatação, terminal, project.json, version.json e relatório."""

import json

import pytest

from ai_video_editor.cli import build_parser, handle_edit, handle_revise
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectManifest, ProjectReviser
from ai_video_editor.pipeline.schema import RunTimings, StageTiming
from ai_video_editor.pipeline.timing import (
    TimingRecorder,
    build_timings_section,
    format_duration,
    format_ratio,
    summarize_run,
)
from ai_video_editor.revision.providers.mock import MockReviser
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from tests.test_edit_pipeline import short_video  # noqa: F401  (fixture)
from tests.test_revision import _edited_template, edited, lesson_video  # noqa: F401  (fixtures)


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0,0s"),
        (8.43, "8,4s"),
        (59.94, "59,9s"),
        (59.96, "1min"),
        (60, "1min"),
        (65, "1min 05s"),
        (754, "12min 34s"),
        (720, "12min"),
        (3419.6, "57min"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_summary_line_matches_the_expected_wording():
    timings = RunTimings(total_sec=1260, video_duration_sec=3420)
    assert format_ratio(0.37) == "0,37×"
    assert summarize_run(timings) == "57min de vídeo processados em 21min (0,37× o tempo real)"
    assert summarize_run(RunTimings(total_sec=3)) is None


def test_timings_section_lists_status_and_time_per_stage():
    timings = RunTimings(
        total_sec=15.2,
        video_duration_sec=60,
        stages=[
            StageTiming(name="probe", status="cached"),
            StageTiming(name="transcription", status="ran", elapsed_sec=12.34),
            StageTiming(name="director", status="skipped"),
        ],
    )
    text = build_timings_section(timings)
    assert "## Tempo de processamento" in text
    assert "| Metadados | cache | — |" in text
    assert "| Transcrição | executada | 12,3s |" in text
    assert "| Diretor | pulada | — |" in text
    assert "| **Total** | | **15,2s** |" in text


def test_recorder_counts_the_stage_in_progress():
    recorder = TimingRecorder()
    recorder.record("probe", "cached")
    snap = recorder.snapshot(in_progress=("report", 0.0))
    assert [s.name for s in snap.stages] == ["probe", "report"]
    assert snap.stages[0].elapsed_sec is None and snap.stages[1].elapsed_sec is not None


@pytest.mark.slow
def test_cli_edit_shows_times_and_persists_them(tmp_path, short_video, capsys):  # noqa: F811
    args = build_parser().parse_args(["edit", str(short_video), "--projects-dir", str(tmp_path / "projects")])
    assert handle_edit(args) == 0
    out = capsys.readouterr().out
    assert "[2/8] Silêncios: ok — " in out
    assert "[6/8] Renderização: ok — " in out and "s)" in out
    assert "de vídeo processados em" in out and "o tempo real)" in out

    project = next((tmp_path / "projects").glob("*/project.json"))
    manifest = ProjectManifest.load(project)
    run = manifest.last_run
    assert run is not None and run.total_sec > 0 and run.video_duration_sec == pytest.approx(12.5, abs=0.5)
    by_name = {s.name: s for s in run.stages}
    assert by_name["render"].status == "ran" and by_name["render"].elapsed_sec > 0
    assert by_name["transcription"].status == "skipped"
    assert run.total_sec >= by_name["render"].elapsed_sec
    assert "last_run" in json.loads(project.read_text(encoding="utf-8"))

    version = json.loads((project.parent / "versions" / "v1" / "version.json").read_text(encoding="utf-8"))
    assert version["timings"]["total_sec"] == run.total_sec

    report = (project.parent / "edit_report.md").read_text(encoding="utf-8")
    assert "## Tempo de processamento" in report
    assert "| Renderização | executada |" in report and "| **Total** |" in report

    # Segunda execução: tudo vem do cache, sem tempo de processamento por etapa.
    assert handle_edit(args) == 0
    out = capsys.readouterr().out
    assert "[6/8] Renderização: em cache" in out
    rerun = ProjectManifest.load(project).last_run
    assert {s.name: s.status for s in rerun.stages}["render"] == "cached"
    assert all(s.elapsed_sec is None for s in rerun.stages if s.status == "cached")


@pytest.mark.slow
def test_pipeline_result_marks_cached_stages(tmp_path, short_video):  # noqa: F811
    pipeline = EditPipeline(projects_root=tmp_path / "p", transcription_provider=MockTranscriptionProvider())
    pipeline.run(short_video, EditPipelineParams())
    result = pipeline.run(short_video, EditPipelineParams(threshold_db=-30.0))
    status = {s.name: s.status for s in result.timings.stages}
    assert status["probe"] == "cached" and status["silences"] == "ran" and status["render"] == "ran"
    report = result.report_path.read_text(encoding="utf-8")
    assert "| Metadados | cache | — |" in report


@pytest.mark.slow
def test_revise_records_times_in_terminal_version_json_project_and_report(edited, capsys):  # noqa: F811
    args = build_parser().parse_args(
        ["revise", str(edited.project), "restaura tudo", "--reviser", "mock", "--from", "v1"]
    )
    assert handle_revise(args) == 0
    out = capsys.readouterr().out
    assert "[revisão] render: " in out and "s)" in out
    assert "de vídeo processados em" in out

    manifest = ProjectManifest.load(edited.project / "project.json")
    v2 = next(v for v in manifest.versions if v.number == 2)
    assert v2.timings is not None and manifest.last_run == v2.timings
    status = {s.name: s.status for s in v2.timings.stages}
    assert status["probe"] == status["silences"] == status["transcription"] == "cached"
    assert status["reviser"] == status["plan"] == status["render"] == "ran"
    assert v2.timings.total_sec >= sum(s.elapsed_sec or 0 for s in v2.timings.stages) - 0.01

    version = json.loads((edited.project / "versions" / "v2" / "version.json").read_text(encoding="utf-8"))
    assert version["timings"]["stages"]
    report = (edited.project / "versions" / "v2" / "edit_report.md").read_text(encoding="utf-8")
    assert "## Tempo de processamento" in report and "| Transcrição | cache | — |" in report
    # v1 mantém os tempos do edit original
    assert manifest.versions[0].timings is not None
