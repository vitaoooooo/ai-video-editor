"""Linha de andamento ao vivo do `edit`/`revise` e callbacks de progresso (relógio e stdout falsos)."""

import io
import types

import pytest

from ai_video_editor.cli import EditProgressPrinter, RevisionProgressPrinter, build_parser
from ai_video_editor.intelligence.exceptions import AIUnavailableError
from ai_video_editor.intelligence.retry import RetryEvent
from ai_video_editor.live_status import LiveStatus, describe_progress, estimate_remaining
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectReviser, StageProgress
from ai_video_editor.revision.providers.mock import MockReviser
from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from ai_video_editor.transcription.schema import TranscriptionConfig
from tests.test_edit_pipeline import short_video  # noqa: F401  (fixture)
from tests.test_revision import _edited_template, edited, lesson_video  # noqa: F401  (fixtures)


class FakeTerminal(io.StringIO):
    def __init__(self, tty: bool = True):
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def screen(output: str) -> list[str]:
    """O que fica visível num console: `\\r` volta ao início da linha e sobrescreve."""
    lines = [""]
    col = 0
    for ch in output:
        if ch == "\r":
            col = 0
        elif ch == "\n":
            lines.append("")
            col = 0
        else:
            line = lines[-1]
            lines[-1] = line[:col] + ch + line[col + 1:]
            col += 1
    return [line.rstrip() for line in lines if line.strip()]


def live_status(tty: bool = True, **kwargs):
    term, clock = FakeTerminal(tty), FakeClock()
    status = LiveStatus(stream=term, clock=clock, width=200, auto_refresh=False, **kwargs)
    return status, term, clock


def test_live_line_is_rewritten_in_place_with_elapsed_time():
    status, term, clock = live_status()
    status.begin("[3/8] Transcrição")
    clock.now += 1
    status.refresh()
    clock.now += 132
    status.refresh()
    assert screen(term.getvalue()) == ["[3/8] Transcrição... 2min 13s"]
    assert term.getvalue().count("\n") == 0


def test_final_line_replaces_the_live_line():
    status, term, clock = live_status()
    status.begin("[6/8] Renderização")
    status.update(done=3, total=258, unit="items")
    clock.now += 75
    status.refresh()
    status.end("[6/8] Renderização: ok — 12,0s (1min 15s)")
    assert screen(term.getvalue()) == ["[6/8] Renderização: ok — 12,0s (1min 15s)"]
    assert term.getvalue().endswith("[6/8] Renderização: ok — 12,0s (1min 15s)\n")


def test_shorter_line_erases_the_rest_of_the_previous_one():
    status, term, clock = live_status()
    status.begin("[4/8] Diretor")
    status.update(note="gemini:gemini-3.8-flash, tentativa 2/5")
    status.refresh()
    status.update(note="x")
    status.refresh()
    assert screen(term.getvalue()) == ["[4/8] Diretor... 0,0s — x"]


def test_line_is_truncated_to_the_terminal_width():
    term = FakeTerminal()
    status = LiveStatus(stream=term, clock=FakeClock(), width=20, auto_refresh=False)
    status.begin("[3/8] Transcrição com nome comprido")
    assert all(len(line) <= 19 for line in term.getvalue().split("\r"))


def test_not_a_terminal_prints_only_whole_lines():
    status, term, clock = live_status(tty=False)
    assert status.live is False
    status.begin("[3/8] Transcrição")
    status.update(done=50, total=100)
    clock.now += 30
    status.refresh()
    status.message("[3/8] Transcrição: GPU indisponível; transcrevendo na CPU.")
    status.end("[3/8] Transcrição: ok — 12 segmentos (30,0s)")
    assert "\r" not in term.getvalue()
    assert term.getvalue().splitlines() == [
        "[3/8] Transcrição...",
        "[3/8] Transcrição: GPU indisponível; transcrevendo na CPU.",
        "[3/8] Transcrição: ok — 12 segmentos (30,0s)",
    ]


def test_no_live_option_disables_the_live_line_on_a_terminal():
    parser = build_parser()
    assert parser.parse_args(["edit", "v.mp4", "--no-live"]).no_live is True
    assert parser.parse_args(["revise", "p", "x", "--no-live"]).no_live is True
    assert parser.parse_args(["edit", "v.mp4"]).no_live is False
    assert LiveStatus(stream=FakeTerminal(), live=False).live is False
    assert LiveStatus(stream=FakeTerminal()).live is True


def test_warning_in_the_middle_of_a_stage_stays_above_the_live_line():
    status, term, clock = live_status()
    status.begin("[4/8] Diretor")
    clock.now += 5
    status.refresh()
    status.message("[4/8] Diretor: Gemini indisponível (503), nova tentativa em 4s (2/5)")
    clock.now += 5
    status.refresh()
    assert screen(term.getvalue()) == [
        "[4/8] Diretor: Gemini indisponível (503), nova tentativa em 4s (2/5)",
        "[4/8] Diretor... 10,0s",
    ]
    status.end("[4/8] Diretor: ok — 3 de 10 frases removidas (12,0s)")
    assert screen(term.getvalue())[0].startswith("[4/8] Diretor: Gemini indisponível")
    assert screen(term.getvalue())[1] == "[4/8] Diretor: ok — 3 de 10 frases removidas (12,0s)"


def test_progress_text_for_transcription_and_render():
    assert describe_progress(StageProgress(done=23 * 60 + 10, total=57 * 60, unit="sec")) == "40% (23min de 57min)"
    assert describe_progress(StageProgress(done=118, total=258, unit="items")) == "118/258 trechos (45%)"
    assert describe_progress(StageProgress(done=1, total=0)) == ""
    assert describe_progress(StageProgress()) == ""


def test_remaining_time_only_after_ten_percent():
    assert estimate_remaining(60, None) is None
    assert estimate_remaining(60, 0.05) is None
    assert estimate_remaining(60, 0.20) == pytest.approx(240)

    status, term, clock = live_status()
    status.begin("[3/8] Transcrição")
    clock.now += 60
    status.update(done=5, total=100)
    assert "faltam" not in status.render()
    status.update(done=20, total=100)
    assert status.render() == "[3/8] Transcrição... 1min — 20% (20,0s de 1min), faltam ~4min"


def test_note_survives_progress_updates_and_resets_on_next_stage():
    status, _, _ = live_status()
    status.begin("[revisão] revisor")
    status.update(note="mock:mock")
    assert status.render() == "[revisão] revisor... 0,0s — mock:mock"
    status.apply(StageProgress(done=1, total=4, unit="items"))
    assert status.render() == "[revisão] revisor... 0,0s — 1/4 trechos (25%), faltam ~1s, mock:mock"
    status.end("[revisão] revisor: pedido analisado (1,0s)")
    status.begin("[revisão] render")
    assert status.render() == "[revisão] render... 0,0s"


def test_close_on_ctrl_c_leaves_no_dangling_line_and_stops_the_thread():
    term, clock = FakeTerminal(), FakeClock()
    status = LiveStatus(stream=term, clock=clock, width=200, interval_sec=0.01)
    with pytest.raises(KeyboardInterrupt):
        with status:
            status.begin("[6/8] Renderização")
            thread = status._thread
            assert thread is not None and thread.is_alive()
            clock.now += 7
            raise KeyboardInterrupt
    assert not thread.is_alive() and status._thread is None
    assert term.getvalue().endswith("\n")
    assert screen(term.getvalue()) == ["[6/8] Renderização: interrompida (7,0s)"]


def test_close_on_error_marks_the_open_stage_as_failed():
    status, term, _ = live_status()
    with pytest.raises(RuntimeError):
        with status:
            status.begin("[revisão] render")
            raise RuntimeError("ffmpeg quebrou")
    assert screen(term.getvalue()) == ["[revisão] render: FALHOU (0,0s)"]
    assert not status.active


def test_edit_printer_drives_the_live_line():
    status, term, clock = live_status()
    printer = EditProgressPrinter(status)
    printer("probe", "cached", "")
    printer("transcription", "start", "")
    printer.progress("transcription", StageProgress(done=30, total=120, unit="sec"))
    printer("transcription", "notice", "GPU indisponível (sem cuDNN); transcrevendo na CPU.")
    clock.now += 10
    status.refresh()
    assert screen(term.getvalue()) == [
        "[1/8] Metadados: em cache",
        "[3/8] Transcrição: GPU indisponível (sem cuDNN); transcrevendo na CPU.",
        "[3/8] Transcrição... 10,0s — 25% (30,0s de 2min), faltam ~30s",
    ]
    printer("transcription", "done", "12 segmentos (pt) (10,0s)")
    assert screen(term.getvalue())[-1] == "[3/8] Transcrição: ok — 12 segmentos (pt) (10,0s)"


def test_revision_printer_shows_model_retries_and_steps():
    status, term, clock = live_status()
    printer = RevisionProgressPrinter(status)
    printer.model = "gemini:gemini-3.8-flash"
    printer("revisor", "v1 + pedido")
    printer.start("revisor")
    chain = types.SimpleNamespace(current=types.SimpleNamespace(label="gemini:gemini-3.8-flash"), switches=[])
    event = RetryEvent(AIUnavailableError("gemini"), 4.0, 2, 5)
    printer.retry(chain, event)
    clock.now += 3
    status.refresh()
    assert screen(term.getvalue()) == [
        "[revisão] revisor: v1 + pedido",
        f"[revisão] revisor: {event.message}",
        "[revisão] revisor... 3,0s — gemini:gemini-3.8-flash, tentativa 2/5",
    ]
    printer("revisor", "pedido analisado (3,0s)")
    assert screen(term.getvalue())[-1] == "[revisão] revisor: pedido analisado (3,0s)"


def test_director_retry_and_switch_update_model_and_attempt():
    events, progress = [], []
    pipeline = EditPipeline(
        on_progress=lambda stage, event, detail: events.append((stage, event, detail)),
        on_stage_progress=lambda stage, p: progress.append((stage, p.note)),
    )
    director = pipeline.director_for(EditPipelineParams(director="gemini", director_model="gemini-3.8-flash"))
    chain = director.chain
    event = RetryEvent(AIUnavailableError("gemini"), 4.0, 2, 5)
    chain.retry.on_retry(event)
    assert events == [("director", "retry", event.message)]
    assert progress == [("director", "gemini:gemini-3.8-flash, tentativa 2/5")]

    chain.switches.append(types.SimpleNamespace(to_model="gemini:gemini-3.6-flash"))
    chain.on_event("gemini-3.8-flash indisponível → trocando para gemini-3.6-flash")
    assert events[-1][1] == "notice"
    assert progress[-1] == ("director", "gemini:gemini-3.6-flash, tentativa 1")


def test_mock_transcriber_reports_progress(tmp_path):
    calls = []
    MockTranscriptionProvider(sentences=["um dois", "três"]).transcribe(
        tmp_path / "x.wav", TranscriptionConfig(), on_progress=lambda done, total: calls.append((done, total))
    )
    # Total = duração do áudio simulado (inclui a pausa final).
    assert calls == [(pytest.approx(0.8), pytest.approx(2.2)), (pytest.approx(1.7), pytest.approx(2.2))]


def test_faster_whisper_reports_end_of_each_segment():
    raw = [
        types.SimpleNamespace(start=0.0, end=12.0, text="a", words=[]),
        types.SimpleNamespace(start=12.0, end=61.0, text="b", words=[]),
    ]
    calls = []
    segments = FasterWhisperProvider._convert_segments(iter(raw), (lambda d, t: calls.append((d, t)), 60.0))
    assert len(segments) == 2
    assert calls == [(12.0, 60.0), (60.0, 60.0)]


@pytest.mark.slow
def test_pipeline_forwards_transcription_and_render_progress(tmp_path, short_video):  # noqa: F811
    progress: list[tuple[str, StageProgress]] = []
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(),
        on_stage_progress=lambda stage, p: progress.append((stage, p)),
    )
    pipeline.run(short_video, EditPipelineParams(transcribe=True))

    transcription = [p for stage, p in progress if stage == "transcription"]
    assert transcription and all(p.unit == "sec" for p in transcription)
    fractions = [p.fraction for p in transcription]
    assert fractions == sorted(fractions) and 0 < fractions[-1] <= 1
    render = [p for stage, p in progress if stage == "render"]
    assert render[0].done == 0 and render[-1].done == render[-1].total > 0
    assert all(p.unit == "items" for p in render)


@pytest.mark.slow
def test_revision_announces_steps_and_render_progress(edited):  # noqa: F811
    starts, progress = [], []
    ProjectReviser(
        MockReviser(),
        on_step_start=starts.append,
        on_stage_progress=lambda step, p: progress.append((step, p)),
    ).revise(edited.project, "restaura tudo", 1)
    assert starts == ["revisor", "plano", "render", "legendas", "relatório"]
    assert progress and all(step == "render" for step, _ in progress)
    assert progress[-1][1].done == progress[-1][1].total > 0
