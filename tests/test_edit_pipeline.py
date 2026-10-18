"""Testes de ponta a ponta do pipeline `edit` com vídeos sintéticos e transcrição Mock."""

import pytest

from ai_video_editor.analyzer.probe import VideoProbe
from ai_video_editor.cli import build_parser, handle_edit
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectManifest
from ai_video_editor.pipeline import runner as runner_module
from ai_video_editor.planning import NoSpeechDetectedError, SilenceCutParams
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from ai_video_editor.transcription.schema import Transcript
from tests.conftest import create_tone_silence_video


class CountingExecutor(RenderExecutor):
    """RenderExecutor real que conta quantas vezes renderizou."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def execute(self, plan):
        self.calls += 1
        return super().execute(plan)


@pytest.fixture
def count_silence_calls(monkeypatch):
    calls = {"n": 0}
    original = runner_module.detect_silences

    def spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "detect_silences", spy)
    return calls


@pytest.fixture(scope="module")
def talking_video(tmp_path_factory):
    """60s: tom de 4s alternado com silêncio de 2s."""
    path = tmp_path_factory.mktemp("videos") / "Aula Teste.mp4"
    return create_tone_silence_video(path, [("tone", 4.0), ("silence", 2.0)] * 10, width=320, height=240)


@pytest.fixture(scope="module")
def short_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "curto.mp4"
    return create_tone_silence_video(path, [("tone", 2.0), ("silence", 1.5)] * 3 + [("tone", 2.0)])


def test_end_to_end_silence_cut(tmp_path, talking_video):
    events = []
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(),
        on_progress=lambda stage, event, detail: events.append((stage, event)),
    )
    result = pipeline.run(talking_video, EditPipelineParams(transcribe=True))

    # 10 falas de 4s; padding 0.15 antes (exceto a 1ª, em 0s) e 0.20 depois; silêncio final cortado
    expected = 4.2 + 9 * 4.35
    assert result.final_duration_sec == pytest.approx(expected, abs=0.3)
    probed = VideoProbe().probe(result.output_path)
    assert probed.duration_sec == pytest.approx(expected, abs=0.3)
    assert (probed.video_stream.width, probed.video_stream.height) == (320, 240)
    assert probed.video_stream.fps == pytest.approx(30.0)

    project = result.project_dir
    assert project.parent == (tmp_path / "projects").resolve()
    assert project.name.startswith("aula-teste-")
    for name in ("project.json", "audio_analysis.json", "transcript.json", "edit_plan.json", "edit_report.md"):
        assert (project / name).is_file(), name
    assert Transcript.load(project / "transcript.json").model_used == "mock"

    manifest = ProjectManifest.load(project / "project.json")
    assert manifest.video_path == talking_video.resolve()
    assert manifest.params.threshold_db == -35.0
    assert manifest.params.cut == SilenceCutParams()
    assert manifest.params.transcribe is True
    assert all(record.status == "done" for record in manifest.stages.values())
    assert all(record.fingerprint for record in manifest.stages.values())
    assert manifest.stages["plan"].metrics["segments"] == 10
    assert manifest.stages["render"].artifact == result.output_path.name
    assert manifest.stages["render"].metrics["duration_sec"] == pytest.approx(expected, abs=0.3)

    report = (project / "edit_report.md").read_text(encoding="utf-8")
    assert "Duração original | 01:00.0" in report
    assert f"({result.final_duration_sec:.2f}s)" in report
    removed_pct = (60.0 - result.final_duration_sec) / 60.0 * 100
    assert f"({removed_pct:.1f}%)" in report
    assert "Número de cortes | 10 |" in report
    assert "| padding_before_sec | 0.15s |" in report
    assert "| threshold_db | -35.0 dB |" in report
    # Primeiro corte: 4.20s → 5.85s
    assert "| 1 | 00:04.2" in report and "| 00:05.8" in report
    assert report.count("\n| ") >= 10 + 7 + 10  # resumo + parâmetros + linhas de cortes

    assert ("render", "done") in events and ("report", "done") in events


def test_cache_skips_unchanged_stages_and_threshold_invalidates(
    tmp_path, short_video, count_silence_calls
):
    executor = CountingExecutor()
    pipeline = EditPipeline(projects_root=tmp_path / "projects", render_executor=executor)

    first = pipeline.run(short_video, EditPipelineParams())
    assert first.outcomes == {
        "probe": "ran",
        "silences": "ran",
        "transcription": "skipped",
        "plan": "ran",
        "render": "ran",
        "report": "ran",
    }
    assert count_silence_calls["n"] == 1 and executor.calls == 1
    mtime = first.output_path.stat().st_mtime_ns

    second = pipeline.run(short_video, EditPipelineParams())
    assert second.outcomes["silences"] == "cached"
    assert second.outcomes["plan"] == "cached"
    assert second.outcomes["render"] == "cached"
    assert second.outcomes["report"] == "cached"
    assert count_silence_calls["n"] == 1 and executor.calls == 1
    assert second.output_path.stat().st_mtime_ns == mtime
    assert second.final_duration_sec == first.final_duration_sec

    third = pipeline.run(short_video, EditPipelineParams(threshold_db=-50.0))
    assert third.outcomes["probe"] == "cached"
    assert third.outcomes["silences"] == "ran"
    assert third.outcomes["plan"] == "ran"
    assert third.outcomes["render"] == "ran"
    assert third.outcomes["report"] == "ran"
    assert count_silence_calls["n"] == 2 and executor.calls == 2
    manifest = ProjectManifest.load(third.manifest_path)
    assert manifest.params.threshold_db == -50.0

    # Só mudar o padding refaz plano e render, mas não os silêncios
    fourth = pipeline.run(
        short_video,
        EditPipelineParams(threshold_db=-50.0, cut=SilenceCutParams(padding_after_sec=0.3)),
    )
    assert fourth.outcomes["silences"] == "cached"
    assert fourth.outcomes["plan"] == "ran" and fourth.outcomes["render"] == "ran"

    forced = pipeline.run(
        short_video,
        EditPipelineParams(threshold_db=-50.0, cut=SilenceCutParams(padding_after_sec=0.3)),
        force=True,
    )
    assert set(forced.outcomes.values()) == {"ran", "skipped"}
    assert count_silence_calls["n"] == 3 and executor.calls == 4


def test_transcription_cached_and_redone_on_language_change(tmp_path, short_video):
    calls = {"n": 0}

    class CountingMock(MockTranscriptionProvider):
        def transcribe(self, media_path, config):
            calls["n"] += 1
            return super().transcribe(media_path, config)

    pipeline = EditPipeline(
        projects_root=tmp_path / "projects", transcription_provider=CountingMock()
    )
    params = EditPipelineParams(transcribe=True)
    pipeline.run(short_video, params)
    result = pipeline.run(short_video, params)
    assert result.outcomes["transcription"] == "cached"
    assert calls["n"] == 1

    params.transcription.language = "en"
    result = pipeline.run(short_video, params)
    assert result.outcomes["transcription"] == "ran"
    assert result.outcomes["render"] == "cached"  # a transcrição ainda não afeta os cortes
    assert Transcript.load(result.project_dir / "transcript.json").language == "en"


def test_all_silent_video_fails_clearly(tmp_path):
    video = create_tone_silence_video(tmp_path / "mudo.mp4", [("silence", 3.0)])
    pipeline = EditPipeline(projects_root=tmp_path / "projects")
    with pytest.raises(NoSpeechDetectedError, match="Nenhum trecho de fala"):
        pipeline.run(video)

    manifests = list((tmp_path / "projects").glob("*/project.json"))
    assert len(manifests) == 1
    manifest = ProjectManifest.load(manifests[0])
    assert manifest.stages["silences"].status == "done"
    assert manifest.stages["plan"].status == "failed"
    assert "Nenhum trecho de fala" in manifest.stages["plan"].error


def test_cli_edit(tmp_path, short_video, capsys):
    args = build_parser().parse_args(
        ["edit", str(short_video), "--projects-dir", str(tmp_path / "projects"), "--padding-after", "0.25"]
    )
    assert handle_edit(args) == 0
    out = capsys.readouterr().out
    assert "[2/6] Silêncios: ok" in out
    assert "[3/6] Transcrição: pulada" in out
    assert "[5/6] Renderização: ok" in out
    assert "Edição concluída!" in out

    assert handle_edit(args) == 0
    assert "[5/6] Renderização: em cache" in capsys.readouterr().out

    manifest = ProjectManifest.load(next((tmp_path / "projects").glob("*/project.json")))
    assert manifest.params.cut.padding_after_sec == 0.25


def test_cli_edit_missing_video(tmp_path, capsys):
    args = build_parser().parse_args(["edit", str(tmp_path / "nao_existe.mp4")])
    assert handle_edit(args) == 1
    assert "não encontrado" in capsys.readouterr().err
