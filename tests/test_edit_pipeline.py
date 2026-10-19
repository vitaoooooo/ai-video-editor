"""Testes de ponta a ponta do pipeline `edit` com vídeos sintéticos e transcrição Mock."""

import pytest

from ai_video_editor.analyzer.probe import VideoProbe
from ai_video_editor.cli import build_parser, handle_edit
from ai_video_editor.core.schema import EditPlan
from ai_video_editor.director import DirectorDecision, DirectorRun, SentenceRemoval, UserBrief
from ai_video_editor.director.providers.mock import MockDirector
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectManifest
from ai_video_editor.pipeline import runner as runner_module
from ai_video_editor.planning import NoSpeechDetectedError, SilenceCutParams
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.subtitles import SubtitleParams
from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment, Word
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
    assert manifest.stages["director"].status == "skipped"
    done = {name: r for name, r in manifest.stages.items() if name != "director"}
    assert all(record.status == "done" for record in done.values())
    assert all(record.fingerprint for record in done.values())
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
        "director": "skipped",
        "plan": "ran",
        "render": "ran",
        "subtitles": "skipped",
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
    assert "[2/8] Silêncios: ok" in out
    assert "[3/8] Transcrição: pulada" in out
    assert "[6/8] Renderização: ok" in out
    assert "Edição concluída!" in out

    assert handle_edit(args) == 0
    assert "[6/8] Renderização: em cache" in capsys.readouterr().out

    manifest = ProjectManifest.load(next((tmp_path / "projects").glob("*/project.json")))
    assert manifest.params.cut.padding_after_sec == 0.25


def test_cli_edit_missing_video(tmp_path, capsys):
    args = build_parser().parse_args(["edit", str(tmp_path / "nao_existe.mp4")])
    assert handle_edit(args) == 1
    assert "não encontrado" in capsys.readouterr().err


def test_stage_error_details_reach_cli_and_manifest(tmp_path, short_video, capsys, monkeypatch):
    class BrokenModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("motivo-real-xyz")

    monkeypatch.setattr(FasterWhisperProvider, "_load_model", lambda self, config: BrokenModel())
    projects = tmp_path / "projects"
    args = build_parser().parse_args(
        ["edit", str(short_video), "--projects-dir", str(projects), "--transcribe"]
    )
    assert handle_edit(args) == 1
    assert "motivo-real-xyz" in capsys.readouterr().err
    manifest = ProjectManifest.load(next(projects.glob("*/project.json")))
    assert "motivo-real-xyz" in manifest.stages["transcription"].error


# --- Diretor de IA ---


@pytest.fixture(scope="module")
def lesson_video(tmp_path_factory):
    """16.5s: quatro falas de 3s (0–3, 4.5–7.5, 9–12, 13.5–16.5) separadas por 1.5s de silêncio."""
    path = tmp_path_factory.mktemp("videos") / "aula-diretor.mp4"
    return create_tone_silence_video(path, [("tone", 3.0), ("silence", 1.5)] * 3 + [("tone", 3.0)])


def _lesson_transcript(video) -> Transcript:
    """Tempos do Whisper levemente deslocados das falas reais; a frase 1 repete a 0."""
    texts = [
        "Hoje vamos falar de cortes.",
        "Hoje vamos falar de cortes!",
        "Primeiro abrimos o editor.",
        "Obrigado por assistir.",
    ]
    starts = [0.1, 4.6, 9.1, 13.6]
    return Transcript(
        source_path=video,
        language="pt",
        duration_sec=16.5,
        model_used="mock",
        segments=[
            TranscriptSegment(id=i, start_sec=s, end_sec=s + 2.8, text=t)
            for i, (s, t) in enumerate(zip(starts, texts))
        ],
    )


def test_end_to_end_director_removes_sentences(tmp_path, lesson_video):
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(custom_transcript=_lesson_transcript(lesson_video)),
    )
    params = EditPipelineParams(transcribe=True, director="mock")
    result = pipeline.run(lesson_video, params)

    # Sem o Diretor: 3.2 + 3.35 + 3.35 + 3.15 = 13.05s. A frase 1 sai com a pausa: corte 3.2 → 8.85
    expected = 3.2 + 3.35 + 3.15
    assert result.outcomes["director"] == "ran"
    assert result.final_duration_sec == pytest.approx(expected, abs=0.3)
    assert VideoProbe().probe(result.output_path).duration_sec == pytest.approx(expected, abs=0.3)
    assert result.warnings == []

    plan = EditPlan.model_validate_json((result.project_dir / "edit_plan.json").read_text(encoding="utf-8"))
    assert all(seg.end_sec <= 4.5 or seg.start_sec >= 7.5 for seg in plan.segments)
    assert [(s.start_sec, s.end_sec) for s in plan.segments] == [(0.0, 3.2), (8.85, 12.2), (13.35, 16.5)]

    run = DirectorRun.model_validate_json(
        (result.project_dir / "director_decision.json").read_text(encoding="utf-8")
    )
    assert [s.id for s, _ in run.removed_sentences] == [1]
    manifest = ProjectManifest.load(result.manifest_path)
    assert manifest.stages["director"].metrics["removed"] == 1
    assert manifest.stages["plan"].metrics["director_cuts"] == 1

    report = (result.project_dir / "edit_report.md").read_text(encoding="utf-8")
    assert "## Decisões do Diretor" in report
    assert "| 1 | 00:04.600 | 00:07.400 | repetition | Hoje vamos falar de cortes! | Repete a frase 0. |" in report
    assert "| Diretor | mock (mock) |" in report

    # Cache: nada muda; mudar o briefing refaz o Diretor
    again = pipeline.run(lesson_video, params)
    assert again.outcomes["director"] == "cached" and again.outcomes["render"] == "cached"
    params.brief = UserBrief(intensity="medium")
    changed = pipeline.run(lesson_video, params)
    assert changed.outcomes["transcription"] == "cached"
    assert changed.outcomes["director"] == "ran"


def test_director_fallback_keeps_silence_cut_and_warns(tmp_path, lesson_video):
    invalid = DirectorDecision(removals=[SentenceRemoval(sentence_id=99, category="other", reason="x")], summary="x")
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(custom_transcript=_lesson_transcript(lesson_video)),
        edit_director=MockDirector(responses=[invalid, invalid]),
    )
    result = pipeline.run(lesson_video, EditPipelineParams(transcribe=True, director="gemini"))

    assert result.final_duration_sec == pytest.approx(13.05, abs=0.3)
    assert len(result.warnings) == 1 and "só o corte de silêncio" in result.warnings[0]
    report = (result.project_dir / "edit_report.md").read_text(encoding="utf-8")
    assert "> **Aviso:** Diretor ignorado" in report
    assert "inválida (AIResponseValidationError)" in report


def test_director_requires_transcription(tmp_path, short_video):
    pipeline = EditPipeline(projects_root=tmp_path / "projects")
    result = pipeline.run(short_video, EditPipelineParams(director="mock"))
    assert result.outcomes["director"] == "skipped"
    assert not (result.project_dir / "director_decision.json").exists()


def test_cli_edit_director_options(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, video_path, params, force=False):
        captured["params"] = params
        raise RuntimeError("parar aqui")

    monkeypatch.setattr(EditPipeline, "run", fake_run)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    args = build_parser().parse_args(
        [
            "edit", str(video), "--transcribe", "--director", "gemini", "--intensity", "aggressive",
            "--target-duration", "240", "--instructions", "tire as piadas", "--director-model", "gemini-2.5-pro",
        ]
    )
    assert handle_edit(args) == 1
    params = captured["params"]
    assert params.director == "gemini" and params.director_model == "gemini-2.5-pro"
    assert params.brief == UserBrief(
        intensity="aggressive", target_duration_sec=240, remove_fillers=True, instructions="tire as piadas"
    )


def test_old_manifest_without_director_stage_loads(tmp_path):
    manifest = ProjectManifest(video_path=tmp_path / "v.mp4", video_hash="abc", params=EditPipelineParams())
    data = manifest.model_dump(mode="json")
    del data["stages"]["director"]
    loaded = ProjectManifest.model_validate(data)
    assert loaded.stages["director"].status == "pending"
    assert list(loaded.stages) == list(manifest.stages)


# --- Legendas ---


def _words(text: str, start: float, step: float = 0.4) -> list[Word]:
    return [
        Word(text=t, start_sec=start + i * step, end_sec=start + (i + 1) * step - 0.05)
        for i, t in enumerate(text.split())
    ]


def _lesson_transcript_with_words(video) -> Transcript:
    texts = [
        "Hoje vamos falar de cortes.",
        "Frase repetida que sai do video.",
        "Primeiro abrimos o editor.",
        "Obrigado por assistir.",
    ]
    starts = [0.1, 4.6, 9.1, 13.6]
    segments = []
    for i, (start, text) in enumerate(zip(starts, texts)):
        words = _words(text, start)
        segments.append(
            TranscriptSegment(id=i, start_sec=start, end_sec=words[-1].end_sec, text=text, words=words)
        )
    return Transcript(source_path=video, language="pt", duration_sec=16.5, model_used="mock", segments=segments)


def _srt_blocks(path):
    return [b for b in path.read_text(encoding="utf-8").strip().split("\n\n") if b]


def test_end_to_end_subtitles_follow_director_cuts(tmp_path, lesson_video):
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(lesson_video)
        ),
        edit_director=MockDirector(removals=[SentenceRemoval(sentence_id=1, category="repetition", reason="Repete.")]),
    )
    result = pipeline.run(lesson_video, EditPipelineParams(transcribe=True, director="mock"))

    srt = result.subtitles_path
    assert srt == result.project_dir / "aula-diretor_edited.srt"
    assert srt.parent == result.output_path.parent and srt.stem == result.output_path.stem
    content = srt.read_text(encoding="utf-8")
    assert "Hoje vamos falar de cortes." in content and "Obrigado por assistir." in content
    assert "repetida" not in content

    duration = VideoProbe().probe(result.output_path).duration_sec
    for line in content.splitlines():
        if " --> " in line:
            start, end = (_to_sec(t) for t in line.split(" --> "))
            assert 0 <= start < end <= duration + 0.05

    manifest = ProjectManifest.load(result.manifest_path)
    assert manifest.stages["subtitles"].status == "done"
    assert manifest.stages["subtitles"].artifact == srt.name
    blocks = manifest.stages["subtitles"].metrics["blocks"]
    assert blocks == len(_srt_blocks(srt)) == 3
    report = result.report_path.read_text(encoding="utf-8")
    assert f"| Legendas | `aula-diretor_edited.srt` ({blocks} blocos) |" in report


def _to_sec(stamp: str) -> float:
    h, m, rest = stamp.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def test_subtitles_cache_and_max_chars_invalidation(tmp_path, lesson_video):
    executor = CountingExecutor()
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(lesson_video)
        ),
        render_executor=executor,
    )
    params = EditPipelineParams(transcribe=True)
    first = pipeline.run(lesson_video, params)
    assert first.outcomes["subtitles"] == "ran"
    mtime = first.subtitles_path.stat().st_mtime_ns

    second = pipeline.run(lesson_video, params)
    assert second.outcomes["subtitles"] == "cached" and second.outcomes["report"] == "cached"
    assert first.subtitles_path.stat().st_mtime_ns == mtime

    params = EditPipelineParams(transcribe=True, subtitle=SubtitleParams(max_chars_per_line=10))
    third = pipeline.run(lesson_video, params)
    assert third.outcomes["subtitles"] == "ran" and third.outcomes["report"] == "ran"
    assert third.outcomes["plan"] == "cached" and third.outcomes["render"] == "cached"
    assert third.outcomes["transcription"] == "cached"
    assert executor.calls == 1
    assert max(len(line) for line in third.subtitles_path.read_text(encoding="utf-8").splitlines()
               if line and " --> " not in line and not line.isdigit()) <= 10


def test_subtitles_skipped_without_transcript_or_when_disabled(tmp_path, lesson_video):
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(lesson_video)
        ),
    )
    plain = pipeline.run(lesson_video, EditPipelineParams())
    assert plain.outcomes["subtitles"] == "skipped" and plain.subtitles_path is None
    assert not plain.subtitles_path and not list(plain.project_dir.glob("*.srt"))
    assert "Legendas" not in plain.report_path.read_text(encoding="utf-8")

    with_srt = pipeline.run(lesson_video, EditPipelineParams(transcribe=True))
    assert with_srt.subtitles_path.is_file()

    off = pipeline.run(lesson_video, EditPipelineParams(transcribe=True, subtitles=False))
    assert off.outcomes["subtitles"] == "skipped" and off.subtitles_path is None
    assert not list(off.project_dir.glob("*.srt"))
    assert ProjectManifest.load(off.manifest_path).stages["subtitles"].status == "skipped"
    assert "Legendas" not in off.report_path.read_text(encoding="utf-8")


def test_cli_edit_subtitle_options(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, video_path, params, force=False):
        captured["params"] = params
        raise RuntimeError("parar aqui")

    monkeypatch.setattr(EditPipeline, "run", fake_run)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    parser = build_parser()

    assert handle_edit(parser.parse_args(["edit", str(video)])) == 1
    assert captured["params"].subtitles is True
    assert captured["params"].subtitle == SubtitleParams()

    args = parser.parse_args(["edit", str(video), "--no-subtitles", "--subtitle-max-chars", "30"])
    assert handle_edit(args) == 1
    assert captured["params"].subtitles is False
    assert captured["params"].subtitle.max_chars_per_line == 30
