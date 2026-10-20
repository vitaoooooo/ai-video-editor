"""Testes do ciclo de revisão (versões, `revise`, `versions`) com vídeo sintético e provedores mock."""

import json
import shutil
from types import SimpleNamespace

import pytest

from ai_video_editor.analyzer.probe import VideoProbe
from ai_video_editor.cli import build_parser, handle_revise, handle_versions
from ai_video_editor.director import SentenceRemoval
from ai_video_editor.director.providers.mock import MockDirector
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectManifest, ProjectReviser
from ai_video_editor.planning import SilenceCutParams, apply_pause_preset
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.revision import (
    RequestHistoryItem,
    RevisionContext,
    RevisionDecision,
    RevisionRejectedError,
    SentenceRestoration,
    SentenceState,
    run_reviser,
)
from ai_video_editor.revision.providers.gemini import GeminiReviser, build_reviser_prompt
from ai_video_editor.revision.providers.mock import MockReviser
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from tests.conftest import create_tone_silence_video
from tests.test_edit_pipeline import CountingExecutor, _lesson_transcript_with_words


@pytest.fixture(scope="module")
def lesson_video(tmp_path_factory):
    """16.5s: quatro falas de 3s (0–3, 4.5–7.5, 9–12, 13.5–16.5) separadas por 1.5s de silêncio."""
    path = tmp_path_factory.mktemp("videos") / "aula-revisao.mp4"
    return create_tone_silence_video(path, [("tone", 3.0), ("silence", 1.5)] * 3 + [("tone", 3.0)])


def _removal(sid: int, reason: str = "Motivo.") -> SentenceRemoval:
    return SentenceRemoval(sentence_id=sid, category="other", reason=reason)


# Restaura a frase 1 (removida na v1) e remove a 3: respeita o limite de 40% do tempo de fala.
_swap_1_for_3 = RevisionDecision(
    removals=[SentenceRemoval(sentence_id=3, category="other", reason="Despedida.")],
    restorations=[SentenceRestoration(sentence_id=1, reason="Volta.")],
    summary="Troquei a 1 pela 3.",
)


@pytest.fixture(scope="module")
def _edited_template(tmp_path_factory, lesson_video):
    """Roda o `edit` (v1) uma única vez por módulo; cada teste trabalha numa cópia."""
    root = tmp_path_factory.mktemp("edited_template")
    pipeline = EditPipeline(
        projects_root=root / "projects",
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(lesson_video)
        ),
        edit_director=MockDirector(removals=[_removal(1, "Repete.")]),
    )
    result = pipeline.run(lesson_video, EditPipelineParams(transcribe=True, director="mock"))
    return root, result


@pytest.fixture
def edited(tmp_path, lesson_video, _edited_template):
    """Projeto com v1 pronto: o Diretor mock removeu a frase 1 (cópia isolada em tmp_path)."""
    root, template = _edited_template
    shutil.copytree(root / "projects", tmp_path / "projects")

    def relocate(path):
        return tmp_path / path.relative_to(root) if path is not None else None

    result = template.model_copy(
        update={
            "project_dir": relocate(template.project_dir),
            "manifest_path": relocate(template.manifest_path),
            "output_path": relocate(template.output_path),
            "report_path": relocate(template.report_path),
            "subtitles_path": relocate(template.subtitles_path),
        }
    )
    return SimpleNamespace(result=result, project=result.project_dir, video=lesson_video)


def _srt(path) -> str:
    return path.read_text(encoding="utf-8")


def _revise(project, reviser, request="pedido", from_version=None, executor=None):
    return ProjectReviser(reviser, render_executor=executor).revise(project, request, from_version)


@pytest.mark.slow
def test_edit_creates_v1_with_all_artifacts(edited):
    manifest = ProjectManifest.load(edited.result.manifest_path)
    assert edited.result.version == 1
    assert manifest.current_version == 1 and [v.number for v in manifest.versions] == [1]
    v1 = manifest.versions[0]
    assert v1.kind == "edit" and v1.parent is None and v1.pacing == "normal"
    assert v1.duration_sec == pytest.approx(edited.result.final_duration_sec, abs=0.01)
    assert v1.removed_sentences == 1 and v1.cuts >= 1

    vdir = edited.project / "versions" / "v1"
    for name in (
        "director_decision.json", "edit_plan.json", "edit_report.md", "version.json",
        v1.video, v1.subtitles,
    ):
        assert (vdir / name).is_file(), name
    plan = json.loads((vdir / "edit_plan.json").read_text(encoding="utf-8"))
    assert plan["output_path"].replace("\\", "/").endswith("versions/v1/" + v1.video)


@pytest.mark.slow
def test_edit_rerun_does_not_duplicate_but_changed_params_add_a_version(tmp_path, lesson_video):
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(lesson_video)
        ),
        edit_director=MockDirector(removals=[_removal(1)]),
    )
    params = EditPipelineParams(transcribe=True, director="mock")
    first = pipeline.run(lesson_video, params)
    again = pipeline.run(lesson_video, params)
    assert first.version == again.version == 1

    changed = pipeline.run(
        lesson_video, params.model_copy(update={"cut": SilenceCutParams(padding_after_sec=0.4)})
    )
    assert changed.version == 2
    manifest = ProjectManifest.load(changed.manifest_path)
    assert manifest.current_version == 2 and manifest.versions[1].parent == 1


@pytest.mark.slow
def test_revise_restores_one_sentence_and_removes_another_keeping_v1_intact(edited):
    project = edited.project
    v1_dir = project / "versions" / "v1"
    v1_files = {p.name: p.read_bytes() for p in v1_dir.iterdir()}
    root_video = edited.result.output_path
    root_mtime = root_video.stat().st_mtime_ns

    executor = CountingExecutor()
    reviser = MockReviser(
        RevisionDecision(
            removals=[_removal(3, "Despedida desnecessária.")],
            restorations=[SentenceRestoration(sentence_id=1, reason="O usuário quer essa frase.")],
            summary="Restaurei a frase 1 e tirei a despedida.",
        )
    )
    result = _revise(project, reviser, "mantém a frase repetida e tira o final", executor=executor)

    assert executor.calls == 1
    assert result.version.number == 2 and result.base_version == 1
    assert [c.id for c in result.removed] == [3] and [c.id for c in result.restored] == [1]
    assert result.duration_before_sec > 0 and result.duration_after_sec > 0

    v2_dir = project / "versions" / "v2"
    video = VideoProbe().probe(result.video_path)
    assert video.duration_sec == pytest.approx(result.duration_after_sec, abs=0.3)
    # v2: 3.2 + 3.35 + 3.35 das três primeiras falas + o resto do áudio depois da frase 3
    # (no transcript sintético ela termina em 14.75s, antes do fim da fala real em 16.5s)
    assert video.duration_sec == pytest.approx(3.2 + 3.35 + 3.35 + (16.5 - 14.75), abs=0.3)
    srt = _srt(result.subtitles_path)
    assert "Frase repetida que sai do video." in srt and "Obrigado por assistir." not in srt
    assert "Hoje vamos falar de cortes." in srt and "Primeiro abrimos o editor." in srt

    decision = json.loads((v2_dir / "director_decision.json").read_text(encoding="utf-8"))
    assert [r["sentence_id"] for r in decision["decision"]["removals"]] == [3]

    report = result.report_path.read_text(encoding="utf-8")
    assert "## Revisão" in report
    assert "| Pedido | mantém a frase repetida e tira o final |" in report
    assert "Restaurei a frase 1 e tirei a despedida." in report
    assert "| 1 | Frase repetida que sai do video. | removida | mantida | O usuário quer essa frase. |" in report
    assert "| 3 | Obrigado por assistir. | mantida | removida | Despedida desnecessária. |" in report

    # v1 e os artefatos da raiz continuam intactos
    assert {p.name: p.read_bytes() for p in v1_dir.iterdir()} == v1_files
    assert root_video.stat().st_mtime_ns == root_mtime
    srt_v1 = _srt(v1_dir / ProjectManifest.load(edited.result.manifest_path).versions[0].subtitles)
    assert "Frase repetida" not in srt_v1 and "Obrigado por assistir." in srt_v1

    manifest = ProjectManifest.load(edited.result.manifest_path)
    assert manifest.current_version == 2
    v2 = manifest.versions[1]
    assert v2.kind == "revision" and v2.parent == 1 and v2.request.startswith("mantém")


@pytest.mark.slow
def test_revise_sends_states_brief_history_and_request_to_the_reviser(edited):
    reviser = MockReviser()
    _revise(edited.project, reviser, "restaura tudo")
    states, context = reviser.calls[0]
    assert [s.removed for s in states] == [False, True, False, False]
    assert states[1].reason == "Repete."
    assert context.request == "restaura tudo" and context.base_version == 1
    assert context.history == [RequestHistoryItem(version=1, request="Edição inicial")]

    reviser2 = MockReviser(RevisionDecision(removals=[_removal(3)], summary="x"))
    _revise(edited.project, reviser2, "outro pedido")
    assert [h.request for h in reviser2.calls[0][1].history] == ["Edição inicial", "restaura tudo"]
    assert reviser2.calls[0][1].base_version == 2


@pytest.mark.slow
def test_revise_from_v1_after_v2_exists_and_v3_descends_from_v1(edited):
    project = edited.project
    _revise(project, MockReviser(_swap_1_for_3), "tira o final")

    # base v1: a frase 3 está mantida e a 1 removida, mesmo a v2 sendo a atual
    reviser = MockReviser(
        RevisionDecision(
            removals=[],
            restorations=[SentenceRestoration(sentence_id=1, reason="Volta.")],
            summary="Tudo de volta.",
        )
    )
    result = _revise(project, reviser, "restaura", from_version=1)
    states, context = reviser.calls[0]
    assert [s.removed for s in states] == [False, True, False, False]
    assert context.base_version == 1
    assert result.version.number == 3 and result.version.parent == 1
    assert result.base_version == 1

    manifest = ProjectManifest.load(edited.result.manifest_path)
    assert manifest.current_version == 3
    assert "Obrigado por assistir." in _srt(result.subtitles_path)
    assert "Frase repetida que sai do video." in _srt(result.subtitles_path)


@pytest.mark.slow
def test_revise_from_unknown_version_fails_without_creating_anything(edited):
    from ai_video_editor.revision import ProjectStateError

    with pytest.raises(ProjectStateError, match="v7"):
        _revise(edited.project, MockReviser(), "x", from_version=7)
    assert not (edited.project / "versions" / "v2").exists()


@pytest.mark.slow
def test_invalid_ids_after_retry_create_no_version(edited):
    manifest_before = (edited.project / "project.json").read_text(encoding="utf-8")
    bad = RevisionDecision(removals=[_removal(99)], summary="id inexistente")
    reviser = MockReviser(responses=[bad, bad])
    executor = CountingExecutor()

    with pytest.raises(RevisionRejectedError, match="não existe"):
        _revise(edited.project, reviser, "tira a parte inventada", executor=executor)

    assert len(reviser.calls) == 2
    assert reviser.calls[1][1].previous_error and "99" in reviser.calls[1][1].previous_error
    assert executor.calls == 0
    assert not (edited.project / "versions" / "v2").exists()
    assert (edited.project / "project.json").read_text(encoding="utf-8") == manifest_before


@pytest.mark.slow
def test_retry_succeeds_when_second_answer_is_valid(edited):
    bad = RevisionDecision(removals=[_removal(99)], summary="x")
    good = _swap_1_for_3
    result = _revise(edited.project, MockReviser(responses=[bad, good]), "tira o final")
    assert result.version.number == 2


def test_restorations_must_match_the_diff():
    states = [
        SentenceState(sentence=s, removed=s.id == 1)
        for s in build_sentences_for_tests()
    ]
    context = RevisionContext(video_name="a.mp4", duration_sec=10, base_version=1, request="x")
    from ai_video_editor.director import UserBrief

    # frase 1 volta, mas o revisor esqueceu de listá-la em restorations
    forgot = RevisionDecision(removals=[], summary="x")
    with pytest.raises(RevisionRejectedError, match="restorations"):
        run_reviser(MockReviser(decision=forgot), states, UserBrief(), context)


@pytest.mark.slow
def test_infeasible_request_is_explained_and_creates_no_version(edited):
    reviser = MockReviser(
        RevisionDecision(feasible=False, problem="Ninguém fala do site neste vídeo.", summary="")
    )
    with pytest.raises(RevisionRejectedError, match="Ninguém fala do site"):
        _revise(edited.project, reviser, "tira a parte do site")
    assert len(reviser.calls) == 1  # pedido impossível não gasta nova tentativa
    assert not (edited.project / "versions" / "v2").exists()


@pytest.mark.slow
def test_request_that_changes_nothing_is_rejected(edited):
    reviser = MockReviser(RevisionDecision(removals=[_removal(1)], summary="Já está assim."))
    with pytest.raises(RevisionRejectedError, match="não muda nada"):
        _revise(edited.project, reviser, "deixa como está")
    assert not (edited.project / "versions" / "v2").exists()


@pytest.mark.slow
def test_removing_everything_is_rejected(edited):
    everything = RevisionDecision(removals=[_removal(i) for i in range(4)], summary="tudo")
    with pytest.raises(RevisionRejectedError):
        _revise(edited.project, MockReviser(everything), "apaga tudo")
    assert not (edited.project / "versions" / "v2").exists()


@pytest.mark.slow
def test_render_failure_leaves_no_half_version(edited):
    class FailingExecutor(RenderExecutor):
        def execute(self, plan):
            raise RuntimeError("ffmpeg quebrou")

    before = (edited.project / "project.json").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        _revise(edited.project, MockReviser(), "restaura", executor=FailingExecutor())
    assert not (edited.project / "versions" / "v2").exists()
    assert (edited.project / "project.json").read_text(encoding="utf-8") == before


def test_pause_presets_scale_the_cut_params():
    base = SilenceCutParams()
    assert apply_pause_preset(base, "normal") == base
    natural, tight = apply_pause_preset(base, "natural"), apply_pause_preset(base, "tight")
    assert natural.padding_after_sec > base.padding_after_sec > tight.padding_after_sec
    assert natural.padding_before_sec > base.padding_before_sec > tight.padding_before_sec
    assert natural.min_gap_to_cut_sec > base.min_gap_to_cut_sec > tight.min_gap_to_cut_sec
    assert natural.min_keep_sec == base.min_keep_sec == tight.min_keep_sec


@pytest.mark.slow
def test_pause_presets_change_the_plan_and_the_inherited_pacing(edited):
    project = edited.project
    keep = RevisionDecision(removals=[_removal(1)], summary="só o ritmo")

    def plan_segments(version: int):
        data = json.loads((project / "versions" / f"v{version}" / "edit_plan.json").read_text(encoding="utf-8"))
        return [(s["start_sec"], s["end_sec"]) for s in data["segments"]]

    natural = _revise(project, MockReviser(keep.model_copy(update={"pacing": "natural"})), "mais respirado")
    tight = _revise(project, MockReviser(keep.model_copy(update={"pacing": "tight"})), "mais seco", from_version=1)
    assert natural.version.pacing == "natural" and tight.version.pacing == "tight"
    assert (natural.pacing_before, natural.pacing_after) == ("normal", "natural")

    # mesmas frases removidas: só a folga das pausas muda, então natural > normal > tight
    assert natural.duration_after_sec > natural.duration_before_sec > tight.duration_after_sec
    assert plan_segments(2) != plan_segments(1) != plan_segments(3)
    assert "| Ritmo das pausas | normal → natural |" in natural.report_path.read_text(encoding="utf-8")

    # sem `pacing`, a revisão herda o ritmo da versão base
    inherited = _revise(project, MockReviser(_swap_1_for_3), "tira o final")
    assert inherited.version.pacing == "tight" and inherited.base_version == 3


@pytest.mark.slow
def test_legacy_project_without_versions_is_migrated_to_v1(edited):
    manifest_path = edited.result.manifest_path
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    del raw["versions"], raw["current_version"]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    import shutil

    shutil.rmtree(edited.project / "versions")

    manifest = ProjectManifest.load(manifest_path)
    assert manifest.versions == [] and manifest.current_version is None

    result = _revise(edited.project, MockReviser(), "restaura")
    assert result.base_version == 1 and result.version.number == 2
    migrated = ProjectManifest.load(manifest_path)
    assert [v.number for v in migrated.versions] == [1, 2]
    v1 = migrated.versions[0]
    assert v1.kind == "edit" and v1.removed_sentences == 1
    assert (edited.project / "versions" / "v1" / v1.video).is_file()
    assert (edited.project / "versions" / "v1" / "director_decision.json").is_file()

    # e o `edit` repetido num projeto migrado não cria versão duplicada
    pipeline = EditPipeline(
        projects_root=edited.project.parent,
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(edited.video)
        ),
        edit_director=MockDirector(removals=[_removal(1, "Repete.")]),
    )
    again = pipeline.run(edited.video, EditPipelineParams(transcribe=True, director="mock"))
    assert again.version == 1
    assert [v.number for v in ProjectManifest.load(manifest_path).versions] == [1, 2]


@pytest.mark.slow
def test_legacy_project_with_silence_cut_only_can_be_revised(tmp_path, lesson_video):
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(
            custom_transcript=_lesson_transcript_with_words(lesson_video)
        ),
    )
    result = pipeline.run(lesson_video, EditPipelineParams(transcribe=True))
    assert not (result.project_dir / "versions" / "v1" / "director_decision.json").exists()

    revised = _revise(
        result.project_dir, MockReviser(RevisionDecision(removals=[_removal(3)], summary="sem despedida")), "tira o final"
    )
    assert [c.id for c in revised.removed] == [3] and revised.restored == []
    assert revised.duration_after_sec < revised.duration_before_sec


@pytest.mark.slow
def test_revise_requires_a_transcript(tmp_path, lesson_video):
    from ai_video_editor.revision import ProjectStateError

    result = EditPipeline(projects_root=tmp_path / "projects").run(lesson_video, EditPipelineParams())
    with pytest.raises(ProjectStateError, match="transcript.json"):
        _revise(result.project_dir, MockReviser(), "x")


def test_revise_on_a_folder_that_is_not_a_project(tmp_path):
    from ai_video_editor.revision import ProjectStateError

    with pytest.raises(ProjectStateError, match="project.json"):
        _revise(tmp_path, MockReviser(), "x")


@pytest.mark.slow
def test_cli_versions_and_revise(edited, capsys):
    parser = build_parser()
    args = parser.parse_args(["revise", str(edited.project), "restaura tudo", "--reviser", "mock", "--from", "v1"])
    assert args.from_version == "v1"

    class Quiet(MockReviser):
        pass

    import ai_video_editor.cli as cli

    assert handle_revise(args) == 0
    out = capsys.readouterr().out
    assert "Versão v2 criada (a partir da v1)." in out
    assert "Restauradas:" in out and "Frase repetida" in out
    assert "(v1 -> v2)" in out

    assert handle_versions(parser.parse_args(["versions", str(edited.project)])) == 0
    out = capsys.readouterr().out
    assert "Edição inicial" in out and "restaura tudo" in out
    assert "*v2" in out and " v1" in out
    assert cli  # módulo importado acima


@pytest.mark.slow
def test_cli_revise_rejection_exits_1_and_explains(edited, capsys):
    parser = build_parser()
    args = parser.parse_args(["revise", str(edited.project), "deixa", "--reviser", "mock"])
    # o mock padrão mantém tudo igual → pedido sem efeito
    assert handle_revise(args) == 1
    err = capsys.readouterr().err
    assert "Versão não criada" in err and "não muda nada" in err
    assert not (edited.project / "versions" / "v2").exists()


def test_cli_revise_on_missing_project(tmp_path, capsys):
    args = build_parser().parse_args(["revise", str(tmp_path / "nada"), "x", "--reviser", "mock"])
    assert handle_revise(args) == 1
    assert "não é uma pasta de projeto" in capsys.readouterr().err


# --- GeminiReviser (sem rede) ---


def build_sentences_for_tests():
    from ai_video_editor.director import Sentence

    texts = ["Primeira frase.", "Segunda frase.", "Terceira frase."]
    return [
        Sentence(id=i, start_sec=i * 4.0, end_sec=i * 4.0 + 3.0, text=t, segment_ids=[i])
        for i, t in enumerate(texts)
    ]


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=json.dumps(self.payload),
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidate_token_count=None, candidates_token_count=5),
        )


def _states(removed=(1,)):
    return [SentenceState(sentence=s, removed=s.id in removed, category="repetition" if s.id in removed else None,
                          reason="Repete a 0." if s.id in removed else None) for s in build_sentences_for_tests()]


def test_gemini_reviser_prompt_has_states_brief_history_and_request():
    from ai_video_editor.director import UserBrief

    context = RevisionContext(
        video_name="aula.mp4", duration_sec=12, language="pt", base_version=2, base_pacing="tight",
        request="corta menos",
        history=[RequestHistoryItem(version=1, request="Edição inicial"), RequestHistoryItem(version=2, request="mais seco")],
        previous_error="sentence_id 9 não existe",
    )
    prompt = build_reviser_prompt(_states(), UserBrief(intensity="medium", instructions="sem piadas"), context)
    assert "[1] (3.0s) REMOVIDA [repetition: Repete a 0.] — Segunda frase." in prompt
    assert "[0] (3.0s) mantida — Primeira frase." in prompt
    assert "- v2: mais seco" in prompt and "- v1: Edição inicial" in prompt
    assert "NOVO PEDIDO DO USUÁRIO:\ncorta menos" in prompt
    assert "Intensidade: medium" in prompt and "Pedido original: sem piadas" in prompt
    assert "Ritmo atual das pausas: tight" in prompt and "versão v2" in prompt
    assert "sentence_id 9 não existe" in prompt


def test_gemini_reviser_parses_json_and_uses_director_thinking_config():
    from ai_video_editor.director import UserBrief

    payload = {
        "removals": [{"sentence_id": 2, "category": "other", "reason": "Fora do tema."}],
        "restorations": [{"sentence_id": 1, "reason": "Pedido."}],
        "pacing": "natural",
        "summary": "Troquei.",
    }
    models = _FakeModels(payload)
    reviser = GeminiReviser(client=SimpleNamespace(models=models), thinking="low", timeout_sec=30)
    context = RevisionContext(video_name="a.mp4", duration_sec=12, base_version=1, request="x")
    decision = reviser.revise(_states(), UserBrief(), context)

    assert decision.pacing == "natural" and decision.removals[0].sentence_id == 2
    assert decision.restorations[0].sentence_id == 1 and decision.feasible
    config = models.calls[0]["config"]
    assert config.thinking_config.thinking_level.value == "LOW"
    assert config.response_schema is RevisionDecision
    assert reviser.last_input_tokens == 10 and reviser.last_output_tokens == 5
    assert reviser.thinking_label == "low (thinking_level=LOW)"


def test_gemini_reviser_invalid_json_triggers_retry_then_rejection():
    from ai_video_editor.director import UserBrief

    models = _FakeModels({})
    models.generate_content = lambda **kw: SimpleNamespace(text="não é json", usage_metadata=None)
    reviser = GeminiReviser(client=SimpleNamespace(models=models))
    context = RevisionContext(video_name="a.mp4", duration_sec=12, base_version=1, request="x")
    with pytest.raises(RevisionRejectedError, match="JSON inválido"):
        run_reviser(reviser, _states(), UserBrief(), context)


def test_gemini_reviser_escalates_thinking_like_the_director():
    from ai_video_editor.director import UserBrief

    class Models(_FakeModels):
        def generate_content(self, **kwargs):
            level = kwargs["config"].thinking_config.thinking_level.value
            if level == "MINIMAL":
                self.calls.append(kwargs)
                raise Exception("400 INVALID_ARGUMENT. Thinking level MINIMAL is not supported by this model.")
            return super().generate_content(**kwargs)

    models = Models({"removals": [], "summary": "x"})
    reviser = GeminiReviser(client=SimpleNamespace(models=models), thinking="off")
    context = RevisionContext(video_name="a.mp4", duration_sec=12, base_version=1, request="x")
    run = run_reviser(reviser, _states(removed=()), UserBrief(), context)
    assert run.decision is not None
    assert run.thinking == "off → LOW (MINIMAL não suportado)"


def test_mock_reviser_is_deterministic():
    from ai_video_editor.director import UserBrief

    context = RevisionContext(video_name="a.mp4", duration_sec=12, base_version=1, request="restaura e deixa tight")
    a = MockReviser().revise(_states(), UserBrief(), context)
    b = MockReviser().revise(_states(), UserBrief(), context)
    assert a == b and a.pacing == "tight" and [r.sentence_id for r in a.restorations] == [1]
    assert AIResponseValidationError  # import usado em outros testes de erro
