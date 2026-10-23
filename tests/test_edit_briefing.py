"""Testes do `edit` lendo o briefing do Produtor: precedência, reaproveitamento, --sem-briefing, origem e cache."""

import json
from pathlib import Path

import pytest

from ai_video_editor import cli as cli_module
from ai_video_editor.catalog import MockSpeechDetector
from ai_video_editor.cli import build_parser, handle_briefing, handle_edit
from ai_video_editor.director.schema import UserBrief
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectManifest
from ai_video_editor.pipeline.briefing import EditBriefing, briefing_location, prepare_briefing, resolve_settings
from ai_video_editor.pipeline.project import fingerprint
from ai_video_editor.pipeline.schema import settings_fingerprint
from ai_video_editor.producer import BRIEFING_FILE, Briefing, BriefingContent, MaterialSummary
from ai_video_editor.producer import material as material_module
from ai_video_editor.producer.material import Material
from ai_video_editor.producer.providers.mock import MockProducer
from ai_video_editor.producer.schema import RoleSummary
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider


def _material(name: str = "v.mp4", clips: int = 1) -> MaterialSummary:
    return MaterialSummary(
        kind="arquivo" if clips == 1 else "pasta", name=name, clips=clips, duration_sec=100.0,
        roles=[RoleSummary(role="principal", clips=clips, duration_sec=100.0, clips_with_speech=clips)],
    )


def _briefing(**update) -> Briefing:
    data = dict(
        video_type="aula", recipe="fala", understood="Entendi: aula sem enrolação.", intensity="medium",
        pace="tight", director_instructions="tire as gaguejadas", target_duration_sec=60.0,
        origin="ia", request="aula curta", material=_material(),
    )
    data.update(update)
    return Briefing(**data)


def _content(**update) -> BriefingContent:
    data = dict(
        video_type="aula", recipe="fala", understood="Entendi: aula sem enrolação.", intensity="medium",
        pace="tight", director_instructions="tire as gaguejadas", target_duration_sec=2.0,
    )
    data.update(update)
    return BriefingContent(**data)


# --- precedência (função pura) ---


def test_cli_beats_briefing_beats_default():
    found = EditBriefing(_briefing(), "reaproveitado")
    settings = resolve_settings(found, "fala")
    assert [(s.value, s.origin) for s in (settings.recipe, settings.intensity, settings.pace)] == [
        ("fala", "briefing"), ("medium", "briefing"), ("tight", "briefing"),
    ]
    assert (settings.instructions.value, settings.instructions.origin) == ("tire as gaguejadas", "briefing")
    assert (settings.target_duration_sec.value, settings.target_duration_sec.origin) == (60.0, "briefing")
    assert settings.understood == "Entendi: aula sem enrolação." and settings.uses_briefing

    cli = resolve_settings(
        found, "fala", recipe="fala", intensity="aggressive", pace="natural", instructions="corte as piadas",
        target_duration_sec=30.0,
    )
    assert {s.origin for s in (cli.recipe, cli.intensity, cli.pace, cli.instructions, cli.target_duration_sec)} == {"cli"}
    assert (cli.intensity.value, cli.pace.value, cli.instructions.value) == ("aggressive", "natural", "corte as piadas")

    none = resolve_settings(None, "vlog")
    assert [(s.value, s.origin) for s in (none.recipe, none.intensity, none.pace, none.instructions)] == [
        ("vlog", "padrao"), ("light", "padrao"), ("normal", "padrao"), (None, "padrao"),
    ]
    assert none.briefing == "sem_briefing" and not none.uses_briefing


def test_empty_briefing_fields_fall_back_to_default():
    settings = resolve_settings(EditBriefing(_briefing(director_instructions="", target_duration_sec=None), "gerado"), "fala")
    assert settings.instructions.origin == "padrao" and settings.instructions.value is None
    assert settings.target_duration_sec.origin == "padrao"


def test_fallback_briefing_recipe_does_not_override_the_edit_rule():
    """A regra sem IA não sabe da análise visual; o padrão do `edit` (vlog_cobertura) vale mais."""
    fallback = _briefing(origin="fallback", recipe="vlog")
    assert resolve_settings(EditBriefing(fallback, "gerado"), "vlog_cobertura").recipe.origin == "padrao"
    forced = _briefing(origin="fallback", recipe="vlog", forced_recipe="vlog")
    settings = resolve_settings(EditBriefing(forced, "gerado"), "vlog_cobertura")
    assert (settings.recipe.value, settings.recipe.origin) == ("vlog", "briefing")
    # os outros valores do briefing de fallback continuam valendo
    assert settings.intensity.origin == "briefing"


def test_same_briefing_gives_the_same_fingerprint():
    first = resolve_settings(EditBriefing(_briefing(), "gerado"), "fala")
    again = _briefing().model_copy(update={"created_at": _briefing().created_at.replace(year=2030)})
    reused = resolve_settings(EditBriefing(again, "reaproveitado", notes=["algum aviso"]), "fala")
    assert settings_fingerprint(first) == settings_fingerprint(reused)
    other = resolve_settings(EditBriefing(_briefing(intensity="aggressive"), "gerado"), "fala")
    assert settings_fingerprint(first) != settings_fingerprint(other)
    assert settings_fingerprint(resolve_settings(None, "fala")) == ()


def test_pacing_normal_keeps_params_fingerprint_and_cut():
    params = EditPipelineParams()
    assert "pacing" not in params.fingerprint_view()
    assert params.effective_cut == params.cut
    tight = EditPipelineParams(pacing="tight")
    assert tight.fingerprint_view()["pacing"] == "tight"
    assert tight.effective_cut.padding_after_sec < params.cut.padding_after_sec
    assert fingerprint(params.effective_cut) != fingerprint(tight.effective_cut)


# --- briefing.json reaproveitado ou refeito ---


@pytest.fixture
def mock_vad(monkeypatch):
    monkeypatch.setattr(material_module, "FasterWhisperVAD", lambda: MockSpeechDetector(segments=[(0.0, 2.5)]))


class CountingFactory:
    def __init__(self, responses=None):
        self.producers: list[MockProducer] = []
        self.responses = responses

    def __call__(self) -> MockProducer:
        producer = MockProducer(responses=self.responses)
        self.producers.append(producer)
        return producer

    @property
    def calls(self) -> int:
        return sum(len(p.calls) for p in self.producers)


def test_briefing_is_reused_while_the_request_is_the_same(tmp_path, synthetic_video_with_audio, mock_vad):
    projects = tmp_path / "projects"
    factory = CountingFactory([_content()])
    video = synthetic_video_with_audio

    first = prepare_briefing(video, "aula curta", projects, factory)
    assert first.status == "gerado" and factory.calls == 1
    assert first.path == briefing_location(video, projects) and first.path.is_file()
    assert first.briefing.intensity == "medium" and first.briefing.request == "aula curta"

    same = prepare_briefing(video, "  aula curta ", projects, factory)
    assert same.status == "reaproveitado" and factory.calls == 1 and same.briefing == first.briefing

    # sem --pedido, o briefing.json atual também vale (ex.: feito pelo comando `briefing`)
    assert prepare_briefing(video, None, projects, factory).status == "reaproveitado" and factory.calls == 1

    changed = prepare_briefing(video, "aula curtíssima", projects, factory)
    assert changed.status == "gerado" and factory.calls == 2
    assert changed.notes == ["briefing refeito: o pedido mudou"]
    assert Briefing.load(first.path).request == "aula curtíssima"


def test_no_request_and_no_briefing_is_today_behavior(tmp_path, synthetic_video_with_audio):
    factory = CountingFactory()
    found = prepare_briefing(synthetic_video_with_audio, None, tmp_path / "projects", factory)
    assert found.briefing is None and found.status == "sem_briefing" and factory.calls == 0 and not found.notes
    assert not found.path.exists()


def test_folder_briefing_is_redone_when_the_material_changes(tmp_path, monkeypatch):
    folder = tmp_path / "Viagem"
    folder.mkdir()
    projects = tmp_path / "projects"
    project_dir = briefing_location(folder, projects).parent
    summary = {"value": _material("Viagem", clips=3)}
    monkeypatch.setattr(
        material_module, "load_material",
        lambda target, root, **kw: Material(summary=summary["value"], project_dir=project_dir),
    )
    factory = CountingFactory([_content(recipe="vlog", target_duration_sec=None)])
    assert prepare_briefing(folder, "vlog curto", projects, factory).status == "gerado"
    assert prepare_briefing(folder, "vlog curto", projects, factory).status == "reaproveitado"

    summary["value"] = _material("Viagem", clips=4)  # um clipe novo (ou papel trocado no catalog)
    ignored = prepare_briefing(folder, None, projects, factory)
    assert ignored.briefing is None and ignored.status == "sem_briefing"
    assert "o material mudou" in ignored.notes[0]
    redone = prepare_briefing(folder, "vlog curto", projects, factory)
    assert redone.status == "gerado" and factory.calls == 2 and "o material mudou" in redone.notes[0]


# --- CLI ---


@pytest.fixture
def captured_run(monkeypatch):
    captured = {}

    def fake_run(self, video_path, params, force=False, **kwargs):
        captured["params"] = params
        captured.update(kwargs)
        raise RuntimeError("parar aqui")

    monkeypatch.setattr(EditPipeline, "run", fake_run)
    return captured


@pytest.fixture
def producer_factory(monkeypatch):
    factory = CountingFactory([_content()])
    monkeypatch.setattr(cli_module, "_build_producer", lambda args: factory())
    return factory


def _edit(video: Path, projects: Path, *extra: str) -> int:
    return handle_edit(build_parser().parse_args(["edit", str(video), "--projects-dir", str(projects), "--no-live", *extra]))


def test_cli_edit_uses_the_briefing(tmp_path, synthetic_video_with_audio, mock_vad, captured_run, producer_factory, capsys):
    projects = tmp_path / "projects"
    assert _edit(synthetic_video_with_audio, projects, "--pedido", "aula curta") == 1  # o fake_run para aqui
    params, settings = captured_run["params"], captured_run["settings"]
    assert params.brief == UserBrief(intensity="medium", target_duration_sec=2.0, instructions="tire as gaguejadas")
    assert params.pacing == "tight"
    assert settings.briefing == "gerado" and settings.intensity.origin == "briefing" and settings.recipe.origin == "briefing"
    out = capsys.readouterr().out
    assert "Entendi: aula sem enrolação." in out
    assert "Briefing: gerado (IA;" in out
    assert "  Intensidade: medium (briefing)" in out
    assert "  Duração alvo (informada ao Diretor): 2s (briefing)" in out
    assert "  Instruções ao Diretor: tire as gaguejadas (briefing)" in out

    # opção explícita da CLI vale mais; o briefing é reaproveitado (mesmo pedido)
    assert _edit(
        synthetic_video_with_audio, projects, "--pedido", "aula curta", "--intensidade", "aggressive", "--ritmo", "natural",
        "--instructions", "corte as piadas",
    ) == 1
    params, settings = captured_run["params"], captured_run["settings"]
    assert producer_factory.calls == 1 and settings.briefing == "reaproveitado"
    assert (params.brief.intensity, params.pacing, params.brief.instructions) == ("aggressive", "natural", "corte as piadas")
    assert (settings.intensity.origin, settings.pace.origin, settings.instructions.origin) == ("cli", "cli", "cli")
    assert settings.target_duration_sec.origin == "briefing"
    out = capsys.readouterr().out
    assert "  Ritmo das pausas: natural (opção da CLI)" in out and "[produtor]" not in out

    # pedido novo → Produtor de novo
    assert _edit(synthetic_video_with_audio, projects, "--pedido", "outra coisa") == 1
    assert producer_factory.calls == 2 and captured_run["settings"].briefing == "gerado"
    assert "briefing refeito: o pedido mudou" in capsys.readouterr().err


def test_cli_edit_sem_briefing(tmp_path, synthetic_video_with_audio, mock_vad, captured_run, producer_factory, capsys):
    projects = tmp_path / "projects"
    assert _edit(synthetic_video_with_audio, projects, "--pedido", "aula curta") == 1
    assert producer_factory.calls == 1
    capsys.readouterr()
    assert _edit(synthetic_video_with_audio, projects, "--pedido", "tire as piadas", "--sem-briefing") == 1
    params, settings = captured_run["params"], captured_run["settings"]
    assert producer_factory.calls == 1 and settings.briefing == "desligado" and not settings.uses_briefing
    # sem briefing, o pedido vai direto ao Diretor, como antes
    assert params.brief == UserBrief(instructions="tire as piadas") and params.pacing == "normal"
    assert settings.instructions.origin == "cli" and settings.intensity.origin == "padrao"
    assert "Entendi" not in capsys.readouterr().out
    # o briefing.json continua lá, intacto
    assert Briefing.load(briefing_location(synthetic_video_with_audio, projects)).request == "aula curta"


def test_cli_edit_without_request_is_unchanged(tmp_path, synthetic_video_with_audio, captured_run, producer_factory, capsys):
    projects = tmp_path / "projects"
    assert _edit(synthetic_video_with_audio, projects) == 1
    assert captured_run["params"].brief == UserBrief() and captured_run["params"].pacing == "normal"
    assert captured_run["settings"].briefing == "sem_briefing" and producer_factory.calls == 0
    assert not briefing_location(synthetic_video_with_audio, projects).exists()
    assert "Entendi" not in capsys.readouterr().out


def test_cli_edit_refuses_a_briefing_recipe_for_other_material(
    tmp_path, synthetic_video_with_audio, mock_vad, captured_run, monkeypatch, capsys
):
    factory = CountingFactory([_content(recipe="vlog", target_duration_sec=None)])
    monkeypatch.setattr(cli_module, "_build_producer", lambda args: factory())
    projects = tmp_path / "projects"
    assert _edit(synthetic_video_with_audio, projects, "--pedido", "vlog") == 1
    err = capsys.readouterr().err
    assert "o briefing (" in err and "escolheu a receita 'vlog', que edita uma pasta de clipes, mas recebido um vídeo" in err
    assert "--receita" in err and "--sem-briefing" in err
    assert "params" not in captured_run  # recusado antes do pipeline
    # --receita resolve sem refazer o briefing
    assert _edit(synthetic_video_with_audio, projects, "--pedido", "vlog", "--receita", "fala") == 1
    assert captured_run["settings"].recipe.origin == "cli" and factory.calls == 1


def test_cli_edit_refuses_coverage_recipe_without_visual(tmp_path, captured_run, monkeypatch, capsys):
    folder = tmp_path / "Viagem"
    folder.mkdir()
    projects = tmp_path / "projects"
    briefing = _briefing(recipe="vlog_cobertura", material=_material("Viagem", clips=3))
    monkeypatch.setattr(
        cli_module, "_edit_briefing",
        lambda args, video: EditBriefing(briefing, "reaproveitado", briefing_location(folder, projects)),
    )
    assert _edit(folder, projects) == 1
    assert "escolheu a receita 'vlog_cobertura', que precisa da análise visual" in capsys.readouterr().err


# --- execução de verdade: origem no project.json/relatório e cache ---


@pytest.mark.slow
def test_edit_records_origins_and_same_briefing_keeps_the_cache(
    tmp_path, synthetic_video_with_audio, mock_vad, monkeypatch, capsys
):
    monkeypatch.setattr(
        EditPipeline, "transcription_provider", property(lambda self: MockTranscriptionProvider(["Olá pessoal.", "Tchau."]))
    )
    projects = tmp_path / "projects"
    video = synthetic_video_with_audio
    base = ["--transcribe", "--director", "mock", "--pedido", "tire as pausas"]
    assert _edit(video, projects, *base) == 0
    out = capsys.readouterr().out
    assert "[8/8] Relatório: ok" in out

    project_dir = briefing_location(video, projects).parent
    manifest = ProjectManifest.load(project_dir / "project.json")
    settings = manifest.settings
    assert settings.briefing == "gerado" and settings.recipe.origin == "briefing"
    assert settings.intensity.origin == "briefing" and settings.pace.origin == "briefing"
    raw = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    assert raw["settings"]["instructions"] == {"value": "tire as pausas", "origin": "briefing"}
    report = (project_dir / "edit_report.md").read_text(encoding="utf-8")
    assert "## Briefing" in report and "| Intensidade | light | briefing |" in report
    assert "| Pedido | tire as pausas | briefing.json (IA) |" in report

    # o comando `briefing` reescreve um briefing igual (outra data): nada do edit é refeito
    args = build_parser().parse_args(
        ["briefing", str(video), "--director-provider", "mock", "--pedido", "tire as pausas", "--projects-dir", str(projects)]
    )
    assert handle_briefing(args) == 0
    capsys.readouterr()
    assert _edit(video, projects, *base) == 0
    out = capsys.readouterr().out
    assert out.count(": em cache") == 8, out

    # o ritmo muda o corte: plano em diante refeitos; transcrição e Diretor continuam no cache
    assert _edit(video, projects, *base, "--ritmo", "tight") == 0
    out = capsys.readouterr().out
    assert "[3/8] Transcrição: em cache" in out and "[4/8] Diretor: em cache" in out
    assert "[5/8] Plano de corte: ok" in out
    manifest = ProjectManifest.load(project_dir / "project.json")
    assert manifest.params.pacing == "tight" and manifest.settings.pace.origin == "cli"
    assert manifest.versions[-1].pacing == "tight"
    assert "| Ritmo das pausas | tight (padding e min_gap ajustados) |" in (project_dir / "edit_report.md").read_text(
        encoding="utf-8"
    )


def test_coverage_ignores_briefing_json_when_the_edit_does_not_use_it(tmp_path):
    from types import SimpleNamespace

    from ai_video_editor.pipeline.spine import SpinePipeline

    _briefing(coverage_instructions="drone na abertura", roles_to_use=["principal"]).save(tmp_path / BRIEFING_FILE)
    pipeline = SpinePipeline(projects_root=tmp_path)
    used = pipeline._briefing(SimpleNamespace(project_dir=tmp_path, use_briefing=True))
    assert used == (["drone na abertura"], ["principal"])
    assert pipeline._briefing(SimpleNamespace(project_dir=tmp_path, use_briefing=False)) == ([], [])
