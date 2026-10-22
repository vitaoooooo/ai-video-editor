"""Testes do Produtor: resumo do material, validação, nova tentativa, fallback, --receita e o comando `briefing`."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from ai_video_editor.catalog import MockSpeechDetector
from ai_video_editor.catalog import builder as builder_module
from ai_video_editor.catalog.schema import CatalogClip, ClipMedia, ColorSignals, SpeechDetection
from ai_video_editor.cli import build_parser, handle_briefing
from ai_video_editor.intelligence.exceptions import AIAuthenticationError
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.schema import ModelSpec
from ai_video_editor.pipeline.project import file_hash, project_dir_for
from ai_video_editor.producer import (
    BRIEFING_FILE,
    Briefing,
    BriefingContent,
    registered_recipes,
    run_producer,
    summarize_clips,
    validate_content,
)
from ai_video_editor.producer import material as material_module
from ai_video_editor.producer.prompt import build_producer_prompt
from ai_video_editor.producer.providers.llm import LLMProducer
from ai_video_editor.producer.providers.mock import MockProducer
from ai_video_editor.producer.schema import ProducerContext
from ai_video_editor.recipes import Recipe, RecipeError, register_recipe
from ai_video_editor.recipes import registry as recipe_registry
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path


class MultiClipRecipe(Recipe):
    """Como uma receita de vários clipes se declara para o Produtor."""

    multi_clip: bool = True


@pytest.fixture
def vlog_recipe():
    """Receita `vlog` registrada em tempo de execução (outra tarefa registra a real)."""
    previous = recipe_registry._RECIPES.get("vlog")
    register_recipe(
        MultiClipRecipe(
            name="vlog", description="Vlog: vários clipes principais com cobertura",
            steps=["probe", "silences", "transcription", "director", "plan", "render", "report"],
        )
    )
    yield
    if previous is None:
        recipe_registry._RECIPES.pop("vlog", None)
    else:
        register_recipe(previous)


@pytest.fixture(autouse=True)
def _without_vlog(request):
    """Sem a fixture `vlog_recipe` (ou `real_vlog`), o teste roda sem `vlog` registrada."""
    if {"vlog_recipe", "real_vlog"} & set(request.fixturenames):
        yield
        return
    saved = recipe_registry._RECIPES.pop("vlog", None)
    yield
    if saved is not None:
        register_recipe(saved)


def _clip(role, n, duration=60.0, speech=0.8, audio=True, folder=None):
    status = "analisada" if audio and speech is not None else ("desconhecido" if audio else "sem_audio")
    return CatalogClip(
        id=f"{role[0].upper()}{n:03d}", role=role, folder=folder or role, relative_path=f"{role}/c{n}.mp4",
        size_bytes=1,
        media=ClipMedia(duration_sec=duration, width=1920, height=1080, fps=30, codec="h264", has_audio=audio),
        speech=SpeechDetection(
            status=status, ratio=speech if status == "analisada" else None,
            speech_sec=duration * speech if status == "analisada" else None,
        ),
        color=ColorSignals(frames=3, luma_spread=0.2, saturation=0.02) if role == "broll" else None,
    )


def _vlog_material():
    clips = [_clip("principal", n) for n in range(1, 4)]
    clips += [_clip("drone", 1, duration=30, audio=False), _clip("broll", 1, duration=20, speech=0.0)]
    return summarize_clips(clips, "pasta", "Viagem")


def _single_material(speech=0.8):
    return summarize_clips([_clip("principal", 1, duration=300, speech=speech)], "arquivo", "aula.mp4")


def _content(**changes):
    data = dict(
        video_type="vlog de viagem", recipe="vlog", understood="Entendi: vlog de viagem curto.",
        intensity="medium", pace="tight", director_instructions="tire as repetições",
        coverage_instructions="abrir com drone", roles_to_use=["principal", "drone"],
    )
    data.update(changes)
    return BriefingContent(**data)


# --- resumo e receitas ---


def test_summary_is_per_role_without_paths():
    material = _vlog_material()
    assert material.kind == "pasta" and material.clips == 5 and material.duration_sec == 230
    principal = material.role("principal")
    assert (principal.clips, principal.duration_sec, principal.clips_with_speech, principal.speech_share) == (3, 180, 3, 0.8)
    assert principal.formats == ["1920x1080 30fps h264"]
    drone = material.role("drone")
    assert drone.speech_share is None and drone.clips_with_speech == 0 and drone.speech_unknown == 0
    assert material.role("broll").log_suspects == 1 and material.role("broll").clips_with_speech == 0
    assert "c1.mp4" not in material.model_dump_json()


def test_recipes_are_read_from_the_registry_at_call_time(vlog_recipe):
    infos = {r.name: r for r in registered_recipes()}
    assert infos["fala"].multi_clip is False and infos["fala"].needs_speech is True
    assert infos["vlog"].multi_clip is True and infos["vlog"].needs_speech is True
    register_recipe(Recipe(name="paisagem", description="Só imagens", steps=["probe", "silences", "plan", "render"]))
    try:
        assert registered_recipes()[-1].name == "vlog"
        assert next(r for r in registered_recipes() if r.name == "paisagem").needs_speech is False
    finally:
        recipe_registry._RECIPES.pop("paisagem")


def test_prompt_has_summary_recipes_request_and_previous_error(vlog_recipe):
    prompt = build_producer_prompt(
        "vlog de 5 minutos", _vlog_material(), registered_recipes(),
        ProducerContext(forced_recipe="vlog", previous_error="a receita 'x' não existe"),
    )
    assert "papel principal: 3 clipe(s), 3min00s, fala em 80% do tempo, 3/3 clipes com fala" in prompt
    assert "papel drone: 1 clipe(s), 0min30s, sem áudio" in prompt
    assert "1 com suspeita de perfil log" in prompt
    assert "- vlog: Vlog: vários clipes principais com cobertura (vários clipes; precisa de fala)" in prompt
    assert "- fala: " in prompt and "(1 vídeo; precisa de fala)" in prompt
    assert "Use exatamente ela" in prompt and "a receita 'x' não existe" in prompt
    assert prompt.rstrip().endswith("Responda em JSON no schema pedido.")
    assert "vlog de 5 minutos" in prompt and "c1.mp4" not in prompt


# --- validação ---


def test_validation_recipe_roles_duration_and_compatibility(vlog_recipe):
    recipes = registered_recipes()
    material = _vlog_material()
    assert validate_content(_content(), material, recipes) == []
    assert validate_content(_content(recipe="cinema"), material, recipes) == [
        "a receita 'cinema' não existe (disponíveis: fala, vlog)"
    ]
    problems = validate_content(_content(recipe="fala"), material, recipes)
    assert problems == [
        "a receita 'fala' edita 1 vídeo, mas o material tem 3 clipes com fala; escolha uma receita de vários clipes"
    ]
    problems = validate_content(_content(roles_to_use=["principal", "pov"]), material, recipes)
    assert problems == ["papéis que não existem no material: pov (existem: principal, drone, broll)"]
    # principal + drone = 210s
    assert "não é menor que o material a usar (210s)" in validate_content(
        _content(target_duration_sec=210), material, recipes
    )[0]
    assert validate_content(_content(target_duration_sec=120), material, recipes) == []

    no_speech = summarize_clips([_clip("drone", 1, audio=False), _clip("drone", 2, audio=False)], "pasta", "Drone")
    assert validate_content(_content(recipe="vlog", roles_to_use=[]), no_speech, recipes) == [
        "a receita 'vlog' precisa de fala, mas nenhum clipe do material tem fala"
    ]


# --- execução ---


@pytest.fixture
def real_vlog():
    """A receita `vlog` de verdade, registrada por `recipes/registry.py` (espinha falada multiclipe)."""
    recipe = recipe_registry.get_recipe("vlog")
    assert recipe.multi_clip and "catalog" in recipe.steps
    return recipe


def test_real_vlog_recipe_is_accepted_for_folder_with_several_speaking_clips(real_vlog):
    """Pasta com 3 principais falados: a `vlog` real (multi_clip=True) é válida e vem da IA, sem fallback."""
    material = _vlog_material()
    assert material.role("principal").clips_with_speech == 3
    assert validate_content(_content(), material, registered_recipes()) == []
    briefing = run_producer(MockProducer(responses=[_content()]), "vlog de viagem curto", material)
    assert briefing.origin == "ia" and briefing.origin_reason == "" and briefing.recipe == "vlog"
    # A `fala` (1 vídeo) não serve para vários clipes com fala.
    assert validate_content(_content(recipe="fala"), material, registered_recipes())


def test_valid_briefing_from_ai(vlog_recipe):
    producer = MockProducer(responses=[_content(understood="vlog curto com drone")])
    briefing = run_producer(producer, "vlog curto, abre com drone", _vlog_material())
    assert briefing.origin == "ia" and briefing.origin_reason == ""
    assert briefing.recipe == "vlog" and briefing.understood == "Entendi: vlog curto com drone"
    assert (briefing.intensity, briefing.pace) == ("medium", "tight")
    assert briefing.request == "vlog curto, abre com drone"
    assert len(producer.calls) == 1 and len(briefing.attempts) == 1 and briefing.attempts[0].success
    estimate = briefing.estimate
    assert (estimate.clips, estimate.clips_to_transcribe, estimate.audio_to_transcribe_sec) == (4, 3, 180)
    assert (estimate.ai_steps, estimate.ai_calls, estimate.ai_calls_max, estimate.producer_calls) == (["director"], 1, 2, 1)


def test_missing_roles_default_to_all_roles(vlog_recipe):
    briefing = run_producer(MockProducer(responses=[_content(roles_to_use=[])]), "x", _vlog_material())
    assert briefing.roles_to_use == ["principal", "drone", "broll"]


def test_unknown_recipe_retries_with_error_then_falls_back(vlog_recipe):
    producer = MockProducer(responses=[_content(recipe="cinema")])
    briefing = run_producer(producer, "faz um vlog animado", _vlog_material())
    assert len(producer.calls) == 2
    assert producer.calls[0].previous_error is None
    assert "a receita 'cinema' não existe" in producer.calls[1].previous_error
    assert briefing.origin == "fallback" and "resposta inválida em 2 tentativas" in briefing.origin_reason
    assert briefing.recipe == "vlog" and briefing.director_instructions == "faz um vlog animado"
    assert briefing.warnings[0].startswith("Briefing sem IA: resposta inválida")
    assert [a.success for a in briefing.attempts] == [False, False]


def test_retry_fixes_the_answer(vlog_recipe):
    producer = MockProducer(responses=[_content(recipe="cinema"), _content()])
    briefing = run_producer(producer, "vlog", _vlog_material())
    assert briefing.origin == "ia" and [a.success for a in briefing.attempts] == [False, True]


def test_incompatible_recipe_falls_back_to_fala_with_warning_without_vlog():
    producer = MockProducer(responses=[_content(recipe="fala")])
    briefing = run_producer(producer, "vlog", _vlog_material())
    assert briefing.origin == "fallback" and briefing.recipe == "fala"
    assert "edita 1 vídeo, mas o material tem 3 clipes com fala" in briefing.origin_reason
    assert any("A receita 'vlog' não está registrada; usando 'fala'" in w for w in briefing.warnings)
    assert briefing.roles_to_use == ["principal"]


def test_material_without_speech_falls_back_with_warning():
    material = summarize_clips([_clip("drone", n, audio=False) for n in (1, 2)], "pasta", "Drone")
    briefing = run_producer(MockProducer(responses=[_content(recipe="fala", roles_to_use=[])]), "", material)
    assert briefing.origin == "fallback" and briefing.recipe == "fala"
    assert "precisa de fala, mas nenhum clipe do material tem fala" in briefing.origin_reason
    assert any("Nenhum clipe principal com fala" in w for w in briefing.warnings)


def test_empty_request_and_single_spoken_video_skips_ai():
    producer = MockProducer(responses=[AssertionError("não devia chamar a IA")])
    briefing = run_producer(producer, "  ", _single_material())
    assert producer.calls == [] and briefing.attempts == []
    assert briefing.origin == "fallback" and "pedido vazio e 1 vídeo falado" in briefing.origin_reason
    assert briefing.recipe == "fala" and briefing.roles_to_use == ["principal"]
    assert briefing.understood.startswith("Entendi: editar o vídeo falado")
    assert not any(w.startswith("Briefing sem IA") for w in briefing.warnings)
    assert briefing.estimate.audio_to_transcribe_sec == 300 and briefing.estimate.producer_calls == 0


def test_empty_request_with_a_folder_still_asks_the_ai(vlog_recipe):
    producer = MockProducer()
    briefing = run_producer(producer, "", _vlog_material())
    assert len(producer.calls) == 1 and briefing.origin == "ia" and briefing.recipe == "vlog"


def test_forced_recipe_overrides_ai_choice_and_warns_if_incompatible(vlog_recipe):
    producer = MockProducer(responses=[_content(recipe="vlog")])
    briefing = run_producer(producer, "só a fala", _vlog_material(), forced_recipe="fala")
    assert briefing.origin == "ia" and briefing.recipe == "fala" and briefing.forced_recipe == "fala"
    assert producer.calls[0].forced_recipe == "fala" and len(producer.calls) == 1
    assert any(w.startswith("Receita escolhida com --receita: a receita 'fala' edita 1 vídeo") for w in briefing.warnings)
    with pytest.raises(RecipeError, match="Receita desconhecida: 'cinema'"):
        run_producer(producer, "", _vlog_material(), forced_recipe="cinema")


def test_forced_recipe_in_fallback(vlog_recipe):
    briefing = run_producer(None, "corta tudo", _vlog_material(), forced_recipe="vlog")
    assert briefing.origin == "fallback" and briefing.recipe == "vlog" and briefing.forced_recipe == "vlog"
    assert "--director-provider none" in briefing.origin_reason


def test_provider_error_falls_back_with_warning():
    producer = MockProducer(responses=[AIAuthenticationError("gemini", details="Variável de ambiente GEMINI_API_KEY não encontrada.")])
    briefing = run_producer(producer, "aula sem enrolação", _single_material())
    assert len(producer.calls) == 1 and briefing.origin == "fallback" and briefing.recipe == "fala"
    assert "IA indisponível (Variável de ambiente GEMINI_API_KEY não encontrada.)" in briefing.origin_reason
    assert briefing.warnings[0].startswith("Briefing sem IA: IA indisponível")


def test_llm_producer_through_model_chain(vlog_recipe):
    client = MockLLMClient(responses=[{"video_type": "x", "recipe": "nada", "understood": "Entendi: x"}, _content()])
    chain = ModelChain([ModelSpec(provider="mock", model_id="m1")], clients={"mock": client})
    briefing = run_producer(LLMProducer(chain), "vlog", _vlog_material())
    assert briefing.origin == "ia" and briefing.provider == "mock" and briefing.model_chain == ["mock:m1"]
    assert briefing.effective_model == "mock:m1"
    assert len(client.calls) == 2 and "a receita 'nada' não existe" in client.calls[1][0]


def test_briefing_round_trip(tmp_path, vlog_recipe):
    briefing = run_producer(MockProducer(responses=[_content()]), "vlog", _vlog_material())
    briefing.save(tmp_path / BRIEFING_FILE)
    loaded = Briefing.load(tmp_path / BRIEFING_FILE)
    assert loaded == briefing and json.loads((tmp_path / BRIEFING_FILE).read_text(encoding="utf-8"))["schema_version"] == 1


# --- comando briefing ---


def test_cli_briefing_single_file(tmp_path, synthetic_video_with_audio, capsys, monkeypatch):
    monkeypatch.setattr(material_module, "FasterWhisperVAD", lambda: MockSpeechDetector(segments=[(0.0, 2.5)]))
    projects = tmp_path / "projects"
    args = build_parser().parse_args(
        ["briefing", str(synthetic_video_with_audio), "--director-provider", "mock", "--projects-dir", str(projects)]
    )
    assert handle_briefing(args) == 0
    out = capsys.readouterr().out
    assert "Entendi: editar o vídeo falado" in out
    assert "Receita: fala (regra sem IA: pedido vazio e 1 vídeo falado; a IA não foi chamada)" in out
    assert "Metas: duração livre | intensidade light | ritmo normal" in out
    assert "Estimativa: 1 clipe(s); 3,0s de áudio a transcrever (1 clipe(s)); 1 chamada(s) de IA" in out
    expected = project_dir_for(synthetic_video_with_audio, file_hash(synthetic_video_with_audio), projects).resolve()
    briefing = Briefing.load(expected / BRIEFING_FILE)
    assert briefing.material.kind == "arquivo" and briefing.material.name == synthetic_video_with_audio.name

    # com pedido, a IA (mock) é chamada; --receita força a receita
    args = build_parser().parse_args(
        ["briefing", str(synthetic_video_with_audio), "--director-provider", "mock", "--projects-dir", str(projects),
         "--pedido", "tire as gaguejadas", "--receita", "fala"]
    )
    assert handle_briefing(args) == 0
    out = capsys.readouterr().out
    assert "Receita: fala (--receita; resto por IA (mock))" in out
    assert "Diretor de cortes: tire as gaguejadas" in out
    assert Briefing.load(expected / BRIEFING_FILE).request == "tire as gaguejadas"


def test_cli_briefing_rejects_bad_input(tmp_path, synthetic_video_with_audio, capsys):
    parser = build_parser()
    assert handle_briefing(parser.parse_args(["briefing", str(tmp_path / "nao-existe")])) == 1
    assert "não encontrado" in capsys.readouterr().err
    args = parser.parse_args(["briefing", str(synthetic_video_with_audio), "--receita", "cinema"])
    assert handle_briefing(args) == 1
    assert "Receita desconhecida: 'cinema'" in capsys.readouterr().err


def _make_clip(path: Path, duration: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(get_ffmpeg_path()), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=160x120:rate=15",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac", "-shortest",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    os.utime(path, (1_700_000_000, 1_700_000_000))


@pytest.mark.slow
def test_cli_briefing_folder_builds_then_reuses_catalog(tmp_path, capsys, monkeypatch, vlog_recipe):
    detector = MockSpeechDetector(segments=[(0.0, 1.0)])
    monkeypatch.setattr(builder_module, "FasterWhisperVAD", lambda: detector)
    root = tmp_path / "Viagem"
    _make_clip(root / "Osmo Pocket - PRINCIPAL" / "OsmoPocket_FullHD (4).mp4")
    _make_clip(root / "Osmo Pocket - PRINCIPAL" / "OsmoPocket_FullHD (15).mp4")
    _make_clip(root / "Imagens de Drone" / "Drone (1).mp4")
    argv = ["briefing", str(root), "--director-provider", "none", "--projects-dir", str(tmp_path / "projects"), "--no-live"]

    assert handle_briefing(build_parser().parse_args(argv)) == 0
    out = capsys.readouterr().out
    assert "[4/4] Catálogo: ok" in out and "Catálogo (gerado):" in out
    assert "Receita: vlog (regra sem IA: Produtor de IA desligado (--director-provider none))" in out
    assert "Papéis: principal, drone" in out
    calls = len(detector.calls)

    assert handle_briefing(build_parser().parse_args(argv)) == 0
    out = capsys.readouterr().out
    assert "Catálogo (reaproveitado):" in out and "[1/4]" not in out and len(detector.calls) == calls
    briefing_path = next((tmp_path / "projects").glob("viagem-*/briefing.json"))
    assert (briefing_path.parent / "catalog" / "catalog.json").is_file()
    assert Briefing.load(briefing_path).material.role("principal").clips == 2
