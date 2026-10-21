"""Testes das receitas (combinações validadas de peças) e da numeração de etapas derivada delas."""

import io

import pytest

from ai_video_editor.cli import EditProgressPrinter, build_parser, handle_edit
from ai_video_editor.live_status import LiveStatus
from ai_video_editor.pipeline import STAGES, EditPipeline
from ai_video_editor.recipes import PIECES, Recipe, RecipeError, get_recipe, recipe_names


def test_fala_is_exactly_the_original_edit_stages():
    fala = get_recipe("fala")
    assert fala.steps == list(STAGES)
    assert set(PIECES) == set(STAGES)
    assert "fala" in recipe_names()
    assert fala.position("director") == 4
    assert fala.dependencies("plan") == ["probe", "silences", "director"]


def test_unknown_recipe_lists_available():
    with pytest.raises(RecipeError, match="Receita desconhecida: 'gameplay'.*fala"):
        get_recipe("gameplay")


@pytest.mark.parametrize(
    "steps, requires, message",
    [
        (["probe", "voar"], {}, "etapa desconhecida 'voar'"),
        (["probe", "probe"], {}, "etapa repetida: probe"),
        (["probe", "silences", "plan"], {"render": ["plan"]}, "cita a etapa 'render', que não está"),
        (["probe", "silences"], {"silences": ["teletransporte"]}, "depende de 'teletransporte', que não existe"),
        (["probe", "plan"], {}, "dependência faltando: 'plan' precisa de 'silences'"),
        (["probe", "transcription", "director"], {"transcription": ["transcription"]}, "depende de si mesma"),
        (["silences", "probe"], {}, "ordem impossível: 'silences' precisa vir depois de 'probe'"),
        (
            ["probe", "silences", "plan", "render"],
            {"silences": ["render"]},
            "dependência circular entre 'silences' e 'render'",
        ),
    ],
)
def test_invalid_recipes_fail_with_clear_message(steps, requires, message):
    with pytest.raises(ValueError, match=message):
        Recipe(name="teste", description="x", steps=steps, requires=requires)


def test_pipeline_receives_recipe_by_name_or_object(tmp_path):
    assert EditPipeline(projects_root=tmp_path).recipe.name == "fala"
    custom = Recipe(name="so-metadados", description="x", steps=["probe"])
    assert EditPipeline(projects_root=tmp_path, recipe=custom).recipe is custom
    with pytest.raises(RecipeError):
        EditPipeline(projects_root=tmp_path, recipe="nao-existe")


def test_progress_numbering_comes_from_recipe():
    out = io.StringIO()
    printer = EditProgressPrinter(LiveStatus(stream=out, live=False))
    printer("director", "skipped", "use --director para ativar")
    short = Recipe(name="curta", description="x", steps=["probe", "silences", "plan", "render"])
    EditProgressPrinter(LiveStatus(stream=out, live=False), short)("render", "cached", "")
    assert out.getvalue().splitlines() == [
        "[4/8] Diretor: pulada (use --director para ativar)",
        "[4/4] Renderização: em cache",
    ]


def test_cli_edit_receita_is_validated(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    args = build_parser().parse_args(["edit", str(video), "--receita", "vlog"])
    assert handle_edit(args) == 1
    assert "Receita desconhecida: 'vlog'" in capsys.readouterr().err
    assert build_parser().parse_args(["edit", str(video)]).receita == "fala"
