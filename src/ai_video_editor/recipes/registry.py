"""Registro de receitas: nome → `Recipe` validada."""

from ai_video_editor.recipes.schema import Recipe, RecipeError

DEFAULT_RECIPE = "fala"

_RECIPES: dict[str, Recipe] = {}


def register_recipe(recipe: Recipe) -> None:
    """Registra (ou substitui) uma receita; ela já foi validada ao ser criada."""
    _RECIPES[recipe.name] = recipe


def recipe_names() -> list[str]:
    return sorted(_RECIPES)


def get_recipe(name: str) -> Recipe:
    try:
        return _RECIPES[name]
    except KeyError:
        raise RecipeError(
            f"Receita desconhecida: '{name}' (receitas disponíveis: {', '.join(recipe_names())})."
        ) from None


# Vídeo falado (tutorial, aula, pessoa falando): exatamente as etapas do `edit` original.
register_recipe(
    Recipe(
        name="fala",
        description="Vídeo falado: corta pausas e, com o Diretor, frases repetidas ou recomeçadas",
        steps=["probe", "silences", "transcription", "director", "plan", "render", "subtitles", "report"],
        requires={
            # O plano aplica as remoções do Diretor (quando ele roda).
            "plan": ["director"],
            # O relatório compara a transcrição e as decisões com o plano final.
            "report": ["transcription", "director", "subtitles"],
        },
    )
)
