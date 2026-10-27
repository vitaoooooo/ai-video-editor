"""Registro de receitas: nome → `Recipe` validada."""

from ai_video_editor.recipes.schema import Recipe, RecipeError

DEFAULT_RECIPE = "fala"
# Receita padrão de `edit <pasta>` (projeto com vários clipes) e a usada quando a pasta já tem a
# análise visual (`visual <pasta>`), que acrescenta a cobertura.
DEFAULT_FOLDER_RECIPE = "vlog"
COVERAGE_FOLDER_RECIPE = "vlog_cobertura"

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
            "silences": ["probe"],
            "transcription": ["probe"],
            # O plano aplica as remoções do Diretor (quando ele roda).
            "plan": ["director"],
            "render": ["plan"],
            "subtitles": ["plan"],
            # O relatório compara a transcrição e as decisões com o plano final.
            "report": ["transcription", "director", "subtitles"],
        },
    )
)


# Vlog / pasta de clipes: a espinha falada (clipes principais com fala, em ordem, cada um sem pausas e
# sem frases ruins) montada numa Timeline. Cobertura (drone/b-roll por cima) ainda não entra.
register_recipe(
    Recipe(
        name="vlog",
        description=(
            "Pasta de clipes (vlog, viagem, evento): junta os clipes principais com fala, na ordem do "
            "catálogo, cortando pausas, frases ruins e tomadas repetidas; sem cobertura de b-roll/drone"
        ),
        steps=["catalog", "silences", "transcription", "director", "timeline", "render", "subtitles", "report"],
        multi_clip=True,
        requires={
            "silences": ["catalog"],
            "transcription": ["catalog"],
            # A Timeline aplica as remoções do Diretor (quando ele roda).
            "timeline": ["director"],
            "render": ["timeline"],
            "subtitles": ["timeline"],
            "report": ["transcription", "director", "subtitles"],
        },
    )
)


# Vlog com cobertura: a espinha da `vlog` + a análise visual já feita (`visual <pasta>`) + o Diretor de
# cobertura (B-roll/drone/POV por cima da fala, respiros entre as falas), entre a Timeline e o render.
register_recipe(
    Recipe(
        name="vlog_cobertura",
        description=(
            "Pasta de clipes com análise visual (vlog, viagem): a espinha falada da `vlog` mais cobertura de "
            "drone/b-roll/POV por cima da fala (obrigatória onde a imagem do principal é ruim) e respiros"
        ),
        steps=[
            "catalog", "silences", "transcription", "director", "timeline", "visual", "coverage",
            "render", "subtitles", "report",
        ],
        multi_clip=True,
        requires={
            "silences": ["catalog"],
            "transcription": ["catalog"],
            "timeline": ["director"],
            # Cobertura escolhe frases por ID: precisa da transcrição (e das remoções do Diretor).
            "coverage": ["transcription", "director"],
            "render": ["coverage"],
            "subtitles": ["coverage"],
            "report": ["transcription", "director", "subtitles"],
        },
    )
)


# Editor de IA (qualquer material: pasta ou um vídeo = 1 clipe): o roteiro unificado (todos os clipes, de
# qualquer papel) e a IA decidindo tudo por IDs em passadas (resumos, estrutura, detalhe por bloco); o código
# só valida e converte em Timeline. Fora do Produtor até ser comparada com a `vlog_cobertura` no material real.
register_recipe(
    Recipe(
        name="editor_ia",
        description=(
            "Editor de IA para qualquer material (pasta de clipes ou um vídeo): roteiro unificado e a IA decidindo "
            "ordem, cortes, coberturas e respiros por IDs, em passadas"
        ),
        steps=["catalog", "roteiro", "editor", "timeline", "render", "subtitles", "report"],
        multi_clip=True,
        single_file=True,
        producer=False,
        requires={
            "timeline": ["editor"],
            "render": ["timeline"],
            "subtitles": ["timeline"],
            "report": ["editor", "subtitles"],
        },
    )
)
