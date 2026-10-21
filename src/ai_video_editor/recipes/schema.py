"""Peças (etapas) conhecidas e o modelo `Recipe`, uma combinação validada de peças.

Uma receita é dado, não código: lista ordenada de etapas e as dependências entre elas. Só entram
peças do catálogo `PIECES`; o pipeline executa cada peça com código determinístico. Quem escolher
receitas (o Produtor, na rodada 2) nunca inventa etapas.
"""

from pydantic import BaseModel, Field, model_validator


class Piece(BaseModel):
    """Uma etapa que o código sabe executar."""

    name: str
    description: str
    needs: list[str] = Field(
        default_factory=list, description="Etapas que esta peça sempre exige antes dela, em qualquer receita"
    )


PIECES: dict[str, Piece] = {
    p.name: p
    for p in (
        Piece(name="probe", description="Metadados do vídeo (ffprobe) e rotação"),
        Piece(name="silences", description="Detecção de silêncios no áudio", needs=["probe"]),
        Piece(name="transcription", description="Transcrição com tempos por palavra", needs=["probe"]),
        Piece(name="director", description="Diretor de IA: frases a remover, por ID", needs=["transcription"]),
        Piece(name="plan", description="Plano de corte (EditPlan)", needs=["probe", "silences"]),
        Piece(name="render", description="Renderização com FFmpeg", needs=["plan"]),
        Piece(name="subtitles", description="Legendas .srt no tempo editado", needs=["transcription", "plan"]),
        Piece(name="report", description="Relatório, transcrição comparativa e linha do tempo", needs=["render"]),
    )
}


class RecipeError(ValueError):
    """Receita inválida ou inexistente."""


class Recipe(BaseModel):
    """Combinação ordenada de peças. `requires[etapa]` lista as etapas que precisam vir antes dela."""

    name: str = Field(min_length=1)
    description: str
    steps: list[str] = Field(min_length=1)
    requires: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Recipe":
        check_recipe(self)
        return self

    def dependencies(self, step: str) -> list[str]:
        """Dependências da etapa: as da peça somadas às declaradas na receita."""
        merged = list(PIECES[step].needs) if step in PIECES else []
        merged += [d for d in self.requires.get(step, []) if d not in merged]
        return merged

    def position(self, step: str) -> int:
        """Número da etapa (a partir de 1), para `[4/8] Diretor`."""
        return self.steps.index(step) + 1


def check_recipe(recipe: Recipe) -> None:
    """Valida peças, dependências e ordem; erro em português dizendo o que corrigir."""
    prefix = f"Receita '{recipe.name}'"
    known = ", ".join(PIECES)
    for step in recipe.steps:
        if step not in PIECES:
            raise RecipeError(f"{prefix}: etapa desconhecida '{step}' (etapas conhecidas: {known}).")
    duplicated = sorted({s for s in recipe.steps if recipe.steps.count(s) > 1})
    if duplicated:
        raise RecipeError(f"{prefix}: etapa repetida: {', '.join(duplicated)}.")
    for step, deps in recipe.requires.items():
        if step not in recipe.steps:
            raise RecipeError(f"{prefix}: 'requires' cita a etapa '{step}', que não está em 'steps'.")
        for dep in deps:
            if dep not in PIECES:
                raise RecipeError(f"{prefix}: a etapa '{step}' depende de '{dep}', que não existe (etapas conhecidas: {known}).")

    order = {step: i for i, step in enumerate(recipe.steps)}
    for step in recipe.steps:
        for dep in recipe.dependencies(step):
            if dep == step:
                raise RecipeError(f"{prefix}: a etapa '{step}' depende de si mesma.")
            if dep not in order:
                raise RecipeError(f"{prefix}: dependência faltando: '{step}' precisa de '{dep}', que não está na receita.")
            if order[dep] > order[step]:
                if _reaches(recipe, dep, step):
                    raise RecipeError(
                        f"{prefix}: ordem impossível: dependência circular entre '{step}' e '{dep}'."
                    )
                raise RecipeError(f"{prefix}: ordem impossível: '{step}' precisa vir depois de '{dep}'.")


def _reaches(recipe: Recipe, start: str, target: str) -> bool:
    """`start` depende (direta ou indiretamente) de `target`?"""
    seen: set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        for dep in recipe.dependencies(current):
            if dep == target:
                return True
            if dep not in seen:
                seen.add(dep)
                pending.append(dep)
    return False
