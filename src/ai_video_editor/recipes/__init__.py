"""Receitas: combinações validadas de peças (etapas) do pipeline."""

from ai_video_editor.recipes.registry import DEFAULT_RECIPE, get_recipe, recipe_names, register_recipe
from ai_video_editor.recipes.schema import PIECES, Piece, Recipe, RecipeError, check_recipe

__all__ = [
    "DEFAULT_RECIPE",
    "PIECES",
    "Piece",
    "Recipe",
    "RecipeError",
    "check_recipe",
    "get_recipe",
    "recipe_names",
    "register_recipe",
]
