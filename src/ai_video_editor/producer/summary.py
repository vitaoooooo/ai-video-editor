"""Entrada do Produtor: resumo compacto do catálogo e as receitas registradas.

O resumo nunca leva a lista de clipes nem caminhos: só, por papel, quantidade, duração, fala e
formatos. As receitas são lidas do registro na hora da chamada (outras tarefas registram receitas
novas sem mexer aqui).
"""

from typing import Literal

from ai_video_editor.catalog.schema import ROLES, CatalogClip
from ai_video_editor.catalog.speech import SPEECH_MIN_RATIO
from ai_video_editor.producer.schema import MaterialSummary, RecipeInfo, RoleSummary
from ai_video_editor.recipes import get_recipe, recipe_names
from ai_video_editor.recipes.schema import Recipe


def has_speech(clip: CatalogClip) -> bool:
    speech = clip.speech
    return speech is not None and speech.status == "analisada" and (speech.ratio or 0.0) >= SPEECH_MIN_RATIO


def speech_unknown(clip: CatalogClip) -> bool:
    """Clipe com áudio cuja fala não foi analisada (VAD indisponível): pode ter fala."""
    return clip.media is not None and clip.media.has_audio and (
        clip.speech is None or clip.speech.status == "desconhecido"
    )


def _format(clip: CatalogClip) -> str:
    m = clip.media
    return f"{m.resolution} {m.fps:g}fps {m.codec}"


def summarize_clips(clips: list[CatalogClip], kind: Literal["arquivo", "pasta"], name: str) -> MaterialSummary:
    """Resumo por papel dos clipes válidos (com ID)."""
    valid = [c for c in clips if c.error is None and c.media is not None]
    roles: list[RoleSummary] = []
    for role in ROLES:
        members = [c for c in valid if c.role == role]
        if not members:
            continue
        analyzed = [c for c in members if c.speech and c.speech.status == "analisada"]
        analyzed_sec = sum(c.media.duration_sec for c in analyzed)
        share = None
        if analyzed and analyzed_sec > 0:
            share = round(sum(c.speech.speech_sec or 0.0 for c in analyzed) / analyzed_sec, 3)
        speaking = [c for c in members if has_speech(c) or speech_unknown(c)]
        roles.append(
            RoleSummary(
                role=role,
                clips=len(members),
                duration_sec=round(sum(c.media.duration_sec for c in members), 2),
                speech_share=share,
                clips_with_speech=sum(1 for c in members if has_speech(c)),
                speech_unknown=sum(1 for c in members if speech_unknown(c)),
                speech_clips_duration_sec=round(sum(c.media.duration_sec for c in speaking), 2),
                formats=sorted({_format(c) for c in members}),
                log_suspects=sum(1 for c in members if c.color and c.color.log_suspect),
            )
        )
    return MaterialSummary(
        kind=kind, name=name, clips=len(valid),
        duration_sec=round(sum(c.media.duration_sec for c in valid), 2), roles=roles,
    )


def recipe_info(recipe: Recipe) -> RecipeInfo:
    """Como o Produtor enxerga uma receita.

    Uma receita que edita vários clipes declara o atributo `multi_clip = True`; sem ele, é tratada
    como receita de 1 vídeo (como a `fala`). Precisa de fala quando declara `needs_speech`, ou, sem a
    declaração, quando tem a etapa de transcrição.
    """
    multi_clip = getattr(recipe, "multi_clip", None)
    needs_speech = getattr(recipe, "needs_speech", None)
    return RecipeInfo(
        name=recipe.name,
        description=recipe.description,
        steps=list(recipe.steps),
        multi_clip=bool(multi_clip),
        needs_speech="transcription" in recipe.steps if needs_speech is None else bool(needs_speech),
    )


def registered_recipes() -> list[RecipeInfo]:
    """Receitas registradas agora, em ordem alfabética."""
    return [recipe_info(get_recipe(name)) for name in recipe_names()]
