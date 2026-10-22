"""Validação do briefing, fallback determinístico, estimativa de trabalho e execução do Produtor."""

import time
from typing import Callable, Optional

from ai_video_editor.intelligence.exceptions import AIProviderError, AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics
from ai_video_editor.producer.provider import BriefingProducer
from ai_video_editor.producer.schema import (
    Briefing,
    BriefingContent,
    MaterialSummary,
    ProducerContext,
    RecipeInfo,
    WorkEstimate,
)
from ai_video_editor.producer.summary import registered_recipes
from ai_video_editor.recipes import DEFAULT_RECIPE
from ai_video_editor.recipes.schema import RecipeError

MAX_ATTEMPTS = 2
# Receita do fallback para pasta com papel principal falado, se estiver registrada.
VLOG_RECIPE = "vlog"
# Peças que chamam IA durante a edição (entram na estimativa de chamadas).
AI_PIECES = ("director",)
# Mesmo número de tentativas do Diretor por etapa de IA (uma nova tentativa com o erro).
AI_ATTEMPTS_PER_STEP = 2
UNDERSTOOD_PREFIX = "Entendi: "


def _possible_speech(material: MaterialSummary, role: Optional[str] = None) -> int:
    """Clipes com fala detectada ou desconhecida (no papel, ou em todos)."""
    roles = [material.role(role)] if role else material.roles
    return sum(r.clips_with_speech + r.speech_unknown for r in roles if r is not None)


def is_single_spoken_video(material: MaterialSummary) -> bool:
    """1 clipe, com fala detectada (ou fala não analisada)."""
    return material.clips == 1 and _possible_speech(material) == 1


def recipe_problems(recipe: RecipeInfo, material: MaterialSummary) -> list[str]:
    """Incompatibilidades da receita com o material (vazio = compatível)."""
    problems: list[str] = []
    speaking = _possible_speech(material)
    if recipe.needs_speech and speaking == 0:
        problems.append(f"a receita '{recipe.name}' precisa de fala, mas nenhum clipe do material tem fala")
    if not recipe.multi_clip and speaking > 1:
        problems.append(
            f"a receita '{recipe.name}' edita 1 vídeo, mas o material tem {speaking} clipes com fala; "
            "escolha uma receita de vários clipes"
        )
    return problems


def _used_duration(content: BriefingContent, material: MaterialSummary) -> float:
    roles = [r for r in material.roles if not content.roles_to_use or r.role in content.roles_to_use]
    return sum(r.duration_sec for r in roles)


def validate_content(
    content: BriefingContent, material: MaterialSummary, recipes: list[RecipeInfo], forced: Optional[str] = None
) -> list[str]:
    """Problemas do briefing além do Pydantic (vazio = válido).

    Com `forced`, a incompatibilidade da receita com o material não é erro da IA (vira aviso depois).
    """
    by_name = {r.name: r for r in recipes}
    problems: list[str] = []
    recipe = by_name.get(content.recipe)
    if recipe is None:
        problems.append(f"a receita '{content.recipe}' não existe (disponíveis: {', '.join(by_name)})")
    elif forced is None:
        problems += recipe_problems(recipe, material)
    unknown = [role for role in content.roles_to_use if role not in material.role_names]
    if unknown:
        problems.append(
            f"papéis que não existem no material: {', '.join(unknown)} (existem: {', '.join(material.role_names)})"
        )
    target = content.target_duration_sec
    if target is not None:
        available = _used_duration(content, material)
        if target <= 0:
            problems.append("target_duration_sec precisa ser maior que zero")
        elif target >= available:
            problems.append(
                f"a duração alvo ({target:.0f}s) não é menor que o material a usar ({available:.0f}s); "
                "deixe target_duration_sec vazio ou use um valor menor"
            )
    if not content.video_type.strip():
        problems.append("video_type está vazio")
    if not content.understood.strip():
        problems.append("understood está vazio")
    return problems


def _normalize(content: BriefingContent, material: MaterialSummary) -> BriefingContent:
    """Ajustes sem decisão: prefixo "Entendi: ", papéis padrão, textos sem espaços sobrando."""
    understood = content.understood.strip()
    if not understood.lower().startswith("entendi"):
        understood = UNDERSTOOD_PREFIX + understood
    roles = list(dict.fromkeys(content.roles_to_use)) or list(material.role_names)
    return content.model_copy(
        update={
            "understood": understood,
            "video_type": content.video_type.strip(),
            "roles_to_use": roles,
            "director_instructions": content.director_instructions.strip(),
            "coverage_instructions": content.coverage_instructions.strip(),
        }
    )


def fallback_content(
    request: str, material: MaterialSummary, recipes: list[RecipeInfo], forced: Optional[str] = None
) -> BriefingContent:
    """Briefing sem IA: receita pela regra do material e o pedido inteiro para o Diretor.

    - `--receita` → ela.
    - 1 clipe → `fala` (receita padrão).
    - Pasta com papel principal falado → `vlog`, se registrada; senão `fala`, com aviso.
    - Sem fala no principal → `fala`, com aviso.
    """
    names = {r.name for r in recipes}
    warnings: list[str] = []
    if forced:
        recipe, video_type = forced, "vídeo"
    elif material.clips <= 1:
        recipe, video_type = DEFAULT_RECIPE, "vídeo falado"
        if _possible_speech(material) == 0:
            warnings.append("Nenhuma fala detectada no vídeo; a receita fala pode não ter o que cortar.")
    elif _possible_speech(material, "principal") > 0:
        if VLOG_RECIPE in names:
            recipe, video_type = VLOG_RECIPE, "vlog"
        else:
            recipe, video_type = DEFAULT_RECIPE, "vlog"
            warnings.append(
                f"A receita '{VLOG_RECIPE}' não está registrada; usando '{DEFAULT_RECIPE}', que edita 1 vídeo "
                "por vez."
            )
    else:
        recipe, video_type = DEFAULT_RECIPE, "vídeo"
        warnings.append(
            "Nenhum clipe principal com fala; nenhuma receita registrada combina com este material. "
            f"Usando '{DEFAULT_RECIPE}'; confira os papéis (--papel no catalog)."
        )
    info = next((r for r in recipes if r.name == recipe), None)
    if info is not None and not info.multi_clip:
        roles = ["principal"] if material.role("principal") else list(material.role_names)
    else:
        roles = list(material.role_names)
    text = request.strip()
    if material.clips <= 1:
        what = "o vídeo falado"
    else:
        what = f"os {material.clips} clipes" + (f" como {video_type}" if video_type != "vídeo" else "")
    if text:
        understood = f"{UNDERSTOOD_PREFIX}editar {what} seguindo o pedido: \"{text}\"."
    else:
        understood = f"{UNDERSTOOD_PREFIX}editar {what}, cortando pausas e erros evidentes, sem pedido específico."
    return BriefingContent(
        video_type=video_type, recipe=recipe, understood=understood, intensity="light", pace="normal",
        director_instructions=text, roles_to_use=roles, warnings=warnings,
    )


def estimate_work(content: BriefingContent, material: MaterialSummary, recipes: list[RecipeInfo]) -> WorkEstimate:
    """Clipes, áudio a transcrever e chamadas de IA previstas para a edição (aproximado, sem preços)."""
    recipe = next((r for r in recipes if r.name == content.recipe), None)
    roles = [r for r in material.roles if r.role in content.roles_to_use]
    steps = recipe.steps if recipe else []
    transcribes = "transcription" in steps
    ai_steps = [s for s in steps if s in AI_PIECES]
    return WorkEstimate(
        clips=sum(r.clips for r in roles),
        clips_to_transcribe=sum(r.clips_with_speech + r.speech_unknown for r in roles) if transcribes else 0,
        audio_to_transcribe_sec=round(sum(r.speech_clips_duration_sec for r in roles), 2) if transcribes else 0.0,
        ai_steps=ai_steps,
        ai_calls=len(ai_steps),
        ai_calls_max=len(ai_steps) * AI_ATTEMPTS_PER_STEP,
    )


def _record_model(briefing: Briefing, producer: BriefingProducer) -> None:
    if producer.effective_model_id:
        briefing.effective_model = f"{producer.effective_provider}:{producer.effective_model_id}"
    briefing.model_switches = list(producer.model_switches)


def run_producer(
    producer: Optional[BriefingProducer],
    request: str,
    material: MaterialSummary,
    forced_recipe: Optional[str] = None,
    recipes: Optional[list[RecipeInfo]] = None,
) -> Briefing:
    """Pede o briefing à IA, valida e tenta de novo uma vez enviando o erro ao modelo.

    Sem Produtor (`None`), ou com pedido vazio e 1 vídeo falado, a IA não é chamada. Resposta
    inválida duas vezes, ou falha de comunicação com a IA, leva ao fallback determinístico; o
    motivo fica em `origin_reason` e nos avisos.

    Raises:
        RecipeError: `forced_recipe` não está registrada.
    """
    recipes = registered_recipes() if recipes is None else recipes
    names = [r.name for r in recipes]
    if forced_recipe is not None and forced_recipe not in names:
        raise RecipeError(f"Receita desconhecida: '{forced_recipe}' (receitas disponíveis: {', '.join(names)}).")
    base = {"request": request, "forced_recipe": forced_recipe, "material": material}

    def fallback(reason: str, warn: bool) -> Briefing:
        content = _normalize(fallback_content(request, material, recipes, forced_recipe), material)
        warnings = ([f"Briefing sem IA: {reason}."] if warn else []) + content.warnings
        return Briefing(**{**content.model_dump(), "warnings": warnings}, origin="fallback", origin_reason=reason, **base)

    if producer is None:
        briefing = fallback("Produtor de IA desligado (--director-provider none)", warn=False)
    elif not request.strip() and is_single_spoken_video(material):
        briefing = fallback("pedido vazio e 1 vídeo falado; a IA não foi chamada", warn=False)
    else:
        briefing = _ask(producer, request, material, recipes, forced_recipe, fallback)
        briefing.provider = producer.provider_name
        briefing.model_chain = list(producer.model_chain) or [f"{producer.provider_name}:{producer.model_id}"]
        _record_model(briefing, producer)

    if forced_recipe is not None:
        forced = next(r for r in recipes if r.name == forced_recipe)
        briefing.warnings += [f"Receita escolhida com --receita: {p}." for p in recipe_problems(forced, material)]
    briefing.estimate = estimate_work(briefing, material, recipes)
    briefing.estimate.producer_calls = sum(a.api_attempts for a in briefing.attempts)
    return briefing


def _ask(
    producer: BriefingProducer,
    request: str,
    material: MaterialSummary,
    recipes: list[RecipeInfo],
    forced: Optional[str],
    fallback: Callable[[str, bool], Briefing],
) -> Briefing:
    context = ProducerContext(forced_recipe=forced)
    errors: list[str] = []
    attempts: list[AIRunMetrics] = []
    for _ in range(MAX_ATTEMPTS):
        producer.last_input_tokens = None
        producer.last_output_tokens = None
        producer.last_api_attempts = 1
        started = time.perf_counter()
        error: Optional[AIProviderError] = None
        content: Optional[BriefingContent] = None
        try:
            content = producer.propose(request, material, recipes, context)
            if forced:
                content = content.model_copy(update={"recipe": forced})
            content = _normalize(content, material)
            problems = validate_content(content, material, recipes, forced)
            if problems:
                raise AIResponseValidationError("producer", details="; ".join(problems))
        except AIProviderError as exc:
            error = exc
        attempts.append(
            AIRunMetrics(
                provider=producer.effective_provider or producer.provider_name,
                model_id=producer.effective_model_id or producer.model_id,
                frames_sent=0,
                latency_sec=round(time.perf_counter() - started, 3),
                input_tokens=producer.last_input_tokens,
                output_tokens=producer.last_output_tokens,
                api_attempts=max(1, producer.last_api_attempts),
                success=error is None,
                error_type=type(error).__name__ if error else None,
            )
        )
        if error is None:
            return Briefing(
                **content.model_dump(), origin="ia", attempts=attempts, request=request,
                forced_recipe=forced, material=material,
            )
        message = error.details or str(error)
        if not isinstance(error, AIResponseValidationError):
            briefing = fallback(f"IA indisponível ({message})", warn=True)
            briefing.attempts = attempts
            return briefing
        errors.append(message)
        context = context.model_copy(update={"previous_error": message})
    reasons = " | ".join(dict.fromkeys(errors))  # sem repetir o mesmo erro
    briefing = fallback(f"resposta inválida em {MAX_ATTEMPTS} tentativas ({reasons})", warn=True)
    briefing.attempts = attempts
    return briefing
