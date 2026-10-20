"""Validação das decisões do Diretor e execução com nova tentativa e fallback."""

import time

from ai_video_editor.director.provider import EditDirector
from ai_video_editor.director.schema import (
    DirectorContext,
    DirectorDecision,
    DirectorRun,
    Sentence,
    UserBrief,
)
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics

# Fração máxima do tempo de fala que o Diretor pode remover sem uma duração alvo que exija mais.
MAX_REMOVAL_RATIO = 0.40
# Folga sobre a fração exigida pela duração alvo (o modelo não acerta a duração exata).
TARGET_RATIO_MARGIN = 0.10
MAX_ATTEMPTS = 2


def speech_duration(sentences: list[Sentence]) -> float:
    return sum(s.duration for s in sentences)


def allowed_removal_ratio(sentences: list[Sentence], brief: UserBrief) -> float:
    """40% do tempo de fala, ou mais se a duração alvo pedir isso."""
    total = speech_duration(sentences)
    if brief.target_duration_sec is None or total <= 0 or brief.target_duration_sec >= total:
        return MAX_REMOVAL_RATIO
    needed = 1.0 - brief.target_duration_sec / total
    return min(1.0, max(MAX_REMOVAL_RATIO, needed + TARGET_RATIO_MARGIN))


def validate_decision(
    decision: DirectorDecision, sentences: list[Sentence], brief: UserBrief
) -> None:
    """Confere a decisão contra as frases e o briefing.

    Raises:
        AIResponseValidationError: ID inexistente ou repetido, enrolação removida com
            `remove_fillers=False`, ou remoção acima do limite de tempo de fala.
    """
    by_id = {s.id: s for s in sentences}
    seen: set[int] = set()
    problems: list[str] = []
    for removal in decision.removals:
        if removal.sentence_id not in by_id:
            problems.append(f"sentence_id {removal.sentence_id} não existe")
        elif removal.sentence_id in seen:
            problems.append(f"sentence_id {removal.sentence_id} aparece mais de uma vez")
        seen.add(removal.sentence_id)
        if removal.category == "filler" and not brief.remove_fillers:
            problems.append(
                f"sentence_id {removal.sentence_id}: o usuário pediu para não remover enrolação (filler)"
            )
    if problems:
        raise AIResponseValidationError("director", details="; ".join(problems))

    total = speech_duration(sentences)
    removed = sum(by_id[i].duration for i in seen)
    limit = allowed_removal_ratio(sentences, brief)
    if total > 0 and removed / total > limit + 1e-9:
        raise AIResponseValidationError(
            "director",
            details=(
                f"remoção de {removed:.1f}s ({removed / total:.0%} do tempo de fala) passa do limite "
                f"de {limit:.0%}; remova menos frases"
            ),
        )


def run_director(
    director: EditDirector,
    sentences: list[Sentence],
    brief: UserBrief,
    context: DirectorContext,
) -> DirectorRun:
    """Pede a decisão ao Diretor, valida e tenta de novo uma vez enviando o erro ao modelo.

    Se a segunda resposta também for inválida, devolve um `DirectorRun` sem decisão e com
    um aviso: o pipeline segue só com o corte de silêncio. Erros de comunicação
    (autenticação, cota, timeout) não são tratados aqui e sobem para o chamador.
    """
    run = DirectorRun(
        provider=director.provider_name, model_id=director.model_id,
        thinking=director.thinking_label, brief=brief, sentences=sentences
    )
    if not sentences:
        run.decision = DirectorDecision(summary="Transcrição sem frases; nada a decidir.")
        return run

    errors: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        director.last_input_tokens = None
        director.last_output_tokens = None
        director.last_api_attempts = 1
        started = time.perf_counter()
        error: AIResponseValidationError | None = None
        try:
            decision = director.decide(sentences, brief, context)
            validate_decision(decision, sentences, brief)
        except AIResponseValidationError as exc:
            error = exc
        if director.effective_thinking_label:
            run.thinking = director.effective_thinking_label
        run.attempts.append(
            AIRunMetrics(
                provider=director.provider_name,
                model_id=director.model_id,
                frames_sent=0,
                latency_sec=round(time.perf_counter() - started, 3),
                input_tokens=director.last_input_tokens,
                output_tokens=director.last_output_tokens,
                api_attempts=director.last_api_attempts,
                success=error is None,
                error_type=type(error).__name__ if error else None,
            )
        )
        if error is None:
            run.decision = decision
            return run
        message = error.details or str(error)
        errors.append(message)
        context = context.model_copy(update={"previous_error": message})

    run.warnings.append(
        "Diretor ignorado: resposta inválida em "
        f"{MAX_ATTEMPTS} tentativas ({' | '.join(errors)}). Aplicado só o corte de silêncio."
    )
    return run
