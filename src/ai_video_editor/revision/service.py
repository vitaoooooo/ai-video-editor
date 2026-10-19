"""Validação das decisões do revisor e execução com nova tentativa."""

import time

from ai_video_editor.director.schema import DirectorDecision, UserBrief
from ai_video_editor.director.service import validate_decision
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics
from ai_video_editor.revision.exceptions import RevisionRejectedError
from ai_video_editor.revision.provider import EditReviser
from ai_video_editor.revision.schema import RevisionContext, RevisionDecision, RevisionRun, SentenceState

MAX_ATTEMPTS = 2


def removed_ids_of(states: list[SentenceState]) -> set[int]:
    return {s.sentence.id for s in states if s.removed}


def validate_revision(decision: RevisionDecision, states: list[SentenceState], brief: UserBrief) -> None:
    """Mesmas regras do Diretor (IDs existentes e únicos, limite de remoção) mais a coerência das restaurações.

    Raises:
        AIResponseValidationError: decisão inválida (o revisor recebe o erro e tenta de novo).
    """
    sentences = [s.sentence for s in states]
    validate_decision(DirectorDecision(removals=decision.removals, summary=decision.summary), sentences, brief)

    problems: list[str] = []
    known = {s.sentence.id for s in states}
    restored = [r.sentence_id for r in decision.restorations]
    if len(set(restored)) != len(restored):
        problems.append("restorations repete sentence_id")
    unknown = sorted(set(restored) - known)
    if unknown:
        problems.append(f"restorations referencia IDs inexistentes: {unknown}")
    expected = removed_ids_of(states) - {r.sentence_id for r in decision.removals}
    if set(restored) - set(unknown) != expected:
        problems.append(
            "restorations deve listar exatamente as frases removidas na versão base que voltam "
            f"({sorted(expected)}), mas lista {sorted(restored)}"
        )
    if problems:
        raise AIResponseValidationError("reviser", details="; ".join(problems))


def run_reviser(
    reviser: EditReviser,
    states: list[SentenceState],
    brief: UserBrief,
    context: RevisionContext,
) -> RevisionRun:
    """Pede a revisão, valida e tenta de novo uma vez enviando o erro ao modelo.

    Raises:
        RevisionRejectedError: pedido impossível/ambíguo (`feasible=False`) ou resposta
            inválida nas duas tentativas. Nenhuma versão deve ser criada.
        AIProviderError: erros de comunicação (autenticação, cota, timeout) sobem direto.
    """
    run = RevisionRun(provider=reviser.provider_name, model_id=reviser.model_id, thinking=reviser.thinking_label)
    errors: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        reviser.last_input_tokens = None
        reviser.last_output_tokens = None
        started = time.perf_counter()
        error: AIResponseValidationError | None = None
        decision = None
        try:
            decision = reviser.revise(states, brief, context)
            if decision.feasible:
                validate_revision(decision, states, brief)
        except AIResponseValidationError as exc:
            error = exc
        if reviser.effective_thinking_label:
            run.thinking = reviser.effective_thinking_label
        run.attempts.append(
            AIRunMetrics(
                provider=reviser.provider_name,
                model_id=reviser.model_id,
                frames_sent=0,
                latency_sec=round(time.perf_counter() - started, 3),
                input_tokens=reviser.last_input_tokens,
                output_tokens=reviser.last_output_tokens,
                success=error is None,
                error_type=type(error).__name__ if error else None,
            )
        )
        if error is None:
            if not decision.feasible:
                raise RevisionRejectedError(
                    decision.problem or decision.summary or "o revisor não conseguiu atender ao pedido"
                )
            run.decision = decision
            return run
        message = error.details or str(error)
        errors.append(message)
        context = context.model_copy(update={"previous_error": message})

    raise RevisionRejectedError(
        f"resposta inválida do revisor em {MAX_ATTEMPTS} tentativas ({' | '.join(errors)}). "
        "Nenhuma versão foi criada; reformule o pedido."
    )
