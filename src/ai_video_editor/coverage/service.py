"""Validação da decisão de cobertura, preenchimento determinístico das janelas obrigatórias e execução
com nova tentativa e fallback."""

import time
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import (
    AcceptedBreath,
    AcceptedCoverage,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    CoverageRun,
    VisualShot,
)
from ai_video_editor.director.service import record_effective_model
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics

# Duração de uma cobertura no vídeo final. Uma frase sozinha pode passar do máximo (não dá para
# dividir uma frase por ID); o código encadeia outra tomada se a escolhida for curta.
MIN_COVERAGE_SEC = 1.5
MAX_COVERAGE_SEC = 8.0
# Tamanho dos respiros (limitado ao tamanho da tomada) e o mínimo de tomada para um respiro.
BREATH_SECONDS: dict[str, float] = {"short": 2.0, "medium": 3.5, "long": 5.0}
MIN_BREATH_SHOT_SEC = 1.0
# No máximo um respiro a cada tantos segundos de espinha (pelo menos 1).
BREATH_EVERY_SEC = 30.0
# A última tomada de um clipe termina na duração do contêiner, que pode passar do último quadro.
SOURCE_END_MARGIN_SEC = 0.1
MAX_ATTEMPTS = 2

ProblemKind = Literal["invalid", "missing_window"]


@dataclass(frozen=True)
class Problem:
    kind: ProblemKind
    message: str


def shot_span(shot: VisualShot) -> tuple[float, float]:
    """Trecho da tomada que pode ir para a timeline (tempo do clipe)."""
    end = shot.end_sec - (SOURCE_END_MARGIN_SEC if shot.last_in_clip else 0.0)
    return shot.start_sec, max(shot.start_sec, end)


def shot_length(shot: VisualShot) -> float:
    lo, hi = shot_span(shot)
    return hi - lo


def max_breaths(context: CoverageContext) -> int:
    return max(1, int(context.spine_duration_sec // BREATH_EVERY_SEC))


def _span_sec(context: CoverageContext, i: int, j: int) -> float:
    s = context.sentences
    return (s[j].end_frame - s[i].start_frame) / context.fps


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


def coverage_problems(decision: CoverageDecision, context: CoverageContext, excluded: dict[str, str]) -> list[Problem]:
    """Tudo o que está errado na decisão; lista vazia = válida.

    `excluded` explica as tomadas que existem mas não podem ser cobertura (sem análise, inutilizável,
    papel não permitido, principal com fala), para a mensagem da nova tentativa.
    """
    order = {s.id: i for i, s in enumerate(context.sentences)}
    candidates = {c.id: c for c in context.candidates}
    windows = {w.id: w for w in context.windows}
    problems: list[str] = []
    used: dict[str, str] = {}
    spans: list[tuple[int, int, str]] = []

    def check_shot(label: str, shot_id: str) -> Optional[VisualShot]:
        if shot_id in used:
            problems.append(f"{label}: a tomada {shot_id} já foi usada em {used[shot_id]} (cada tomada no máximo uma vez)")
        used.setdefault(shot_id, label)
        if shot_id in candidates:
            return candidates[shot_id]
        if shot_id in excluded:
            problems.append(f"{label}: a tomada {shot_id} não pode ser cobertura ({excluded[shot_id]})")
        else:
            problems.append(f"{label}: a tomada {shot_id} não existe")
        return None

    for n, c in enumerate(decision.coverages, start=1):
        label = f"cobertura {n} ({c.from_sentence}–{c.to_sentence})"
        missing = [x for x in (c.from_sentence, c.to_sentence) if x not in order]
        for x in missing:
            problems.append(f"{label}: a frase {x} não existe (ou foi removida do vídeo)")
        shot = check_shot(label, c.shot_id)
        for w in c.windows:
            if w not in windows:
                problems.append(f"{label}: a janela obrigatória {w} não existe")
        if missing:
            continue
        i, j = order[c.from_sentence], order[c.to_sentence]
        if i > j:
            problems.append(f"{label}: {c.from_sentence} vem depois de {c.to_sentence}")
            continue
        speaking = {context.sentences[k].clip_id for k in range(i, j + 1)}
        if shot is not None and shot.clip_id in speaking:
            problems.append(f"{label}: a tomada {c.shot_id} é do clipe {shot.clip_id}, que está falando nesse trecho")
        duration = _span_sec(context, i, j)
        if duration < MIN_COVERAGE_SEC - 1e-9:
            problems.append(f"{label}: dura {_fmt(duration)}, menos que o mínimo de {_fmt(MIN_COVERAGE_SEC)}; inclua frases vizinhas")
        elif duration > MAX_COVERAGE_SEC + 1e-9 and i != j:
            problems.append(f"{label}: dura {_fmt(duration)}, mais que o máximo de {_fmt(MAX_COVERAGE_SEC)}; divida em coberturas menores")
        for w in c.windows:
            if w in windows and not {order[x] for x in windows[w].sentence_ids} & set(range(i, j + 1)):
                problems.append(f"{label}: diz atender {w}, mas não cobre nenhuma frase dela ({windows[w].span_label})")
        spans.append((i, j, label))

    spans.sort()
    for (_, prev_end, prev), (start, _, current) in zip(spans, spans[1:]):
        if start <= prev_end:
            problems.append(f"{current} se sobrepõe a {prev}")

    breath_after: set[int] = set()
    for n, b in enumerate(decision.breaths, start=1):
        label = f"respiro {n} (depois de {b.after_sentence})"
        shot = check_shot(label, b.shot_id)
        if shot is not None and shot_length(shot) < MIN_BREATH_SHOT_SEC:
            problems.append(f"{label}: a tomada {b.shot_id} tem {_fmt(shot_length(shot))}, menos que {_fmt(MIN_BREATH_SHOT_SEC)}")
        if b.after_sentence not in order:
            problems.append(f"{label}: a frase {b.after_sentence} não existe (ou foi removida do vídeo)")
            continue
        k = order[b.after_sentence]
        if k in breath_after:
            problems.append(f"{label}: já há um respiro depois de {b.after_sentence}")
        breath_after.add(k)
        inside = next((lab for i, j, lab in spans if i <= k < j), None)
        if inside:
            problems.append(f"{label}: cai no meio de {inside}")
    if len(decision.breaths) > max_breaths(context):
        problems.append(
            f"{len(decision.breaths)} respiros; o máximo é {max_breaths(context)} "
            f"(um a cada {BREATH_EVERY_SEC:.0f}s de fala)"
        )

    result = [Problem("invalid", p) for p in problems]
    covered = {k for i, j, _ in spans for k in range(i, j + 1)}
    for w in context.windows:
        lacking = [x for x in w.sentence_ids if order[x] not in covered]
        if lacking:
            result.append(
                Problem(
                    "missing_window",
                    f"janela obrigatória {w.id} ({w.span_label}, {', '.join(w.reasons)}) sem cobertura: "
                    f"faltam {', '.join(lacking)}",
                )
            )
    return result


def window_cover(context: CoverageContext, coverages: Iterable[CoverageChoice]) -> dict[str, list[str]]:
    """Janela → IDs das coberturas (C1, C2...) que tocam as frases dela."""
    order = {s.id: i for i, s in enumerate(context.sentences)}
    spans = [(f"C{n}", set(range(order[c.from_sentence], order[c.to_sentence] + 1))) for n, c in enumerate(coverages, 1)]
    return {
        w.id: [cid for cid, idxs in spans if idxs & {order[x] for x in w.sentence_ids}]
        for w in context.windows
    }


def best_free_shot(
    candidates: Iterable[VisualShot], need_sec: float, speaking: set[str], used: set[str]
) -> Optional[VisualShot]:
    """Tomada livre de maior beleza que caiba (se nenhuma couber, a de maior beleza: o resto é encadeado),
    fora dos clipes que estão falando."""
    free = [c for c in candidates if c.id not in used and c.clip_id not in speaking]
    if not free:
        return None
    return max(free, key=lambda c: (shot_length(c) >= need_sec - 1e-9, c.beauty or 0, c.usable == "sim", shot_length(c)))


def _runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for k in sorted(indices):
        if runs and runs[-1][-1] == k - 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


def _chunks(context: CoverageContext, run: list[int], covered: set[int], breath_after: set[int]) -> list[tuple[int, int]]:
    """Divide frases seguidas em coberturas de até `MAX_COVERAGE_SEC`; uma curta demais pega frases vizinhas
    livres até o mínimo (sem atravessar um respiro)."""
    chunks = []
    i, last = run[0], run[-1]
    while i <= last:
        j = i
        while j + 1 <= last and j not in breath_after and _span_sec(context, i, j + 1) <= MAX_COVERAGE_SEC:
            j += 1
        while _span_sec(context, i, j) < MIN_COVERAGE_SEC and j + 1 < len(context.sentences) and (
            j + 1 not in covered and j not in breath_after and _span_sec(context, i, j + 1) <= MAX_COVERAGE_SEC
        ):
            j += 1
        while _span_sec(context, i, j) < MIN_COVERAGE_SEC and i - 1 >= 0 and (
            i - 1 not in covered and i - 1 not in breath_after and _span_sec(context, i - 1, j) <= MAX_COVERAGE_SEC
        ):
            i -= 1
        chunks.append((i, j))
        covered.update(range(i, j + 1))
        i = j + 1
    return chunks


def fill_windows(
    context: CoverageContext,
    coverages: list[AcceptedCoverage],
    breaths: list[AcceptedBreath],
    origin: str = "fallback",
) -> tuple[list[AcceptedCoverage], list[str]]:
    """Cobre as frases das janelas obrigatórias que ficaram descobertas com a tomada livre de maior beleza.

    Devolve as coberturas (as de entrada + as novas, em ordem) e avisos (janela sem tomada livre,
    cobertura abaixo do mínimo).
    """
    order = {s.id: i for i, s in enumerate(context.sentences)}
    covered = {k for c in coverages for k in range(order[c.from_sentence], order[c.to_sentence] + 1)}
    used = {c.shot_id for c in coverages} | {b.shot_id for b in breaths}
    breath_after = {order[b.after_sentence] for b in breaths}
    result = list(coverages)
    warnings: list[str] = []
    for window in context.windows:
        lacking = [order[x] for x in window.sentence_ids if order[x] not in covered]
        for run in _runs(lacking):
            for i, j in _chunks(context, run, covered, breath_after):
                first, last = context.sentences[i], context.sentences[j]
                span = _span_sec(context, i, j)
                speaking = {context.sentences[k].clip_id for k in range(i, j + 1)}
                shot = best_free_shot(context.candidates, span, speaking, used)
                if shot is None:
                    warnings.append(f"janela {window.id} ({first.id}–{last.id}): nenhuma tomada livre para cobrir")
                    continue
                if span < MIN_COVERAGE_SEC:
                    warnings.append(f"cobertura de {first.id}–{last.id} ({window.id}) com {_fmt(span)}, abaixo do mínimo")
                used.add(shot.id)
                result.append(
                    AcceptedCoverage(
                        from_sentence=first.id, to_sentence=last.id, shot_id=shot.id, windows=[window.id],
                        reason=f"Cobre {window.id} ({', '.join(window.reasons)}) com a tomada livre de maior beleza.",
                        origin=origin,
                    )
                )
    result.sort(key=lambda c: order[c.from_sentence])
    return result, warnings


def excluded_shots(context: CoverageContext, all_shots: Iterable[VisualShot]) -> dict[str, str]:
    """Tomadas que existem mas não são candidatas, com o motivo (para a mensagem de erro)."""
    candidate_ids = {c.id for c in context.candidates}
    reasons: dict[str, str] = {}
    for shot in all_shots:
        if shot.id in candidate_ids:
            continue
        if shot.status != "analisada":
            reasons[shot.id] = "sem análise da IA"
        elif shot.usable == "nao":
            reasons[shot.id] = "imagem inutilizável (usable=nao)"
        elif shot.role not in context.roles:
            reasons[shot.id] = f"papel {shot.role} não permitido"
        else:
            reasons[shot.id] = "principal com fala"
    return reasons


def _new_run(director: Optional[CoverageDirector], context: CoverageContext) -> CoverageRun:
    return CoverageRun(
        provider=director.provider_name if director else "none",
        model_id=director.model_id if director else "none",
        thinking=director.thinking_label if director else None,
        model_chain=list(director.model_chain) if director else [],
        instructions=context.instructions,
        roles=context.roles,
        candidates=len(context.candidates),
        windows=context.windows,
        spine_duration_sec=round(context.spine_duration_sec, 3),
    )


def _finish(run: CoverageRun, context: CoverageContext) -> CoverageRun:
    run.window_cover = window_cover(context, run.coverages)
    missing = run.uncovered_windows
    if missing:
        run.warnings.append(f"janelas obrigatórias sem cobertura: {', '.join(missing)}")
    return run


def run_coverage(
    director: Optional[CoverageDirector], context: CoverageContext, all_shots: Iterable[VisualShot]
) -> CoverageRun:
    """Pede a decisão ao Diretor de cobertura, valida e tenta de novo uma vez com o erro.

    - Sem Diretor (`none`) ou sem tomadas candidatas: só as janelas obrigatórias, pelo código.
    - Resposta inválida duas vezes: se o único problema são janelas obrigatórias descobertas, a decisão
      da IA fica e o código completa (`ia+fallback`); senão, só as janelas obrigatórias (`fallback`).
    Erros de comunicação (autenticação, cota, timeout) sobem para o chamador, como no Diretor.
    """
    run = _new_run(director, context)
    if director is None or not context.candidates or not context.sentences:
        run.origin = "fallback"
        if director is None:
            run.origin_reason = "IA de cobertura desligada (--coverage-provider none): só as janelas obrigatórias"
        else:
            run.origin_reason = "nenhuma tomada candidata a cobertura" if context.sentences else "nenhuma frase mantida"
        run.coverages, warnings = fill_windows(context, [], [])
        run.warnings += warnings
        run.summary = f"{len(run.coverages)} cobertura(s) das janelas obrigatórias, escolhidas pelo código."
        return _finish(run, context)

    excluded = excluded_shots(context, all_shots)
    errors: list[str] = []
    decision: Optional[CoverageDecision] = None
    problems: list[Problem] = []
    for _ in range(MAX_ATTEMPTS):
        director.last_input_tokens = None
        director.last_output_tokens = None
        director.last_api_attempts = 1
        started = time.perf_counter()
        error: Optional[AIResponseValidationError] = None
        try:
            decision = director.decide(context)
            problems = coverage_problems(decision, context, excluded)
            if problems:
                error = AIResponseValidationError("coverage", details="; ".join(p.message for p in problems))
        except AIResponseValidationError as exc:
            error, decision, problems = exc, None, []
        if director.effective_thinking_label:
            run.thinking = director.effective_thinking_label
        record_effective_model(run, director)
        run.attempts.append(
            AIRunMetrics(
                provider=run.effective_provider or director.provider_name,
                model_id=run.effective_model_id or director.model_id,
                frames_sent=0,
                latency_sec=round(time.perf_counter() - started, 3),
                input_tokens=director.last_input_tokens,
                output_tokens=director.last_output_tokens,
                api_attempts=director.last_api_attempts,
                success=error is None,
                error_type=type(error).__name__ if error else None,
            )
        )
        if error is None and decision is not None:
            run.origin = "ia"
            run.summary = decision.summary.strip()
            run.coverages = [AcceptedCoverage(**c.model_dump()) for c in decision.coverages]
            run.breaths = [AcceptedBreath(**b.model_dump()) for b in decision.breaths]
            run.coverages.sort(key=lambda c: [s.id for s in context.sentences].index(c.from_sentence))
            return _finish(run, context)
        message = error.details or str(error)
        errors.append(message)
        context = context.model_copy(update={"previous_error": message})

    if decision is not None and problems and all(p.kind == "missing_window" for p in problems):
        run.origin = "ia+fallback"
        run.origin_reason = (
            f"a IA deixou janelas obrigatórias descobertas em {MAX_ATTEMPTS} tentativas; o código completou"
        )
        run.summary = decision.summary.strip()
        accepted = [AcceptedCoverage(**c.model_dump()) for c in decision.coverages]
        run.breaths = [AcceptedBreath(**b.model_dump()) for b in decision.breaths]
        run.coverages, warnings = fill_windows(context, accepted, run.breaths)
        run.warnings.append(f"Cobertura completada pelo código: {' | '.join(errors)}")
    else:
        run.origin = "fallback"
        run.origin_reason = f"resposta inválida em {MAX_ATTEMPTS} tentativas: só as janelas obrigatórias"
        run.coverages, warnings = fill_windows(context, [], [])
        run.summary = f"{len(run.coverages)} cobertura(s) das janelas obrigatórias, escolhidas pelo código."
        run.warnings.append(
            f"Cobertura da IA ignorada: resposta inválida em {MAX_ATTEMPTS} tentativas ({' | '.join(errors)}). "
            "Aplicadas só as janelas obrigatórias."
        )
    run.warnings += warnings
    return _finish(run, context)
