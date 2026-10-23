"""Validação item a item da decisão de cobertura, segundos decididos pelo código, fallback com tomadas
relacionadas e execução com nova tentativa.

A IA escolhe frases e tomadas (IDs); os segundos são do código. Por isso a validação recusa só o que a
IA pode corrigir e o código não decide sozinho, e recusa **o item**, não a resposta:

- **Item descartado (com o motivo):** frase ou tomada inexistente, tomada não candidata, do clipe que
  fala, repetida, cobertura sobreposta a outra, respiro inválido, "manter" numa janela que exige
  cobertura. Numa cobertura com várias tomadas, a tomada inválida sai e o resto fica (ajuste).
- **Duração (ajuste, nunca recusa):** cada tomada mostra até `MAX_COVERAGE_SEC`; um trecho mais longo é
  dividido entre as tomadas listadas. Se elas não bastam, numa cobertura que toca uma janela com imagem
  ruim o código completa com tomadas livres parecidas (mesmo papel ou tags; nunca do clipe que fala); fora
  de janela, a cobertura é **encurtada** (cobre do início até onde as tomadas dão): ali a pessoa falando é
  uma imagem boa, e uma tomada que a IA não escolheu pode não ter nada a ver com a fala. Trecho menor que
  `MIN_COVERAGE_SEC` estende para a frase vizinha livre; sem vizinha, o item é descartado.
- **Nova tentativa** só quando sobra janela que exige cobertura descoberta ou mais de
  `RETRY_DISCARD_RATIO` dos itens foram descartados; ela leva só os problemas e os itens já aceitos, e
  pede só o que falta.
- **Fallback (código):** janela sem decisão da IA é coberta só com tomada relacionada ao que está na
  tela (`related.py`); sem nenhuma, a imagem original fica, com aviso. Nunca uma tomada "aleatória"
  só por beleza. Entre as relacionadas: não repetir o clipe do pedaço anterior quando houver
  alternativa, maior relação, espalhar entre os papéis (drone/b-roll/POV/principal) e, por fim, beleza.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.related import related_shots, specific_tags
from ai_video_editor.coverage.schema import (
    AcceptedBreath,
    AcceptedCoverage,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    CoverageRun,
    DiscardedItem,
    ItemNote,
    MandatoryWindow,
    VisualShot,
    WindowOutcome,
)
from ai_video_editor.director.service import record_effective_model
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics

# Duração de uma cobertura no vídeo final (mínimo) e de cada tomada dentro dela (máximo).
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
# Nova tentativa quando mais que esta fração dos itens da resposta foi descartada.
RETRY_DISCARD_RATIO = 0.5
# Fallback numa janela que não exige cobertura: sobra de imagem ruim tolerada no fim do trecho.
MAX_UNCOVERED_SEC = 0.5
_EPS = 1e-9


def shot_span(shot: VisualShot) -> tuple[float, float]:
    """Trecho da tomada que pode ir para a timeline (tempo do clipe)."""
    end = shot.end_sec - (SOURCE_END_MARGIN_SEC if shot.last_in_clip else 0.0)
    return shot.start_sec, max(shot.start_sec, end)


def shot_length(shot: VisualShot) -> float:
    lo, hi = shot_span(shot)
    return hi - lo


def shot_frames(shot: VisualShot, fps: int) -> int:
    """Frames que a tomada pode dar a uma cobertura: até `MAX_COVERAGE_SEC`, limitado à tomada."""
    lo, hi = shot_span(shot)
    return min(int(math.floor((hi - lo) * fps + 1e-6)), round(MAX_COVERAGE_SEC * fps))


def max_breaths(context: CoverageContext) -> int:
    return max(1, int(context.spine_duration_sec // BREATH_EVERY_SEC))


def _span_frames(context: CoverageContext, i: int, j: int) -> int:
    s = context.sentences
    return s[j].end_frame - s[i].start_frame


def _span_sec(context: CoverageContext, i: int, j: int) -> float:
    return _span_frames(context, i, j) / context.fps


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


def _span_label(context: CoverageContext, i: int, j: int) -> str:
    a, b = context.sentences[i].id, context.sentences[j].id
    return a if i == j else f"{a}–{b}"


def _order(context: CoverageContext) -> dict[str, int]:
    return {s.id: i for i, s in enumerate(context.sentences)}


@dataclass
class CoverageState:
    """O que já foi aceito (entre tentativas e no fallback): coberturas, respiros, janelas mantidas, descartes."""

    context: CoverageContext
    coverages: list[AcceptedCoverage] = field(default_factory=list)
    breaths: list[AcceptedBreath] = field(default_factory=list)
    kept: dict[str, WindowOutcome] = field(default_factory=dict)
    discarded: list[DiscardedItem] = field(default_factory=list)
    used: dict[str, str] = field(default_factory=dict)  # tomada → item que a usa
    labels: list[tuple[int, int, str]] = field(default_factory=list)  # (i, j, rótulo) das coberturas
    # Tomadas citadas na resposta em análise: o código não as usa para completar outro item.
    reserved: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.order = _order(self.context)
        self.candidates = {c.id: c for c in self.context.candidates}
        self.screens = {s.id: s for s in self.context.screen_shots}

    def covered(self) -> set[int]:
        return {k for i, j, _ in self.labels for k in range(i, j + 1)}

    def breath_after(self) -> set[int]:
        return {self.order[b.after_sentence] for b in self.breaths}

    def role_uses(self) -> Counter:
        return Counter(self.candidates[x].role for c in self.coverages for x in c.shot_ids if x in self.candidates)

    def last_clip_before(self, i: int) -> Optional[str]:
        """Clipe da última tomada da cobertura que termina logo antes da frase `i` (pedaço anterior)."""
        for c in self.coverages:
            if self.order[c.to_sentence] == i - 1 and c.shot_ids[-1] in self.candidates:
                return self.candidates[c.shot_ids[-1]].clip_id
        return None

    def add_coverage(self, coverage: AcceptedCoverage, label: str) -> None:
        self.coverages.append(coverage)
        self.labels.append((self.order[coverage.from_sentence], self.order[coverage.to_sentence], label))
        for shot_id in coverage.shot_ids:
            self.used[shot_id] = label

    def add_breath(self, breath: AcceptedBreath, label: str) -> None:
        self.breaths.append(breath)
        self.used[breath.shot_id] = label

    def window_indices(self, window: MandatoryWindow) -> list[int]:
        return [self.order[x] for x in window.sentence_ids]

    def windows_touching(self, i: int, j: int) -> list[str]:
        return [w.id for w in self.context.windows if any(i <= k <= j for k in self.window_indices(w))]

    def window_screens(self, windows: Iterable[MandatoryWindow]) -> list[VisualShot]:
        """Tomadas ruins das janelas (o que está na tela ali): a referência da relação."""
        ids = dict.fromkeys(x for w in windows for x in w.shot_ids)
        return [self.screens[x] for x in ids if x in self.screens]


def _extend(state: CoverageState, i: int, j: int, shot_clips: set[str]) -> Optional[tuple[int, int]]:
    """Estende um trecho curto para frases vizinhas livres (a mais curta das duas; empate: a seguinte) até o
    mínimo; None se não der.

    Não atravessa respiro nem cobertura aceita e não pega frase de um clipe usado nas tomadas.
    """
    context, covered, after = state.context, state.covered(), state.breath_after()
    sentences = context.sentences
    while _span_sec(context, i, j) < MIN_COVERAGE_SEC - _EPS:
        options = []
        nxt, prv = j + 1, i - 1
        if nxt < len(sentences) and nxt not in covered and j not in after and sentences[nxt].clip_id not in shot_clips:
            options.append((_span_frames(context, i, nxt), 0, (i, nxt)))
        if prv >= 0 and prv not in covered and prv not in after and sentences[prv].clip_id not in shot_clips:
            options.append((_span_frames(context, prv, j), 1, (prv, j)))
        if not options:
            return None
        i, j = min(options)[2]
    return i, j


def _choose(
    state: CoverageState, pool: list[tuple[VisualShot, tuple]], need: int, prev_clip: Optional[str]
) -> list[VisualShot]:
    """Tomadas do `pool` (com a pontuação de relação) até somar `need` frames: clipe diferente do pedaço
    anterior > relação > papel menos usado até aqui (espalha drone/b-roll/POV) > beleza."""
    pool, chosen, total, uses = list(pool), [], 0, state.role_uses()
    while total < need and pool:
        best = max(
            pool,
            key=lambda x: (x[0].clip_id != prev_clip, x[1], -uses[x[0].role], x[0].beauty or 0, x[0].usable == "sim", shot_length(x[0])),
        )
        pool.remove(best)
        shot = best[0]
        chosen.append(shot)
        total += shot_frames(shot, state.context.fps)
        prev_clip = shot.clip_id
        uses[shot.role] += 1
    return chosen


def _free(state: CoverageState, speaking: set[str], exclude: Iterable[str] = ()) -> list[VisualShot]:
    skip = set(exclude) | state.reserved
    return [
        c for c in state.context.candidates
        if c.id not in state.used and c.id not in skip and c.clip_id not in speaking and shot_length(c) >= MIN_COVERAGE_SEC
    ]


def _similar_pool(state: CoverageState, shots: list[VisualShot], speaking: set[str], screens: list[VisualShot]) -> list[tuple[VisualShot, tuple]]:
    """Tomadas livres parecidas com as escolhidas (mesmo papel ou tags específicas em comum) ou relacionadas
    com a tela."""
    roles = {s.role for s in shots}
    tags = set(specific_tags(t for s in shots for t in s.tags))
    related = {c.id: r.score for c, r in related_shots(_free(state, speaking, [s.id for s in shots]), screens)}
    pool = []
    for c in _free(state, speaking, [s.id for s in shots]):
        common = len(tags & set(specific_tags(c.tags)))
        if c.id in related or c.role in roles or common:
            pool.append((c, (related.get(c.id, (0, 0)), common, c.role in roles)))
    return pool


def _plan(
    state: CoverageState,
    label: str,
    i: int,
    j: int,
    shots: list[VisualShot],
    reason: str,
    notes: list[ItemNote],
    origin: str,
    complete: bool = True,
) -> AcceptedCoverage:
    """Cobertura aceita com a lista final de tomadas: tira as que sobram, completa (janela) ou encurta."""
    context = state.context
    need = _span_frames(context, i, j)
    span = _fmt(need / context.fps)
    min_frames = MIN_COVERAGE_SEC * context.fps
    shots = list(shots)
    dropped = []
    while len(shots) > 1 and need < len(shots) * min_frames - _EPS:
        dropped.append(shots.pop().id)
    if dropped:
        notes.append(ItemNote(kind="duração", message=f"{label}: {span} não comportam todas as tomadas; retiradas {', '.join(dropped)}"))
    windows = state.windows_touching(i, j)
    capacity = sum(shot_frames(s, context.fps) for s in shots)
    added: list[VisualShot] = []
    if capacity < need and complete and windows:
        speaking = {context.sentences[k].clip_id for k in range(i, j + 1)}
        touched = [w for w in context.windows if w.id in windows]
        pool = _similar_pool(state, shots, speaking, state.window_screens(touched))
        added = _choose(state, pool, need - capacity, shots[-1].clip_id)
        capacity += sum(shot_frames(s, context.fps) for s in added)
        if added:
            notes.append(
                ItemNote(
                    kind="duração",
                    message=f"{label}: {span} com {len(shots)} tomada(s); completada com {', '.join(s.id for s in added)}",
                )
            )
    if capacity < need:
        where = "" if windows else " (fora de janela com imagem ruim o código não acrescenta tomadas)"
        notes.append(
            ItemNote(
                kind="duração",
                message=f"{label}: encurtada; as tomadas cobrem {_fmt(capacity / context.fps)} de {span}{where}",
            )
        )
    if origin == "ia" and notes:
        origin = "ia-ajustada"
    return AcceptedCoverage(
        from_sentence=context.sentences[i].id, to_sentence=context.sentences[j].id,
        shot_ids=[s.id for s in shots + added], added_shots=[s.id for s in added], reason=reason,
        windows=windows, origin=origin, adjustments=notes,
    )


def _shot_problem(state: CoverageState, shot_id: str, speaking: set[str], excluded: dict[str, str]) -> Optional[tuple[str, str]]:
    """(categoria, mensagem) se a tomada não pode ser usada aqui; None se pode."""
    if shot_id in state.used:
        return "tomada repetida", f"a tomada {shot_id} já foi usada em {state.used[shot_id]} (cada tomada no máximo uma vez)"
    shot = state.candidates.get(shot_id)
    if shot is None:
        if shot_id in excluded:
            return "tomada não candidata", f"a tomada {shot_id} não pode ser cobertura ({excluded[shot_id]})"
        return "ID inexistente", f"a tomada {shot_id} não existe"
    if shot.clip_id in speaking:
        return "clipe que fala", f"a tomada {shot_id} é do clipe {shot.clip_id}, que está falando nesse trecho"
    return None


def review_decision(
    decision: CoverageDecision, state: CoverageState, excluded: dict[str, str], attempt: int
) -> tuple[int, list[DiscardedItem]]:
    """Aceita os itens válidos da resposta (ajustando a duração) em `state`; devolve (nº de itens, descartados)."""
    context, order = state.context, state.order
    windows = {w.id: w for w in context.windows}
    discarded: list[DiscardedItem] = []

    def discard(label: str, kind: str, message: str) -> None:
        discarded.append(DiscardedItem(attempt=attempt, item=label, kind=kind, message=f"{label}: {message}"))

    items = 0
    state.reserved = {x for c in decision.coverages for x in c.shot_ids} | {b.shot_id for b in decision.breaths}
    for n, c in enumerate(decision.coverages, start=1):
        items += 1
        span = c.from_sentence if c.from_sentence == c.to_sentence else f"{c.from_sentence}–{c.to_sentence}"
        label = f"cobertura {n} ({span})" + (f" [tentativa {attempt}]" if attempt > 1 else "")
        missing = [x for x in (c.from_sentence, c.to_sentence) if x not in order]
        if missing:
            discard(label, "ID inexistente", f"a frase {missing[0]} não existe (ou foi removida do vídeo)")
            continue
        i, j = order[c.from_sentence], order[c.to_sentence]
        if i > j:
            discard(label, "ordem", f"{c.from_sentence} vem depois de {c.to_sentence}")
            continue
        other = next((lab for a, b, lab in state.labels if a <= j and i <= b), None)
        if other:
            discard(label, "sobreposição", f"se sobrepõe a {other}")
            continue
        speaking = {context.sentences[k].clip_id for k in range(i, j + 1)}
        shots, problems = [], []
        for shot_id in dict.fromkeys(c.shot_ids):
            problem = _shot_problem(state, shot_id, speaking, excluded)
            if problem:
                problems.append(problem)
            else:
                shots.append(state.candidates[shot_id])
        if not shots:
            discard(label, problems[0][0], "; ".join(m for _, m in problems))
            continue
        notes = [ItemNote(kind=k, message=f"{label}: {m}; retirada") for k, m in problems]
        if _span_sec(context, i, j) < MIN_COVERAGE_SEC - _EPS:
            extended = _extend(state, i, j, {s.clip_id for s in shots})
            if extended is None:
                discard(
                    label, "curta demais",
                    f"dura {_fmt(_span_sec(context, i, j))}, menos que o mínimo de {_fmt(MIN_COVERAGE_SEC)}, "
                    "e não há frase vizinha livre para estender",
                )
                continue
            notes.append(
                ItemNote(
                    kind="duração",
                    message=f"{label}: {_fmt(_span_sec(context, i, j))} é menos que o mínimo; estendida para "
                    f"{_span_label(context, *extended)} ({_fmt(_span_sec(context, *extended))})",
                )
            )
            i, j = extended
        state.add_coverage(_plan(state, label, i, j, shots, c.reason, notes, "ia"), label)

    for n, b in enumerate(decision.breaths, start=1):
        items += 1
        label = f"respiro {n} (depois de {b.after_sentence})" + (f" [tentativa {attempt}]" if attempt > 1 else "")
        if b.after_sentence not in order:
            discard(label, "ID inexistente", f"a frase {b.after_sentence} não existe (ou foi removida do vídeo)")
            continue
        k = order[b.after_sentence]
        problem = _shot_problem(state, b.shot_id, set(), excluded)
        if problem:
            discard(label, *problem)
        elif shot_length(state.candidates[b.shot_id]) < MIN_BREATH_SHOT_SEC:
            discard(
                label, "tomada curta",
                f"a tomada {b.shot_id} tem {_fmt(shot_length(state.candidates[b.shot_id]))}, menos que {_fmt(MIN_BREATH_SHOT_SEC)}",
            )
        elif k in state.breath_after():
            discard(label, "respiro repetido", f"já há um respiro depois de {b.after_sentence}")
        elif inside := next((lab for i, j, lab in state.labels if i <= k < j), None):
            discard(label, "respiro no meio de cobertura", f"cai no meio de {inside}")
        elif len(state.breaths) >= max_breaths(context):
            discard(label, "respiros demais", f"o máximo é {max_breaths(context)} (um a cada {BREATH_EVERY_SEC:.0f}s de fala)")
        else:
            state.add_breath(AcceptedBreath(**b.model_dump()), label)

    covered = state.covered()
    for d in decision.window_decisions:
        items += 1
        label = f"janela {d.window} ({'manter' if d.action == 'keep' else 'cobrir'})"
        window = windows.get(d.window)
        if window is None:
            discard(label, "ID inexistente", f"a janela {d.window} não existe")
        elif d.action == "keep":
            if window.required:
                discard(
                    label, "janela exige cobertura",
                    f"{window.id} exige cobertura ({', '.join(window.reasons)}): a imagem não pode ficar",
                )
            elif not covered & set(state.window_indices(window)):
                state.kept[window.id] = WindowOutcome(window=window.id, status="mantida", reason=d.reason, origin="ia")
    state.reserved = set()
    state.discarded += discarded
    return items, discarded


def missing_required(state: CoverageState) -> list[tuple[MandatoryWindow, list[str]]]:
    """Janelas que exigem cobertura com frases ainda descobertas (e quais)."""
    covered = state.covered()
    result = []
    for w in state.context.windows:
        lacking = [x for x in w.sentence_ids if state.order[x] not in covered]
        if w.required and lacking:
            result.append((w, lacking))
    return result


def undecided_windows(state: CoverageState) -> list[MandatoryWindow]:
    """Janelas sem decisão: as que exigem cobertura e estão incompletas; as outras sem cobertura nem "manter"."""
    covered = state.covered()
    result = []
    for w in state.context.windows:
        touched = covered & set(state.window_indices(w))
        if (w.required and len(touched) < len(w.sentence_ids)) or (not w.required and not touched and w.id not in state.kept):
            result.append(w)
    return result


def _runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for k in sorted(indices):
        if runs and runs[-1][-1] == k - 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


def fill_windows(state: CoverageState, windows: Iterable[MandatoryWindow], origin: str = "fallback") -> list[str]:
    """Cobre as frases descobertas das janelas só com tomadas relacionadas às tomadas ruins da janela (o que
    está na tela); devolve avisos.

    Sem nenhuma relacionada livre, a imagem original fica. Numa janela que não exige cobertura, o código só
    cobre se as relacionadas dão conta do trecho inteiro (meia cobertura deixaria a imagem ruim à mostra
    do mesmo jeito); numa que exige, cobre até onde elas dão.
    """
    context, warnings = state.context, []

    def keep(window: MandatoryWindow, reason: str) -> None:
        state.kept.setdefault(window.id, WindowOutcome(window=window.id, status="mantida", reason=reason, origin="fallback"))

    for window in windows:
        covered = state.covered()
        lacking = [k for k in state.window_indices(window) if k not in covered]
        screens = state.window_screens([window])
        plans, short = [], None
        for run in _runs(lacking):
            i, j = run[0], run[-1]
            label = f"janela {window.id} ({_span_label(context, i, j)})"
            if _span_sec(context, i, j) < MIN_COVERAGE_SEC - _EPS:
                extended = _extend(state, i, j, set())
                if extended is None:
                    short = f"{label}: {_fmt(_span_sec(context, i, j))}, curta demais e sem frase vizinha livre"
                    continue
                i, j = extended
            speaking = {context.sentences[k].clip_id for k in range(i, j + 1)}
            related = related_shots(_free(state, speaking), screens)
            need = _span_frames(context, i, j)
            shots = _choose(state, [(c, r.score) for c, r in related], need, state.last_clip_before(i))
            state.reserved |= {s.id for s in shots}
            capacity = sum(shot_frames(s, context.fps) for s in shots)
            if not shots or (not window.required and capacity < need - MAX_UNCOVERED_SEC * context.fps):
                short = (
                    f"{label}: nenhuma tomada relacionada livre (vizinha na gravação ou com tags em comum)" if not shots
                    else f"{label}: as tomadas relacionadas livres cobrem só {_fmt(capacity / context.fps)} de {_fmt(need / context.fps)}"
                )
            plans.append((i, j, label, shots, {c.id: r.label for c, r in related}))
        state.reserved = set()
        if short and not window.required:
            warnings.append(f"{short}; imagem original mantida")
            keep(window, "nenhuma tomada relacionada ao que está na tela cobre o trecho")
            continue
        if short:
            warnings.append(f"{short}; janela que exige cobertura fica com a imagem original nesse trecho")
            keep(window, "nenhuma tomada relacionada ao que está na tela")
        for i, j, label, shots, why in plans:
            if not shots:
                continue
            reason = f"Cobre {window.id} ({', '.join(window.reasons)}) com tomada relacionada: " + "; ".join(
                f"{s.id} ({why[s.id]})" for s in shots
            )
            state.add_coverage(_plan(state, label, i, j, shots, reason, [], origin, complete=False), label)
    return warnings


def window_cover(context: CoverageContext, coverages: Iterable[CoverageChoice]) -> dict[str, list[str]]:
    """Janela → IDs das coberturas (C1, C2...) que tocam as frases dela."""
    order = _order(context)
    spans = [(f"C{n}", set(range(order[c.from_sentence], order[c.to_sentence] + 1))) for n, c in enumerate(coverages, 1)]
    return {
        w.id: [cid for cid, idxs in spans if idxs & {order[x] for x in w.sentence_ids}]
        for w in context.windows
    }


def best_free_shot(
    candidates: Iterable[VisualShot], need_sec: float, speaking: set[str], used: set[str]
) -> Optional[VisualShot]:
    """Tomada livre de maior beleza que caiba, fora dos clipes que estão falando (usada pelo mock)."""
    free = [c for c in candidates if c.id not in used and c.clip_id not in speaking]
    if not free:
        return None
    return max(free, key=lambda c: (shot_length(c) >= need_sec - _EPS, c.beauty or 0, c.usable == "sim", shot_length(c)))


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


def _counts(counter: Counter) -> str:
    if len(counter) == 1:
        return next(iter(counter))
    return ", ".join(f"{kind}: {n}" for kind, n in counter.most_common())


def ai_digest(run: CoverageRun) -> Optional[str]:
    """Uma linha para o terminal: itens da IA, ajustados e descartados por categoria (None se nada mudou)."""
    adjusted = [x for x in [*run.coverages, *run.breaths] if x.origin == "ia-ajustada"]
    if not run.ai_items or not (adjusted or run.discarded):
        return None
    parts = [f"cobertura da IA: {run.ai_items} itens"]
    if adjusted:
        kinds = Counter(k for x in adjusted for k in dict.fromkeys(n.kind for n in x.adjustments))
        parts.append(f"{len(adjusted)} ajustados ({_counts(kinds)})")
    if run.discarded:
        parts.append(f"{len(run.discarded)} descartados ({_counts(Counter(d.kind for d in run.discarded))})")
    return ", ".join(parts) + "; detalhes no edit_report.md"


def _finish(run: CoverageRun, state: CoverageState) -> CoverageRun:
    context, order = state.context, state.order
    run.coverages = sorted(state.coverages, key=lambda c: order[c.from_sentence])
    run.breaths = sorted(state.breaths, key=lambda b: order[b.after_sentence])
    run.discarded = state.discarded
    run.kept_windows = state.kept
    run.window_cover = window_cover(context, run.coverages)
    by_id = {f"C{n}": c for n, c in enumerate(run.coverages, start=1)}
    run.window_outcomes = []
    for w in context.windows:
        covers = run.window_cover.get(w.id, [])
        if covers:
            first = by_id[covers[0]]
            run.window_outcomes.append(WindowOutcome(window=w.id, status="coberta", by=covers, reason=first.reason, origin=first.origin))
        elif w.id in state.kept:
            run.window_outcomes.append(state.kept[w.id])
        else:
            run.window_outcomes.append(WindowOutcome(window=w.id, status="descoberta", reason="sem decisão"))
    run.notes = [n.message for c in [*run.coverages, *run.breaths] for n in c.adjustments] + [d.message for d in run.discarded]
    digest = ai_digest(run)
    if digest:
        run.warnings.insert(0, digest)
    missing = [w.id for w, _ in missing_required(state)]
    if missing:
        run.warnings.append(f"janelas que exigem cobertura sem cobertura completa: {', '.join(missing)}")
    return run


def _accepted_lines(state: CoverageState) -> list[str]:
    """Itens já aceitos, para a nova tentativa (a IA não os repete nem reusa as tomadas)."""
    lines = [
        f"cobertura {c.from_sentence}–{c.to_sentence}: {', '.join(c.shot_ids)}"
        for c in sorted(state.coverages, key=lambda c: state.order[c.from_sentence])
    ]
    lines += [f"respiro depois de {b.after_sentence}: {b.shot_id}" for b in state.breaths]
    lines += [f"janela {w} mantida" for w, o in state.kept.items() if o.origin == "ia"]
    return lines


def run_coverage(
    director: Optional[CoverageDirector], context: CoverageContext, all_shots: Iterable[VisualShot]
) -> CoverageRun:
    """Pede a decisão ao Diretor de cobertura, aceita item a item e, se preciso, tenta de novo uma vez
    só com os problemas e pedindo só o que falta; o que ficar sem decisão vai para o fallback.

    - Sem Diretor (`none`) ou sem tomadas candidatas: só o fallback nas janelas com imagem ruim.
    - Nova tentativa: janela que exige cobertura descoberta, mais de `RETRY_DISCARD_RATIO` dos itens
      descartados ou resposta que não pôde ser interpretada.
    Erros de comunicação (autenticação, cota, timeout) sobem para o chamador, como no Diretor.
    """
    run = _new_run(director, context)
    state = CoverageState(context)
    if director is None or not context.candidates or not context.sentences:
        run.origin = "fallback"
        if director is None:
            run.origin_reason = "IA de cobertura desligada (--coverage-provider none): só as janelas com imagem ruim"
        else:
            run.origin_reason = "nenhuma tomada candidata a cobertura" if context.sentences else "nenhuma frase mantida"
        run.warnings += fill_windows(state, context.windows)
        run.summary = f"{len(state.coverages)} cobertura(s) das janelas com imagem ruim, escolhidas pelo código."
        return _finish(run, state)

    excluded = excluded_shots(context, all_shots)
    ask = context
    errors: list[str] = []
    summaries: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        director.last_input_tokens = None
        director.last_output_tokens = None
        director.last_api_attempts = 1
        started = time.perf_counter()
        error: Optional[AIResponseValidationError] = None
        decision: Optional[CoverageDecision] = None
        items, discarded = 0, []
        try:
            decision = director.decide(ask)
        except AIResponseValidationError as exc:
            error = exc
            errors.append(exc.details or str(exc))
        run.responses.append(decision)
        if decision is not None:
            items, discarded = review_decision(decision, state, excluded, attempt)
            run.ai_items += items
            if decision.summary.strip():
                summaries.append(decision.summary.strip())
        missing = missing_required(state)
        too_many = items > 0 and len(discarded) > RETRY_DISCARD_RATIO * items
        retry = error is not None or bool(missing) or too_many
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
                success=not retry,
                error_type=type(error).__name__ if error else ("RespostaIncompleta" if retry else None),
            )
        )
        if not retry or attempt == MAX_ATTEMPTS:
            break
        problems = [error.details or str(error)] if error else []
        problems += [d.message for d in discarded]
        problems += [
            f"janela {w.id} ({w.span_label}, {', '.join(w.reasons)}) exige cobertura: faltam {', '.join(lacking)}"
            for w, lacking in missing
        ]
        if too_many:
            problems.append(f"{len(discarded)} de {items} itens descartados: substitua-os por itens válidos")
        ask = context.model_copy(update={"previous_error": "\n".join(f"- {p}" for p in problems), "accepted_items": _accepted_lines(state)})

    from_ai = bool(state.coverages or state.breaths or any(o.origin == "ia" for o in state.kept.values()))
    pending = undecided_windows(state)
    run.warnings += fill_windows(state, pending)
    run.summary = " ".join(summaries)
    if not from_ai:
        run.origin = "fallback"
        run.origin_reason = f"nenhum item aproveitável da IA em {len(run.attempts)} tentativa(s): só as janelas com imagem ruim"
        if errors:
            run.warnings.append(f"Cobertura da IA ignorada: {' | '.join(errors)}")
        run.summary = f"{len(state.coverages)} cobertura(s) das janelas com imagem ruim, escolhidas pelo código."
    elif pending:
        run.origin = "ia+fallback"
        run.origin_reason = (
            f"janelas sem decisão da IA ({', '.join(w.id for w in pending)}): o código cobriu com tomada relacionada ou manteve"
        )
    else:
        run.origin = "ia"
    return _finish(run, state)
