"""Cobertura pelo conteúdo da fala: validação item a item com evidências conferidas pelo código, segundos
decididos pelo código, limite de cobertura, tapa de tela uniforme e execução com nova tentativa.

A IA escolhe frases e tomadas (IDs) e cita evidências; o código confere, decide os segundos e recusa **o
item**, não a resposta:

- **Evidências (`evidence.py`):** cada cobertura traz um trecho literal da fala coberta e uma tag ou
  trecho da descrição de cada tomada, e as duas precisam falar da mesma coisa (palavra de conteúdo em
  comum, com plural e sinônimos simples). Sem isso, o item sai com o motivo; tomada da lista sem a evidência
  de imagem sai da lista. Na dúvida, a pessoa continua na tela.
- **Pessoa falando para a câmera** (tomada principal com rosto e utilizável): só com evidência forte (a
  palavra em comum é tag da tomada, o assunto principal da imagem).
- **Limite:** a cobertura de conteúdo soma no máximo `MAX_COVERAGE_RATIO` da espinha; o que passa sai,
  da evidência mais fraca para a mais forte, com aviso.
- **Imagem ruim (`BadSpan`):** não obriga cobertura. Uma cobertura que toca um trecho ruim fica limitada
  aos segundos ruins. Sem cobertura válida, a imagem original fica, com aviso. Só a tela uniforme (parede,
  tampa) pode ser tapada sem evidência de fala: com tomada relacionada (`related.py`), pelo tempo dela, e o
  código completa ou faz o tapa sozinho (fallback).
- **Duração:** cada tomada mostra até `MAX_COVERAGE_SEC`; um trecho mais longo é dividido entre as tomadas
  listadas e, se elas não bastam, a cobertura é **encurtada** em volta de onde a evidência é dita (nunca
  completada com tomada que a IA não escolheu, salvo na tela uniforme). Trecho menor que `MIN_COVERAGE_SEC`
  estende para a frase vizinha livre; sem vizinha, o item sai.
- **Respiros:** só entre blocos de assunto (depois da última frase de um bloco), com tomada que tenha
  relação com o bloco anterior ou o seguinte (evidência de imagem ligada ao assunto/lugar do bloco, ou
  tomada relacionada à tela dele).
- **Nova tentativa** só quando a resposta não pôde ser interpretada ou mais de `RETRY_DISCARD_RATIO` dos
  itens foram descartados; ela leva só os problemas e os itens já aceitos.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ai_video_editor.coverage import evidence as ev
from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.related import relation, related_shots
from ai_video_editor.coverage.schema import (
    AcceptedBreath,
    AcceptedCoverage,
    BadSpan,
    BadSpanOutcome,
    CoverageContext,
    CoverageDecision,
    CoverageRun,
    DiscardedItem,
    Evidence,
    ItemNote,
    Topic,
    VisualShot,
)
from ai_video_editor.coverage.topics import run_topics
from ai_video_editor.director.service import record_effective_model
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics

# Duração de uma cobertura no vídeo final (mínimo) e de cada tomada dentro dela (máximo).
MIN_COVERAGE_SEC = 1.5
MAX_COVERAGE_SEC = 8.0
# Cobertura de conteúdo somada: no máximo esta fração da espinha (a pessoa também conta a história).
MAX_COVERAGE_RATIO = 0.30
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
# Trecho com imagem ruim "coberto" mesmo com esta sobra descoberta.
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


def coverage_limit_frames(context: CoverageContext) -> int:
    return int(MAX_COVERAGE_RATIO * context.spine_duration_sec * context.fps)


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


def _span_label(context: CoverageContext, i: int, j: int) -> str:
    a, b = context.sentences[i].id, context.sentences[j].id
    return a if i == j else f"{a}–{b}"


def _frames(context: CoverageContext, i: int, j: int) -> tuple[int, int]:
    return context.sentences[i].start_frame, context.sentences[j].end_frame


@dataclass
class CoverageState:
    """O que já foi aceito (entre tentativas e no fallback): coberturas, respiros e descartes."""

    context: CoverageContext
    coverages: list[AcceptedCoverage] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # rótulo de cada cobertura (mensagens)
    breaths: list[AcceptedBreath] = field(default_factory=list)
    discarded: list[DiscardedItem] = field(default_factory=list)
    used: dict[str, str] = field(default_factory=dict)  # tomada → item que a usa
    # Tomadas citadas na resposta em análise: o código não as usa para completar outro item.
    reserved: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.order = {s.id: i for i, s in enumerate(self.context.sentences)}
        self.candidates = {c.id: c for c in self.context.candidates}
        self.screens = {s.id: s for s in self.context.screen_shots}
        self.topic_of: dict[int, Topic] = {}
        for topic in self.context.topics:
            for k in range(self.order[topic.from_sentence], self.order[topic.to_sentence] + 1):
                self.topic_of[k] = topic

    def overlapping(self, start: int, end: int) -> Optional[str]:
        return next((lab for c, lab in zip(self.coverages, self.labels) if c.start_frame < end and start < c.end_frame), None)

    def breath_after(self) -> set[int]:
        return {self.order[b.after_sentence] for b in self.breaths}

    def role_uses(self) -> Counter:
        return Counter(self.candidates[x].role for c in self.coverages for x in c.shot_ids if x in self.candidates)

    def add_coverage(self, coverage: AcceptedCoverage, label: str) -> None:
        self.coverages.append(coverage)
        self.labels.append(label)
        for shot_id in coverage.shot_ids:
            self.used[shot_id] = label

    def remove_coverage(self, coverage: AcceptedCoverage) -> str:
        k = next(n for n, c in enumerate(self.coverages) if c is coverage)
        label = self.labels.pop(k)
        self.coverages.pop(k)
        for shot_id in coverage.shot_ids:
            self.used.pop(shot_id, None)
        return label

    def add_breath(self, breath: AcceptedBreath, label: str) -> None:
        self.breaths.append(breath)
        self.used[breath.shot_id] = label

    def spans_touching(self, start: int, end: int, uniform: Optional[bool] = None) -> list[BadSpan]:
        return [
            s for s in self.context.bad_spans
            if s.start_frame < end and start < s.end_frame and (uniform is None or s.uniform == uniform)
        ]

    def span_screens(self, spans: Iterable[BadSpan]) -> list[VisualShot]:
        ids = dict.fromkeys(x for s in spans for x in s.shot_ids)
        return [self.screens[x] for x in ids if x in self.screens]

    def topic_screens(self, topic: Topic) -> list[VisualShot]:
        i, j = self.order[topic.from_sentence], self.order[topic.to_sentence]
        ids = dict.fromkeys(x for k in range(i, j + 1) for x in self.context.sentences[k].screen)
        return [self.screens[x] for x in ids if x in self.screens]


def _extend(state: CoverageState, i: int, j: int, shot_clips: set[str]) -> Optional[tuple[int, int]]:
    """Estende um trecho curto para frases vizinhas livres (a mais curta das duas; empate: a seguinte) até o
    mínimo; None se não der.

    Não atravessa respiro nem cobertura aceita e não pega frase de um clipe usado nas tomadas.
    """
    context, after = state.context, state.breath_after()
    sentences, fps = context.sentences, context.fps

    def free(k: int) -> bool:
        return state.overlapping(sentences[k].start_frame, sentences[k].end_frame) is None and sentences[k].clip_id not in shot_clips

    while (sentences[j].end_frame - sentences[i].start_frame) / fps < MIN_COVERAGE_SEC - _EPS:
        options = []
        nxt, prv = j + 1, i - 1
        if nxt < len(sentences) and free(nxt) and j not in after:
            options.append((sentences[nxt].end_frame - sentences[i].start_frame, 0, (i, nxt)))
        if prv >= 0 and free(prv) and prv not in after:
            options.append((sentences[j].end_frame - sentences[prv].start_frame, 1, (prv, j)))
        if not options:
            return None
        i, j = min(options)[2]
    return i, j


def _window(length: int, center: float, lo: int, hi: int) -> tuple[int, int]:
    """Trecho de `length` frames em volta de `center`, dentro de [lo, hi)."""
    length = min(length, hi - lo)
    start = int(round(center - length / 2))
    start = min(max(start, lo), hi - length)
    return start, start + length


def _choose(state: CoverageState, pool: list[tuple[VisualShot, tuple]], need: int, prev_clip: Optional[str]) -> list[VisualShot]:
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


def _trim_shots(context: CoverageContext, label: str, shots: list[VisualShot], need: int, notes: list[ItemNote]) -> list[VisualShot]:
    """Tira do fim as tomadas que não cabem (cada uma precisa de pelo menos `MIN_COVERAGE_SEC`)."""
    shots, dropped = list(shots), []
    while len(shots) > 1 and need < len(shots) * MIN_COVERAGE_SEC * context.fps - _EPS:
        dropped.append(shots.pop().id)
    if dropped:
        notes.append(ItemNote(kind="duração", message=f"{label}: {_fmt(need / context.fps)} não comportam todas as tomadas; retiradas {', '.join(dropped)}"))
    return shots


def _place(
    state: CoverageState, label: str, i: int, j: int, shots: list[VisualShot], hit: Optional[ev.SpeechHit],
    notes: list[ItemNote], uniform: bool = False,
) -> Optional[tuple[int, int, list[str]]]:
    """(início, fim, trechos ruins tocados) da cobertura na espinha; None se ficou curta demais.

    Tocando imagem ruim (ou numa tela uniforme), só os segundos ruins; senão as frases inteiras. Se as
    tomadas não bastam, encurta em volta de onde a evidência é dita (na tela uniforme não: o código completa
    depois, `_complete_uniform`).
    """
    context, fps = state.context, state.context.fps
    a, b = _frames(context, i, j)
    spans = state.spans_touching(a, b, uniform=True if uniform else None)
    lo, hi = a, b
    if spans:
        lo, hi = max(a, min(s.start_frame for s in spans)), min(b, max(s.end_frame for s in spans))
        if (lo, hi) != (a, b):
            notes.append(
                ItemNote(
                    kind="imagem ruim",
                    message=f"{label}: limitada aos segundos com imagem ruim ({', '.join(s.id for s in spans)}, {_fmt((hi - lo) / fps)})",
                )
            )
    capacity = hi - lo if uniform else sum(shot_frames(s, fps) for s in shots)
    length = min(hi - lo, capacity)
    min_frames = round(MIN_COVERAGE_SEC * fps)
    if length < hi - lo:
        notes.append(ItemNote(kind="duração", message=f"{label}: encurtada; as tomadas cobrem {_fmt(capacity / fps)} de {_fmt((hi - lo) / fps)}"))
    if hit is not None and not spans:
        s = context.sentences[i + hit.sentence]
        center = s.start_frame + hit.fraction * (s.end_frame - s.start_frame)
    else:
        center = lo + length / 2
    start, end = _window(length, center, lo, hi)
    if end - start < min_frames:
        if capacity < min_frames or b - a < min_frames:
            return None
        start, end = _window(min_frames, (start + end) / 2, a, b)
    return start, end, [s.id for s in state.spans_touching(start, end)]


def _uniform_relation(state: CoverageState, shots: list[VisualShot], spans: list[BadSpan], label: str, notes: list[ItemNote]) -> list[tuple[VisualShot, str]]:
    """Tomadas relacionadas com a tela uniforme (as outras saem da lista, com nota)."""
    screens = state.span_screens(spans)
    kept, dropped = [], []
    for shot in shots:
        r = relation(shot, screens)
        if r is None:
            dropped.append(shot.id)
        else:
            kept.append((shot, r.label))
    if dropped and kept:
        notes.append(ItemNote(kind="evidência", message=f"{label}: sem relação com a tela uniforme; retiradas {', '.join(dropped)}"))
    return kept


def _complete_uniform(state: CoverageState, label: str, shots: list[VisualShot], spans: list[BadSpan], need: int, notes: list[ItemNote]) -> list[VisualShot]:
    """Na tela uniforme, o código completa com tomadas relacionadas livres (nunca do clipe que fala)."""
    fps = state.context.fps
    capacity = sum(shot_frames(s, fps) for s in shots)
    if capacity >= need:
        return []
    speaking = {s.clip_id for s in spans}
    pool = [(c, r.score) for c, r in related_shots(_free(state, speaking, [s.id for s in shots]), state.span_screens(spans))]
    added = _choose(state, pool, need - capacity, shots[-1].clip_id if shots else None)
    if added:
        notes.append(ItemNote(kind="duração", message=f"{label}: tela uniforme completada com {', '.join(s.id for s in added)}"))
    return added


def review_decision(
    decision: CoverageDecision, state: CoverageState, excluded: dict[str, str], attempt: int
) -> tuple[int, list[DiscardedItem]]:
    """Aceita os itens válidos da resposta (evidências conferidas, segundos do código) em `state`; devolve
    (nº de itens, descartados)."""
    context, order = state.context, state.order
    fps = context.fps
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

        texts = [context.sentences[k].text for k in range(i, j + 1)]
        hit = ev.find_speech(texts, c.speech_evidence) if ev.concepts(c.speech_evidence) else None
        a, b = _frames(context, i, j)
        uniform_spans = state.spans_touching(a, b, uniform=True)
        if hit is None and uniform_spans:
            # Sem evidência de fala: só um tapa da tela uniforme, com tomada relacionada.
            related = _uniform_relation(state, shots, uniform_spans, label, notes)
            if not related:
                discard(
                    label, "evidência",
                    "sem evidência de fala válida e nenhuma tomada relacionada à tela uniforme "
                    f"({', '.join(s.id for s in uniform_spans)})",
                )
                continue
            shots = [s for s, _ in related]
            placed = _place(state, label, i, j, shots, None, notes, uniform=True)
            if placed is None:
                discard(label, "curta demais", "a tela uniforme é curta demais para uma cobertura")
                continue
            start, end, touched = placed
            shots = _trim_shots(context, label, shots, end - start, notes)
            added = _complete_uniform(state, label, shots, uniform_spans, end - start, notes)
            capacity = sum(shot_frames(s, fps) for s in shots + added)
            if capacity < end - start:
                notes.append(ItemNote(kind="duração", message=f"{label}: encurtada; as tomadas relacionadas cobrem {_fmt(capacity / fps)} de {_fmt((end - start) / fps)}"))
                end = start + capacity
                touched = [s.id for s in state.spans_touching(start, end)]
            other = state.overlapping(start, end)
            if other:
                discard(label, "sobreposição", f"se sobrepõe a {other}")
                continue
            ev_note = Evidence(relation="; ".join(f"{s.id} ({why})" for s, why in related))
            state.add_coverage(
                AcceptedCoverage(
                    from_sentence=c.from_sentence, to_sentence=c.to_sentence, shot_ids=[s.id for s in shots + added],
                    added_shots=[s.id for s in added], reason=c.reason, kind="tela_uniforme", evidence=ev_note,
                    origin="ia-ajustada" if notes else "ia", adjustments=notes, start_frame=start, end_frame=end, bad_spans=touched,
                ),
                label,
            )
            continue
        if hit is None:
            if not c.speech_evidence.strip():
                discard(label, "evidência de fala", "sem evidência de fala (um trecho copiado das frases cobertas)")
            else:
                discard(label, "evidência de fala", f"a evidência de fala '{c.speech_evidence}' não está no texto de {span}")
            continue
        matching = [s for s in shots if ev.image_matches(c.image_evidence, s.description, s.tags)]
        if not matching:
            what = f"'{c.image_evidence}'" if c.image_evidence.strip() else "(vazia)"
            discard(
                label, "evidência de imagem",
                f"a evidência de imagem {what} não está na análise de {', '.join(s.id for s in shots)} (descrição e tags)",
            )
            continue
        if len(matching) < len(shots):
            dropped = [s.id for s in shots if s not in matching]
            notes.append(ItemNote(kind="evidência", message=f"{label}: {', '.join(dropped)} não mostra(m) '{c.image_evidence}'; retirada(s)"))
        shots = matching
        shared = ev.shared_concepts(c.speech_evidence, c.image_evidence)
        if not shared:
            discard(
                label, "evidência sem relação",
                f"'{c.speech_evidence}' e '{c.image_evidence}' não falam da mesma coisa (nenhuma palavra de conteúdo em comum)",
            )
            continue
        strong = any(ev.is_strong(shared, s.tags) for s in shots)
        if (b - a) / fps < MIN_COVERAGE_SEC - _EPS:
            extended = _extend(state, i, j, {s.clip_id for s in shots})
            if extended is None:
                discard(
                    label, "curta demais",
                    f"dura {_fmt((b - a) / fps)}, menos que o mínimo de {_fmt(MIN_COVERAGE_SEC)}, e não há frase vizinha livre para estender",
                )
                continue
            notes.append(
                ItemNote(
                    kind="duração",
                    message=f"{label}: {_fmt((b - a) / fps)} é menos que o mínimo; estendida para "
                    f"{_span_label(context, *extended)} ({_fmt((_frames(context, *extended)[1] - _frames(context, *extended)[0]) / fps)})",
                )
            )
            hit = ev.SpeechHit(sentence=hit.sentence + (i - extended[0]), fraction=hit.fraction)
            i, j = extended
        placed = _place(state, label, i, j, shots, hit, notes)
        if placed is None:
            discard(label, "curta demais", "o trecho com imagem ruim e as tomadas não chegam ao mínimo de uma cobertura")
            continue
        start, end, touched = placed
        camera = [s.id for s in context.sentences if s.to_camera and s.start_frame < end and start < s.end_frame]
        if camera and not strong:
            discard(
                label, "pessoa para a câmera",
                f"{', '.join(camera)}: a pessoa fala para a câmera; só com evidência forte (a palavra em comum, "
                f"{', '.join(shared)}, precisa ser tag da tomada)",
            )
            continue
        other = state.overlapping(start, end)
        if other:
            discard(label, "sobreposição", f"se sobrepõe a {other}")
            continue
        shots = _trim_shots(context, label, shots, end - start, notes)
        state.add_coverage(
            AcceptedCoverage(
                from_sentence=context.sentences[i].id, to_sentence=context.sentences[j].id, shot_ids=[s.id for s in shots],
                speech_evidence=c.speech_evidence, image_evidence=c.image_evidence, reason=c.reason,
                evidence=Evidence(speech=c.speech_evidence, image=c.image_evidence, shared=[ev.label(w) for w in shared], strong=strong),
                origin="ia-ajustada" if notes else "ia", adjustments=notes, start_frame=start, end_frame=end, bad_spans=touched,
            ),
            label,
        )

    ends = {order[t.to_sentence]: t for t in context.topics[:-1]}
    wall = _wall_shots(state)
    for n, b in enumerate(decision.breaths, start=1):
        items += 1
        label = f"respiro {n} (depois de {b.after_sentence})" + (f" [tentativa {attempt}]" if attempt > 1 else "")
        if b.after_sentence not in order:
            discard(label, "ID inexistente", f"a frase {b.after_sentence} não existe (ou foi removida do vídeo)")
            continue
        k = order[b.after_sentence]
        if k not in ends:
            allowed = ", ".join(t.to_sentence for t in context.topics[:-1][:6]) or "nenhuma"
            discard(label, "respiro fora da troca de assunto", f"{b.after_sentence} não fecha um bloco de assunto (respiros só depois de: {allowed}...)")
            continue
        problem = _shot_problem(state, b.shot_id, set(), excluded)
        if problem:
            discard(label, *problem)
            continue
        if b.shot_id in wall:
            discard(label, "reservada para tela uniforme", f"a tomada {b.shot_id} é a relacionada que tapa a tela uniforme {wall[b.shot_id]}")
            continue
        shot = state.candidates[b.shot_id]
        if shot_length(shot) < MIN_BREATH_SHOT_SEC:
            discard(label, "tomada curta", f"a tomada {b.shot_id} tem {_fmt(shot_length(shot))}, menos que {_fmt(MIN_BREATH_SHOT_SEC)}")
            continue
        why = _breath_relation(state, shot, b.image_evidence, [ends[k], state.topic_of.get(k + 1)])
        sentences = context.sentences
        if why is None:
            discard(
                label, "respiro sem relação",
                f"a tomada {b.shot_id} não tem relação com os blocos vizinhos (evidência de imagem ligada ao assunto ou "
                "tomada relacionada à tela deles)",
            )
        elif k in state.breath_after():
            discard(label, "respiro repetido", f"já há um respiro depois de {b.after_sentence}")
        elif k + 1 < len(sentences) and (inside := _covering(state, sentences[k].end_frame, sentences[k + 1].start_frame)):
            discard(label, "respiro no meio de cobertura", f"cai no meio de {inside}")
        elif len(state.breaths) >= max_breaths(context):
            discard(label, "respiros demais", f"o máximo é {max_breaths(context)} (um a cada {BREATH_EVERY_SEC:.0f}s de fala)")
        else:
            state.add_breath(AcceptedBreath(**b.model_dump(), relation=why), label)
    state.reserved = set()
    state.discarded += discarded
    return items, discarded


def _wall_shots(state: CoverageState) -> dict[str, str]:
    """Tomadas que o código usaria para tapar as telas uniformes ainda descobertas (tomada → trecho): tapar a
    parede vale mais que um respiro."""
    reserved: dict[str, str] = {}
    for span in state.context.bad_spans:
        if not span.uniform or state.overlapping(span.start_frame, span.end_frame):
            continue
        free = [
            c for c in state.context.candidates
            if c.id not in state.used and c.id not in reserved and c.clip_id != span.clip_id and shot_length(c) >= MIN_COVERAGE_SEC
        ]
        related = related_shots(free, state.span_screens([span]))
        for shot in _choose(state, [(c, r.score) for c, r in related], span.end_frame - span.start_frame, None):
            reserved[shot.id] = span.id
    return reserved


def _covering(state: CoverageState, end_frame: int, next_start: int) -> Optional[str]:
    """Cobertura que passa por cima da troca entre duas frases (do fim de uma ao início da seguinte)."""
    return next((lab for c, lab in zip(state.coverages, state.labels) if c.start_frame < end_frame and c.end_frame > next_start), None)


def _breath_relation(state: CoverageState, shot: VisualShot, image_evidence: str, topics: list[Optional[Topic]]) -> Optional[str]:
    """Por que a tomada do respiro tem a ver com o bloco anterior ou o seguinte (None = não tem)."""
    for topic in [t for t in topics if t is not None]:
        about = f"{topic.topic} {topic.place}"
        if ev.image_matches(image_evidence, shot.description, shot.tags) and ev.shared_concepts(image_evidence, about):
            return f"'{image_evidence}' ↔ {topic.id} ({topic.topic})"
    for topic in [t for t in topics if t is not None]:
        r = relation(shot, state.topic_screens(topic))
        if r is not None:
            return f"relacionada a {topic.id} ({r.label})"
    return None


def apply_limit(state: CoverageState) -> list[str]:
    """Tira coberturas de conteúdo até a soma caber em `MAX_COVERAGE_RATIO` da espinha: sai primeiro a que
    não toca imagem ruim, depois a de evidência fraca, depois a mais longa; devolve avisos."""
    context = state.context
    limit = coverage_limit_frames(context)
    content = [c for c in state.coverages if c.kind == "conteudo"]
    total = sum(c.end_frame - c.start_frame for c in content)
    warnings = []
    ranked = sorted(content, key=lambda c: (bool(c.bad_spans), c.evidence.strong, -(c.end_frame - c.start_frame)))
    for coverage in ranked:
        if total <= limit:
            break
        label = state.remove_coverage(coverage)
        seconds = (coverage.end_frame - coverage.start_frame) / context.fps
        total -= coverage.end_frame - coverage.start_frame
        message = (
            f"{label}: ficou de fora pelo limite de cobertura ({MAX_COVERAGE_RATIO:.0%} do vídeo, "
            f"{_fmt(limit / context.fps)}); {_fmt(seconds)} com {', '.join(coverage.shot_ids)}"
        )
        state.discarded.append(DiscardedItem(attempt=0, item=label, kind="limite de cobertura", message=message))
        warnings.append(message)
    return warnings


def fill_uniform(state: CoverageState, origin: str = "fallback") -> list[str]:
    """Tapa as telas uniformes ainda descobertas com tomadas relacionadas (vizinha na gravação ou tags em
    comum com a tela), pelo tempo da tela uniforme; sem nenhuma, a imagem original fica, com aviso."""
    context, fps, warnings = state.context, state.context.fps, []
    for span in context.bad_spans:
        if not span.uniform or state.overlapping(span.start_frame, span.end_frame):
            continue
        i, j = state.order[span.sentence_ids[0]], state.order[span.sentence_ids[-1]]
        label = f"tela uniforme {span.id} ({span.span_label})"
        related = related_shots(_free(state, {span.clip_id}), state.span_screens([span]))
        if not related:
            warnings.append(f"{label}: nenhuma tomada relacionada livre (vizinha na gravação ou com tags em comum); imagem original mantida")
            continue
        a, b = _frames(context, i, j)
        need = span.end_frame - span.start_frame
        start, end = span.start_frame, span.end_frame
        min_frames = round(MIN_COVERAGE_SEC * fps)
        if need < min_frames:
            start, end = _window(min_frames, (start + end) / 2, a, b)
        shots = _choose(state, [(c, r.score) for c, r in related], end - start, None)
        notes: list[ItemNote] = []
        capacity = sum(shot_frames(s, fps) for s in shots)
        if capacity < end - start:
            notes.append(ItemNote(kind="duração", message=f"{label}: encurtada; as tomadas relacionadas cobrem {_fmt(capacity / fps)} de {_fmt((end - start) / fps)}"))
            end = start + capacity
        if state.overlapping(start, end) or end - start < 1:
            continue
        why = {c.id: r.label for c, r in related}
        state.add_coverage(
            AcceptedCoverage(
                from_sentence=span.sentence_ids[0], to_sentence=span.sentence_ids[-1], shot_ids=[s.id for s in shots],
                reason=f"Tapa a tela uniforme ({span.id}) com tomada relacionada: " + "; ".join(f"{s.id} ({why[s.id]})" for s in shots),
                kind="tela_uniforme", origin=origin, adjustments=notes, evidence=Evidence(relation="; ".join(why[s.id] for s in shots)),
                start_frame=start, end_frame=end, bad_spans=[s.id for s in state.spans_touching(start, end)],
            ),
            label,
        )
    return warnings


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
        bad_spans=context.bad_spans,
        fps=context.fps,
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


def span_outcomes(state: CoverageState) -> list[BadSpanOutcome]:
    fps = state.context.fps
    outcomes = []
    for span in state.context.bad_spans:
        hits = [(n, c) for n, c in enumerate(state.coverages, start=1) if c.start_frame < span.end_frame and span.start_frame < c.end_frame]
        covered = sum(min(c.end_frame, span.end_frame) - max(c.start_frame, span.start_frame) for _, c in hits)
        missing = (span.end_frame - span.start_frame - covered) / fps
        if not hits:
            status, reason = "mantido", "nenhuma imagem que combine com a fala" + ("" if not span.uniform else " nem tomada relacionada")
        elif missing <= MAX_UNCOVERED_SEC + _EPS:
            status, reason = "coberto", ""
        else:
            status, reason = "parcial", f"{_fmt(missing)} ficam com a imagem original"
        outcomes.append(BadSpanOutcome(span=span.id, status=status, by=[f"C{n}" for n, _ in hits], reason=reason))
    return outcomes


def _finish(run: CoverageRun, state: CoverageState) -> CoverageRun:
    from ai_video_editor.coverage.plan import plan_lines

    context, order = state.context, state.order
    pairs = sorted(zip(state.coverages, state.labels), key=lambda x: x[0].start_frame)
    state.coverages, state.labels = [c for c, _ in pairs], [lab for _, lab in pairs]
    run.coverages = state.coverages
    run.breaths = sorted(state.breaths, key=lambda b: order[b.after_sentence])
    run.discarded = state.discarded
    run.topics = context.topics
    run.span_outcomes = span_outcomes(state)
    run.covered_sec = round(sum(c.end_frame - c.start_frame for c in run.coverages if c.kind == "conteudo") / context.fps, 3)
    run.notes = [n.message for c in [*run.coverages, *run.breaths] for n in c.adjustments] + [d.message for d in run.discarded]
    digest = ai_digest(run)
    if digest:
        run.warnings.insert(0, digest)
    spans = {s.id: s for s in context.bad_spans}
    for o in run.span_outcomes:
        if o.status != "coberto":
            span = spans[o.span]
            what = "tela uniforme" if span.uniform else ", ".join(span.reasons)
            head = "imagem ruim mantida" if o.status == "mantido" else "imagem ruim coberta em parte"
            run.warnings.append(f"{head} em {span.id} ({span.span_label}, {_fmt(span.duration_sec(context.fps))}, {what}): {o.reason}")
    run.plan = plan_lines(run, context)
    return run


def _accepted_lines(state: CoverageState) -> list[str]:
    """Itens já aceitos, para a nova tentativa (a IA não os repete nem reusa as tomadas)."""
    lines = [
        f"cobertura {c.from_sentence}–{c.to_sentence}: {', '.join(c.shot_ids)}"
        for c in sorted(state.coverages, key=lambda c: c.start_frame)
    ]
    lines += [f"respiro depois de {b.after_sentence}: {b.shot_id}" for b in state.breaths]
    return lines


def run_coverage(
    director: Optional[CoverageDirector], context: CoverageContext, all_shots: Iterable[VisualShot]
) -> CoverageRun:
    """Mapa de assuntos (1ª chamada) e cobertura (2ª), item a item, com uma nova tentativa só com os
    problemas; depois o limite de cobertura e o tapa das telas uniformes que sobraram (código).

    - Sem Diretor (`none`) ou sem tomadas candidatas: mapa por clipe e só o tapa das telas uniformes.
    - Erros de comunicação (autenticação, cota, timeout) sobem para o chamador, como no Diretor.
    """
    run = _new_run(director, context)
    topics = run_topics(director, context, run)
    context = context.model_copy(update={"topics": topics, "video_summary": run.video_summary})
    state = CoverageState(context)
    if director is None or not context.candidates or not context.sentences:
        run.origin = "fallback"
        if director is None:
            run.origin_reason = "IA de cobertura desligada (--coverage-provider none): só o tapa das telas uniformes"
        else:
            run.origin_reason = "nenhuma tomada candidata a cobertura" if context.sentences else "nenhuma frase mantida"
        run.warnings += fill_uniform(state)
        run.summary = f"{len(state.coverages)} tapa(s) de tela uniforme, escolhidos pelo código."
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
        too_many = items > 0 and len(discarded) > RETRY_DISCARD_RATIO * items
        retry = error is not None or too_many
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
        if too_many:
            problems.append(
                f"{len(discarded)} de {items} itens descartados: corrija as evidências (trecho literal da fala e "
                "tag/descrição da tomada, falando da mesma coisa) ou deixe a pessoa na tela"
            )
        ask = context.model_copy(update={"previous_error": "\n".join(f"- {p}" for p in problems), "accepted_items": _accepted_lines(state)})

    from_ai = bool(state.coverages or state.breaths)
    run.warnings += apply_limit(state)
    before = len(state.coverages)
    run.warnings += fill_uniform(state)
    run.summary = " ".join(summaries)
    if not from_ai:
        run.origin = "fallback"
        run.origin_reason = f"nenhum item aproveitável da IA em {len(run.attempts)} tentativa(s): só o tapa das telas uniformes"
        if errors:
            run.warnings.append(f"Cobertura da IA ignorada: {' | '.join(errors)}")
        if not run.summary:
            run.summary = f"{len(state.coverages)} tapa(s) de tela uniforme, escolhidos pelo código."
    elif len(state.coverages) > before:
        run.origin = "ia+fallback"
        run.origin_reason = "telas uniformes sem cobertura da IA: o código tapou com tomada relacionada"
    else:
        run.origin = "ia"
    return _finish(run, state)
