"""Validação das respostas do Editor de IA: só segurança, item a item (nenhuma regra editorial).

Estrutura (passada 1): IDs de clipes existem; cada clipe no máximo uma vez entre blocos e "fora";
bloco sem clipe válido é descartado. Sem nenhum bloco válido, a resposta inteira é recusada.

Detalhe (passada 2), cada item sozinho (um item inválido é descartado com o motivo):

- manter faixa `de`/`ate`: as duas frases existem, são do mesmo clipe, em ordem, de um clipe do pedaço (e
  dos momentos do pedaço, se o clipe foi dividido); nenhuma frase já está no vídeo;
- manter momento: existe, é de um clipe do pedaço, não foi usado, `duracao_sec` dentro do momento;
- respiro: `depois_de` é uma frase mantida ou um momento de manter deste pedaço; momento do acervo
  pré-selecionado ou de um clipe do pedaço, não usado; não entra no meio de uma cobertura;
- cobertura: frases mantidas e seguidas na faixa principal (sem respiro no meio) e não cobertas antes;
  momentos do acervo pré-selecionado ou de outro clipe do pedaço (nunca o clipe que está falando), não
  usados; evidências de verdade (`material.matched_words`: pelo menos 1 palavra de conteúdo da evidência
  da fala no texto das frases cobertas, e da evidência da imagem na descrição/lugar/tags de CADA momento;
  momento sem palavra em comum sai da cobertura, e a cobertura sem momento algum é descartada).

Em todos: o mesmo trecho de um arquivo não entra duas vezes no vídeo (intervalos por clipe, tolerância
`OVERLAP_TOLERANCE_SEC`). A resposta do pedaço é recusada (nova tentativa com a lista de erros) quando
metade ou mais dos itens é inválida ou quando nada da faixa principal sobra.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial, matched_words, moment_text
from ai_video_editor.editor.schema import (
    Block,
    BlockAnswer,
    BreathAnswer,
    BreathChoice,
    CoverageAnswer,
    CoverageChoice,
    Discarded,
    KeepAnswer,
    KeepRange,
    OutClip,
    StructureAnswer,
)

# Sobreposição tolerada entre dois usos do mesmo arquivo (tempos do Whisper e das fotos não coincidem).
OVERLAP_TOLERANCE_SEC = 0.25
# Folga na comparação da duração pedida com a do momento.
DURATION_TOLERANCE_SEC = 0.05
# Fração de itens inválidos a partir da qual a resposta do pedaço é recusada.
RETRY_INVALID_RATIO = 0.5


def _short(item) -> str:
    data = item.model_dump(exclude_none=True, exclude_defaults=True)
    text = ", ".join(f"{k}={v}" for k, v in data.items())
    return text[:160] or "(vazio)"


# --- passada 1 ---


@dataclass
class StructureCheck:
    blocks: list[Block]
    out: list[OutClip]
    discarded: list[Discarded]

    @property
    def errors(self) -> list[str]:
        return [f"{d.kind} ({d.item}): {d.reason}" for d in self.discarded]


def validate_structure(answer: StructureAnswer, material: EditorMaterial) -> StructureCheck:
    known = set(material.clip_ids)
    seen: set[str] = set()
    blocks: list[Block] = []
    out: list[OutClip] = []
    discarded: list[Discarded] = []
    for raw in answer.blocos:
        clips: list[str] = []
        for clip_id in raw.clipes:
            if clip_id not in known:
                discarded.append(Discarded(kind="bloco", item=f"{raw.id or '?'}: {clip_id}", reason=f"clipe {clip_id} não existe"))
            elif clip_id in seen:
                discarded.append(Discarded(kind="bloco", item=f"{raw.id or '?'}: {clip_id}", reason=f"clipe {clip_id} já usado"))
            else:
                seen.add(clip_id)
                clips.append(clip_id)
        if not clips:
            discarded.append(Discarded(kind="bloco", item=_short(raw), reason="nenhum clipe válido"))
            continue
        target = raw.duracao_alvo_sec if raw.duracao_alvo_sec and raw.duracao_alvo_sec > 0 else None
        blocks.append(Block(
            id=f"B{len(blocks) + 1:02d}", title=raw.titulo, subject=raw.assunto, clip_ids=clips, target_sec=target,
        ))
    for raw in answer.fora:
        if raw.clipe not in known:
            discarded.append(Discarded(kind="fora", item=_short(raw), reason=f"clipe {raw.clipe or '(vazio)'} não existe"))
        elif raw.clipe in seen:
            discarded.append(Discarded(kind="fora", item=_short(raw), reason=f"clipe {raw.clipe} já está num bloco ou em fora"))
        else:
            seen.add(raw.clipe)
            out.append(raw)
    return StructureCheck(blocks=blocks, out=out, discarded=discarded)


# --- passada 2 ---


@dataclass
class Usage:
    """O que já está no vídeo (todos os pedaços aceitos até aqui)."""

    sentences: set[str] = field(default_factory=set)
    moments: set[str] = field(default_factory=set)
    covered: set[str] = field(default_factory=set)
    intervals: dict[str, list[tuple[float, float, str]]] = field(default_factory=dict)  # clipe → (a, b, uso)
    kept_order: list[str] = field(default_factory=list)  # frases mantidas na ordem do vídeo
    used_sec: float = 0.0

    def copy(self) -> "Usage":
        return copy.deepcopy(self)

    def conflict(self, clip_id: str, start: float, end: float) -> Optional[str]:
        for a, b, what in self.intervals.get(clip_id, []):
            if min(b, end) - max(a, start) > OVERLAP_TOLERANCE_SEC:
                return what
        return None

    def take(self, clip_id: str, start: float, end: float, what: str) -> None:
        self.intervals.setdefault(clip_id, []).append((start, end, what))


@dataclass
class Unit:
    """Pedaço da passada 2: um bloco ou, se não coube, parte dele (clipes inteiros ou momentos de um clipe)."""

    unit_id: str
    block: Block
    clip_ids: list[str]
    moment_ids: Optional[list[str]] = None  # só estes momentos (clipe gigante dividido)
    part: int = 1
    parts: int = 1

    def sentence_ids(self, material: EditorMaterial) -> list[str]:
        if self.moment_ids is not None:
            return [s.id for s in material.moment_sentences(self.moment_ids) if s.clip_id in self.clip_ids]
        return [s.id for c in self.clip_ids for s in material.clip_sentences.get(c, [])]

    def moments(self, material: EditorMaterial):
        moments = [m for c in self.clip_ids for m in material.clip_moments.get(c, [])]
        if self.moment_ids is not None:
            wanted = set(self.moment_ids)
            moments = [m for m in moments if m.id in wanted]
        return moments


@dataclass
class DetailCheck:
    keep: list[KeepRange] = field(default_factory=list)
    coverages: list[CoverageChoice] = field(default_factory=list)
    breaths: list[BreathChoice] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    items: int = 0
    usage: Optional[Usage] = None

    @property
    def errors(self) -> list[str]:
        return [f"{d.kind} ({d.item}): {d.reason}" for d in self.discarded]

    @property
    def refused(self) -> Optional[str]:
        """Por que a resposta inteira é recusada (None = aceita, com os itens inválidos descartados)."""
        if not self.keep:
            return "nada válido em manter"
        invalid = sum(1 for d in self.discarded if d.kind in ("manter", "cobertura", "respiro"))
        if self.items and invalid / self.items >= RETRY_INVALID_RATIO:
            return f"{invalid} de {self.items} itens inválidos"
        return None


def _moment_span(moment, duration: Optional[float]) -> tuple[float, float]:
    length = moment.duration if duration is None else min(duration, moment.duration)
    return moment.start_sec, round(moment.start_sec + length, 3)


def _duration_problem(moment, duration: Optional[float]) -> Optional[str]:
    if duration is None:
        return None
    if duration <= 0:
        return f"duracao_sec {duration} inválida"
    if duration > moment.duration + DURATION_TOLERANCE_SEC:
        return f"duracao_sec {duration:g} maior que o momento {moment.id} ({moment.duration:g}s)"
    return None


def validate_detail(
    answer: BlockAnswer, unit: Unit, material: EditorMaterial, library_ids: list[str], usage: Usage
) -> DetailCheck:
    """Confere a resposta do pedaço contra uma CÓPIA do uso; `check.usage` é o uso com os itens aceitos."""
    state = usage.copy()
    check = DetailCheck(items=len(answer.manter) + len(answer.coberturas) + len(answer.respiros), usage=state)
    unit_clips = set(unit.clip_ids)
    unit_sentences = set(unit.sentence_ids(material))
    unit_moments = {m.id for m in unit.moments(material)}
    library = set(library_ids)
    # Faixa principal deste pedaço: frases em ordem, None onde há um momento (quebra a sequência).
    sequence: list[Optional[str]] = []
    keys: dict[str, int] = {}  # ID que `depois_de` pode citar → posição na sequência

    def drop(kind: str, item, reason: str) -> None:
        check.discarded.append(Discarded(kind=kind, item=_short(item), reason=reason))

    for raw in answer.manter:
        problem = _keep_problem(raw, material, unit, unit_clips, unit_sentences, unit_moments, state)
        if problem:
            drop("manter", raw, problem)
            continue
        if raw.momento:
            moment = material.moments[raw.momento]
            start, end = _moment_span(moment, raw.duracao_sec)
            state.moments.add(moment.id)
            state.take(moment.clip_id, start, end, f"momento {moment.id}")
            state.used_sec += end - start
            sequence.append(None)
            keys[moment.id] = len(sequence) - 1
            check.keep.append(KeepRange(
                kind="momento", clip_id=moment.clip_id, moment_id=moment.id, duration_sec=raw.duracao_sec,
            ))
            continue
        sentences = material.sentence_range(raw.de, raw.ate)
        ids = [s.id for s in sentences]
        state.sentences.update(ids)
        state.kept_order += ids
        state.take(sentences[0].clip_id, sentences[0].start_sec, sentences[-1].end_sec, f"frases {raw.de}–{raw.ate}")
        state.used_sec += sum(s.end_sec - s.start_sec for s in sentences)
        for sid in ids:
            sequence.append(sid)
            keys[sid] = len(sequence) - 1
        check.keep.append(KeepRange(
            kind="frases", clip_id=sentences[0].clip_id, from_sentence=raw.de, to_sentence=raw.ate, sentence_ids=ids,
        ))

    allowed_moments = library | unit_moments
    breath_after: set[int] = set()
    for raw in answer.respiros:
        problem = _breath_problem(raw, material, keys, allowed_moments, state)
        if problem:
            drop("respiro", raw, problem)
            continue
        moment = material.moments[raw.momento]
        start, end = _moment_span(moment, raw.duracao_sec)
        state.moments.add(moment.id)
        state.take(moment.clip_id, start, end, f"respiro {moment.id}")
        state.used_sec += end - start
        breath_after.add(keys[raw.depois_de])
        check.breaths.append(BreathChoice(
            after=raw.depois_de, moment_id=moment.id, duration_sec=raw.duracao_sec, reason=raw.motivo,
        ))

    position = {sid: i for i, sid in enumerate(sequence) if sid is not None}
    for raw in answer.coberturas:
        result = _coverage(raw, material, position, sequence, breath_after, allowed_moments, state, check)
        if isinstance(result, str):
            drop("cobertura", raw, result)
            continue
        check.coverages.append(result)
    return check


def _keep_problem(raw: KeepAnswer, material, unit: Unit, unit_clips, unit_sentences, unit_moments, state: Usage) -> Optional[str]:
    if raw.momento:
        if raw.de or raw.ate:
            return "item com frases e momento ao mesmo tempo"
        moment = material.moments.get(raw.momento)
        if moment is None:
            return f"momento {raw.momento} não existe"
        if moment.clip_id not in unit_clips or raw.momento not in unit_moments:
            return f"momento {raw.momento} não é deste bloco"
        if raw.momento in state.moments:
            return f"momento {raw.momento} já usado no vídeo"
        problem = _duration_problem(moment, raw.duracao_sec)
        if problem:
            return problem
        start, end = _moment_span(moment, raw.duracao_sec)
        if end - start <= 0:
            return f"momento {raw.momento} sem duração"
        clash = state.conflict(moment.clip_id, start, end)
        return f"o trecho de {raw.momento} já está no vídeo ({clash})" if clash else None
    if not raw.de and not raw.ate:
        return "item sem frases nem momento"
    de, ate = raw.de or raw.ate, raw.ate or raw.de
    raw.de, raw.ate = de, ate
    for sid in (de, ate):
        if sid not in material.sentences:
            return f"frase {sid} não existe"
    first, last = material.sentences[de], material.sentences[ate]
    if first.clip_id != last.clip_id:
        return f"{de} e {ate} são de clipes diferentes"
    if first.index > last.index:
        return f"{de} vem depois de {ate}"
    if first.clip_id not in unit_clips:
        return f"{de} é do clipe {first.clip_id}, fora deste bloco"
    sentences = material.sentence_range(de, ate)
    outside = [s.id for s in sentences if s.id not in unit_sentences]
    if outside:
        return f"frases fora deste pedaço do bloco: {', '.join(outside[:3])}"
    repeated = [s.id for s in sentences if s.id in state.sentences]
    if repeated:
        return f"frase já no vídeo: {', '.join(repeated[:3])}"
    clash = state.conflict(first.clip_id, first.start_sec, last.end_sec)
    return f"o trecho de {de}–{ate} já está no vídeo ({clash})" if clash else None


def _breath_problem(raw: BreathAnswer, material, keys: dict[str, int], allowed, state: Usage) -> Optional[str]:
    if not raw.depois_de or raw.depois_de not in keys:
        return f"depois_de {raw.depois_de or '(vazio)'} não é uma frase ou momento mantido neste bloco"
    if not raw.momento:
        return "respiro sem momento"
    moment = material.moments.get(raw.momento)
    if moment is None:
        return f"momento {raw.momento} não existe"
    if raw.momento not in allowed:
        return f"momento {raw.momento} não está no acervo oferecido nem no bloco"
    if raw.momento in state.moments:
        return f"momento {raw.momento} já usado no vídeo"
    problem = _duration_problem(moment, raw.duracao_sec)
    if problem:
        return problem
    start, end = _moment_span(moment, raw.duracao_sec)
    clash = state.conflict(moment.clip_id, start, end)
    return f"o trecho de {raw.momento} já está no vídeo ({clash})" if clash else None


def _coverage(
    raw: CoverageAnswer, material, position, sequence, breath_after: set[int], allowed, state: Usage, check: DetailCheck
):
    if not raw.de or not raw.ate:
        return "cobertura sem de/ate"
    sentences = material.sentence_range(raw.de, raw.ate)
    if not sentences:
        return f"{raw.de}–{raw.ate} não é uma faixa de frases do mesmo clipe, em ordem"
    ids = [s.id for s in sentences]
    missing = [sid for sid in ids if sid not in position]
    if missing:
        return f"cobertura sobre frase não mantida neste bloco: {', '.join(missing[:3])}"
    spots = [position[sid] for sid in ids]
    if any(b != a + 1 for a, b in zip(spots, spots[1:])):
        return f"as frases {raw.de}–{raw.ate} não estão seguidas na faixa principal"
    if any(p in breath_after for p in spots[:-1]):
        return "há um respiro no meio das frases cobertas"
    covered = [sid for sid in ids if sid in state.covered]
    if covered:
        return f"frase já coberta: {', '.join(covered[:3])}"
    speech = " ".join(s.text for s in sentences)
    speech_words = matched_words(raw.evidencia_fala, speech)
    if not speech_words:
        return f"evidencia_fala ({raw.evidencia_fala!r}) não aparece no texto de {raw.de}–{raw.ate}"
    speaker = sentences[0].clip_id
    kept: list[str] = []
    image_words: dict[str, list[str]] = {}
    for moment_id in raw.momentos:
        moment = material.moments.get(moment_id)
        reason = None
        if moment is None:
            reason = "não existe"
        elif moment.clip_id == speaker:
            reason = "é do clipe que está falando"
        elif moment_id not in allowed:
            reason = "não está no acervo oferecido nem no bloco"
        elif moment_id in state.moments or moment_id in kept:
            reason = "já usado no vídeo"
        elif state.conflict(moment.clip_id, moment.start_sec, moment.end_sec):
            reason = "o trecho já está no vídeo"
        else:
            found = matched_words(raw.evidencia_imagem, moment_text(moment))
            if not found:
                reason = f"evidencia_imagem ({raw.evidencia_imagem!r}) não aparece na descrição/lugar/tags"
            else:
                image_words[moment_id] = found
        if reason:
            check.warnings.append(f"cobertura {raw.de}–{raw.ate}: momento {moment_id} retirado ({reason})")
            continue
        kept.append(moment_id)
    if not kept:
        return "nenhum momento válido na cobertura"
    for moment_id in kept:
        moment = material.moments[moment_id]
        state.moments.add(moment_id)
        state.take(moment.clip_id, moment.start_sec, moment.end_sec, f"cobertura {moment_id}")
    state.covered.update(ids)
    return CoverageChoice(
        from_sentence=raw.de, to_sentence=raw.ate, sentence_ids=ids, clip_id=speaker, moment_ids=kept,
        speech_evidence=raw.evidencia_fala, image_evidence=raw.evidencia_imagem, reason=raw.motivo,
        matched_speech=speech_words, matched_image=image_words,
    )


def neutral_detail(unit: Unit, material: EditorMaterial, usage: Usage) -> BlockAnswer:
    """Fallback neutro do pedaço (a IA falhou de vez): todas as frases livres dos clipes, em ordem, sem
    cobertura nem respiros; um clipe sem fala entra com todos os seus momentos livres, em ordem."""
    keep: list[KeepAnswer] = []
    unit_sentences = set(unit.sentence_ids(material))
    for clip_id in unit.clip_ids:
        clip_sentences = material.clip_sentences.get(clip_id, [])
        if not clip_sentences:
            keep += [
                KeepAnswer(momento=m.id) for m in unit.moments(material)
                if m.clip_id == clip_id and m.id not in usage.moments and not usage.conflict(clip_id, m.start_sec, m.end_sec)
            ]
            continue
        run: list[str] = []
        for sentence in clip_sentences:
            if sentence.id in unit_sentences and sentence.id not in usage.sentences:
                run.append(sentence.id)
                continue
            if run:
                keep.append(KeepAnswer(de=run[0], ate=run[-1]))
                run = []
        if run:
            keep.append(KeepAnswer(de=run[0], ate=run[-1]))
    return BlockAnswer(manter=keep, notas="fallback neutro: a IA falhou neste bloco")


def neutral_structure(material: EditorMaterial) -> StructureAnswer:
    """Fallback neutro da estrutura: um bloco por clipe com fala, na ordem do catálogo, sem "fora" (sem
    nenhum clipe com fala, um bloco por clipe)."""
    speaking = [c for c in material.clip_ids if material.has_speech(c)] or material.clip_ids
    return StructureAnswer(
        blocos=[{"id": f"B{n:02d}", "clipes": [c]} for n, c in enumerate(speaking, start=1)],
        notas="fallback neutro: a IA falhou na estrutura",
    )
