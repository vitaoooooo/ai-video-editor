"""Validação das respostas do Editor de IA: só segurança, item a item (nenhuma regra editorial).

Estrutura (passada 1): IDs de clipes existem; cada clipe no máximo uma vez entre blocos e "fora"; "fora"
tem motivo; bloco sem clipe válido é descartado. **Completude:** todo clipe COM fala está num bloco ou em
"fora" (o esquecido é um item inválido: gera a nova tentativa e, se persistir, vai para um bloco próprio na
ordem do catálogo, com aviso — `place_forgotten`). Clipes sem fala podem ficar no acervo.

Detalhe (passada 2): por padrão FICA TUDO (as frases livres do pedaço, na ordem dos clipes e do roteiro);
a resposta diz o que sai e o que entra a mais. Cada item sozinho (o inválido é descartado com o motivo):

- cortar `de`/`ate` (ou uma frase): as frases existem, do mesmo clipe, em ordem, deste pedaço e ainda
  livres;
- ordem: clipes do pedaço (os que faltarem vão no fim, na ordem da estrutura);
- incluir momento: existe, é de um clipe do pedaço, NÃO tem frase (`Moment.sentence_ids` vazio), não foi
  usado, `duracao_sec` dentro do momento, `depois_de` = frase que ficou, momento incluído antes ou vazio
  (começo do bloco);
- respiro: `depois_de` é uma frase que ficou ou um momento incluído deste pedaço; momento do acervo
  pré-selecionado ou de um clipe do pedaço, não usado; não entra no meio de uma cobertura;
- cobertura: frases que ficaram, seguidas na faixa principal (sem respiro/incluído no meio) e não cobertas
  antes; momentos do acervo pré-selecionado ou de outro clipe do pedaço (nunca o clipe que está falando),
  não usados. **Evidência** (as duas cortadas nas primeiras `EVIDENCE_MAX_WORDS` palavras): TODAS as palavras
  de conteúdo de `evidencia_fala` estão nas frases cobertas e TODAS as de `evidencia_imagem` estão na
  descrição/lugar/tags dos momentos usados (a evidência da imagem não pode vir de outro momento); momento
  sem nenhuma palavra da evidência sai da cobertura. **Tipo:** "mostra o que é dito" exige ao menos uma
  palavra de conteúdo em comum entre `evidencia_fala` e `evidencia_imagem`; "mostra o lugar" exige que o
  lugar de cada momento usado tenha palavra em comum com o lugar do momento da 1ª frase coberta (momento
  com lugar diferente sai da cobertura). Palavras comparadas sem acento e sem caixa, com o radical simples
  de `material.same_word` (plural/singular; ônibus/ómnibus).

Antes de tudo, os IDs da resposta passam por `editor/ids.py` (`P009.S025` → `S025` só se P009 é o clipe da
frase; caixa, espaços e zeros à esquerda; faixa num campo só); as trocas ficam em `normalized`. ID recusado
leva na mensagem o formato certo (vai para a nova tentativa).

Em todos: o mesmo trecho de um arquivo não entra duas vezes no vídeo (intervalos por clipe, tolerância
`OVERLAP_TOLERANCE_SEC`). A resposta é recusada (nova tentativa com a lista de erros) quando nada fica no
pedaço ou metade ou mais dos itens é inválida; é inutilizável (fallback neutro) quando nada fica ou quando
tinha itens e nenhum era válido.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.ids import (
    anchor_hint,
    clip_hint,
    moment_hint,
    normalize_detail,
    normalize_structure,
    sentence_hint,
)
from ai_video_editor.editor.material import EditorMaterial, common_words, content_words, matched_words, moment_text
from ai_video_editor.editor.schema import (
    COVERAGE_PLACE,
    COVERAGE_SAYS,
    Block,
    BlockAnswer,
    BreathAnswer,
    BreathChoice,
    CoverageAnswer,
    CoverageChoice,
    CutAnswer,
    CutChoice,
    Discarded,
    IncludeAnswer,
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
# Evidências da cobertura: só as primeiras palavras contam (o pedido pede até 6, copiadas).
EVIDENCE_MAX_WORDS = 6


def _short(item) -> str:
    data = item.model_dump(exclude_none=True, exclude_defaults=True)
    text = ", ".join(f"{k}={v}" for k, v in data.items())
    return text[:160] or "(vazio)"


def first_words(text: str, limit: int = EVIDENCE_MAX_WORDS) -> str:
    return " ".join(text.split()[:limit])


# --- passada 1 ---


@dataclass
class StructureCheck:
    blocks: list[Block]
    out: list[OutClip]
    discarded: list[Discarded]
    forgotten: list[str] = field(default_factory=list)  # clipes com fala nem em bloco nem em "fora"
    normalized: list[str] = field(default_factory=list)  # IDs trocados pelo formato do roteiro ("bruto → ID")

    @property
    def errors(self) -> list[str]:
        return [f"{d.kind} ({d.item}): {d.reason}" for d in self.discarded]

    @property
    def refused(self) -> Optional[str]:
        """Por que pedir de novo (None = aceita, com os itens inválidos descartados)."""
        if not self.blocks:
            return "nenhum bloco válido"
        if self.forgotten:
            return (
                f"clipes com fala esquecidos (ponha cada um num bloco ou em \"fora\" com motivo): "
                f"{', '.join(self.forgotten)}"
            )
        return None


def validate_structure(answer: StructureAnswer, material: EditorMaterial) -> StructureCheck:
    answer, normalized = normalize_structure(answer, material)
    known = set(material.clip_ids)
    seen: set[str] = set()
    blocks: list[Block] = []
    out: list[OutClip] = []
    discarded: list[Discarded] = []
    for raw in answer.blocos:
        clips: list[str] = []
        for clip_id in raw.clipes:
            if clip_id not in known:
                discarded.append(Discarded(kind="bloco", item=f"{raw.id or '?'}: {clip_id}", reason=f"clipe {clip_id} não existe ({clip_hint(clip_id, material)})"))
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
            discarded.append(Discarded(kind="fora", item=_short(raw), reason=f"clipe {raw.clipe or '(vazio)'} não existe ({clip_hint(raw.clipe, material)})"))
        elif raw.clipe in seen:
            discarded.append(Discarded(kind="fora", item=_short(raw), reason=f"clipe {raw.clipe} já está num bloco ou em fora"))
        elif not raw.motivo:
            discarded.append(Discarded(kind="fora", item=_short(raw), reason=f"clipe {raw.clipe} em fora sem motivo"))
        else:
            seen.add(raw.clipe)
            out.append(raw)
    forgotten = [c for c in material.clip_ids if material.has_speech(c) and c not in seen]
    for clip_id in forgotten:
        discarded.append(Discarded(kind="faltando", item=clip_id, reason=f"clipe {clip_id} tem fala e não está em bloco nem em fora"))
    return StructureCheck(blocks=blocks, out=out, discarded=discarded, forgotten=forgotten, normalized=normalized)


def place_forgotten(blocks: list[Block], forgotten: list[str], material: EditorMaterial) -> list[Block]:
    """Cada clipe com fala esquecido vira um bloco próprio, na ordem do catálogo: logo depois do bloco que tem
    o clipe anterior a ele no catálogo (sem nenhum anterior, no começo). Blocos renumerados na ordem final."""
    order = {clip_id: n for n, clip_id in enumerate(material.clip_ids)}
    result = list(blocks)
    for clip_id in sorted(forgotten, key=order.__getitem__):
        position = 0
        for k, block in enumerate(result):
            if any(order[c] < order[clip_id] for c in block.clip_ids):
                position = k + 1
        result.insert(position, Block(id="", title=f"(clipe esquecido {clip_id})", clip_ids=[clip_id]))
    return [b.model_copy(update={"id": f"B{n:02d}"}) for n, b in enumerate(result, start=1)]


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

    def silent_moments(self, material: EditorMaterial, usage: "Usage"):
        """Momentos sem frase do pedaço, ainda não usados (os que `incluir` aceita)."""
        return [m for m in self.moments(material) if not m.sentence_ids and m.id not in usage.moments]


def sentences_sec(material: EditorMaterial, ids) -> float:
    return sum(material.sentences[s].end_sec - material.sentences[s].start_sec for s in ids)


@dataclass
class DetailCheck:
    keep: list[KeepRange] = field(default_factory=list)
    cuts: list[CutChoice] = field(default_factory=list)
    clip_order: list[str] = field(default_factory=list)
    coverages: list[CoverageChoice] = field(default_factory=list)
    breaths: list[BreathChoice] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    items: int = 0
    result_sec: float = 0.0  # frases que ficaram + incluídos + respiros
    usage: Optional[Usage] = None
    normalized: list[str] = field(default_factory=list)  # IDs trocados pelo formato do roteiro ("bruto → ID")

    @property
    def errors(self) -> list[str]:
        return [f"{d.kind} ({d.item}): {d.reason}" for d in self.discarded]

    @property
    def invalid(self) -> int:
        return len(self.discarded)

    @property
    def refused(self) -> Optional[str]:
        """Por que pedir de novo (None = aceita, com os itens inválidos descartados)."""
        if not self.keep:
            return "nada fica no bloco (tudo cortado ou nenhum momento incluído)"
        if self.items and self.invalid / self.items >= RETRY_INVALID_RATIO:
            return f"{self.invalid} de {self.items} itens inválidos"
        return None

    @property
    def unusable(self) -> Optional[str]:
        """Por que nem a resposta da última tentativa serve (None = aceita os itens válidos)."""
        if not self.keep:
            return "nada fica no bloco"
        if self.items and self.invalid >= self.items:
            return "nenhum item válido"
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
    answer, normalized = normalize_detail(answer, material)
    state = usage.copy()
    check = DetailCheck(
        normalized=normalized,
        items=len(answer.cortar) + len(answer.incluir) + len(answer.coberturas) + len(answer.respiros)
        + (1 if answer.ordem else 0),
        usage=state,
    )
    unit_clips = set(unit.clip_ids)
    free = [s for s in unit.sentence_ids(material) if s not in usage.sentences]
    free_set = set(free)
    unit_moments = {m.id for m in unit.moments(material)}
    library = set(library_ids)

    def drop(kind: str, item, reason: str) -> None:
        check.discarded.append(Discarded(kind=kind, item=_short(item) if not isinstance(item, str) else item, reason=reason))

    # Ordem dos clipes: a da IA (só clipes do pedaço), completada com a da estrutura.
    order: list[str] = []
    bad = [c for c in answer.ordem if c not in unit_clips]
    if bad:
        unknown = [c for c in bad if c not in material.clips]
        hint = f" ({clip_hint(unknown[0], material)}: {', '.join(unit.clip_ids)})" if unknown else ""
        drop("ordem", ", ".join(answer.ordem), f"clipes fora deste bloco: {', '.join(bad[:3])}{hint}")
    else:
        order = list(dict.fromkeys(answer.ordem))
    check.clip_order = order + [c for c in unit.clip_ids if c not in order]

    # Cortes.
    cut: set[str] = set()
    for raw in answer.cortar:
        problem = _cut_problem(raw, material, unit_clips, free_set)
        if problem:
            drop("cortar", raw, problem)
            continue
        ids = [s.id for s in material.sentence_range(raw.de, raw.ate)]
        cut.update(ids)
        check.cuts.append(CutChoice(from_sentence=raw.de, to_sentence=raw.ate, sentence_ids=ids, reason=raw.motivo))
    kept = [s for s in free if s not in cut]
    for sid in kept:
        sentence = material.sentences[sid]
        state.sentences.add(sid)
        state.take(sentence.clip_id, sentence.start_sec, sentence.end_sec, f"frase {sid}")
    kept_set = set(kept)

    # Incluídos (momentos sem fala), ancorados numa frase que ficou ou num incluído anterior.
    includes: dict[Optional[str], list[IncludeAnswer]] = {}
    included: set[str] = set()
    for raw in answer.incluir:
        problem = _include_problem(raw, material, unit_clips, unit_moments, kept_set, included, state)
        if problem:
            drop("incluir", raw, problem)
            continue
        moment = material.moments[raw.momento]
        start, end = _moment_span(moment, raw.duracao_sec)
        state.moments.add(moment.id)
        state.take(moment.clip_id, start, end, f"momento {moment.id}")
        included.add(moment.id)
        includes.setdefault(raw.depois_de, []).append(raw)

    # Faixa principal: incluídos do começo, depois os clipes na ordem; uma faixa de frases quebra onde há
    # frase cortada/fora do pedaço ou um incluído ancorado.
    sequence: list[Optional[str]] = []  # frases em ordem; None onde há um momento (quebra a sequência)
    keys: dict[str, int] = {}  # ID que `depois_de` pode citar → posição na sequência

    def emit_includes(anchor: Optional[str]) -> None:
        for raw in includes.pop(anchor, []):
            moment = material.moments[raw.momento]
            sequence.append(None)
            keys[moment.id] = len(sequence) - 1
            check.keep.append(KeepRange(
                kind="momento", clip_id=moment.clip_id, moment_id=moment.id, duration_sec=raw.duracao_sec,
            ))
            emit_includes(moment.id)

    emit_includes(None)
    for clip_id in check.clip_order:
        run: list[str] = []

        def close() -> None:
            if run:
                check.keep.append(KeepRange(
                    kind="frases", clip_id=clip_id, from_sentence=run[0], to_sentence=run[-1], sentence_ids=list(run),
                ))
                state.kept_order.extend(run)
                run.clear()

        for sentence in material.clip_sentences.get(clip_id, []):
            if sentence.id not in kept_set:
                close()
                continue
            run.append(sentence.id)
            sequence.append(sentence.id)
            keys[sentence.id] = len(sequence) - 1
            if sentence.id in includes:
                close()
                emit_includes(sentence.id)
        close()

    allowed_moments = library | unit_moments
    breath_after: set[int] = set()
    breaths_sec = 0.0
    for raw in answer.respiros:
        problem = _breath_problem(raw, material, keys, allowed_moments, state)
        if problem:
            drop("respiro", raw, problem)
            continue
        moment = material.moments[raw.momento]
        start, end = _moment_span(moment, raw.duracao_sec)
        state.moments.add(moment.id)
        state.take(moment.clip_id, start, end, f"respiro {moment.id}")
        breaths_sec += end - start
        breath_after.add(keys[raw.depois_de])
        check.breaths.append(BreathChoice(
            after=raw.depois_de, moment_id=moment.id, duration_sec=raw.duracao_sec, reason=raw.motivo,
        ))

    position = {sid: i for i, sid in enumerate(sequence) if sid is not None}
    for raw in answer.coberturas:
        result = _coverage(raw, material, position, breath_after, allowed_moments, state, check)
        if isinstance(result, str):
            drop("cobertura", raw, result)
            continue
        check.coverages.append(result)

    included_sec = sum(
        _moment_span(material.moments[k.moment_id], k.duration_sec)[1] - material.moments[k.moment_id].start_sec
        for k in check.keep if k.kind == "momento"
    )
    check.result_sec = round(sentences_sec(material, kept) + included_sec + breaths_sec, 3)
    state.used_sec += check.result_sec
    return check


def _cut_problem(raw: CutAnswer, material, unit_clips, free: set[str]) -> Optional[str]:
    if not raw.de and not raw.ate:
        return "corte sem frases"
    de, ate = raw.de or raw.ate, raw.ate or raw.de
    raw.de, raw.ate = de, ate
    for sid in (de, ate):
        if sid not in material.sentences:
            return f"frase {sid} não existe ({sentence_hint(sid, material)})"
    first, last = material.sentences[de], material.sentences[ate]
    if first.clip_id != last.clip_id:
        return f"{de} e {ate} são de clipes diferentes"
    if first.index > last.index:
        return f"{de} vem depois de {ate}"
    if first.clip_id not in unit_clips:
        return f"{de} é do clipe {first.clip_id}, fora deste bloco"
    outside = [s.id for s in material.sentence_range(de, ate) if s.id not in free]
    if outside:
        return f"frases fora deste pedaço do bloco ou já no vídeo: {', '.join(outside[:3])}"
    return None


def _include_problem(raw: IncludeAnswer, material, unit_clips, unit_moments, kept: set[str], included: set[str], state: Usage) -> Optional[str]:
    if not raw.momento:
        return "incluir sem momento"
    moment = material.moments.get(raw.momento)
    if moment is None:
        return f"momento {raw.momento} não existe ({moment_hint(raw.momento, material)})"
    if moment.clip_id not in unit_clips or raw.momento not in unit_moments:
        return f"momento {raw.momento} não é deste bloco"
    if moment.sentence_ids:
        return f"momento {raw.momento} tem fala ({', '.join(moment.sentence_ids[:3])}): só entram momentos sem fala"
    if raw.momento in state.moments:
        return f"momento {raw.momento} já usado no vídeo"
    if raw.depois_de is not None and raw.depois_de not in kept and raw.depois_de not in included:
        return f"depois_de {raw.depois_de} não é uma frase que ficou nem um momento incluído antes{_hint(raw.depois_de, material)}"
    problem = _duration_problem(moment, raw.duracao_sec)
    if problem:
        return problem
    start, end = _moment_span(moment, raw.duracao_sec)
    if end - start <= 0:
        return f"momento {raw.momento} sem duração"
    clash = state.conflict(moment.clip_id, start, end)
    return f"o trecho de {raw.momento} já está no vídeo ({clash})" if clash else None


def _breath_problem(raw: BreathAnswer, material, keys: dict[str, int], allowed, state: Usage) -> Optional[str]:
    if not raw.depois_de or raw.depois_de not in keys:
        return (
            f"depois_de {raw.depois_de or '(vazio)'} não é uma frase que ficou nem um momento incluído neste bloco"
            + (_hint(raw.depois_de, material) if raw.depois_de else "")
        )
    if not raw.momento:
        return "respiro sem momento"
    moment = material.moments.get(raw.momento)
    if moment is None:
        return f"momento {raw.momento} não existe ({moment_hint(raw.momento, material)})"
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


def _hint(raw: str, material) -> str:
    hint = anchor_hint(raw, material)
    return f" ({hint})" if hint else ""


def _missing(evidence: str, text: str) -> list[str]:
    """Palavras de conteúdo da evidência que NÃO estão no texto."""
    found = set(matched_words(evidence, text))
    return [w for w in content_words(evidence) if w not in found]


def _coverage(raw: CoverageAnswer, material, position, breath_after: set[int], allowed, state: Usage, check: DetailCheck):
    if not raw.de or not raw.ate:
        return "cobertura sem de/ate"
    for sid in (raw.de, raw.ate):
        if sid not in material.sentences:
            return f"frase {sid} não existe ({sentence_hint(sid, material)})"
    sentences = material.sentence_range(raw.de, raw.ate)
    if not sentences:
        return f"{raw.de}–{raw.ate} não é uma faixa de frases do mesmo clipe, em ordem"
    ids = [s.id for s in sentences]
    missing = [sid for sid in ids if sid not in position]
    if missing:
        return f"cobertura sobre frase que não fica neste bloco: {', '.join(missing[:3])}"
    spots = [position[sid] for sid in ids]
    if any(b != a + 1 for a, b in zip(spots, spots[1:])):
        return f"as frases {raw.de}–{raw.ate} não estão seguidas na faixa principal"
    if any(p in breath_after for p in spots[:-1]):
        return "há um respiro no meio das frases cobertas"
    covered = [sid for sid in ids if sid in state.covered]
    if covered:
        return f"frase já coberta: {', '.join(covered[:3])}"
    if raw.tipo is None:
        return f'tipo ausente ou desconhecido (use "{COVERAGE_SAYS}" ou "{COVERAGE_PLACE}")'
    speech_evidence, image_evidence = first_words(raw.evidencia_fala), first_words(raw.evidencia_imagem)
    speech = " ".join(s.text for s in sentences)
    speech_words = matched_words(speech_evidence, speech)
    if not speech_words:
        return f"evidencia_fala ({speech_evidence!r}) não aparece no texto de {raw.de}–{raw.ate}"
    not_said = _missing(speech_evidence, speech)
    if not_said:
        return f"evidencia_fala cita palavras que não estão em {raw.de}–{raw.ate}: {', '.join(not_said[:4])}"
    if not content_words(image_evidence):
        return "evidencia_imagem sem palavras de conteúdo"
    common: list[str] = []
    if raw.tipo == COVERAGE_SAYS:
        common = common_words(speech_evidence, image_evidence)
        if not common:
            return (
                f'"{COVERAGE_SAYS}" sem palavra em comum entre evidencia_fala ({speech_evidence!r}) e '
                f"evidencia_imagem ({image_evidence!r})"
            )
    speaker = sentences[0].clip_id
    speech_moment = material.moments.get(sentences[0].moment_ids[0]) if sentences[0].moment_ids else None
    speech_place = speech_moment.place if speech_moment is not None else ""
    if raw.tipo == COVERAGE_PLACE and not content_words(speech_place):
        return f'"{COVERAGE_PLACE}": o momento da fala ({speech_moment.id if speech_moment else "?"}) não tem lugar'
    kept: list[str] = []
    image_words: dict[str, list[str]] = {}
    for moment_id in raw.momentos:
        moment = material.moments.get(moment_id)
        reason = None
        if moment is None:
            reason = f"não existe; {moment_hint(moment_id, material)}"
        elif moment.clip_id == speaker:
            reason = "é do clipe que está falando"
        elif moment_id not in allowed:
            reason = "não está no acervo oferecido nem no bloco"
        elif moment_id in state.moments or moment_id in kept:
            reason = "já usado no vídeo"
        elif state.conflict(moment.clip_id, moment.start_sec, moment.end_sec):
            reason = "o trecho já está no vídeo"
        else:
            found = matched_words(image_evidence, moment_text(moment))
            place_common = common_words(moment.place, speech_place) if raw.tipo == COVERAGE_PLACE else []
            if not found:
                reason = f"evidencia_imagem ({image_evidence!r}) não aparece na descrição/lugar/tags"
            elif raw.tipo == COVERAGE_PLACE and not place_common:
                reason = f"lugar {moment.place or '(vazio)'!r} diferente do lugar da fala ({speech_place!r})"
            else:
                image_words[moment_id] = found
                common += [w for w in place_common if w not in common]
        if reason:
            check.warnings.append(f"cobertura {raw.de}–{raw.ate}: momento {moment_id} retirado ({reason})")
            continue
        kept.append(moment_id)
    if not kept:
        return "nenhum momento válido na cobertura"
    used_text = " ".join(moment_text(material.moments[m]) for m in kept)
    not_shown = _missing(image_evidence, used_text)
    if not_shown:
        sources = sorted({
            m.id for m in material.moments.values()
            if m.id not in kept and any(matched_words(w, moment_text(m)) for w in not_shown)
        })
        where = f" (aparece em {', '.join(sources[:3])})" if sources else ""
        return (
            f"evidencia_imagem cita palavras que não estão nos momentos usados ({', '.join(kept)}): "
            f"{', '.join(not_shown[:4])}{where}"
        )
    for moment_id in kept:
        moment = material.moments[moment_id]
        state.moments.add(moment_id)
        state.take(moment.clip_id, moment.start_sec, moment.end_sec, f"cobertura {moment_id}")
    state.covered.update(ids)
    return CoverageChoice(
        from_sentence=raw.de, to_sentence=raw.ate, sentence_ids=ids, clip_id=speaker, moment_ids=kept, kind=raw.tipo,
        common_words=common, speech_evidence=speech_evidence, image_evidence=image_evidence, reason=raw.motivo,
        matched_speech=speech_words, matched_image=image_words,
    )


def neutral_detail(unit: Unit, material: EditorMaterial, usage: Usage) -> BlockAnswer:
    """Fallback neutro do pedaço (a IA falhou de vez): nada cortado (todas as frases livres, na ordem), sem
    cobertura nem respiros; os momentos livres de um clipe sem fala entram em ordem, no lugar do clipe."""
    includes: list[dict] = []
    anchor: Optional[str] = None
    unit_sentences = set(unit.sentence_ids(material))
    for clip_id in unit.clip_ids:
        clip_sentences = material.clip_sentences.get(clip_id, [])
        if clip_sentences:
            free = [s.id for s in clip_sentences if s.id in unit_sentences and s.id not in usage.sentences]
            anchor = free[-1] if free else anchor
            continue
        for moment in unit.moments(material):
            if moment.clip_id == clip_id and moment.id not in usage.moments and not usage.conflict(clip_id, moment.start_sec, moment.end_sec):
                includes.append({"momento": moment.id, "depois_de": anchor})
                anchor = moment.id
    return BlockAnswer.model_validate({"incluir": includes, "notas": "fallback neutro: a IA falhou neste bloco"})


def neutral_structure(material: EditorMaterial) -> StructureAnswer:
    """Fallback neutro da estrutura: um bloco por clipe com fala, na ordem do catálogo, sem "fora" (sem
    nenhum clipe com fala, um bloco por clipe)."""
    speaking = [c for c in material.clip_ids if material.has_speech(c)] or material.clip_ids
    return StructureAnswer(
        blocos=[{"id": f"B{n:02d}", "clipes": [c]} for n, c in enumerate(speaking, start=1)],
        notas="fallback neutro: a IA falhou na estrutura",
    )
