"""Validação das respostas do Editor de IA: só segurança, item a item (nenhuma regra editorial).

Estrutura (passada 1): IDs de clipes existem; cada clipe no máximo uma vez entre blocos e "fora"; "fora"
tem motivo; bloco sem clipe válido é descartado. **Completude:** todo clipe COM fala está num bloco ou em
"fora" (o esquecido é um item inválido: gera a nova tentativa e, se persistir, vai para um bloco próprio na
ordem do catálogo, com aviso — `place_forgotten`). Clipes sem fala podem ficar no acervo.

Passada 2, três perguntas por bloco; cada item sozinho (o inválido é descartado com o motivo):

- 2a cortes (`validate_cuts`): por padrão FICA TUDO; cada item é UMA frase que existe, deste clipe (ou parte)
  e ainda livre; faixa ou várias frases num item são recusadas (uma frase por item); repetida, descartada.
- 2b inclusões (`validate_inserts`): `depois_do_trecho` inteiro de 0 (começo) ao último trecho (fim); letra
  da lista, no máximo uma vez; `duracao_sec` maior que o momento vira o momento inteiro (com aviso); o
  trecho do arquivo não pode estar no vídeo. Candidato do bloco entra como momento; do acervo, como respiro.
- 2c coberturas (`validate_coverages`): trecho perguntado, no máximo uma vez; "nenhuma" vale; letra entre os
  candidatos DAQUELE trecho, no máximo uma vez no bloco; **evidência**: todas as palavras de conteúdo de
  `palavra` estão na fala do trecho e na descrição/lugar/tags do momento ("mostra o que é dito") ou no lugar
  do momento e no lugar de um momento da fala ("mostra o lugar"); sem acento/caixa, mesmo radical
  (`material.same_word`). A cobertura começa na frase com a palavra (lugar: a 1ª frase daquele lugar) e
  segue pelas frases seguintes do trecho enquanto couberem na duração do momento (`coverage_span`).

Antes de tudo, os IDs da 2a passam por `editor/ids.py` (`P009.S025` → `S025` só se P009 é o clipe da frase;
caixa, espaços, zeros à esquerda e lixo de markdown); as trocas ficam em `normalized`. ID recusado leva na
mensagem o formato certo (vai para a nova tentativa).

Em todas: o mesmo trecho de um arquivo não entra duas vezes no vídeo (intervalos por clipe, tolerância
`OVERLAP_TOLERANCE_SEC`). Uma resposta é recusada (nova tentativa com a lista de erros) com metade ou mais
dos itens inválida; é inutilizável (fallback neutro) quando tinha itens e nenhum era válido. A 2b de um
bloco sem fala também é recusada se não escolhe nada (o bloco ficaria vazio).
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.ids import clip_hint, normalize_cuts, normalize_structure, sentence_hint, sentence_tokens
from ai_video_editor.editor.material import EditorMaterial, content_words, matched_words, moment_text
from ai_video_editor.editor.schema import (
    COVERAGE_PLACE,
    COVERAGE_SAYS,
    Block,
    Candidate,
    CoveragesAnswer,
    CoverageChoice,
    CutChoice,
    CutsAnswer,
    Discarded,
    InsertChoice,
    InsertsAnswer,
    OutClip,
    StructureAnswer,
    Trecho,
)

# Sobreposição tolerada entre dois usos do mesmo arquivo (tempos do Whisper e das fotos não coincidem).
OVERLAP_TOLERANCE_SEC = 0.25
# Folga na comparação da duração pedida com a do momento (e da cobertura com a duração das frases).
DURATION_TOLERANCE_SEC = 0.05
# Fração de itens inválidos a partir da qual a resposta é recusada.
RETRY_INVALID_RATIO = 0.5
# Palavra em comum da cobertura: só as primeiras palavras contam (o pedido pede a palavra copiada).
EVIDENCE_MAX_WORDS = 6
# A cobertura pega mais uma frase do trecho enquanto o fim dela não passa da duração do momento + isto.
COVERAGE_EXTEND_SEC = 0.5


def _short(item) -> str:
    data = item.model_dump(exclude_none=True, exclude_defaults=True)
    text = ", ".join(f"{k}={v}" for k, v in data.items())
    return text[:160] or "(vazio)"


def first_words(text: str, limit: int = EVIDENCE_MAX_WORDS) -> str:
    return " ".join(text.split()[:limit])


@dataclass
class _Check:
    """Base das conferências da passada 2: descartes, itens e as regras de recusa."""

    discarded: list[Discarded] = field(default_factory=list)
    items: int = 0

    @property
    def errors(self) -> list[str]:
        return [f"{d.kind} ({d.item}): {d.reason}" for d in self.discarded]

    @property
    def invalid(self) -> int:
        return len(self.discarded)

    @property
    def refused(self) -> Optional[str]:
        """Por que pedir de novo (None = aceita, com os itens inválidos descartados)."""
        if self.items and self.invalid / self.items >= RETRY_INVALID_RATIO:
            return f"{self.invalid} de {self.items} itens inválidos"
        return None

    @property
    def unusable(self) -> Optional[str]:
        """Por que nem a resposta da última tentativa serve (None = aceita os itens válidos)."""
        if self.items and self.invalid >= self.items:
            return "nenhum item válido"
        return None

    def drop(self, kind: str, item, reason: str) -> None:
        self.discarded.append(Discarded(kind=kind, item=item if isinstance(item, str) else _short(item), reason=reason))


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


def sentences_sec(material: EditorMaterial, ids) -> float:
    return sum(material.sentences[s].end_sec - material.sentences[s].start_sec for s in ids)


def take_sentences(usage: Usage, ids, material: EditorMaterial) -> None:
    """Marca as frases como usadas no vídeo (e o trecho do arquivo delas)."""
    for sid in ids:
        sentence = material.sentences[sid]
        usage.sentences.add(sid)
        usage.take(sentence.clip_id, sentence.start_sec, sentence.end_sec, f"frase {sid}")
        usage.kept_order.append(sid)


def letters(count: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ...: as letras dos candidatos na ordem da lista."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    for n in range(count):
        result.append(alphabet[n] if n < 26 else alphabet[n // 26 - 1] + alphabet[n % 26])
    return result


# --- 2a: cortes ---


@dataclass
class CutsCheck(_Check):
    cuts: list[CutChoice] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)  # frases que ficam, na ordem do roteiro
    result_sec: float = 0.0
    normalized: list[str] = field(default_factory=list)

    @property
    def refused(self) -> Optional[str]:
        if not self.kept:
            return "todas as frases do clipe foram cortadas (um clipe que sai inteiro vai em \"fora\" na estrutura)"
        return super().refused


def validate_cuts(answer: CutsAnswer, free: list[str], material: EditorMaterial) -> CutsCheck:
    """`free`: as frases livres do clipe (ou da parte), em ordem. O que não for cortado fica."""
    answer, normalized = normalize_cuts(answer, material)
    check = CutsCheck(items=len(answer.cortar), normalized=normalized)
    free_set = set(free)
    cut: dict[str, CutChoice] = {}
    for raw in answer.cortar:
        sid = raw.frase
        if not sid:
            check.drop("cortar", raw, "item sem frase")
        elif len(sentence_tokens(sid)) > 1:
            check.drop("cortar", raw, f"{sid}: uma frase por item (para cortar várias, um item para cada frase)")
        elif sid not in material.sentences:
            check.drop("cortar", raw, f"frase {sid} não existe ({sentence_hint(sid, material)})")
        elif sid not in free_set:
            owner = material.sentences[sid].clip_id
            check.drop("cortar", raw, f"frase {sid} não é deste clipe nem desta parte (é do clipe {owner})")
        elif sid in cut:
            check.drop("cortar", raw, f"frase {sid} repetida")
        else:
            cut[sid] = CutChoice(sentence_id=sid, reason=raw.motivo)
    check.cuts = [cut[s] for s in free if s in cut]
    check.kept = [s for s in free if s not in cut]
    check.result_sec = round(sentences_sec(material, check.kept), 3)
    return check


def build_trechos(clip_order: list[str], kept: set[str], material: EditorMaterial) -> list[Trecho]:
    """Trechos de fala mantidos, na ordem do roteiro (clipes do bloco em ordem, frases em ordem); uma frase
    cortada ou fora do bloco quebra o trecho. Numerados a partir de 1."""
    trechos: list[Trecho] = []
    for clip_id in clip_order:
        run: list[str] = []
        for sentence in [*material.clip_sentences.get(clip_id, []), None]:
            if sentence is not None and sentence.id in kept:
                run.append(sentence.id)
                continue
            if run:
                trechos.append(Trecho(
                    number=len(trechos) + 1, clip_id=clip_id, sentence_ids=list(run),
                    seconds=round(sentences_sec(material, run), 3),
                ))
                run = []
    return trechos


# --- 2b: inclusões e respiros ---


@dataclass
class InsertsCheck(_Check):
    inserts: list[InsertChoice] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    added_sec: float = 0.0
    usage: Optional[Usage] = None
    has_speech: bool = True  # o bloco tem trechos de fala

    @property
    def refused(self) -> Optional[str]:
        if not self.has_speech and not self.inserts:
            return "o bloco não tem fala e nenhuma imagem foi escolhida: o bloco ficaria vazio"
        return super().refused

    @property
    def unusable(self) -> Optional[str]:
        if not self.has_speech and not self.inserts:
            return "bloco vazio"
        return super().unusable


def _moment_span(moment, duration: Optional[float]) -> tuple[float, float]:
    length = moment.duration if duration is None else min(duration, moment.duration)
    return moment.start_sec, round(moment.start_sec + length, 3)


def validate_inserts(
    answer: InsertsAnswer, trechos: list[Trecho], candidates: list[Candidate], material: EditorMaterial, usage: Usage
) -> InsertsCheck:
    """Confere a 2b contra uma CÓPIA do uso; `check.usage` é o uso com as inclusões aceitas."""
    state = usage.copy()
    check = InsertsCheck(items=len(answer.inserir), usage=state, has_speech=bool(trechos))
    by_letter = {c.letter: c for c in candidates}
    last = len(trechos)
    used: set[str] = set()
    valid_letters = ", ".join(c.letter for c in candidates[:8]) + ("…" if len(candidates) > 8 else "")
    for raw in answer.inserir:
        position = raw.depois_do_trecho
        if position is None or position != int(position):
            check.drop("inserir", raw, f"depois_do_trecho deve ser um número inteiro de 0 a {last}")
            continue
        position = int(position)
        if not 0 <= position <= last:
            check.drop("inserir", raw, f"depois_do_trecho {position} não existe (de 0 = começo a {last} = fim)")
            continue
        if not raw.letra:
            check.drop("inserir", raw, "item sem letra")
            continue
        candidate = by_letter.get(raw.letra)
        if candidate is None:
            check.drop("inserir", raw, f"letra {raw.letra} não está na lista (letras: {valid_letters or 'nenhuma'})")
            continue
        if raw.letra in used:
            check.drop("inserir", raw, f"letra {raw.letra} repetida (cada letra no máximo uma vez)")
            continue
        moment = material.moments[candidate.moment_id]
        duration = raw.duracao_sec
        if duration is not None and duration <= 0:
            duration = None
        if duration is not None and duration > moment.duration + DURATION_TOLERANCE_SEC:
            check.warnings.append(
                f"inserir {raw.letra}: duracao_sec {duration:g} maior que a imagem ({moment.duration:.1f}s); "
                "entra a imagem inteira"
            )
            duration = None
        start, end = _moment_span(moment, duration)
        clash = state.conflict(moment.clip_id, start, end) if moment.id not in state.moments else "já usado"
        if clash:
            check.drop("inserir", raw, f"a imagem {raw.letra} ({moment.id}) já está no vídeo ({clash})")
            continue
        used.add(raw.letra)
        state.moments.add(moment.id)
        state.take(moment.clip_id, start, end, f"momento {moment.id}")
        check.added_sec += end - start
        check.inserts.append(InsertChoice(
            after_trecho=position, letter=raw.letra, moment_id=moment.id, duration_sec=duration,
            breath=candidate.source != "bloco",
        ))
    check.added_sec = round(check.added_sec, 3)
    return check


def neutral_inserts(trechos: list[Trecho], candidates: list[Candidate]) -> InsertsAnswer:
    """Fallback neutro da 2b: nada a mais num bloco com fala; num bloco sem fala, todas as imagens do bloco
    na ordem (o bloco não pode ficar vazio)."""
    if trechos:
        return InsertsAnswer(notas="fallback neutro: nenhuma imagem incluída")
    return InsertsAnswer.model_validate({
        "inserir": [{"depois_do_trecho": 0, "letra": c.letter} for c in candidates if c.source == "bloco"],
        "notas": "fallback neutro: todas as imagens do bloco, em ordem",
    })


# --- 2c: coberturas ---


def _missing(evidence: str, text: str) -> list[str]:
    """Palavras de conteúdo da evidência que NÃO estão no texto."""
    found = set(matched_words(evidence, text))
    return [w for w in content_words(evidence) if w not in found]


def speech_places(trecho: Trecho, material: EditorMaterial) -> list[str]:
    """Lugares dos momentos em que as frases do trecho começam (sem repetir, em ordem)."""
    places: list[str] = []
    for sid in trecho.sentence_ids:
        sentence = material.sentences[sid]
        moment = material.moments.get(sentence.moment_ids[0]) if sentence.moment_ids else None
        if moment is not None and moment.place and moment.place not in places:
            places.append(moment.place)
    return places


def coverage_span(trecho: Trecho, candidate: Candidate, word: str, material: EditorMaterial) -> list[str]:
    """Frases cobertas: da 1ª frase do trecho com a palavra ("mostra o que é dito") ou do lugar do momento
    ("mostra o lugar"), seguindo enquanto o fim da frase cabe na duração do momento (+ `COVERAGE_EXTEND_SEC`)."""
    ids = trecho.sentence_ids
    moment = material.moments[candidate.moment_id]
    start = 0
    for k, sid in enumerate(ids):
        sentence = material.sentences[sid]
        if candidate.kind == COVERAGE_PLACE:
            where = material.moments.get(sentence.moment_ids[0]) if sentence.moment_ids else None
            hit = where is not None and bool(matched_words(moment.place, where.place))
        else:
            hit = bool(matched_words(word, sentence.text))
        if hit:
            start = k
            break
    first = material.sentences[ids[start]]
    span = [ids[start]]
    for sid in ids[start + 1:]:
        if material.sentences[sid].end_sec - first.start_sec > moment.duration + COVERAGE_EXTEND_SEC:
            break
        span.append(sid)
    return span


@dataclass
class CoverageCheck(_Check):
    coverages: list[CoverageChoice] = field(default_factory=list)
    nones: list[int] = field(default_factory=list)
    usage: Optional[Usage] = None


def validate_coverages(
    answer: CoveragesAnswer,
    questions: dict[int, list[Candidate]],
    trechos: list[Trecho],
    material: EditorMaterial,
    usage: Usage,
) -> CoverageCheck:
    """`questions`: trecho perguntado → candidatos dele (com letra). Confere contra uma CÓPIA do uso."""
    state = usage.copy()
    check = CoverageCheck(items=len(answer.coberturas), usage=state)
    by_number = {t.number: t for t in trechos}
    seen_trechos: set[int] = set()
    used_letters: set[str] = set()
    for raw in answer.coberturas:
        number = raw.trecho
        if number is None or number != int(number) or int(number) not in questions:
            asked = ", ".join(str(n) for n in questions)
            shown = int(number) if number is not None and number == int(number) else number
            check.drop("cobertura", raw, f"trecho {shown} não foi perguntado (trechos: {asked})")
            continue
        number = int(number)
        if number in seen_trechos:
            check.drop("cobertura", raw, f"trecho {number} repetido (um item por trecho)")
            continue
        seen_trechos.add(number)
        if raw.none:
            check.nones.append(number)
            continue
        options = {c.letter: c for c in questions[number]}
        candidate = options.get(raw.letra)
        if candidate is None:
            check.drop("cobertura", raw, f"letra {raw.letra} não é candidata do trecho {number} "
                                         f"(letras: {', '.join(options)} ou \"nenhuma\")")
            continue
        if raw.letra in used_letters:
            check.drop("cobertura", raw, f"letra {raw.letra} já usada em outro trecho")
            continue
        trecho = by_number[number]
        moment = material.moments[candidate.moment_id]
        word = first_words(raw.palavra)
        if not content_words(word):
            check.drop("cobertura", raw, f"sem a palavra em comum (copie uma de: {', '.join(candidate.words)})")
            continue
        speech = " ".join(material.sentences[s].text for s in trecho.sentence_ids)
        if candidate.kind == COVERAGE_PLACE:
            places = " ".join(speech_places(trecho, material))
            missing = _missing(word, moment.place) or _missing(word, places)
            if missing:
                check.drop("cobertura", raw, f"palavra {word!r} não está no lugar da imagem ({moment.place!r}) e no "
                                             f"da fala ({places!r}): {', '.join(missing)}")
                continue
        else:
            not_said = _missing(word, speech)
            if not_said:
                check.drop("cobertura", raw, f"palavra {word!r} não está na fala do trecho {number}: {', '.join(not_said)}")
                continue
            not_shown = _missing(word, moment_text(moment))
            if not_shown:
                check.drop("cobertura", raw, f"palavra {word!r} não está na descrição/lugar/tags da imagem "
                                             f"{raw.letra}: {', '.join(not_shown)}")
                continue
        if moment.id in state.moments or state.conflict(moment.clip_id, moment.start_sec, moment.end_sec):
            check.drop("cobertura", raw, f"a imagem {raw.letra} ({moment.id}) já está no vídeo")
            continue
        span = coverage_span(trecho, candidate, word, material)
        covered = [s for s in span if s in state.covered]
        if covered:
            check.drop("cobertura", raw, f"frase já coberta: {', '.join(covered[:3])}")
            continue
        used_letters.add(raw.letra)
        state.moments.add(moment.id)
        state.take(moment.clip_id, moment.start_sec, moment.end_sec, f"cobertura {moment.id}")
        state.covered.update(span)
        check.coverages.append(CoverageChoice(
            trecho=number, letter=raw.letra, word=word, from_sentence=span[0], to_sentence=span[-1],
            sentence_ids=span, clip_id=trecho.clip_id, moment_ids=[moment.id], kind=candidate.kind or COVERAGE_SAYS,
            common_words=list(candidate.words), speech_evidence=word, image_evidence=word,
        ))
    return check


def neutral_structure(material: EditorMaterial) -> StructureAnswer:
    """Fallback neutro da estrutura: um bloco por clipe com fala, na ordem do catálogo, sem "fora" (sem
    nenhum clipe com fala, um bloco por clipe)."""
    speaking = [c for c in material.clip_ids if material.has_speech(c)] or material.clip_ids
    return StructureAnswer(
        blocos=[{"id": f"B{n:02d}", "clipes": [c]} for n, c in enumerate(speaking, start=1)],
        notas="fallback neutro: a IA falhou na estrutura",
    )
