"""Validação das respostas do Editor de IA: só segurança, item a item (nenhuma regra editorial).

Estrutura (passada 1): IDs de clipes existem; cada clipe no máximo uma vez entre blocos e "fora"; "fora"
tem motivo; bloco sem clipe válido é descartado. A IA usa só o que é pertinente ao pedido: um clipe com
fala que ela não pôs em nenhum bloco nem em "fora" conta como fora, com o motivo `NOT_CHOSEN` (sem nova
tentativa e sem bloco extra). Clipes sem fala que ficam de fora da lista são o acervo.

Passada 2, três perguntas por bloco; cada item sozinho (o inválido é descartado com o motivo):

- 2a cortes (`validate_cuts`): por padrão FICA TUDO; cada item é UMA frase que existe, deste clipe (ou parte)
  e ainda livre; faixa ou várias frases num item são recusadas (uma frase por item); repetida, descartada.
  Todas as frases cortadas sem `fora: true` pede a nova tentativa dizendo isso (`ALL_CUT`); se persistir (ou
  com `fora: true`), o clipe sai do bloco (decisão da IA, registrada).
  Par contínuo (`continuous_groups`): uma frase continua a anterior do mesmo clipe quando a anterior termina
  sem pontuação final ou com reticências, ou quando ela começa com letra minúscula; frases ligadas assim são
  uma unidade. Cortar só parte da unidade → o código corta todas ou mantém todas, pelo que a IA fez com a
  maior parte em segundos (empate = mantém), e registra em `joined`.
- 2b inclusões (`validate_inserts`): `depois_do_trecho` numa das posições oferecidas (`breath_positions`
  para respiro, `own_positions` para imagem do bloco, medidas por `insert_positions`); letra da lista, no
  máximo uma vez; `duracao_sec` maior que o momento vira o momento inteiro (com aviso); o trecho do arquivo
  não pode estar no vídeo. Candidato do bloco entra como momento; do acervo, como respiro: no máximo
  `RESPIRO_MAX_SEC`, nunca dois respiros seguidos, e a soma das inclusões não passa da sobra do bloco (os
  respiros que passam são recusados do último para o primeiro). O trecho do arquivo é a janela estável do
  momento (`steady.steady_window`: sem as pontas de câmera mexida e, quando a imagem é mais longa que o que
  entra, a janela de menor movimento; sem medidas, o respiro fica centrado e a inclusão começa no momento).
- 2c coberturas (`validate_coverages`): trecho perguntado, no máximo uma vez (itens para trechos não perguntados
  são ignorados juntos, sem contar como inválidos: o modelo às vezes continua a lista em loop); "nenhuma" vale; letra entre os
  candidatos DAQUELE trecho, no máximo uma vez no bloco; **evidência**: todas as palavras de conteúdo de
  `palavra` estão na fala do trecho e na descrição/lugar do momento ("mostra o que é dito"; as tags não
  valem) ou no lugar do momento e no lugar de um momento da fala ("mostra o lugar"); sem acento/caixa, mesmo
  radical (`material.same_word`), sem palavras genéricas (`material.GENERIC_WORDS`). A cobertura começa na
  frase com a palavra (lugar: a 1ª frase daquele lugar), só entre as frases ditas para a câmera
  (`talking_sentences`), e segue pelas seguintes enquanto couberem na duração do momento (`coverage_span`).
  Sem palavra em comum que valha, vale a **correspondência explicada**: `mostra` (o que a imagem mostra) tem
  palavra de conteúdo na descrição/lugar do momento (a evidência visual continua obrigatória) e `porque` cita
  palavra de conteúdo da fala do trecho (a cobertura começa na frase com essa palavra). Faltando uma das duas,
  o item é descartado e o trecho fica sem cobertura. Nos dois caminhos, a palavra da fala não pode estar na
  imagem de quem fala (`speaker_text`: a descrição dos momentos do trecho): se a imagem da pessoa já mostra
  aquilo, não cobre.

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

from ai_video_editor.editor.ids import IdResolver, clip_hint, normalize_cuts, normalize_structure, sentence_hint, sentence_tokens
from ai_video_editor.editor.material import (
    EditorMaterial,
    content_words,
    matched_words,
    moment_visible_text,
    same_place,
)
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
    Skipped,
    StructureAdditionsAnswer,
    StructureAnswer,
    Trecho,
)
from ai_video_editor.editor.steady import steady_range, steady_window

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
# Respiro é pausa curta: o momento do acervo escolhido entra com no máximo isto, centrado no momento.
RESPIRO_MAX_SEC = 6.0
# Trecho que termina a menos disto do fim do clipe: a gravação parou no meio da fala (a fala continua).
CLIP_END_SPEECH_SEC = 1.0
# Motivo de "fora" de um clipe com fala que a IA não pôs em nenhum bloco nem em "fora".
NOT_CHOSEN = "não escolhido pela IA"
# Pontuação que fecha uma frase (sem ela, ou com reticências, a fala continua no trecho seguinte).
_FINAL_PUNCTUATION = (".", "!", "?")
_CLOSERS = "\"'”’»)]"


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
    not_chosen: list[str] = field(default_factory=list)  # clipes com fala nem em bloco nem em "fora" (= fora)
    normalized: list[str] = field(default_factory=list)  # IDs trocados pelo formato do roteiro ("bruto → ID")

    @property
    def errors(self) -> list[str]:
        return [f"{d.kind} ({d.item}): {d.reason}" for d in self.discarded]

    @property
    def refused(self) -> Optional[str]:
        """Por que pedir de novo (None = aceita, com os itens inválidos descartados)."""
        if not self.blocks:
            return "nenhum bloco válido"
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
    not_chosen = [c for c in material.clip_ids if material.has_speech(c) and c not in seen]
    out += [OutClip(clipe=c, motivo=NOT_CHOSEN) for c in not_chosen]
    return StructureCheck(blocks=blocks, out=out, discarded=discarded, not_chosen=not_chosen, normalized=normalized)


@dataclass
class AdditionsCheck:
    """Conferência da pergunta de acréscimo: os clipes aceitos (na ordem da resposta) e os códigos ignorados."""

    added: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    normalized: list[str] = field(default_factory=list)


def validate_additions(answer: StructureAdditionsAnswer, offered: list[str], material: EditorMaterial) -> AdditionsCheck:
    """Só clipes da lista oferecida entram (cada um uma vez); o resto é ignorado. Resposta vazia é válida."""
    ids = IdResolver(material)
    check = AdditionsCheck()
    for raw in answer.acrescentar:
        clip_id = ids.clip(raw)
        if clip_id in offered:
            if clip_id not in check.added:
                check.added.append(clip_id)
        elif raw not in check.ignored:
            check.ignored.append(raw)
    check.normalized = ids.fixed
    return check


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


# Erro da nova tentativa da 2a quando a resposta corta todas as frases sem dizer fora: true.
ALL_CUT = "você cortou todas as frases; mantenha as que contam o assunto ou responda fora: true"


def open_ending(text: str) -> Optional[str]:
    """Por que a frase não termina (None = termina): sem pontuação final ou com reticências."""
    text = text.strip().rstrip(_CLOSERS).rstrip()
    if text.endswith(("...", "…")):
        return "termina em reticências"
    if not text.endswith(_FINAL_PUNCTUATION):
        return "sem pontuação final"
    return None


def starts_lowercase(text: str) -> bool:
    """A primeira letra da frase é minúscula (a frase continua a anterior)."""
    first = next((ch for ch in text if ch.isalpha()), "")
    return first.islower()


def continuous_groups(ids: list[str], material: EditorMaterial) -> list[list[str]]:
    """Unidades de frases ligadas (≥ 2), em ordem: B continua A quando B vem logo depois de A no mesmo clipe e
    A termina em aberto (`open_ending`) ou B começa com minúscula. Só entre frases de `ids`."""
    wanted = set(ids)
    groups: list[list[str]] = []
    current: list[str] = []
    for clip_id in dict.fromkeys(material.sentences[s].clip_id for s in ids):
        previous = None
        for sentence in material.clip_sentences.get(clip_id, []):
            linked = (
                previous is not None and previous.id in wanted and sentence.id in wanted
                and (open_ending(previous.text) is not None or starts_lowercase(sentence.text))
            )
            if linked:
                current = current or [previous.id]
                current.append(sentence.id)
            else:
                if len(current) > 1:
                    groups.append(current)
                current = []
            previous = sentence
        if len(current) > 1:
            groups.append(current)
        current = []
    return groups


def continuing_sentences(ids: list[str], material: EditorMaterial) -> list[str]:
    """Frases que continuam na seguinte (todas as de cada unidade, menos a última): a marca do pedido da 2a."""
    return [s for group in continuous_groups(ids, material) for s in group[:-1]]


@dataclass
class CutsCheck(_Check):
    cuts: list[CutChoice] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)  # frases que ficam, na ordem do roteiro
    result_sec: float = 0.0
    normalized: list[str] = field(default_factory=list)
    out: bool = False  # a IA respondeu fora: true (o clipe ou a parte sai do bloco)
    joined: list[str] = field(default_factory=list)  # "par contínuo S000+S001 mantido junto" (ajustes do código)
    limit: str = ""  # a IA disse que a meta não dá sem prejudicar a fala (`limite_meta`)

    @property
    def removed(self) -> bool:
        """O clipe (ou a parte) sai do bloco: fora: true ou nenhuma frase ficou."""
        return self.out or not self.kept

    @property
    def refused(self) -> Optional[str]:
        if not self.out and not self.kept:
            return ALL_CUT
        return super().refused


def validate_cuts(answer: CutsAnswer, free: list[str], material: EditorMaterial) -> CutsCheck:
    """`free`: as frases livres do clipe (ou da parte), em ordem. O que não for cortado fica."""
    answer, normalized = normalize_cuts(answer, material)
    check = CutsCheck(items=len(answer.cortar), normalized=normalized, out=answer.fora, limit=answer.limite_meta)
    if answer.fora:  # o clipe (ou a parte) inteiro sai: os itens de corte não importam
        check.items = 0
        check.cuts = [CutChoice(sentence_id=s, reason="fora: true") for s in free]
        return check
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
    for group in continuous_groups(free, material):
        removed = [s for s in group if s in cut]
        if not removed or len(removed) == len(group):
            continue
        label = "+".join(group)
        if sentences_sec(material, removed) > sentences_sec(material, [s for s in group if s not in cut]):
            for sid in group:
                cut.setdefault(sid, CutChoice(sentence_id=sid, reason=f"par contínuo {label} (cortado junto)"))
            check.joined.append(f"par contínuo {label} cortado junto")
        else:
            for sid in removed:
                del cut[sid]
            check.joined.append(f"par contínuo {label} mantido junto")
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


def speech_continues(trecho: Trecho, material: EditorMaterial) -> Optional[str]:
    """Por que a fala continua depois do trecho (None = termina): a última frase sem pontuação final ou com
    reticências, ou o trecho acaba a menos de `CLIP_END_SPEECH_SEC` do fim do clipe (a gravação parou no meio
    da fala; o Whisper põe ponto final mesmo assim)."""
    last = material.sentences[trecho.sentence_ids[-1]]
    ending = open_ending(last.text)
    if ending:
        return f"fala continua ({ending})"
    clip = material.clips.get(trecho.clip_id)
    if clip is not None and clip.duration_sec - last.end_sec < CLIP_END_SPEECH_SEC:
        return "fala continua (o clipe acaba no meio da fala)"
    return None


def trecho_place(trecho: Trecho, material: EditorMaterial, end: bool) -> str:
    """Lugar do último momento do trecho (`end`) ou do primeiro ("" = sem lugar medido)."""
    sentence = material.sentences[trecho.sentence_ids[-1 if end else 0]]
    if not sentence.moment_ids:
        return ""
    moment = material.moments.get(sentence.moment_ids[-1 if end else 0])
    return moment.place if moment is not None else ""


@dataclass
class InsertPositions:
    """Posições `depois_do_trecho` medidas para a 2b de um bloco (0 = começo; N = depois do trecho N)."""

    breath: list[int] = field(default_factory=list)  # onde pode haver respiro (troca medida)
    own: list[int] = field(default_factory=list)  # onde pode entrar uma imagem do próprio bloco
    refused: list[Skipped] = field(default_factory=list)  # posições de respiro recusadas, com o motivo
    notes: dict[int, str] = field(default_factory=dict)  # posição de respiro → a troca medida


def insert_positions(
    trechos: list[Trecho], material: EditorMaterial, first_block: bool, opening: bool
) -> InsertPositions:
    """Mede onde pode haver inclusão (informação para a 2b; a IA escolhe se usa).

    Respiro: no começo do bloco (exceto no 1º bloco do vídeo, que começa pela 1ª fala ou imagem escolhida, a
    não ser que o pedido peça abertura com paisagem: `opening`) e entre dois trechos cujo lugar muda (lugar do
    último momento do trecho N × do primeiro do N+1, `material.same_place`). Nunca onde a fala continua
    (`speech_continues`) nem no fim do bloco. Imagem do próprio bloco: de 0 ao fim, menos onde a fala continua.
    """
    result = InsertPositions()
    last = len(trechos)
    if first_block and not opening:
        result.refused.append(Skipped(number=0, reason="começo do vídeo (o pedido não pede abertura com paisagem)"))
    else:
        result.breath.append(0)
        result.notes[0] = "começo do bloco"
    result.own.append(0)
    for n in range(1, last):
        before, after = trechos[n - 1], trechos[n]
        continues = speech_continues(before, material)
        if continues:
            result.refused.append(Skipped(number=n, reason=continues))
            continue
        result.own.append(n)
        a, b = trecho_place(before, material, end=True), trecho_place(after, material, end=False)
        if not a or not b:
            result.refused.append(Skipped(number=n, reason="lugar não medido"))
        elif same_place(a, b):
            result.refused.append(Skipped(number=n, reason=f"mesmo lugar ({a})"))
        else:
            result.breath.append(n)
            result.notes[n] = f"troca de lugar: {a} → {b}"
    if last:
        result.own.append(last)
    return result


@dataclass
class InsertsCheck(_Check):
    inserts: list[InsertChoice] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    added_sec: float = 0.0
    breath_sec: float = 0.0
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


def moment_span(moment, duration: Optional[float], material: EditorMaterial) -> tuple[float, float]:
    """Trecho de uma imagem do bloco incluída: o momento sem as pontas de câmera mexida (sem duração pedida) ou a
    janela estável da duração pedida (`steady.steady_window`); sem medidas, a partir do início do momento."""
    a, b = steady_range(material, moment.clip_id, moment.start_sec, moment.end_sec)
    if duration is None or duration >= b - a:
        return a, b
    if not material.clips[moment.clip_id].motion:
        return a, round(a + duration, 3)
    return steady_window(material, moment.clip_id, a, b, duration)


def breath_span(moment, duration: Optional[float] = None, material: Optional[EditorMaterial] = None) -> tuple[float, float]:
    """Trecho do respiro no clipe: no máximo `RESPIRO_MAX_SEC` (e a duração pedida), na janela estável do momento
    (`steady.steady_window`; sem medidas ou sem `material`, centrado no momento)."""
    length = min(moment.duration, RESPIRO_MAX_SEC, duration if duration is not None else moment.duration)
    if material is not None:
        return steady_window(material, moment.clip_id, moment.start_sec, moment.end_sec, length)
    start = moment.start_sec + (moment.duration - length) / 2
    return round(start, 3), round(start + length, 3)


def _positions_text(positions: list[int]) -> str:
    return ", ".join(str(p) for p in positions) or "nenhuma"


def validate_inserts(
    answer: InsertsAnswer,
    trechos: list[Trecho],
    candidates: list[Candidate],
    material: EditorMaterial,
    usage: Usage,
    spare_sec: Optional[float] = None,
    breath_positions: Optional[list[int]] = None,
    own_positions: Optional[list[int]] = None,
) -> InsertsCheck:
    """Confere a 2b contra uma CÓPIA do uso; `check.usage` é o uso com as inclusões aceitas.

    `spare_sec`: sobra do bloco (None = sem limite de tempo): as inclusões aceitas somam no máximo isso, e os
    respiros que passam são recusados do último para o primeiro. `breath_positions`/`own_positions`: as
    posições oferecidas no pedido (None = todas de 0 ao último trecho).
    """
    state = usage.copy()
    check = InsertsCheck(items=len(answer.inserir), usage=state, has_speech=bool(trechos))
    by_letter = {c.letter: c for c in candidates}
    last = len(trechos)
    every = list(range(last + 1))
    breath_ok = every if breath_positions is None else list(breath_positions)
    own_ok = every if own_positions is None else list(own_positions)
    used: set[str] = set()
    valid_letters = ", ".join(c.letter for c in candidates[:8]) + ("…" if len(candidates) > 8 else "")
    accepted: list[tuple[InsertChoice, tuple[str, float, float]]] = []
    previous_breath: dict[int, bool] = {}  # posição → o último item aceito ali é um respiro
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
        breath = candidate.source != "bloco"
        allowed = breath_ok if breath else own_ok
        if position not in allowed:
            what = "respiro (imagem do acervo)" if breath else "imagem do bloco"
            check.drop("inserir", raw, f"posição {position} fora da lista para {what} (posições: {_positions_text(allowed)})")
            continue
        if breath and previous_breath.get(position):
            check.drop("inserir", raw, f"dois respiros seguidos depois do trecho {position}")
            continue
        moment = material.moments[candidate.moment_id]
        duration = raw.duracao_sec
        if duration is not None and duration <= 0:
            duration = None
        if duration is not None and duration > moment.duration + DURATION_TOLERANCE_SEC:
            if not breath:
                check.warnings.append(
                    f"inserir {raw.letra}: duracao_sec {duration:g} maior que a imagem ({moment.duration:.1f}s); "
                    "entra a imagem inteira"
                )
            duration = None
        start, end = breath_span(moment, duration, material) if breath else moment_span(moment, duration, material)
        clash = state.conflict(moment.clip_id, start, end) if moment.id not in state.moments else "já usado"
        if not clash:
            for _, (clip_id, a, b) in accepted:
                if clip_id == moment.clip_id and min(b, end) - max(a, start) > OVERLAP_TOLERANCE_SEC:
                    clash = "outra imagem desta resposta"
        if clash:
            check.drop("inserir", raw, f"a imagem {raw.letra} ({moment.id}) já está no vídeo ({clash})")
            continue
        if breath and moment.duration > RESPIRO_MAX_SEC and (duration is None or duration > RESPIRO_MAX_SEC):
            check.warnings.append(
                f"respiro {raw.letra} ({moment.id}): {moment.duration:.1f}s cortado para {end - start:.1f}s, na janela "
                f"estável do momento ({start:.1f}–{end:.1f} s; respiro = no máximo {RESPIRO_MAX_SEC:g} s)"
            )
        used.add(raw.letra)
        previous_breath[position] = breath
        trimmed = not breath and (start, end) != (moment.start_sec, round(moment.start_sec + (duration or moment.duration), 3))
        choice = InsertChoice(
            after_trecho=position, letter=raw.letra, moment_id=moment.id,
            duration_sec=round(end - start, 3) if breath or trimmed else duration,
            start_sec=start if breath or trimmed else None, breath=breath,
        )
        accepted.append((choice, (moment.clip_id, start, end)))
    if spare_sec is not None:
        total = sum(b - a for _, (_, a, b) in accepted)
        for k in range(len(accepted) - 1, -1, -1):
            if total <= spare_sec + DURATION_TOLERANCE_SEC:
                break
            choice, (_, a, b) = accepted[k]
            if not choice.breath:
                continue
            total -= b - a
            check.drop(
                "inserir", f"depois_do_trecho={choice.after_trecho}, letra={choice.letter}",
                f"respiro {choice.letter} ({b - a:.1f}s) passa da sobra do bloco ({max(spare_sec, 0):.0f} s)",
            )
            accepted.pop(k)
    for choice, (clip_id, a, b) in accepted:
        state.moments.add(choice.moment_id)
        state.take(clip_id, a, b, f"momento {choice.moment_id}")
        check.added_sec += b - a
        if choice.breath:
            check.breath_sec += b - a
        check.inserts.append(choice)
    check.added_sec = round(check.added_sec, 3)
    check.breath_sec = round(check.breath_sec, 3)
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


def speaker_text(sentence_ids: list[str], material: EditorMaterial) -> str:
    """O que a imagem de quem fala já mostra: a descrição dos momentos que as frases atravessam."""
    seen: list[str] = []
    for sid in sentence_ids:
        for moment_id in material.sentences[sid].moment_ids:
            if moment_id not in seen:
                seen.append(moment_id)
    return " ".join(material.moments[m].description for m in seen if m in material.moments)


def talking_sentences(trecho: Trecho, material: EditorMaterial) -> list[str]:
    """Frases do trecho que começam num momento marcado "falando para a câmera" (as únicas que podem ser
    cobertas: nas outras, a própria imagem já mostra o assunto)."""
    result = []
    for sid in trecho.sentence_ids:
        sentence = material.sentences[sid]
        moment = material.moments.get(sentence.moment_ids[0]) if sentence.moment_ids else None
        if moment is not None and moment.talking_to_camera:
            result.append(sid)
    return result


def coverage_span(
    trecho: Trecho, candidate: Candidate, word: str, material: EditorMaterial, allowed: Optional[list[str]] = None
) -> list[str]:
    """Frases cobertas: da 1ª frase do trecho com a palavra ("mostra o que é dito") ou do lugar do momento
    ("mostra o lugar"), seguindo enquanto o fim da frase cabe na duração do momento (+ `COVERAGE_EXTEND_SEC`).
    `allowed`: só essas frases podem ser cobertas (as ditas para a câmera); a cobertura para na primeira fora."""
    ids = trecho.sentence_ids
    ok = set(ids if allowed is None else allowed) or set(ids)
    moment = material.moments[candidate.moment_id]
    start = next((k for k, sid in enumerate(ids) if sid in ok), 0)
    for k, sid in enumerate(ids):
        if sid not in ok:
            continue
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
        if sid not in ok or material.sentences[sid].end_sec - first.start_sec > moment.duration + COVERAGE_EXTEND_SEC:
            break
        span.append(sid)
    return span


@dataclass
class CoverageCheck(_Check):
    coverages: list[CoverageChoice] = field(default_factory=list)
    nones: list[int] = field(default_factory=list)
    usage: Optional[Usage] = None
    ignored: list[str] = field(default_factory=list)  # trechos não perguntados que vieram na resposta

    @property
    def refused(self) -> Optional[str]:
        if self.ignored and not (self.coverages or self.nones or self.discarded):
            return f"nenhum item para os trechos perguntados (vieram só trechos não perguntados: {', '.join(self.ignored[:8])})"
        return super().refused


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
            check.items -= 1  # ignorado (um aviso para todos), não inválido
            check.ignored.append(str(int(number) if number is not None and number == int(number) else number))
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
        talking = talking_sentences(trecho, material) or trecho.sentence_ids
        speech = " ".join(material.sentences[s].text for s in talking)
        speaker = speaker_text(talking, material)
        word = first_words(raw.palavra)
        literal = _literal_problem(word, candidate, moment, trecho, speech, raw.letra, number, material)
        if literal is None and matched_words(word, speaker):
            literal = f"palavra {word!r} já aparece na imagem de quem fala (\"{first_words(speaker, 12)}\"): não cobre"
        semantic = False
        if literal is not None:
            if not (raw.mostra or raw.porque):
                check.drop("cobertura", raw, literal)
                continue
            # Correspondência explícita: o que a imagem mostra (na descrição dela) e o que a fala diz (no porquê).
            shown = matched_words(raw.mostra, moment_visible_text(moment))
            said = [w for w in matched_words(raw.porque, speech) if not matched_words(w, speaker)]
            if not shown:
                check.drop("cobertura", raw, f"mostra {raw.mostra!r} não está na descrição/lugar da imagem {raw.letra} "
                                             f"(\"{first_words(moment_visible_text(moment), 12)}\"): diga o que a imagem mostra")
                continue
            if not said:
                check.drop("cobertura", raw, f"porque {raw.porque!r} não cita nada da fala do trecho {number} que a "
                                             "imagem de quem fala ainda não mostre: diga que parte da fala a imagem ilustra")
                continue
            word, semantic = said[0], True
        if moment.id in state.moments or state.conflict(moment.clip_id, moment.start_sec, moment.end_sec):
            check.drop("cobertura", raw, f"a imagem {raw.letra} ({moment.id}) já está no vídeo")
            continue
        span = coverage_span(trecho, candidate, word, material, talking)
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
            common_words=list(candidate.words), speech_evidence=word, image_evidence=raw.mostra or word,
            reason=raw.porque, semantic=semantic,
        ))
    return check


def _literal_problem(word, candidate, moment, trecho, speech, letter, number, material) -> Optional[str]:
    """Por que a palavra em comum não é evidência (None = é): conteúdo, e está na fala do trecho e na
    descrição/lugar da imagem (ou, para "mostra o lugar", no lugar da imagem e no da fala)."""
    if not content_words(word):
        return f"sem a palavra em comum (copie uma de: {', '.join(candidate.words)})"
    if candidate.kind == COVERAGE_PLACE:
        places = " ".join(speech_places(trecho, material))
        missing = _missing(word, moment.place) or _missing(word, places)
        if missing:
            return (f"palavra {word!r} não está no lugar da imagem ({moment.place!r}) e no da fala ({places!r}): "
                    f"{', '.join(missing)}")
        return None
    not_said = _missing(word, speech)
    if not_said:
        return f"palavra {word!r} não está na fala do trecho {number}: {', '.join(not_said)}"
    not_shown = _missing(word, moment_visible_text(moment))
    if not_shown:
        return f"palavra {word!r} não está na descrição/lugar da imagem {letter}: {', '.join(not_shown)}"
    return None


def neutral_structure(material: EditorMaterial) -> StructureAnswer:
    """Fallback neutro da estrutura: um bloco por clipe com fala, na ordem do catálogo, sem "fora" (sem
    nenhum clipe com fala, um bloco por clipe)."""
    speaking = [c for c in material.clip_ids if material.has_speech(c)] or material.clip_ids
    return StructureAnswer(
        blocos=[{"id": f"B{n:02d}", "clipes": [c]} for n, c in enumerate(speaking, start=1)],
        notas="fallback neutro: a IA falhou na estrutura",
    )
