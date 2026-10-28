"""O roteiro visto pelo Editor de IA: índices por ID, seções de texto, a pré-seleção do acervo e os candidatos
de cobertura medidos.

Nada aqui decide edição: só organiza o que o roteiro já mediu e descreveu, para montar os pedidos e
conferir as respostas. A pré-seleção do acervo é só para caber no contexto, e os candidatos de cobertura
são uma medida (palavra em comum, mesmo lugar): a IA escolhe, e "nenhuma" é sempre uma opção.

Palavra em comum (cobertura e afinidade dos respiros): só palavras de conteúdo (sem `STOPWORDS` nem
`GENERIC_WORDS`) e só o que a imagem mostra de fato — a descrição e o lugar do momento (`moment_visible_text`);
as tags ficam de fora porque têm erros (ex.: "rio" nas tags de uma praça).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ai_video_editor.roteiro.report import clip_section, clock
from ai_video_editor.roteiro.schema import Moment, ProjectRoteiro, RoteiroClip, RoteiroSentence

# Palavras vazias (pt/es/en) que não contam como evidência: artigos, preposições, pronomes, verbos de ligação.
STOPWORDS = frozenset(
    """
    a o as os um uma uns umas de da do das dos dum duma em na no nas nos num numa por pela pelo pelas pelos
    para pra pro com sem sob sobre entre ate desde e ou mas que se nao sim ja la aqui ali ai isso isto
    esse essa esses essas este esta estes estas aquele aquela aqueles aquelas ele ela eles elas eu tu voce
    voces nos vos me te lhe meu minha seu sua nosso nossa dele dela muito muita muitos muitas mais menos
    tao tambem so bem ser estar ter haver foi era sao esta estao tem tinha vai vou fazer feito como quando
    onde porque pois entao agora hoje aqui cada todo toda todos todas outro outra coisa gente tipo assim
    del los las una unos unas por para con sin pero muy esto eso aqui alli es son esta estan hay
    the an and or of to in on at for with from by is are was were be been this that these those it its
    there here very just also so
    """.split()
)
# Palavras genéricas (pt/es/en, sem acento) que não contam como palavra em comum: cabem em quase qualquer fala
# e em quase qualquer imagem ("pessoas" ligava "dois casais" a uma praia com pessoas). Estende `STOPWORDS`.
GENERIC_WORDS = frozenset(
    """
    pessoa pessoas gente coisa coisas lugar lugares dia dias vista vistas paisagem paisagens natureza aqui ali
    ai la aca alli alla acola cosa cosas persona personas paisaje paisajes naturaleza
    people person persons thing things place places day days view views landscape landscapes nature
    """.split()
)
# Uma palavra de conteúdo tem pelo menos isso de letras (a, o, de, em... ficam de fora).
MIN_CONTENT_CHARS = 3
# Radical em comum (uma palavra começa com a outra) a partir de tantas letras: praia/praias, verde/verdes.
MIN_STEM_CHARS = 4
# Uma letra a mais no meio vale como a mesma palavra a partir de tantas letras (ônibus/ómnibus, pt/es).
MIN_INSERTION_CHARS = 6
# Plural → singular (sem acento), na ordem: ões/ães → ão, ais → al, eis → el, ns → m, res/zes/ses → r/z/s, s → nada.
_PLURALS = (("oes", "ao"), ("aes", "ao"), ("ais", "al"), ("eis", "el"), ("ois", "ol"), ("ns", "m"))

_WORD = re.compile(r"[a-z0-9]+")


def plain(text: str) -> str:
    """Minúsculas e sem acentos."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def words(text: str) -> list[str]:
    return _WORD.findall(plain(text))


def content_words(text: str) -> list[str]:
    """Palavras de conteúdo da evidência, sem repetição e em ordem."""
    seen: list[str] = []
    for word in words(text):
        if len(word) >= MIN_CONTENT_CHARS and word not in STOPWORDS and word not in GENERIC_WORDS and word not in seen:
            seen.append(word)
    return seen


def stem(word: str) -> str:
    """Radical simples de uma palavra já sem acento: só tira o plural (cachorros → cachorro, animais → animal)."""
    for suffix, singular in _PLURALS:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)] + singular
    if word.endswith(("res", "zes", "ses")) and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > MIN_CONTENT_CHARS:
        return word[:-1]
    return word


def _one_insertion(short: str, long_: str) -> bool:
    return len(long_) == len(short) + 1 and any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def same_word(a: str, b: str) -> bool:
    """Mesma palavra para a evidência: igual sem o plural, uma começa com a outra (≥ `MIN_STEM_CHARS` letras)
    ou, com ≥ `MIN_INSERTION_CHARS` letras, uma letra a mais (ônibus/ómnibus). Recebe palavras sem acento."""
    if a == b:
        return True
    a, b = stem(a), stem(b)
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= MIN_STEM_CHARS and long_.startswith(short):
        return True
    return len(long_) >= MIN_INSERTION_CHARS and _one_insertion(short, long_)


def matched_words(evidence: str, target: str) -> list[str]:
    """Palavras de conteúdo da evidência que aparecem no texto (sem acento/caixa; vale o mesmo radical)."""
    target_words = set(words(target))
    return [w for w in content_words(evidence) if any(same_word(w, t) for t in target_words)]


def common_words(a: str, b: str) -> list[str]:
    """Palavras de conteúdo de `a` que também são palavras de conteúdo de `b` (mesma regra de `same_word`)."""
    other = content_words(b)
    return [w for w in content_words(a) if any(same_word(w, t) for t in other)]


def moment_text(moment: Moment) -> str:
    """Descrição, lugar e tags do momento (afinidade do acervo; NÃO vale como palavra em comum)."""
    return " ".join([moment.description, moment.place, *moment.tags])


def moment_visible_text(moment: Moment) -> str:
    """O que a palavra em comum pode citar da imagem: só a descrição e o lugar (as tags têm erros)."""
    return " ".join([moment.description, moment.place])


def same_place(a: str, b: str) -> bool:
    """Mesmo lugar: iguais sem acento, caixa e espaços extras (vazio nunca é igual)."""
    a, b = " ".join(plain(a).split()), " ".join(plain(b).split())
    return bool(a) and a == b


# Pedido que pede abertura com paisagem: uma palavra que começa com um radical de cada lista (sem acento).
OPENING_STEMS = ("abert", "abrir", "abre", "abra", "abrindo", "comec", "inici", "intro", "apertura", "empez", "comienz")
LANDSCAPE_STEMS = ("paisag", "paisaj", "drone", "aere", "panoram", "vista", "natureza", "naturaleza", "landscape")


def asks_landscape_opening(request: str) -> bool:
    """O pedido fala em abrir/começar o vídeo E em paisagem/drone/vista (medida no texto, sem decisão)."""
    found = words(request)
    return any(w.startswith(OPENING_STEMS) for w in found) and any(w.startswith(LANDSCAPE_STEMS) for w in found)


def _fmt_sec(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


@dataclass
class EditorMaterial:
    """Índices do roteiro por ID (clipes na ordem do catálogo; frases e momentos na ordem de cada clipe)."""

    roteiro: ProjectRoteiro
    clips: dict[str, RoteiroClip] = field(default_factory=dict)
    sentences: dict[str, RoteiroSentence] = field(default_factory=dict)
    moments: dict[str, Moment] = field(default_factory=dict)
    clip_sentences: dict[str, list[RoteiroSentence]] = field(default_factory=dict)
    clip_moments: dict[str, list[Moment]] = field(default_factory=dict)

    @classmethod
    def from_roteiro(cls, roteiro: ProjectRoteiro) -> "EditorMaterial":
        material = cls(roteiro=roteiro)
        for clip in sorted(roteiro.clips, key=lambda c: c.order):
            material.clips[clip.id] = clip
            material.clip_sentences[clip.id] = []
            material.clip_moments[clip.id] = []
        for sentence in sorted(roteiro.sentences, key=lambda s: s.index):
            material.sentences[sentence.id] = sentence
            material.clip_sentences.setdefault(sentence.clip_id, []).append(sentence)
        for moment in roteiro.moments:
            material.moments[moment.id] = moment
            material.clip_moments.setdefault(moment.clip_id, []).append(moment)
        for moments in material.clip_moments.values():
            moments.sort(key=lambda m: m.start_sec)
        return material

    @property
    def clip_ids(self) -> list[str]:
        return list(self.clips)

    def has_speech(self, clip_id: str) -> bool:
        return bool(self.clip_sentences.get(clip_id))

    def sentence_range(self, from_id: str, to_id: str) -> list[RoteiroSentence]:
        """Frases de `from_id` a `to_id` (mesmo clipe, em ordem); vazio se não for uma faixa válida."""
        a, b = self.sentences.get(from_id), self.sentences.get(to_id)
        if a is None or b is None or a.clip_id != b.clip_id or a.index > b.index:
            return []
        return [s for s in self.clip_sentences[a.clip_id] if a.index <= s.index <= b.index]

    def moment_sentences(self, moment_ids: Iterable[str]) -> list[RoteiroSentence]:
        """Frases que começam nos momentos dados (como no roteiro.md: cada frase no momento em que começa)."""
        wanted = set(moment_ids)
        return [s for s in self.roteiro.sentences if s.moment_ids and s.moment_ids[0] in wanted]

    def section(self, clip_id: str, moment_ids: Optional[list[str]] = None) -> str:
        """Seção do clipe no formato do roteiro.md; com `moment_ids`, só aqueles momentos (e suas frases)."""
        clip = self.clips[clip_id]
        if moment_ids is None:
            return clip_section(self.roteiro, clip)
        wanted = set(moment_ids)
        partial = self.roteiro.model_copy(
            update={
                "moments": [m for m in self.clip_moments[clip_id] if m.id in wanted],
                "sentences": self.moment_sentences(wanted),
            }
        )
        return clip_section(partial, clip)

    def speech_sec(self, clip_id: str) -> float:
        """Soma das durações das frases do clipe."""
        return sum(s.end_sec - s.start_sec for s in self.clip_sentences.get(clip_id, []))

    def structure_line(self, clip_id: str, summary: Optional[str], words: int) -> str:
        """Clipe numa linha para a estrutura: ID, papel, duração, fala (com os segundos de frases), data e o
        resumo nas primeiras `words` palavras (a lista inteira sempre cabe: nunca precisa encurtar depois)."""
        clip = self.clips[clip_id]
        parts = [clip.id, clip.role, clock(clip.duration_sec)]
        if self.has_speech(clip_id):
            parts.append(f"fala sim ({round(self.speech_sec(clip_id))} s de frases)")
        else:
            parts.append("fala não")
        if clip.recorded_at is not None:
            parts.append(clip.recorded_at.strftime("%Y-%m-%d %H:%M"))
        text = " ".join((summary or "(sem resumo)").split())
        cut = text.split()
        if len(cut) > words:
            text = " ".join(cut[:words]) + "…"
        return f"- {' · '.join(parts)} — {text}"

    def clip_line(self, clip_id: str) -> str:
        """Linha curta do clipe: ID, papel, duração, fala, data, ordem do catálogo."""
        clip = self.clips[clip_id]
        parts = [clip.id, clip.role, clock(clip.duration_sec)]
        parts.append("com fala" if self.has_speech(clip_id) else "sem fala")
        if clip.language:
            parts.append(f"idioma {clip.language}")
        if clip.recorded_at is not None:
            parts.append(f"gravado {clip.recorded_at.strftime('%Y-%m-%d %H:%M')}")
        parts.append(f"ordem {clip.order}")
        return " · ".join(parts)

    def moment_line(self, moment: Moment) -> str:
        """Momento do acervo numa linha: ID, duração, lugar, tags, problemas e descrição."""
        parts = [moment.id, _fmt_sec(moment.duration)]
        if moment.place:
            parts.append(f"lugar: {moment.place}")
        if moment.tags:
            parts.append(f"tags: {', '.join(moment.tags)}")
        if moment.sentence_ids:
            parts.append("tem fala")
        if moment.issues:
            parts.append(f"[{', '.join(moment.issues)}]")
        return f"- {' · '.join(parts)} — {moment.description or '(sem descrição)'}"


def library_ranking(
    material: EditorMaterial, block_clip_ids: list[str], candidates: list[Moment]
) -> list[Moment]:
    """Momentos do acervo em ordem de afinidade com o bloco (só para caber no contexto; a IA escolhe).

    Medida: tags em comum com os momentos do bloco + 2 se o lugar é um dos lugares do bloco (sem
    acento/caixa). Empate: ordem do catálogo e do clipe.
    """
    tags: set[str] = set()
    places: set[str] = set()
    for clip_id in block_clip_ids:
        for moment in material.clip_moments.get(clip_id, []):
            tags |= {plain(t) for t in moment.tags}
            if moment.place:
                places.add(plain(moment.place))
    order = {clip_id: n for n, clip_id in enumerate(material.clip_ids)}

    def score(moment: Moment) -> int:
        common = len({plain(t) for t in moment.tags} & tags)
        return common + (2 if moment.place and plain(moment.place) in places else 0)

    return sorted(candidates, key=lambda m: (-score(m), order.get(m.clip_id, 0), m.start_sec))


@dataclass
class CoverageMatch:
    """Candidato de cobertura medido para um trecho: o momento, o tipo e as palavras em comum."""

    moment: Moment
    kind: str  # COVERAGE_SAYS ou COVERAGE_PLACE (textos de `editor.schema`)
    words: list[str]


def coverage_matches(
    speech: str, speech_places: list[str], candidates: Iterable[Moment], limit: int, says: str, place: str
) -> list[CoverageMatch]:
    """Até `limit` momentos com palavra de conteúdo em comum com a fala (`says`: as palavras da fala achadas
    na descrição/lugar, nunca só nas tags; regra de `same_word`, sem palavras genéricas) ou com o mesmo lugar de um momento da fala (`place`:
    palavras em comum entre os lugares). Ordem: mais palavras da fala, depois mais palavras de lugar;
    empate na ordem recebida. Uma medida, não uma escolha: a IA decide se a imagem mostra mesmo aquilo."""
    found: list[tuple[int, int, int, CoverageMatch]] = []
    for n, moment in enumerate(candidates):
        said = common_words(speech, moment_visible_text(moment))
        places = []
        for where in speech_places:
            places += [w for w in common_words(moment.place, where) if w not in places]
        if said:
            found.append((-len(said), -len(places), n, CoverageMatch(moment, says, said)))
        elif places:
            found.append((0, -len(places), n, CoverageMatch(moment, place, places)))
    found.sort(key=lambda item: item[:3])
    return [item[3] for item in found[:limit]]


def breath_affinity(moment: Moment, neighbor_text: str, neighbor_places: list[str]) -> list[str]:
    """Afinidade medida de um momento do acervo com os trechos vizinhos de uma posição de respiro: o lugar
    (igual a um lugar vizinho) e as palavras de conteúdo em comum com a fala/lugares vizinhos (descrição e
    lugar do momento). Vazio = sem afinidade (vai para o fim da lista)."""
    found: list[str] = []
    if moment.place and any(same_place(moment.place, p) for p in neighbor_places):
        found.append(plain(moment.place))
    for word in common_words(neighbor_text, moment_visible_text(moment)):
        if word not in found:
            found.append(word)
    return found
