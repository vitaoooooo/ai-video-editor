"""O roteiro visto pelo Editor de IA: índices por ID, seções de texto e a pré-seleção do acervo.

Nada aqui decide edição: só organiza o que o roteiro já mediu e descreveu, para montar os pedidos e
conferir as respostas. A pré-seleção do acervo é só para caber no contexto (a IA escolhe).
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
# Uma palavra de conteúdo tem pelo menos isso de letras (a, o, de, em... ficam de fora).
MIN_CONTENT_CHARS = 3
# Radical em comum (uma palavra começa com a outra) a partir de tantas letras: praia/praias, verde/verdes.
MIN_STEM_CHARS = 4

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
        if len(word) >= MIN_CONTENT_CHARS and word not in STOPWORDS and word not in seen:
            seen.append(word)
    return seen


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= MIN_STEM_CHARS and long_.startswith(short)


def matched_words(evidence: str, target: str) -> list[str]:
    """Palavras de conteúdo da evidência que aparecem no texto (sem acento/caixa; vale o mesmo radical)."""
    target_words = set(words(target))
    return [w for w in content_words(evidence) if any(_same_word(w, t) for t in target_words)]


def moment_text(moment: Moment) -> str:
    """O que a evidência da imagem pode citar: descrição, lugar e tags do momento."""
    return " ".join([moment.description, moment.place, *moment.tags])


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
