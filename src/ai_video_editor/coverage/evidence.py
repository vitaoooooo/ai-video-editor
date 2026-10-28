"""Evidências da cobertura, conferidas pelo código (sem IA): a fala citada está no texto das frases, a
imagem citada está na análise da tomada e as duas falam da mesma coisa.

- **Comparação de palavras:** sem diferenciar maiúsculas, acentos e pontuação; plural simples (`lagos`,
  `flores`, `cães`) vale como o singular; sinônimos da tabela `SYNONYMS` valem como a palavra principal.
- **Evidência de fala:** trecho literal (palavras seguidas) do texto das frases cobertas.
- **Evidência de imagem:** uma tag da tomada ou um trecho literal da descrição dela.
- **Ligação:** as duas evidências têm uma palavra de conteúdo em comum (fora de `STOPWORDS` e dos
  conceitos genéricos `GENERIC_CONCEPTS`, que aparecem em quase toda tomada). **Forte** quando essa palavra
  é uma tag da tomada (o assunto principal da imagem, não um detalhe da descrição).
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

# Sinônimos simples (palavra principal → variantes), sem acento. O material real mistura português e
# espanhol, por isso as variantes em espanhol dos lugares mais comuns de um vlog de viagem.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "lago": ("lagoa", "laguna"),
    "montanha": ("cordilheira", "morro", "serra", "montana", "cerro"),
    "cachorro": ("cao", "cachorra", "perro", "perra"),
    "ilha": ("isla", "islita"),
    "praia": ("playa",),
    "onibus": ("omnibus", "microonibus", "bus"),
    "trilha": ("sendero",),
}
# Palavras que não dizem o assunto (artigos, preposições, pronomes, verbos muito comuns; pt e es).
STOPWORDS = frozenset(
    """
    que com uma umas uns por para pra pro nos nas dos das num numa sem mas mais ela ele eles elas aos foi
    tem vai ver tao sua seu meu minha nosso nossa isso esse essa este esta aqui ali ai tudo todo toda bem
    muito muita agora entao ser estar esta estamos vamos gente cara olha olhar sim nao como quando onde
    del los las una unos por con pero muy aca alla este esto eso ese hay son fue hoy
    the and with
    """.split()
)
# Conceitos que não provam relação entre a fala e a imagem (estão em quase tudo).
GENERIC_CONCEPTS = frozenset({"pessoa", "rosto", "detalhe", "paisagem", "arvore", "ceu", "imagem", "vista", "lugar", "coisa", "dia"})
# Palavra de conteúdo: pelo menos este número de letras.
MIN_CONCEPT_CHARS = 3

_WORD = re.compile(r"[a-z0-9]+")


def plain(text: str) -> str:
    """Minúsculas, sem acentos."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    """Palavras sem acento, na ordem (pontuação some)."""
    return _WORD.findall(plain(text))


_HEAD = {variant: head for head, variants in SYNONYMS.items() for variant in (head, *variants)}


def forms(word: str) -> frozenset[str]:
    """Formas equivalentes de uma palavra já sem acento: ela, o singular simples e a palavra principal
    dos sinônimos."""
    out = {word}
    if len(word) > 3 and word.endswith("s"):
        out.add(word[:-1])
    if len(word) > 4 and word.endswith("es"):
        out.add(word[:-2])
    if len(word) >= 4 and word.endswith(("oes", "aes")):
        out.add(word[:-3] + "ao")
    out = {word} | {w for w in out if len(w) >= 3}
    return frozenset(out | {_HEAD[w] for w in out if w in _HEAD})


def same_word(a: str, b: str) -> bool:
    return bool(forms(a) & forms(b))


def find_phrase(haystack: list[str], phrase: list[str]) -> Optional[int]:
    """Posição da primeira ocorrência de `phrase` (palavras seguidas) em `haystack`; None se não há."""
    n = len(phrase)
    if not n:
        return None
    for k in range(len(haystack) - n + 1):
        if all(same_word(haystack[k + m], phrase[m]) for m in range(n)):
            return k
    return None


def concepts(text: str) -> list[str]:
    """Palavras de conteúdo do texto (fora de stopwords e de conceitos genéricos)."""
    found = []
    for w in words(text):
        if len(w) < MIN_CONCEPT_CHARS or w in STOPWORDS or forms(w) & GENERIC_CONCEPTS or forms(w) & STOPWORDS:
            continue
        found.append(w)
    return list(dict.fromkeys(found))


def shared_concepts(a: str, b: str) -> list[str]:
    """Palavras de conteúdo de `a` que também estão em `b` (com plural e sinônimos)."""
    others = concepts(b)
    return [w for w in concepts(a) if any(same_word(w, o) for o in others)]


def label(word: str) -> str:
    """Forma mostrada de um conceito: a palavra principal dos sinônimos, se houver."""
    heads = sorted(f for f in forms(word) if f in SYNONYMS)
    return heads[0] if heads else word


@dataclass(frozen=True)
class SpeechHit:
    """Onde a evidência de fala está: índice da frase (entre as cobertas) e a fração do texto dela."""

    sentence: int
    fraction: float


def find_speech(texts: list[str], evidence: str) -> Optional[SpeechHit]:
    """Evidência de fala no texto das frases (em uma delas ou atravessando frases seguidas); None se não está.

    `fraction` = posição do meio do trecho na frase em que ele começa (0 = início, 1 = fim), pela contagem
    de palavras: a fala tem ritmo parecido, então serve para pôr a imagem perto de quando a coisa é dita.
    """
    phrase = words(evidence)
    if not phrase:
        return None
    joined, owner = [], []
    for k, text in enumerate(texts):
        tokens = words(text)
        joined += tokens
        owner += [k] * len(tokens)
    at = find_phrase(joined, phrase)
    if at is None:
        return None
    k = owner[at]
    first = owner.index(k)
    count = owner.count(k)
    middle = min(at + len(phrase) / 2, first + count) - first
    return SpeechHit(sentence=k, fraction=max(0.0, min(1.0, middle / count)))


def image_matches(evidence: str, description: str, tags: Iterable[str]) -> bool:
    """A evidência de imagem é uma tag da tomada ou um trecho literal da descrição."""
    phrase = words(evidence)
    if not phrase:
        return False
    return any(find_phrase(words(h), phrase) is not None for h in [description, *tags])


def tag_concepts(tags: Iterable[str]) -> list[str]:
    return [w for t in tags for w in concepts(t)]


def is_strong(shared: Iterable[str], tags: Iterable[str]) -> bool:
    """Algum conceito em comum entre as evidências é uma tag da tomada."""
    tagged = tag_concepts(tags)
    return any(same_word(w, t) for w in shared for t in tagged)

