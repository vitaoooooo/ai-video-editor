"""Normalização de texto por idioma para o "WER normalizado" do benchmark.

O WER bruto só usa a normalização básica de `wer.normalize_text` (minúsculas, apóstrofos
unificados, pontuação e hífens viram espaço). O WER normalizado aplica, nos DOIS textos
(referência e hipótese), as regras do idioma abaixo, para que diferenças só de grafia não
contem como erro. Idioma sem regras cadastradas = só a normalização básica.

Regras do inglês (`en`):
- Números por extenso viram dígitos: 0–999 ("zero", "seventeen", "forty two", "one hundred and five",
  "a hundred" → 100) e milhares simples ("two thousand five" → 2005). Dígitos já escritos ficam como estão.
  Só se juntam combinações da fala corrente: dezena + unidade ("twenty one"), unidade + "hundred",
  e o resto depois de "hundred"/"thousand". Sequências como "one two three" continuam
  três números separados ("1 2 3").
- Separador de milhar entre dígitos é removido ("1,000" → "1000").
- Grafia: "okay" → "ok".
- Possessivo/plural com apóstrofo: "x's" → "xs", "y's" → "ys" (apóstrofo antes do "s" final some;
  outras contrações como "don't" são mantidas).
- Hífens já viram espaço na normalização básica ("twenty-one" → "twenty one" → 21).

Para um idioma novo (ex.: "pt"), escreva uma função `str -> list[str]` e registre em `NORMALIZERS`.
"""

import re
from typing import Callable, Optional

from ai_video_editor.benchmark.wer import normalize_text

_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_SPELLINGS = {"okay": "ok"}


def _parse_below_100(tokens: list[str], i: int) -> tuple[Optional[int], int]:
    """Lê 0–99 a partir de `i`; devolve (valor, próximo índice) ou (None, i)."""
    if i >= len(tokens):
        return None, i
    word = tokens[i]
    if word in _EN_TENS:
        value = _EN_TENS[word]
        if i + 1 < len(tokens) and 1 <= _EN_UNITS.get(tokens[i + 1], 0) <= 9:
            return value + _EN_UNITS[tokens[i + 1]], i + 2
        return value, i + 1
    if word in _EN_UNITS:
        return _EN_UNITS[word], i + 1
    return None, i


def _parse_below_1000(tokens: list[str], i: int) -> tuple[Optional[int], int]:
    """Lê 0–999 a partir de `i`, com "hundred" ("a hundred", "two hundred and five")."""
    if i >= len(tokens):
        return None, i
    start = i
    multiplier: Optional[int] = None
    if tokens[i] == "a" and i + 1 < len(tokens) and tokens[i + 1] == "hundred":
        multiplier, i = 1, i + 1
    elif 1 <= _EN_UNITS.get(tokens[i], 0) <= 9 and i + 1 < len(tokens) and tokens[i + 1] == "hundred":
        multiplier = _EN_UNITS[tokens[i]]
        i += 1
    if multiplier is None:
        return _parse_below_100(tokens, start)
    value = multiplier * 100
    i += 1  # consome "hundred"
    after_and = i + 1 if i < len(tokens) and tokens[i] == "and" else i
    rest, nxt = _parse_below_100(tokens, after_and)
    if rest is not None:
        return value + rest, nxt
    return value, i


def _parse_number(tokens: list[str], i: int) -> tuple[Optional[int], int]:
    """Lê um número de 0 a 999999 a partir de `i`; devolve (valor, próximo índice)."""
    if tokens[i] == "a" and i + 1 < len(tokens) and tokens[i + 1] == "thousand":
        high, i = 1, i + 1
    else:
        high, nxt = _parse_below_1000(tokens, i)
        if high is None:
            return None, i
        if nxt >= len(tokens) or tokens[nxt] != "thousand":
            return high, nxt
        i = nxt
    # `i` aponta para "thousand"
    value = high * 1000
    i += 1
    after_and = i + 1 if i < len(tokens) and tokens[i] == "and" else i
    rest, nxt = _parse_below_1000(tokens, after_and)
    if rest is not None:
        return value + rest, nxt
    return value, i


def _numbers_to_digits_en(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        value, nxt = _parse_number(tokens, i)
        if value is None:
            out.append(tokens[i])
            i += 1
        else:
            out.append(str(value))
            i = nxt
    return out


def normalize_en(text: str) -> list[str]:
    """Normalização do inglês (ver as regras no topo do módulo)."""
    text = _THOUSANDS_SEPARATOR.sub("", text)
    tokens = []
    for word in normalize_text(text):
        if word.endswith("'s") and len(word) > 2:
            word = word[:-2] + "s"
        tokens.append(_EN_SPELLINGS.get(word, word))
    return _numbers_to_digits_en(tokens)


NORMALIZERS: dict[str, Callable[[str], list[str]]] = {"en": normalize_en}


def has_normalizer(language: Optional[str]) -> bool:
    return bool(language) and language.lower().split("-")[0] in NORMALIZERS


def normalize_words(text: str, language: Optional[str]) -> list[str]:
    """Palavras do texto com as regras do idioma; sem regras, só a normalização básica."""
    if has_normalizer(language):
        return NORMALIZERS[language.lower().split("-")[0]](text)
    return normalize_text(text)
