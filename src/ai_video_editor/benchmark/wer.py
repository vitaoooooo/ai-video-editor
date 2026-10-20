"""WER (Word Error Rate) por distância de edição entre palavras, sem dependências externas.

Limitações conhecidas (a comparação é textual, não semântica):
- Números: "2" (dígito) contra "two" (por extenso) contam como uma substituição, embora a fala seja a mesma.
- Contrações e grafias: "gonna" × "going to", "okay" × "OK", "MatLab" × "matlab" (esta última o
  lowercase resolve) contam como erro quando as grafias diferem.
- Hesitações ("uh", "um") e repetições: a transcrição oficial costuma omiti-las e o Whisper pode
  registrá-las (ou o contrário); isso entra como inserção ou omissão.
- Símbolos falados: "x" × "ex", "minus" × "-" e afins dependem de como cada texto os escreveu.
- Hífens e travessões viram espaço ("right-hand" vira "right hand"); apóstrofos dentro da palavra
  são mantidos ("don't") e os das pontas, descartados.
- A referência não é perfeita: erros e trechos faltando nela aparecem como erro do modelo.
"""

import re
import unicodedata
from collections import Counter

from ai_video_editor.benchmark.schema import SubstitutionCount, WerResult

# Variações tipográficas do apóstrofo, unificadas em "'".
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "´": "'", "`": "'", "ʼ": "'"})

# Tudo que não é letra, dígito, apóstrofo ou espaço vira espaço (inclui hífens e travessões).
_NON_WORD = re.compile(r"[^\w'\s]|_", re.UNICODE)

# Quantas substituições mais frequentes entram no resultado.
TOP_SUBSTITUTIONS = 10

# Códigos das operações do alinhamento.
_MATCH, _SUB, _DEL, _INS = 0, 1, 2, 3


def normalize_text(text: str) -> list[str]:
    """Normaliza o texto e devolve a lista de palavras.

    Minúsculas, apóstrofos unificados, pontuação removida e espaços colapsados.
    """
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    text = _NON_WORD.sub(" ", text)
    words = (word.strip("'") for word in text.split())
    return [word for word in words if word]


def align_words(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, int, int, Counter]:
    """Alinha as palavras e devolve (substituições, omissões, inserções, pares trocados).

    Programação dinâmica clássica (Levenshtein por palavra); no empate, prefere casar/substituir,
    depois omitir, depois inserir. Os pares trocados são (palavra da referência, palavra do modelo).
    """
    # Prefixo e sufixo idênticos não mudam o custo mínimo e encurtam a tabela.
    start = 0
    limit = min(len(reference), len(hypothesis))
    while start < limit and reference[start] == hypothesis[start]:
        start += 1
    ref_end, hyp_end = len(reference), len(hypothesis)
    while (
        ref_end > start and hyp_end > start and reference[ref_end - 1] == hypothesis[hyp_end - 1]
    ):
        ref_end -= 1
        hyp_end -= 1
    ref = reference[start:ref_end]
    hyp = hypothesis[start:hyp_end]
    n, m = len(ref), len(hyp)

    # `ops` guarda a operação escolhida em cada célula (1 byte), para refazer o caminho.
    width = m + 1
    ops = bytearray((n + 1) * width)
    for j in range(1, m + 1):
        ops[j] = _INS
    for i in range(1, n + 1):
        ops[i * width] = _DEL

    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        ref_word = ref[i - 1]
        base = i * width
        for j in range(1, m + 1):
            if ref_word == hyp[j - 1]:
                best, op = previous[j - 1], _MATCH
            else:
                best, op = previous[j - 1] + 1, _SUB
            deletion = previous[j] + 1
            if deletion < best:
                best, op = deletion, _DEL
            insertion = current[j - 1] + 1
            if insertion < best:
                best, op = insertion, _INS
            current[j] = best
            ops[base + j] = op
        previous = current

    substitutions = deletions = insertions = 0
    pairs: Counter = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        op = ops[i * width + j]
        if op == _MATCH:
            i, j = i - 1, j - 1
        elif op == _SUB:
            substitutions += 1
            pairs[(ref[i - 1], hyp[j - 1])] += 1
            i, j = i - 1, j - 1
        elif op == _DEL:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions, pairs


def compute_wer(reference_text: str, hypothesis_text: str) -> WerResult:
    """Calcula o WER de `hypothesis_text` em relação a `reference_text` (ambos normalizados)."""
    reference = normalize_text(reference_text)
    hypothesis = normalize_text(hypothesis_text)
    substitutions, deletions, insertions, pairs = align_words(reference, hypothesis)
    errors = substitutions + deletions + insertions
    if reference:
        wer = errors / len(reference)
    else:
        wer = 0.0 if not hypothesis else 1.0
    top = [
        SubstitutionCount(reference=ref_word, hypothesis=hyp_word, count=count)
        for (ref_word, hyp_word), count in sorted(
            pairs.items(), key=lambda item: (-item[1], item[0])
        )[:TOP_SUBSTITUTIONS]
    ]
    return WerResult(
        wer=wer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference),
        hypothesis_words=len(hypothesis),
        top_substitutions=top,
    )
