"""Montagem das frases (unidades de decisão do Diretor) a partir da transcrição."""

import re
from typing import Optional

from ai_video_editor.director.schema import Sentence, SentenceParams
from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment

# Pontuação final, opcionalmente seguida de aspas ou parênteses de fechamento.
_SENTENCE_END = re.compile(r"[.?!…][\"'”’)\]]*$")


# Pontuação de pausa interna: ponto de corte preferido quando a frase passa do limite.
_SOFT_END = re.compile(r"[,;:][\"'”’)\]]*$")


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def _to_sentence(sentence_id: int, group: list[TranscriptSegment]) -> Sentence:
    return Sentence(
        id=sentence_id,
        start_sec=group[0].start_sec,
        end_sec=group[-1].end_sec,
        text=" ".join(seg.text.strip() for seg in group if seg.text.strip()),
        segment_ids=[seg.id for seg in group],
    )


def _words(group: list[TranscriptSegment]) -> int:
    return sum(len(seg.text.split()) for seg in group)


def _within_limits(group: list[TranscriptSegment], params: SentenceParams) -> bool:
    return (
        group[-1].end_sec - group[0].start_sec <= params.max_sentence_sec
        and _words(group) <= params.max_sentence_words
    )


def _split_index(group: list[TranscriptSegment], params: SentenceParams) -> int:
    """Índice do último segmento da frase a fechar, quando `group` passou do limite.

    Nunca divide dentro de um segmento. Só considera prefixos que cabem no limite (ao menos
    o primeiro segmento). Preferência: fim de segmento com vírgula/;/: (o mais tardio),
    maior pausa até o segmento seguinte, qualquer fim de segmento (o mais tardio).
    """
    candidates = [k for k in range(len(group) - 1) if k == 0 or _within_limits(group[: k + 1], params)]
    commas = [k for k in candidates if _SOFT_END.search(group[k].text.strip())]
    if commas:
        return commas[-1]
    gaps = {k: group[k + 1].start_sec - group[k].end_sec for k in candidates}
    best = max(gaps.values())
    if best > 0:
        return max(k for k, gap in gaps.items() if gap == best)
    return candidates[-1]


def build_sentences(transcript: Transcript, params: Optional[SentenceParams] = None) -> list[Sentence]:
    """Junta segmentos consecutivos até pontuação final (. ? !), pausa longa ou limite de tamanho.

    O Whisper às vezes quebra segmentos no meio da frase; juntar evita que o Diretor
    decida sobre meia frase. Em fala corrida e sem pontuação o limite de duração/palavras
    (`params`) fecha a frase no melhor ponto. Segmentos sem texto são ignorados.
    """
    params = params or SentenceParams()
    sentences: list[Sentence] = []
    group: list[TranscriptSegment] = []
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        if group and segment.start_sec - group[-1].end_sec > params.max_pause_sec:
            sentences.append(_to_sentence(len(sentences), group))
            group = []
        group.append(segment)
        if _ends_sentence(segment.text):
            sentences.append(_to_sentence(len(sentences), group))
            group = []
            continue
        while len(group) > 1 and not _within_limits(group, params):
            k = _split_index(group, params)
            sentences.append(_to_sentence(len(sentences), group[: k + 1]))
            group = group[k + 1 :]
    if group:
        sentences.append(_to_sentence(len(sentences), group))
    return sentences


def split_by_removal(
    sentences: list[Sentence], removed_ids: set[int]
) -> tuple[list[TimeRange], list[TimeRange]]:
    """(intervalos a cortar, intervalos das frases mantidas).

    Frases removidas consecutivas viram um único intervalo, para que a pausa entre
    elas também saia.
    """
    cuts: list[TimeRange] = []
    kept: list[TimeRange] = []
    run: list[Sentence] = []
    for sentence in sentences:
        if sentence.id in removed_ids:
            run.append(sentence)
            continue
        kept.append(TimeRange(start_sec=sentence.start_sec, end_sec=sentence.end_sec))
        if run:
            cuts.append(TimeRange(start_sec=run[0].start_sec, end_sec=run[-1].end_sec))
            run = []
    if run:
        cuts.append(TimeRange(start_sec=run[0].start_sec, end_sec=run[-1].end_sec))
    return cuts, kept
