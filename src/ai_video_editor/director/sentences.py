"""Montagem das frases (unidades de decisão do Diretor) a partir da transcrição."""

import re

from ai_video_editor.director.schema import Sentence
from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment

# Pausa entre segmentos acima da qual a frase termina mesmo sem pontuação.
DEFAULT_MAX_PAUSE_SEC = 1.0

# Pontuação final, opcionalmente seguida de aspas ou parênteses de fechamento.
_SENTENCE_END = re.compile(r"[.?!…][\"'”’)\]]*$")


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


def build_sentences(transcript: Transcript, max_pause_sec: float = DEFAULT_MAX_PAUSE_SEC) -> list[Sentence]:
    """Junta segmentos consecutivos até pontuação final (. ? !) ou pausa maior que `max_pause_sec`.

    O Whisper às vezes quebra segmentos no meio da frase; juntar evita que o Diretor
    decida sobre meia frase. Segmentos sem texto são ignorados.
    """
    sentences: list[Sentence] = []
    group: list[TranscriptSegment] = []
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        if group and segment.start_sec - group[-1].end_sec > max_pause_sec:
            sentences.append(_to_sentence(len(sentences), group))
            group = []
        group.append(segment)
        if _ends_sentence(segment.text):
            sentences.append(_to_sentence(len(sentences), group))
            group = []
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
