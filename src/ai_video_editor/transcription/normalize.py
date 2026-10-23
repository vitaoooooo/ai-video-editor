"""Normalização da saída crua de um provedor de transcrição antes de montar o `Transcript`.

O faster-whisper às vezes devolve um segmento que começa antes do anterior (repetição/alucinação,
emenda das janelas de 30 s ou dos trechos do VAD) e palavras fora de ordem. O `Transcript` recusa
isso, com razão; esta função conserta a saída do provedor sem afrouxar o schema.

Regra (os segmentos são lidos na ordem em que o provedor os emitiu):
- Segmento que começa antes do início do anterior mantido "voltou no tempo": é uma nova decodificação
  de áudio que o anterior já cobre. Se repete o texto do anterior e se sobrepõe a ele, é descartado;
  senão, o início vai para o FIM do anterior (fica só a parte nova, sem sobreposição). Ajustar para o
  início do anterior deixaria dois segmentos sobre o mesmo áudio, e a fala sairia duplicada.
  Não reordenamos: um trecho "anterior" emitido depois quase sempre é essa repetição, e ordená-lo
  esconderia a duplicata sem corrigi-la.
- Segmento ajustado que fica sem duração, ou segmento que tinha palavras e perde todas, é descartado;
  se perdeu algumas, o texto é refeito com as que restaram.
- Palavras de cada segmento ficam em ordem de início (ordenação estável) e dentro do segmento
  (a palavra que começa antes tem o início ajustado; a que fica sem duração é descartada).
- IDs refeitos 0..n-1.
Entrada já em ordem (com palavras em ordem e dentro do segmento) sai idêntica à conversão direta.
"""

import re
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel

from ai_video_editor.transcription.schema import TranscriptSegment, Word

# Menor duração aceita para uma palavra ou segmento; o Whisper às vezes devolve start == end.
MIN_DURATION_SEC = 0.001

_PUNCTUATION = re.compile(r"[^\w\s]")


class SegmentAdjustments(BaseModel):
    """O que a normalização mudou na saída do provedor (tudo zero = nada mudou)."""

    segments_adjusted: int = 0
    segments_dropped_repeated: int = 0
    segments_dropped_empty: int = 0
    words_adjusted: int = 0
    words_dropped: int = 0

    @property
    def changed(self) -> bool:
        return any(self.model_dump().values())

    def summary(self) -> str:
        """Ex.: "2 segmentos com início ajustado, 1 repetição descartada, 3 palavras descartadas"."""
        parts = [
            (self.segments_adjusted, "segmento com início ajustado", "segmentos com início ajustado"),
            (self.segments_dropped_repeated, "repetição descartada", "repetições descartadas"),
            (self.segments_dropped_empty, "segmento vazio descartado", "segmentos vazios descartados"),
            (self.words_adjusted, "palavra ajustada", "palavras ajustadas"),
            (self.words_dropped, "palavra descartada", "palavras descartadas"),
        ]
        return ", ".join(f"{n} {one if n == 1 else many}" for n, one, many in parts if n)


def _same_text(a: str, b: str) -> bool:
    def norm(text: str) -> str:
        return " ".join(_PUNCTUATION.sub(" ", text.casefold()).split())

    return bool(norm(a)) and norm(a) == norm(b)


def _fit_words(
    raw_words: list[Any], start: float, end: float, adjustments: SegmentAdjustments
) -> tuple[list[Word], list[str]]:
    """Palavras em ordem e dentro de [start, end]; devolve também o texto cru das que ficaram."""
    ordered = sorted(raw_words, key=lambda w: w.start)
    moved = {id(a) for a, b in zip(ordered, raw_words) if a is not b}
    # Tolerância no fim: a duração mínima de uma palavra no fim do segmento pode passar dele.
    limit = end + MIN_DURATION_SEC
    words: list[Word] = []
    texts: list[str] = []
    for w in ordered:
        w_start, w_end = w.start, max(w.end, w.start + MIN_DURATION_SEC)
        adjusted = id(w) in moved
        if w_start < start:
            w_start, adjusted = start, True
        if w_end > limit:
            w_end, adjusted = limit, True
        if w_end <= w_start:
            adjustments.words_dropped += 1
            continue
        if adjusted:
            adjustments.words_adjusted += 1
        words.append(Word(text=w.word.strip(), start_sec=w_start, end_sec=w_end, probability=w.probability))
        texts.append(w.word)
    return words, texts


def normalize_segments(
    raw_segments: Iterable[Any], on_segment: Optional[Callable[[Any], None]] = None
) -> tuple[list[TranscriptSegment], SegmentAdjustments]:
    """Converte segmentos no formato do faster-whisper (`start`, `end`, `text`, `words` com `word`,
    `start`, `end`, `probability`) em `TranscriptSegment` válidos, aplicando a regra do módulo.

    `raw_segments` pode ser um gerador (consumido uma vez); `on_segment` é chamado com cada segmento
    cru consumido (progresso), inclusive os descartados.
    """
    adjustments = SegmentAdjustments()
    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        if on_segment is not None:
            on_segment(raw)
        start, end = raw.start, max(raw.end, raw.start + MIN_DURATION_SEC)
        text = raw.text.strip()
        raw_words = list(getattr(raw, "words", None) or [])
        previous = segments[-1] if segments else None
        moved = previous is not None and start < previous.start_sec
        if moved:
            if start < previous.end_sec and _same_text(text, previous.text):
                adjustments.segments_dropped_repeated += 1
                continue
            start = previous.end_sec
            if end <= start:
                adjustments.segments_dropped_empty += 1
                continue
        words, texts = _fit_words(raw_words, start, end, adjustments)
        if raw_words and not words:
            adjustments.segments_dropped_empty += 1
            continue
        if len(words) < len(raw_words):
            text = "".join(texts).strip()
        if moved:
            adjustments.segments_adjusted += 1
        segments.append(TranscriptSegment(id=len(segments), start_sec=start, end_sec=end, text=text, words=words))
    return segments, adjustments


def adjustments_note(summary: str, clip: str = "") -> str:
    """Aviso do `edit` para um resumo de `SegmentAdjustments.summary()` (com o clipe, se houver)."""
    where = f" ({clip})" if clip else ""
    return f"Transcrição{where}: saída do Whisper fora de ordem corrigida — {summary}."
