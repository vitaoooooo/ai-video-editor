"""Remoção de frases inteiras da lista de trechos mantidos.

Os tempos da transcrição não são precisos o bastante para cortar (o Whisper absorve
pausas dentro das palavras), então cada limite de corte é levado para dentro do
silêncio detectado mais próximo. Sem silêncio por perto, usa-se o limite da frase,
protegido pelo padding das frases mantidas vizinhas.

Pedaço de palavra: se o corte ajustado deixa no vídeo uma sobra da frase removida menor
que `MIN_FRAGMENT_SEC` (de um lado do corte), o corte é estendido até o limite da frase,
desde que esse limite não esteja dentro de uma frase mantida. Se estiver, o corte inteiro
é desfeito: a frase fica no vídeo, sem meia palavra solta.
"""

from typing import Optional

from ai_video_editor.audio.schema import SilenceInterval
from ai_video_editor.planning.schema import SentenceCutResult, SilenceCutParams, TimeRange

# Distância máxima entre o limite da frase e um silêncio para ajustar o corte a ele.
DEFAULT_SNAP_WINDOW_SEC = 0.5
# Sobra da frase removida abaixo da qual o resto é tratado como pedaço de palavra.
MIN_FRAGMENT_SEC = 0.25
_EPS = 0.001


def _nearest_silence(
    t: float, silences: list[SilenceInterval], window: float
) -> Optional[SilenceInterval]:
    best: Optional[SilenceInterval] = None
    best_dist = window + _EPS
    for s in silences:
        dist = 0.0 if s.start_sec <= t <= s.end_sec else min(abs(t - s.start_sec), abs(t - s.end_sec))
        if dist < best_dist:
            best, best_dist = s, dist
    return best


def snap_cut(
    cut: TimeRange,
    silences: list[SilenceInterval],
    params: SilenceCutParams,
    prev_kept_end: Optional[float] = None,
    next_kept_start: Optional[float] = None,
    window: float = DEFAULT_SNAP_WINDOW_SEC,
) -> Optional[TimeRange]:
    """Ajusta os limites de um corte de frase(s) para dentro dos silêncios.

    - Início: no silêncio mais próximo (até `window`), a `padding_after_sec` do fim da fala
      anterior. Sem silêncio: o início da frase, mas nunca antes de
      `prev_kept_end + padding_after_sec`.
    - Fim: simétrico, com `padding_before_sec` antes da fala seguinte.
    - Em qualquer caso o corte não invade as frases mantidas vizinhas.

    Retorna None se o corte ficar vazio.
    """
    silence = _nearest_silence(cut.start_sec, silences, window)
    if silence is not None:
        start = min(silence.start_sec + params.padding_after_sec, silence.end_sec)
    else:
        start = cut.start_sec
        if prev_kept_end is not None:
            start = max(start, prev_kept_end + params.padding_after_sec)

    silence = _nearest_silence(cut.end_sec, silences, window)
    if silence is not None:
        end = max(silence.end_sec - params.padding_before_sec, silence.start_sec)
    else:
        end = cut.end_sec
        if next_kept_start is not None:
            end = min(end, next_kept_start - params.padding_before_sec)

    if prev_kept_end is not None:
        start = max(start, prev_kept_end)
    if next_kept_start is not None:
        end = min(end, next_kept_start)
    if end - start <= _EPS:
        return None
    return TimeRange(start_sec=round(start, 3), end_sec=round(end, 3))


def subtract_ranges(
    keep: list[TimeRange], cuts: list[TimeRange], min_keep_sec: float = 0.0
) -> list[TimeRange]:
    """Remove os intervalos `cuts` de `keep`; sobras menores que `min_keep_sec` são descartadas."""
    result: list[TimeRange] = []
    for r in keep:
        pieces = [(r.start_sec, r.end_sec)]
        for c in cuts:
            next_pieces = []
            for a, b in pieces:
                if c.end_sec <= a or c.start_sec >= b:
                    next_pieces.append((a, b))
                    continue
                if c.start_sec > a:
                    next_pieces.append((a, c.start_sec))
                if c.end_sec < b:
                    next_pieces.append((c.end_sec, b))
            pieces = next_pieces
        result += [
            TimeRange(start_sec=round(a, 3), end_sec=round(b, 3))
            for a, b in pieces
            if b - a > max(_EPS, min_keep_sec)
        ]
    return sorted(result, key=lambda r: r.start_sec)


def kept_overlap(start: float, end: float, keep: list[TimeRange]) -> float:
    """Quanto de [start, end] está dentro dos trechos mantidos."""
    return sum(max(0.0, min(end, r.end_sec) - max(start, r.start_sec)) for r in keep)


def _avoid_fragments(
    cut: TimeRange,
    snapped: TimeRange,
    keep: list[TimeRange],
    prev_kept_end: Optional[float],
    next_kept_start: Optional[float],
    min_fragment_sec: float,
) -> SentenceCutResult:
    """Estende o corte até o limite da frase quando a sobra seria um pedaço de palavra.

    A sobra de cada lado é a parte da frase, entre o limite dela e o corte ajustado, que
    continua nos trechos mantidos. Se o limite cai dentro de uma frase mantida, o corte
    inteiro é desfeito (outcome "undone").
    """
    start, end = snapped.start_sec, snapped.end_sec
    extended = False
    head = kept_overlap(cut.start_sec, start, keep)
    if _EPS < head < min_fragment_sec:
        if prev_kept_end is not None and cut.start_sec < prev_kept_end:
            return SentenceCutResult(requested=cut, outcome="undone")
        start, extended = cut.start_sec, True
    tail = kept_overlap(end, cut.end_sec, keep)
    if _EPS < tail < min_fragment_sec:
        if next_kept_start is not None and cut.end_sec > next_kept_start:
            return SentenceCutResult(requested=cut, outcome="undone")
        end, extended = cut.end_sec, True
    return SentenceCutResult(
        requested=cut,
        applied=TimeRange(start_sec=round(start, 3), end_sec=round(end, 3)),
        outcome="extended" if extended else "applied",
    )


def apply_sentence_cuts(
    keep: list[TimeRange],
    cuts: list[TimeRange],
    kept_sentences: list[TimeRange],
    silences: list[SilenceInterval],
    params: SilenceCutParams,
    window: float = DEFAULT_SNAP_WINDOW_SEC,
    min_fragment_sec: float = MIN_FRAGMENT_SEC,
) -> tuple[list[TimeRange], list[SentenceCutResult]]:
    """Ajusta cada corte aos silêncios e o subtrai dos trechos mantidos.

    Args:
        keep: Trechos mantidos pelo corte de silêncio.
        cuts: Intervalos (da transcrição) das frases removidas; frases removidas
            consecutivas devem vir já unidas num só intervalo.
        kept_sentences: Intervalos das frases mantidas, que nunca são invadidos.
        min_fragment_sec: Sobra da frase abaixo da qual o corte é estendido ou desfeito.

    Returns:
        (trechos mantidos atualizados, destino de cada corte pedido em ordem cronológica).
    """
    kept_sorted = sorted(kept_sentences, key=lambda r: r.start_sec)
    results: list[SentenceCutResult] = []
    for cut in sorted(cuts, key=lambda r: r.start_sec):
        # Vizinhas pela ordem de início: tolera frases do Whisper que se sobrepõem um pouco.
        prev_end = max((k.end_sec for k in kept_sorted if k.start_sec < cut.start_sec), default=None)
        next_start = min((k.start_sec for k in kept_sorted if k.start_sec >= cut.start_sec), default=None)
        snapped = snap_cut(cut, silences, params, prev_end, next_start, window)
        if snapped is None:
            results.append(SentenceCutResult(requested=cut, outcome="protected"))
            continue
        results.append(_avoid_fragments(cut, snapped, keep, prev_end, next_start, min_fragment_sec))
    applied = [r.applied for r in results if r.applied is not None]
    return subtract_ranges(keep, applied, params.min_keep_sec), results
