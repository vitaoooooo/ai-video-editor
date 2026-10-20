"""Testes do ajuste dos cortes de frases para dentro dos silêncios detectados."""

from ai_video_editor.audio.schema import SilenceInterval
from ai_video_editor.planning import SilenceCutParams, TimeRange, apply_sentence_cuts, snap_cut, subtract_ranges

PARAMS = SilenceCutParams()  # padding 0.15 antes / 0.20 depois


def tr(a: float, b: float) -> TimeRange:
    return TimeRange(start_sec=a, end_sec=b)


def sil(a: float, b: float) -> SilenceInterval:
    return SilenceInterval.from_bounds(a, b)


def bounds(ranges: list[TimeRange]) -> list[tuple[float, float]]:
    return [(r.start_sec, r.end_sec) for r in ranges]


def test_snap_moves_both_limits_into_nearby_silences():
    # Whisper diz que a frase vai de 10.3 a 14.6; os silêncios reais são 9.5–10.0 e 14.9–15.6
    silences = [sil(9.5, 10.0), sil(14.9, 15.6)]
    cut = snap_cut(tr(10.3, 14.6), silences, PARAMS, prev_kept_end=9.4, next_kept_start=15.7)
    assert bounds([cut]) == [(9.7, 15.45)]  # 9.5 + 0.20 e 15.6 - 0.15


def test_snap_ignores_silences_farther_than_window():
    silences = [sil(8.0, 9.0), sil(15.2, 16.0)]  # a 1.3s e 0.6s dos limites
    cut = snap_cut(tr(10.3, 14.6), silences, PARAMS)
    assert bounds([cut]) == [(10.3, 14.6)]


def test_snap_without_silence_keeps_padding_from_kept_neighbors():
    # Frases coladas: o corte não come o padding das frases mantidas
    cut = snap_cut(tr(10.0, 14.0), [], PARAMS, prev_kept_end=10.0, next_kept_start=14.0)
    assert bounds([cut]) == [(10.2, 13.85)]


def test_snap_never_enters_kept_sentences():
    # O silêncio mais próximo do fim está dentro da frase mantida seguinte (tempos do Whisper)
    silences = [sil(14.3, 14.9)]
    cut = snap_cut(tr(10.0, 14.0), silences, PARAMS, prev_kept_end=9.0, next_kept_start=14.2)
    assert cut.end_sec == 14.2


def test_snap_returns_none_when_cut_vanishes():
    assert snap_cut(tr(10.0, 10.2), [], PARAMS, prev_kept_end=10.0, next_kept_start=10.2) is None


def test_subtract_ranges_splits_and_drops_tiny_leftovers():
    keep = [tr(0, 10), tr(12, 20)]
    result = subtract_ranges(keep, [tr(3, 5), tr(9.95, 12.5)], min_keep_sec=0.1)
    assert bounds(result) == [(0, 3), (5, 9.95), (12.5, 20)]
    assert bounds(subtract_ranges([tr(0, 3.05)], [tr(0.0, 3.0)], min_keep_sec=0.1)) == []


def test_apply_sentence_cuts_removes_sentence_inside_silences():
    # Fala: 0–3, 4.5–7.5, 9–12; silêncios entre elas. Remove a frase do meio.
    silences = [sil(3.0, 4.5), sil(7.5, 9.0)]
    keep = [tr(0, 3.2), tr(4.35, 7.7), tr(8.85, 12)]  # saída do corte de silêncio
    kept_sentences = [tr(0.1, 2.9), tr(9.1, 11.9)]
    keep, results = apply_sentence_cuts(keep, [tr(4.6, 7.4)], kept_sentences, silences, PARAMS)
    assert bounds([r.applied for r in results]) == [(3.2, 8.85)]
    assert results[0].outcome == "applied"
    assert bounds(keep) == [(0, 3.2), (8.85, 12)]
