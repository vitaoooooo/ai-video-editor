"""Normalização da saída crua do faster-whisper (segmentos/palavras fora de ordem) antes do Transcript."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_video_editor.transcription.normalize import SegmentAdjustments, adjustments_note, normalize_segments
from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig


def _w(word, start, end, prob=0.9):
    return SimpleNamespace(word=word, start=start, end=end, probability=prob)


def _seg(start, end, text, words=None):
    return SimpleNamespace(start=start, end=end, text=text, words=words or [])


def _transcript(segments) -> Transcript:
    return Transcript(source_path=Path("v.mp4"), language="pt", duration_sec=60.0, model_used="x", segments=segments)


def test_ordered_input_is_unchanged():
    raw = [
        _seg(0.0, 1.5, " Olá mundo ", [_w(" Olá", 0.0, 0.5), _w(" mundo", 0.6, 1.5)]),
        _seg(1.4, 3.0, " Tchau", [_w(" Tchau", 2.0, 3.0, 0.5)]),  # sobreposição em ordem: não é tocada
        _seg(3.0, 3.0, " a", [_w(" a", 3.0, 3.0)]),  # sem duração: só alargada, como antes
    ]
    segments, adjustments = normalize_segments(raw)
    assert not adjustments.changed and adjustments.summary() == ""
    assert [(s.id, s.start_sec, s.end_sec, s.text) for s in segments] == [
        (0, 0.0, 1.5, "Olá mundo"), (1, 1.4, 3.0, "Tchau"), (2, 3.0, 3.001, "a"),
    ]
    assert [(w.text, w.start_sec, w.end_sec, w.probability) for w in segments[0].words] == [
        ("Olá", 0.0, 0.5, 0.9), ("mundo", 0.6, 1.5, 0.9),
    ]
    assert segments[2].words[0].end_sec == pytest.approx(3.001)


def test_segment_starting_before_previous_moves_to_its_end():
    raw = [
        _seg(10.0, 14.0, " Primeira frase.", [_w(" Primeira", 10.0, 11.0), _w(" frase.", 11.0, 14.0)]),
        _seg(9.0, 18.0, " Mais coisa nova.", [_w(" Mais", 9.0, 13.5), _w(" coisa", 14.5, 16.0), _w(" nova.", 16.0, 18.0)]),
    ]
    segments, adjustments = normalize_segments(raw)
    _transcript(segments)  # passa na validação do Transcript
    assert [(s.start_sec, s.end_sec) for s in segments] == [(10.0, 14.0), (14.0, 18.0)]
    # A palavra que ficou inteira antes do novo início sai, e o texto é refeito com as que restaram.
    assert segments[1].text == "coisa nova." and [w.text for w in segments[1].words] == ["coisa", "nova."]
    assert adjustments == SegmentAdjustments(segments_adjusted=1, words_dropped=1)


def test_word_straddling_new_start_is_clipped():
    raw = [
        _seg(2.0, 5.0, " a b"),
        _seg(1.0, 8.0, " c d", [_w(" c", 4.5, 6.0), _w(" d", 6.0, 8.0)]),
    ]
    segments, adjustments = normalize_segments(raw)
    assert segments[1].start_sec == 5.0 and segments[1].words[0].start_sec == 5.0
    assert segments[1].text == "c d"
    assert adjustments == SegmentAdjustments(segments_adjusted=1, words_adjusted=1)


def test_overlapping_repetition_is_dropped():
    raw = [
        _seg(0.0, 4.0, " Bom dia, pessoal!"),
        _seg(5.0, 9.0, " Vamos começar."),
        _seg(4.0, 10.0, " vamos começar"),  # volta no tempo repetindo o texto (pontuação/caixa à parte)
        _seg(10.0, 12.0, " Fim."),
    ]
    segments, adjustments = normalize_segments(raw)
    assert [s.text for s in segments] == ["Bom dia, pessoal!", "Vamos começar.", "Fim."]
    assert [s.id for s in segments] == [0, 1, 2]
    assert adjustments == SegmentAdjustments(segments_dropped_repeated=1)
    _transcript(segments)


def test_segment_that_vanishes_after_adjustment_is_dropped():
    raw = [
        _seg(20.0, 30.0, " Um trecho longo."),
        _seg(19.0, 25.0, " Outra coisa."),  # inteiro dentro do anterior: sem duração depois do ajuste
        _seg(18.0, 32.0, " Só palavras velhas.", [_w(" Só", 18.0, 28.0), _w(" palavras", 28.0, 30.0)]),
        _seg(32.0, 33.0, " Fim."),
    ]
    segments, adjustments = normalize_segments(raw)
    assert [s.text for s in segments] == ["Um trecho longo.", "Fim."]
    assert [s.id for s in segments] == [0, 1]
    assert adjustments.segments_dropped_empty == 2 and adjustments.words_dropped == 2
    _transcript(segments)


def test_words_out_of_order_are_sorted_and_kept_inside_segment():
    raw = [_seg(1.0, 4.0, " um dois três", [_w(" dois", 2.0, 3.0), _w(" um", 1.0, 2.0), _w(" três", 3.0, 4.5)])]
    segments, adjustments = normalize_segments(raw)
    words = segments[0].words
    assert [w.text for w in words] == ["um", "dois", "três"]
    assert words[-1].end_sec == pytest.approx(4.001)  # preso ao fim do segmento (com a tolerância de 1 ms)
    assert adjustments == SegmentAdjustments(words_adjusted=3)
    assert segments[0].text == "um dois três"  # segmento não movido: texto do provedor intacto


def test_ids_are_rebuilt_and_progress_sees_every_raw_segment():
    raw = [_seg(1.0, 2.0, " a"), _seg(0.5, 1.5, " b"), _seg(2.0, 3.0, " c"), _seg(1.5, 3.5, " c")]
    seen = []
    segments, adjustments = normalize_segments(iter(raw), on_segment=lambda r: seen.append(r.text))
    assert [(s.id, s.text) for s in segments] == [(0, "a"), (1, "c")]
    assert seen == [" a", " b", " c", " c"]
    assert adjustments.summary() == "1 repetição descartada, 1 segmento vazio descartado"


def test_summary_and_note():
    adjustments = SegmentAdjustments(segments_adjusted=2, words_dropped=1)
    assert adjustments.summary() == "2 segmentos com início ajustado, 1 palavra descartada"
    assert adjustments_note(adjustments.summary(), "P012") == (
        "Transcrição (P012): saída do Whisper fora de ordem corrigida — "
        "2 segmentos com início ajustado, 1 palavra descartada."
    )


# --- no provedor (modelo falso, sem download) ---


class _FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        return (s for s in self._segments), SimpleNamespace(language="pt")


def test_faster_whisper_normalizes_and_records_adjustments(monkeypatch, tmp_path):
    raw = [
        _seg(0.0, 2.0, " Olá.", [_w(" Olá.", 0.0, 2.0)]),
        _seg(3.0, 5.0, " Tudo bem?", [_w(" Tudo", 3.0, 4.0), _w(" bem?", 4.0, 5.0)]),
        _seg(1.0, 6.0, " Tudo bem? Sim.", [_w(" Tudo", 1.0, 3.0), _w(" bem?", 3.0, 5.0), _w(" Sim.", 5.0, 6.0)]),
    ]
    media = tmp_path / "a.wav"
    media.write_bytes(b"x")
    monkeypatch.setattr(FasterWhisperProvider, "_load_model", lambda self, config: _FakeModel(raw))
    monkeypatch.setattr(FasterWhisperProvider, "_decode_audio", lambda self, path: [0.0] * 16000 * 7)
    provider = FasterWhisperProvider()

    transcript = provider.transcribe(media, TranscriptionConfig(device="cpu"))

    assert [(s.id, s.start_sec, s.text) for s in transcript.segments] == [(0, 0.0, "Olá."), (1, 3.0, "Tudo bem?"), (2, 5.0, "Sim.")]
    assert provider.last_adjustments == SegmentAdjustments(segments_adjusted=1, words_dropped=2)


def test_faster_whisper_without_adjustments_reports_nothing_changed(monkeypatch, tmp_path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"x")
    monkeypatch.setattr(FasterWhisperProvider, "_load_model", lambda self, config: _FakeModel([_seg(0.0, 1.0, " a")]))
    monkeypatch.setattr(FasterWhisperProvider, "_decode_audio", lambda self, path: [0.0] * 16000)
    provider = FasterWhisperProvider()
    provider.transcribe(media, TranscriptionConfig(device="cpu"))
    assert provider.last_adjustments is not None and not provider.last_adjustments.changed
