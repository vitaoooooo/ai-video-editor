import re
from pathlib import Path

import pytest

from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.subtitles import (
    SubtitleBlock,
    SubtitleParams,
    build_subtitle_blocks,
    format_srt,
    generate_srt,
    remap_time,
)
from ai_video_editor.subtitles.remap import TimelineMapper
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment, Word

FPS = 30.0


def tr(a, b):
    return TimeRange(start_sec=a, end_sec=b)


def make_transcript(words):
    """words: lista de (texto, início, fim) — todas em um único segmento."""
    ws = [Word(text=t, start_sec=s, end_sec=e) for t, s, e in words]
    seg = TranscriptSegment(
        id=0,
        start_sec=ws[0].start_sec,
        end_sec=ws[-1].end_sec,
        text=" ".join(w.text for w in ws),
        words=ws,
    )
    return Transcript(
        source_path=Path("a.mp4"), language="pt", duration_sec=100.0, model_used="mock", segments=[seg]
    )


def spoken(texts, start=0.0, step=0.4):
    return [(t, start + i * step, start + i * step + step * 0.9) for i, t in enumerate(texts)]


# ---------- remapeamento ----------


def test_remap_before_inside_after_cuts():
    ranges = [tr(1.0, 3.0), tr(5.0, 6.0)]
    assert remap_time(1.0, ranges, FPS) == pytest.approx(0.0)
    assert remap_time(2.0, ranges, FPS) == pytest.approx(1.0)
    assert remap_time(5.0, ranges, FPS) == pytest.approx(2.0)
    assert remap_time(5.5, ranges, FPS) == pytest.approx(2.5)


def test_remap_removed_time_has_no_match():
    ranges = [tr(1.0, 3.0), tr(5.0, 6.0)]
    assert remap_time(0.5, ranges, FPS) is None  # antes
    assert remap_time(4.0, ranges, FPS) is None  # entre cortes
    assert remap_time(3.0, ranges, FPS) is None  # fim é exclusivo
    assert remap_time(7.0, ranges, FPS) is None  # depois


def test_remap_quantization_does_not_drift_over_100_ranges():
    dur = 0.105  # 3.15 frames -> round = 3 frames por trecho
    ranges = [tr(i * 1.0, i * 1.0 + dur) for i in range(100)]
    for i, r in enumerate(ranges):
        mapped = remap_time(r.start_sec, ranges, FPS)
        assert abs(mapped * FPS - i * round(dur * FPS)) <= 1
    assert remap_time(ranges[-1].start_sec, ranges, FPS) == pytest.approx(297 / FPS)


def test_remap_matches_renderer_quantization():
    ranges = [tr(0.0, 0.52), tr(2.0, 2.52)]  # 15.6 frames -> 16 frames cada
    assert remap_time(2.0, ranges, FPS) == pytest.approx(16 / FPS)


# ---------- agrupamento ----------


def test_blocks_respect_char_limit():
    t = make_transcript(spoken(["palavra"] * 30, step=0.3))
    params = SubtitleParams(max_chars_per_line=20, max_lines=2)
    blocks = build_subtitle_blocks(t, [tr(0, 20)], FPS, params)
    assert len(blocks) > 1
    for b in blocks:
        assert len(b.lines) <= 2
        assert all(len(line) <= 20 for line in b.lines)
    assert sum(len(line.split()) for b in blocks for line in b.lines) == 30


def test_blocks_break_on_punctuation():
    t = make_transcript(spoken(["Olá", "pessoal.", "Hoje", "vamos", "falar"]))
    blocks = build_subtitle_blocks(t, [tr(0, 10)], FPS, SubtitleParams(min_duration_sec=0.5))
    assert blocks[0].text == "Olá pessoal."
    assert blocks[1].text == "Hoje vamos falar"


def test_block_never_crosses_cut():
    t = make_transcript(spoken(["um", "dois", "três", "quatro"], step=0.5))
    blocks = build_subtitle_blocks(t, [tr(0.0, 1.0), tr(1.0, 2.0)], FPS)
    assert [b.text for b in blocks] == ["um dois", "três quatro"]


def test_removed_words_are_dropped():
    t = make_transcript(spoken(["a", "b", "c", "d"], step=1.0))  # a:0-.9 b:1-1.9 c:2-2.9 d:3-3.9
    blocks = build_subtitle_blocks(t, [tr(0.0, 1.0), tr(3.0, 4.0)], FPS)
    assert [b.text for b in blocks] == ["a", "d"]
    assert blocks[1].start_sec == pytest.approx(1.0)


def test_word_inflated_over_removed_pause_is_kept():
    # Caso real: "first" marcada em 92,70-94,00s, mas a pausa 93,17-93,64s foi cortada.
    t = make_transcript([("this", 92.0, 92.6), ("first", 92.70, 94.00), ("rectangular", 94.0, 95.0)])
    kept = [tr(90.0, 93.17), tr(93.64, 100.0)]
    blocks = build_subtitle_blocks(t, kept, FPS)
    text = " ".join(b.text for b in blocks)
    assert "first" in text and "rectangular" in text
    first_block = next(b for b in blocks if "first" in b.text)
    assert first_block.end_sec <= TimelineMapper(kept, FPS).range_output_bounds(0)[1] + 1e-9


def test_word_crossing_cut_goes_to_range_with_most_of_it():
    t = make_transcript([("mais", 1.8, 2.4)])  # 0,2s no 1º trecho, 0,4s no 2º
    blocks = build_subtitle_blocks(t, [tr(0.0, 2.0), tr(2.0, 3.0)], FPS)
    assert len(blocks) == 1
    assert blocks[0].start_sec >= 2.0 - 1e-9


def test_word_inside_removed_range_is_dropped():
    t = make_transcript([("a", 0.0, 0.5), ("frase", 2.0, 2.6), ("removida", 2.7, 3.2), ("b", 5.0, 5.5)])
    blocks = build_subtitle_blocks(t, [tr(0.0, 1.0), tr(5.0, 6.0)], FPS)
    assert [b.text for b in blocks] == ["a", "b"]


def test_word_with_tiny_overlap_is_dropped():
    t = make_transcript([("quase", 0.9, 1.9)])  # 0,1s mantido de 1,0s: 10% < 30% e < 0,15s
    assert build_subtitle_blocks(t, [tr(0.0, 1.0)], FPS) == []


def test_max_duration_splits_block():
    t = make_transcript(spoken(["x"] * 20, step=0.5))
    blocks = build_subtitle_blocks(t, [tr(0, 11)], FPS, SubtitleParams(max_duration_sec=3.0))
    assert len(blocks) > 1
    assert all(b.end_sec - b.start_sec <= 3.0 + 1e-6 for b in blocks)


def test_min_duration_extended_without_overlap():
    t = make_transcript([("oi", 0.0, 0.2), ("tchau", 5.0, 5.2)])
    blocks = build_subtitle_blocks(t, [tr(0, 6)], FPS)
    assert blocks[0].end_sec - blocks[0].start_sec == pytest.approx(1.0)
    assert blocks[0].end_sec <= blocks[1].start_sec


def test_no_overlap_between_blocks():
    t = make_transcript(spoken(["fala", "rápida"] * 40, step=0.15))
    blocks = build_subtitle_blocks(t, [tr(0, 3), tr(4, 9)], FPS)
    assert len(blocks) > 2
    for prev, nxt in zip(blocks, blocks[1:]):
        assert prev.end_sec <= nxt.start_sec + 1e-9


# ---------- SRT ----------


def test_srt_format(tmp_path):
    words = spoken(["Olá", "mundo."], start=1.0) + spoken(["Segunda", "frase"], start=10.0)
    t = make_transcript(words)
    out = generate_srt(t, [tr(1.0, 3.0), tr(10.0, 12.0)], FPS, tmp_path / "x" / "out.srt")
    entries = out.read_text(encoding="utf-8").strip().split("\n\n")
    assert len(entries) == 2
    stamp = re.compile(r"^(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})$")

    def ms(h, m, s, f):
        return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(f)

    spans = []
    for n, entry in enumerate(entries, start=1):
        lines = entry.split("\n")
        assert lines[0] == str(n)
        m = stamp.match(lines[1])
        assert m
        spans.append((ms(*m.groups()[:4]), ms(*m.groups()[4:])))
        assert lines[2:]
    assert spans[0][0] == 0
    assert spans[1][0] >= spans[0][1]
    assert spans[1][0] == 2000  # 2º trecho começa em 2.0s no vídeo editado
    assert "Olá mundo." in entries[0]


def test_srt_timestamp_hours():
    s = format_srt([SubtitleBlock(index=1, start_sec=3661.5, end_sec=3662.25, lines=["x"])])
    assert "01:01:01,500 --> 01:01:02,250" in s
