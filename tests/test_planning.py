"""Testes unitários do planejador de corte de silêncio (sem FFmpeg)."""

from pathlib import Path

import pytest

from ai_video_editor.audio import AudioAnalysisParams, AudioAnalysisResult, SilenceInterval
from ai_video_editor.audio.silence import compute_speech
from ai_video_editor.planning import (
    NoSpeechDetectedError,
    SilenceCutParams,
    TimeRange,
    build_edit_plan,
    plan_silence_cuts,
    removed_ranges,
)

NO_PADDING = SilenceCutParams(
    padding_before_sec=0.0, padding_after_sec=0.0, min_gap_to_cut_sec=0.0, min_keep_sec=0.0
)


def audio_result(duration: float, silences: list[tuple[float, float]]) -> AudioAnalysisResult:
    intervals = [SilenceInterval.from_bounds(s, e) for s, e in silences]
    return AudioAnalysisResult(
        source_path=Path("video.mp4"),
        duration_sec=duration,
        params=AudioAnalysisParams(threshold_db=-35.0, min_silence_sec=0.5),
        silences=intervals,
        speech=compute_speech(intervals, duration),
    )


def bounds(ranges: list[TimeRange]) -> list[tuple[float, float]]:
    return [(r.start_sec, r.end_sec) for r in ranges]


def test_defaults():
    p = SilenceCutParams()
    assert (p.padding_before_sec, p.padding_after_sec) == (0.15, 0.20)
    assert (p.min_gap_to_cut_sec, p.min_keep_sec) == (0.30, 0.10)


def test_negative_params_rejected():
    with pytest.raises(ValueError):
        SilenceCutParams(padding_before_sec=-0.1)


def test_keeps_speech_without_padding():
    audio = audio_result(10.0, [(2.0, 4.0), (6.0, 7.0)])
    assert bounds(plan_silence_cuts(audio, NO_PADDING)) == [(0.0, 2.0), (4.0, 6.0), (7.0, 10.0)]


def test_padding_expands_each_speech_range():
    audio = audio_result(10.0, [(2.0, 4.0), (6.0, 7.0)])
    keep = plan_silence_cuts(audio, SilenceCutParams())
    assert bounds(keep) == [(0.0, 2.2), (3.85, 6.2), (6.85, 10.0)]


def test_small_pauses_are_merged():
    audio = audio_result(10.0, [(2.0, 2.6), (6.0, 7.0)])
    params = SilenceCutParams(min_gap_to_cut_sec=0.30)
    # 2.0–2.6 → após padding sobra 0.25s de pausa: menor que 0.30, não é cortada
    assert bounds(plan_silence_cuts(audio, params)) == [(0.0, 6.2), (6.85, 10.0)]
    # Com min_gap menor, a mesma pausa é cortada
    params = SilenceCutParams(min_gap_to_cut_sec=0.2)
    assert bounds(plan_silence_cuts(audio, params)) == [(0.0, 2.2), (2.45, 6.2), (6.85, 10.0)]


def test_overlapping_padding_is_merged():
    audio = audio_result(10.0, [(5.0, 5.3)])
    params = SilenceCutParams(padding_before_sec=0.2, padding_after_sec=0.2, min_gap_to_cut_sec=0.0)
    assert bounds(plan_silence_cuts(audio, params)) == [(0.0, 10.0)]


def test_padding_clamped_to_video_bounds():
    # Silêncio no início e no fim: o padding não pode sair de [0, duração]
    audio = audio_result(10.0, [(0.0, 0.1), (9.95, 10.0)])
    keep = plan_silence_cuts(audio, SilenceCutParams(min_gap_to_cut_sec=0.0))
    assert bounds(keep) == [(0.0, 10.0)]


def test_long_silence_at_start_and_end_is_cut():
    audio = audio_result(10.0, [(0.0, 3.0), (8.0, 10.0)])
    assert bounds(plan_silence_cuts(audio, SilenceCutParams())) == [(2.85, 8.2)]


def test_short_edge_gap_is_not_cut():
    # Sobram só 0.25s antes da primeira fala e 0.2s depois da última: ficam no vídeo
    audio = audio_result(10.0, [(0.0, 0.4), (9.6, 10.0)])
    assert bounds(plan_silence_cuts(audio, SilenceCutParams())) == [(0.0, 10.0)]


def test_video_without_silence_keeps_everything():
    audio = audio_result(12.5, [])
    keep = plan_silence_cuts(audio, SilenceCutParams())
    assert bounds(keep) == [(0.0, 12.5)]
    assert removed_ranges(keep, 12.5) == []


def test_all_silence_raises_clear_error():
    audio = audio_result(8.0, [(0.0, 8.0)])
    with pytest.raises(NoSpeechDetectedError, match="Nenhum trecho de fala"):
        plan_silence_cuts(audio, SilenceCutParams())


def test_short_noise_is_discarded_by_min_keep():
    audio = audio_result(10.0, [(0.0, 5.0), (5.05, 10.0)])
    with pytest.raises(NoSpeechDetectedError, match="ruído"):
        plan_silence_cuts(audio, SilenceCutParams(min_keep_sec=0.1))
    keep = plan_silence_cuts(audio, SilenceCutParams(min_keep_sec=0.0))
    assert bounds(keep) == [(4.85, 5.25)]


def test_removed_ranges_is_complement():
    keep = [TimeRange(start_sec=1.0, end_sec=3.0), TimeRange(start_sec=5.0, end_sec=9.0)]
    assert bounds(removed_ranges(keep, 10.0)) == [(0.0, 1.0), (3.0, 5.0), (9.0, 10.0)]


def test_build_edit_plan_uses_source_resolution_and_fps(tmp_path):
    keep = [TimeRange(start_sec=0.0, end_sec=2.0), TimeRange(start_sec=3.0, end_sec=4.5)]
    plan = build_edit_plan(
        Path("in.mp4"), keep, tmp_path / "out.mp4", resolution=(1919, 1080), fps=29.97
    )
    assert plan.target_resolution == (1918, 1080)
    assert plan.target_fps == 30
    assert [(s.start_sec, s.end_sec) for s in plan.segments] == [(0.0, 2.0), (3.0, 4.5)]
    assert all(s.source_path == Path("in.mp4") for s in plan.segments)

    plan = build_edit_plan(Path("in.mp4"), keep, tmp_path / "out.mp4", (1080, 1920), 60.0)
    assert plan.target_resolution == (1080, 1920)
    assert plan.target_fps == 60
