"""Limite de silêncio automático: medição de níveis, regra de escolha e casos de borda."""

import subprocess
from pathlib import Path

import pytest

from ai_video_editor.audio import AudioAnalysisResult, detect_silences
from ai_video_editor.audio.levels import (
    DB_FLOOR,
    choose_threshold,
    measure_levels,
    parse_levels,
    percentile,
)
from ai_video_editor.cli import build_parser
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path

TOL = 0.1


def make_mix(path: Path, segments: list[tuple[str, float]]) -> Path:
    """WAV concatenando 'speech' (tom ~-12 dB), 'noise' (ruído ~-38 dB RMS) e 'digital' (silêncio digital)."""
    sources = {
        "speech": "sine=frequency=440:sample_rate=44100:duration={d},volume=2.0",
        "noise": "anoisesrc=color=white:sample_rate=44100:amplitude=0.0218:duration={d}",
        "digital": "anullsrc=r=44100:cl=mono:d={d}",
    }
    args = [str(get_ffmpeg_path()), "-y"]
    for kind, dur in segments:
        args += ["-f", "lavfi", "-i", sources[kind].format(d=dur)]
    n = len(segments)
    labels = "".join(f"[{i}:a]" for i in range(n))
    args += [
        "-filter_complex", f"{labels}concat=n={n}:v=0:a=1[o]",
        "-map", "[o]", "-ar", "44100", "-ac", "1", str(path),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return path


def _expect(pauses: list[tuple[float, float]], result: AudioAnalysisResult) -> None:
    assert len(result.silences) == len(pauses)
    for got, (start, end) in zip(result.silences, pauses):
        assert got.start_sec == pytest.approx(start, abs=TOL)
        assert got.end_sec == pytest.approx(end, abs=TOL)


# --- unidade: regra de escolha -------------------------------------------------


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([3.0], 90) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0) == 1.0


def test_parse_levels_handles_inf_and_noise_lines():
    out = (
        "frame:0 pts:0 pts_time:0\nlavfi.astats.Overall.Peak_level=-inf\n"
        "lavfi.astats.Overall.Peak_level=-20.5\nx=1\n"
    )
    assert parse_levels(out) == [DB_FLOOR, -20.5]


def test_rule_picks_point_between_floor_and_speech():
    stats = choose_threshold([-40.0] * 50 + [-10.0] * 50)
    # piso -40, fala -10: -40 + 0,35 * 30 = -29,5
    assert (stats.noise_floor_db, stats.speech_level_db) == (-40.0, -10.0)
    assert stats.threshold_db == pytest.approx(-29.5)
    assert stats.reliable and stats.warnings == []


def test_rule_keeps_headroom_below_speech_when_spread_is_small():
    stats = choose_threshold([-40.0] * 50 + [-30.0] * 50)  # 0,35 * 10 = 3,5 dB acima do piso
    assert stats.threshold_db == pytest.approx(-36.5)
    stats = choose_threshold([-40.0] * 50 + [-33.0] * 50)  # -37 passaria de fala - 6 dB = -39
    assert stats.threshold_db == pytest.approx(-39.0)


def test_rule_digital_silence_clamps_to_lower_bound():
    stats = choose_threshold([DB_FLOOR] * 50 + [-20.0] * 50)
    assert stats.noise_floor_db == DB_FLOOR
    assert stats.threshold_db == -60.0  # -90 + 0,35 * 70 = -65,5, limitado a -60
    assert stats.reliable


def test_edge_noise_close_to_speech_warns_and_stays_below_floor():
    stats = choose_threshold([-30.0] * 50 + [-26.0] * 50)
    assert not stats.reliable
    assert stats.threshold_db < stats.noise_floor_db
    assert "pouco confiável" in stats.warnings[0]


def test_edge_no_pauses_constant_level_warns():
    stats = choose_threshold([-20.0] * 100)
    assert not stats.reliable
    assert stats.threshold_db == pytest.approx(-23.0)


def test_edge_almost_all_silence_treats_everything_as_silence():
    stats = choose_threshold([DB_FLOOR] * 98 + [-30.0] * 2)
    assert not stats.reliable
    assert stats.threshold_db > stats.speech_level_db
    assert "quase todo em silêncio" in stats.warnings[0]


def test_edge_no_levels_is_unreliable():
    stats = choose_threshold([])
    assert not stats.reliable and stats.warnings


def test_cli_threshold_accepts_auto_or_number():
    parser = build_parser()
    assert parser.parse_args(["edit", "v.mp4"]).threshold_db == "auto"
    assert parser.parse_args(["edit", "v.mp4", "--threshold-db", "AUTO"]).threshold_db == "auto"
    assert parser.parse_args(["edit", "v.mp4", "--threshold-db", "-42.5"]).threshold_db == -42.5
    with pytest.raises(SystemExit):
        parser.parse_args(["edit", "v.mp4", "--threshold-db", "alto"])


# --- áudio sintético -----------------------------------------------------------

NOISY_PATTERN = [("speech", 2.0), ("noise", 1.0), ("speech", 2.0), ("noise", 1.5), ("speech", 2.0)]
NOISY_PAUSES = [(2.0, 3.0), (5.0, 6.5)]


@pytest.mark.slow
def test_measure_levels_has_one_value_per_50ms_window(tmp_path):
    f = make_mix(tmp_path / "m.wav", NOISY_PATTERN)
    levels = measure_levels(f)
    assert len(levels) == pytest.approx(8.5 / 0.05, abs=3)
    assert max(levels) == pytest.approx(-12.0, abs=1.0)


@pytest.mark.slow
def test_noisy_pauses_fixed_misses_and_auto_finds_them(tmp_path):
    f = make_mix(tmp_path / "noisy.wav", NOISY_PATTERN)

    fixed = detect_silences(f, threshold_db=-35.0)
    assert fixed.params.threshold_mode == "fixed" and fixed.levels is None
    assert len(fixed.silences) == 0  # picos do ruído (~-33 dB) ficam acima de -35 dB

    auto = detect_silences(f)
    assert auto.params.threshold_mode == "auto"
    _expect(NOISY_PAUSES, auto)
    assert auto.levels is not None and auto.levels.reliable
    assert auto.levels.noise_floor_db < auto.params.threshold_db < auto.levels.speech_level_db
    assert auto.levels.speech_level_db == pytest.approx(-12.0, abs=1.5)


@pytest.mark.slow
def test_digital_silence_still_works_in_auto(tmp_path):
    f = make_mix(tmp_path / "digital.wav", [("speech", 2.0), ("digital", 1.5), ("speech", 2.0)])
    auto = detect_silences(f)
    _expect([(2.0, 3.5)], auto)
    assert auto.levels is not None and auto.levels.reliable
    assert auto.levels.noise_floor_db == DB_FLOOR


@pytest.mark.slow
def test_auto_result_roundtrips_with_levels_in_json(tmp_path):
    f = make_mix(tmp_path / "n.wav", NOISY_PATTERN)
    result = detect_silences(f)
    path = result.save_json(tmp_path / "audio_analysis.json")
    loaded = AudioAnalysisResult.load_json(path)
    assert loaded.params.threshold_mode == "auto"
    assert loaded.levels == result.levels
    assert '"noise_floor_db"' in path.read_text(encoding="utf-8")


@pytest.mark.slow
def test_edge_audio_without_pauses_is_flagged_and_not_cut(tmp_path):
    f = make_mix(tmp_path / "continuous.wav", [("speech", 4.0)])
    r = detect_silences(f)
    assert r.silences == []
    assert r.levels is not None and not r.levels.reliable and r.levels.warnings


@pytest.mark.slow
def test_edge_almost_all_silence(tmp_path):
    f = make_mix(tmp_path / "quiet.wav", [("digital", 6.0), ("speech", 0.1), ("digital", 0.5)])
    r = detect_silences(f)
    assert r.levels is not None and not r.levels.reliable
    assert "quase todo em silêncio" in r.levels.warnings[0]
    assert sum(s.duration for s in r.silences) > 5.5
