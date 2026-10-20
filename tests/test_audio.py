"""Testes do módulo de áudio com áudio sintético gerado por FFmpeg."""

import json
import subprocess
import wave
from pathlib import Path

import pytest

from ai_video_editor.audio import (
    AudioAnalysisResult,
    MediaNotFoundError,
    NoAudioStreamError,
    SilenceDetectionError,
    SilenceInterval,
    compute_speech,
    detect_silences,
    extract_audio,
    parse_silencedetect,
)
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path

TOL = 0.1


def make_audio(path: Path, segments: list[tuple[str, float]]) -> Path:
    """Gera WAV concatenando segmentos ('tone' | 'silence', duração)."""
    inputs: list[str] = []
    for kind, dur in segments:
        src = f"sine=frequency=440:duration={dur}" if kind == "tone" else (
            f"anullsrc=r=44100:cl=mono:d={dur}"
        )
        inputs += ["-f", "lavfi", "-i", src]
    n = len(segments)
    args = [str(get_ffmpeg_path()), "-y", *inputs]
    if n > 1:
        labels = "".join(f"[{i}:a]" for i in range(n))
        args += ["-filter_complex", f"{labels}concat=n={n}:v=0:a=1[out]", "-map", "[out]"]
    args += ["-ar", "44100", "-ac", "1", str(path)]
    subprocess.run(args, capture_output=True, check=True)
    return path


@pytest.mark.slow
def test_silence_in_the_middle(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("tone", 2.0), ("silence", 1.5), ("tone", 2.0)])
    r = detect_silences(f)
    assert len(r.silences) == 1
    s = r.silences[0]
    assert s.start_sec == pytest.approx(2.0, abs=TOL)
    assert s.end_sec == pytest.approx(3.5, abs=TOL)
    assert s.duration == pytest.approx(1.5, abs=TOL)
    assert len(r.speech) == 2
    assert r.speech[0].start_sec == 0.0
    assert r.speech[0].end_sec == pytest.approx(2.0, abs=TOL)
    assert r.speech[1].end_sec == pytest.approx(r.duration_sec, abs=0.001)
    assert r.params.threshold_db == -35.0
    assert r.params.min_silence_sec == 0.5


@pytest.mark.slow
def test_silence_at_start(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("silence", 1.5), ("tone", 2.0)])
    r = detect_silences(f)
    assert len(r.silences) == 1
    assert r.silences[0].start_sec == 0.0
    assert r.silences[0].end_sec == pytest.approx(1.5, abs=TOL)
    assert len(r.speech) == 1
    assert r.speech[0].start_sec == pytest.approx(1.5, abs=TOL)


@pytest.mark.slow
def test_silence_until_end_without_silence_end(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("tone", 2.0), ("silence", 1.5)])
    r = detect_silences(f)
    assert len(r.silences) == 1
    assert r.silences[0].start_sec == pytest.approx(2.0, abs=TOL)
    assert r.silences[0].end_sec == pytest.approx(r.duration_sec, abs=0.001)
    assert len(r.speech) == 1
    assert r.speech[0].end_sec == pytest.approx(2.0, abs=TOL)


@pytest.mark.slow
def test_no_silence(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("tone", 3.0)])
    r = detect_silences(f)
    assert r.silences == []
    assert len(r.speech) == 1
    assert r.speech[0].start_sec == 0.0
    assert r.speech[0].end_sec == pytest.approx(r.duration_sec, abs=0.001)


def test_all_silence(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("silence", 3.0)])
    r = detect_silences(f)
    assert len(r.silences) == 1
    assert r.silences[0].start_sec == 0.0
    assert r.silences[0].end_sec == pytest.approx(r.duration_sec, abs=0.001)
    assert r.speech == []


@pytest.mark.slow
def test_min_silence_filters_short_gaps(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("tone", 1.0), ("silence", 0.3), ("tone", 1.0)])
    assert detect_silences(f, min_silence_sec=0.5).silences == []
    assert len(detect_silences(f, min_silence_sec=0.2).silences) == 1


def test_invalid_min_silence(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("tone", 1.0)])
    with pytest.raises(SilenceDetectionError):
        detect_silences(f, min_silence_sec=0)


def test_missing_file(tmp_path):
    with pytest.raises(MediaNotFoundError):
        detect_silences(tmp_path / "nope.wav")


@pytest.mark.slow
def test_video_without_audio_raises(synthetic_video_without_audio, tmp_path):
    with pytest.raises(NoAudioStreamError):
        extract_audio(synthetic_video_without_audio, tmp_path / "out.wav")
    with pytest.raises(NoAudioStreamError):
        detect_silences(synthetic_video_without_audio)


def test_extract_audio_wav_mono(synthetic_video_with_audio, tmp_path):
    out = extract_audio(synthetic_video_with_audio, tmp_path / "sub" / "out.wav")
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getnframes() / 16000 == pytest.approx(3.0, abs=0.2)


@pytest.mark.slow
def test_extract_audio_custom_sample_rate(synthetic_video_with_audio, tmp_path):
    out = extract_audio(synthetic_video_with_audio, tmp_path / "o.wav", sample_rate=8000)
    with wave.open(str(out)) as w:
        assert w.getframerate() == 8000


@pytest.mark.slow
def test_detect_on_video_file(tmp_path):
    from tests.conftest import create_synthetic_video

    v = create_synthetic_video(tmp_path / "v.mp4", duration_sec=2.0)
    r = detect_silences(v)
    assert r.silences == []
    assert r.duration_sec == pytest.approx(2.0, abs=TOL)


def test_json_roundtrip(tmp_path):
    f = make_audio(tmp_path / "a.wav", [("tone", 1.0), ("silence", 1.0), ("tone", 1.0)])
    r = detect_silences(f)
    saved = r.save_json(tmp_path / "out" / "r.json")
    assert json.loads(saved.read_text(encoding="utf-8"))["silences"]
    assert AudioAnalysisResult.load_json(saved) == r


def test_parse_silencedetect_edge_cases():
    log = (
        "[silencedetect @ x] silence_start: -0.00002\n"
        "[silencedetect @ x] silence_end: 1.5 | silence_duration: 1.5\n"
        "[silencedetect @ x] silence_start: 4\n"
    )
    out = parse_silencedetect(log, 6.0)
    assert [(i.start_sec, i.end_sec) for i in out] == [(0.0, 1.5), (4.0, 6.0)]


def test_compute_speech_complement():
    sil = [SilenceInterval.from_bounds(0, 1), SilenceInterval.from_bounds(2, 3)]
    sp = compute_speech(sil, 5.0)
    assert [(i.start_sec, i.end_sec) for i in sp] == [(1.0, 2.0), (3.0, 5.0)]
