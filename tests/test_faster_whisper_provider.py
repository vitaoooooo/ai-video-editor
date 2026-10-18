"""Testes do FasterWhisperProvider com o modelo substituído por um falso (sem rede nem download)."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_video_editor.transcription.exceptions import (
    TranscriptionDependencyError,
    TranscriptionFailedError,
    TranscriptionMediaError,
)
from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider
from ai_video_editor.transcription.schema import TranscriptionConfig


def _w(word, start, end, prob=0.9):
    return SimpleNamespace(word=word, start=start, end=end, probability=prob)


class FakeModel:
    """Imita WhisperModel.transcribe devolvendo (gerador de segmentos, info)."""

    def __init__(self, segments, fail=False):
        self._segments = segments
        self._fail = fail
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self._fail:
            raise RuntimeError("boom")
        info = SimpleNamespace(language="pt", duration=7.5)
        return (s for s in self._segments), info


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"fake")
    return path


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(FasterWhisperProvider, "_load_model", lambda self, config: model)


def test_converts_faster_whisper_output_to_schema(monkeypatch, media):
    model = FakeModel([
        SimpleNamespace(start=0.0, end=1.5, text=" Olá mundo ", words=[_w(" Olá", 0.0, 0.5), _w(" mundo", 0.6, 1.5)]),
        SimpleNamespace(start=2.0, end=3.0, text=" Tchau", words=[_w(" Tchau", 2.0, 3.0, 0.5)]),
    ])
    _patch_model(monkeypatch, model)

    result = FasterWhisperProvider().transcribe(media, TranscriptionConfig(language="pt"))

    assert result.source_path == media
    assert result.language == "pt"
    assert result.duration_sec == 7.5
    assert result.model_used == "faster-whisper:small"
    assert [s.id for s in result.segments] == [0, 1]
    assert result.segments[0].text == "Olá mundo"
    assert [w.text for w in result.segments[0].words] == ["Olá", "mundo"]
    assert result.segments[0].words[1].start_sec == 0.6
    assert result.segments[1].words[0].probability == 0.5
    path, kwargs = model.calls[0]
    assert path == str(media)
    assert kwargs == {"language": "pt", "word_timestamps": True}


def test_auto_language_passes_none(monkeypatch, media):
    model = FakeModel([])
    _patch_model(monkeypatch, model)
    result = FasterWhisperProvider().transcribe(media, TranscriptionConfig())
    assert model.calls[0][1]["language"] is None
    assert result.segments == []


def test_zero_duration_word_is_widened(monkeypatch, media):
    _patch_model(monkeypatch, FakeModel([
        SimpleNamespace(start=1.0, end=1.0, text="a", words=[_w("a", 1.0, 1.0)]),
    ]))
    segment = FasterWhisperProvider().transcribe(media, TranscriptionConfig()).segments[0]
    assert segment.end_sec > segment.start_sec
    assert segment.words[0].end_sec > segment.words[0].start_sec


def test_missing_media_raises(tmp_path):
    with pytest.raises(TranscriptionMediaError):
        FasterWhisperProvider().transcribe(tmp_path / "nao_existe.mp4", TranscriptionConfig())


def test_model_failure_is_wrapped(monkeypatch, media):
    _patch_model(monkeypatch, FakeModel([], fail=True))
    with pytest.raises(TranscriptionFailedError):
        FasterWhisperProvider().transcribe(media, TranscriptionConfig())


def test_missing_dependency_raises_clear_error(monkeypatch, media):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # força ImportError
    with pytest.raises(TranscriptionDependencyError) as exc_info:
        FasterWhisperProvider().transcribe(media, TranscriptionConfig())
    assert "transcription" in str(exc_info.value)


def test_load_model_uses_config(monkeypatch):
    created = {}

    class FakeWhisperModel:
        def __init__(self, size, device, compute_type):
            created.update(size=size, device=device, compute_type=compute_type)

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    FasterWhisperProvider()._load_model(TranscriptionConfig(model_size="tiny", device="cuda", compute_type="float16"))
    assert created == {"size": "tiny", "device": "cuda", "compute_type": "float16"}
