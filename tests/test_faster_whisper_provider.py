"""Testes do FasterWhisperProvider com o modelo substituído por um falso (sem rede nem download)."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai_video_editor.transcription.exceptions import (
    TranscriptionDependencyError,
    TranscriptionFailedError,
    TranscriptionGpuError,
    TranscriptionMediaError,
)
from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider
from ai_video_editor.transcription.schema import TranscriptionConfig
from tests.conftest import create_synthetic_video, create_synthetic_video_no_audio


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
    return create_synthetic_video(tmp_path / "video.mp4", duration_sec=2.0)


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
    assert result.duration_sec == pytest.approx(2.0, abs=0.1)
    assert result.model_used == "faster-whisper:small (cpu, int8)"
    assert [s.id for s in result.segments] == [0, 1]
    assert result.segments[0].text == "Olá mundo"
    assert [w.text for w in result.segments[0].words] == ["Olá", "mundo"]
    assert result.segments[0].words[1].start_sec == 0.6
    assert result.segments[1].words[0].probability == 0.5
    audio, kwargs = model.calls[0]
    assert isinstance(audio, np.ndarray)
    assert kwargs == {"language": "pt", "word_timestamps": True, "vad_filter": True}


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


def test_audio_is_decoded_to_float32_mono_16k(monkeypatch, media):
    model = FakeModel([])
    _patch_model(monkeypatch, model)
    result = FasterWhisperProvider().transcribe(media, TranscriptionConfig(vad_filter=False))
    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 2 * 16000) < 0.1 * 16000
    assert result.duration_sec == pytest.approx(len(audio) / 16000)
    assert kwargs["vad_filter"] is False


def test_video_without_audio_raises_media_error(monkeypatch, tmp_path):
    video = create_synthetic_video_no_audio(tmp_path / "mudo.mp4", duration_sec=1.0)
    _patch_model(monkeypatch, FakeModel([]))
    with pytest.raises(TranscriptionMediaError):
        FasterWhisperProvider().transcribe(video, TranscriptionConfig())


def test_failure_message_includes_details(monkeypatch, media):
    _patch_model(monkeypatch, FakeModel([], fail=True))
    with pytest.raises(TranscriptionFailedError, match="boom"):
        FasterWhisperProvider().transcribe(media, TranscriptionConfig())


# --- GPU / dispositivo -------------------------------------------------------------------


def _fake_whisper_module(monkeypatch, fail_on=(), calls=None):
    """Instala um faster_whisper falso; `fail_on` lista os devices que levantam erro ao carregar."""
    calls = calls if calls is not None else []

    class FakeWhisperModel:
        def __init__(self, size, device, compute_type):
            calls.append((device, compute_type))
            if device in fail_on:
                raise RuntimeError("Library cublas64_12.dll is not found")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls


def _gpu(monkeypatch, available):
    from ai_video_editor.transcription.providers import faster_whisper as fw

    monkeypatch.setattr(fw, "cuda_available", lambda: available)
    monkeypatch.setattr(fw, "register_nvidia_dll_dirs", lambda: [])


def test_auto_with_gpu_uses_cuda_float16(monkeypatch):
    calls = _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, True)
    provider = FasterWhisperProvider()
    provider._load_model(TranscriptionConfig())
    assert calls == [("cuda", "float16")]
    assert provider.effective_settings(TranscriptionConfig()) == ("cuda", "float16")


def test_auto_without_gpu_uses_cpu_int8(monkeypatch):
    calls = _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, False)
    provider = FasterWhisperProvider()
    provider._load_model(TranscriptionConfig())
    assert calls == [("cpu", "int8")]
    assert provider.effective_settings(TranscriptionConfig()) == ("cpu", "int8")


def test_gpu_load_failure_falls_back_to_cpu_with_warning(monkeypatch):
    calls = _fake_whisper_module(monkeypatch, fail_on=("cuda",))
    _gpu(monkeypatch, True)
    warnings = []
    provider = FasterWhisperProvider(on_warning=warnings.append)
    provider._load_model(TranscriptionConfig())
    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert len(warnings) == 1 and "cublas64_12.dll" in warnings[0]
    assert provider._effective == ("cpu", "int8")


def test_explicit_cuda_without_gpu_raises_clear_error(monkeypatch):
    _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, False)
    with pytest.raises(TranscriptionGpuError, match=r"\.\[gpu\]"):
        FasterWhisperProvider()._load_model(TranscriptionConfig(device="cuda"))


def test_explicit_cuda_load_failure_raises_clear_error(monkeypatch):
    _fake_whisper_module(monkeypatch, fail_on=("cuda",))
    _gpu(monkeypatch, True)
    with pytest.raises(TranscriptionGpuError, match="cublas64_12.dll"):
        FasterWhisperProvider()._load_model(TranscriptionConfig(device="cuda"))


def test_explicit_cpu_ignores_gpu(monkeypatch):
    calls = _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, True)
    FasterWhisperProvider()._load_model(TranscriptionConfig(device="cpu", compute_type="int8_float32"))
    assert calls == [("cpu", "int8_float32")]


def test_fallback_to_cpu_replaces_gpu_only_compute_type(monkeypatch):
    calls = _fake_whisper_module(monkeypatch, fail_on=("cuda",))
    _gpu(monkeypatch, True)
    FasterWhisperProvider(on_warning=lambda m: None)._load_model(TranscriptionConfig(compute_type="float16"))
    assert calls == [("cuda", "float16"), ("cpu", "int8")]


def test_transcript_records_effective_device(monkeypatch, media):
    _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, True)

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([]), SimpleNamespace(language="pt", duration=2.0)

    monkeypatch.setattr(
        "ai_video_editor.transcription.providers.faster_whisper.FasterWhisperProvider._run",
        lambda self, model, audio, config: ([], "pt"),
    )
    result = FasterWhisperProvider().transcribe(media, TranscriptionConfig(model_size="tiny"))
    assert result.model_used == "faster-whisper:tiny (cuda, float16)"


def test_inference_failure_on_gpu_retries_on_cpu(monkeypatch, media):
    calls = _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, True)
    warnings = []
    attempts = []

    def fake_run(self, model, audio, config):
        attempts.append(self._effective)
        if len(attempts) == 1:
            raise RuntimeError("cudnn_ops64_9.dll missing")
        return [], "pt"

    monkeypatch.setattr(FasterWhisperProvider, "_run", fake_run)
    result = FasterWhisperProvider(on_warning=warnings.append).transcribe(media, TranscriptionConfig())
    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert result.model_used == "faster-whisper:small (cpu, int8)"
    assert "cudnn_ops64_9.dll" in warnings[0]


def test_inference_failure_on_explicit_cuda_raises(monkeypatch, media):
    _fake_whisper_module(monkeypatch)
    _gpu(monkeypatch, True)

    def fake_run(self, model, audio, config):
        raise RuntimeError("cudnn missing")

    monkeypatch.setattr(FasterWhisperProvider, "_run", fake_run)
    with pytest.raises(TranscriptionGpuError):
        FasterWhisperProvider().transcribe(media, TranscriptionConfig(device="cuda"))
