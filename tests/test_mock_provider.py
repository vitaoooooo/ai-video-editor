"""Testes unitários para o MockMultimodalProvider e simulação de erros de API."""

from pathlib import Path
import pytest

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata, VideoStreamInfo
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.providers.mock import MockMultimodalProvider
from ai_video_editor.intelligence.schema import AIAnalysisConfig


@pytest.fixture
def dummy_metadata():
    return VideoMetadata(
        source_path=Path("dummy.mp4"),
        duration_sec=6.0,
        size_bytes=500000,
        video_stream=VideoStreamInfo(codec_name="h264", width=1280, height=720, fps=30.0),
    )


@pytest.fixture
def dummy_frames(tmp_path: Path):
    return [
        ExtractedFrame(
            frame_index=0,
            target_timestamp_sec=0.0,
            image_path=tmp_path / "f0.jpg",
            width=1280,
            height=720,
            size_bytes=10000,
        ),
        ExtractedFrame(
            frame_index=1,
            target_timestamp_sec=2.5,
            image_path=tmp_path / "f1.jpg",
            width=1280,
            height=720,
            size_bytes=10000,
        ),
    ]


def test_mock_provider_success(dummy_metadata, dummy_frames):
    """Verifica geração correta de observações correlacionadas aos frames de entrada."""
    provider = MockMultimodalProvider()
    cfg = AIAnalysisConfig(model_id="gemini-3.8-flash")

    resp = provider.analyze_frames(dummy_metadata, dummy_frames, cfg)

    assert resp.provider_used == "mock"
    assert resp.model_used == "gemini-3.8-flash"
    assert len(resp.chronological_timeline) == 2
    assert resp.chronological_timeline[0].frame_index == 0
    assert resp.chronological_timeline[0].target_timestamp_sec == 0.0
    assert resp.chronological_timeline[1].frame_index == 1
    assert resp.chronological_timeline[1].target_timestamp_sec == 2.5
    assert len(resp.scene_transitions) >= 1
    assert resp.scene_transitions[0].is_confirmed_exact is False


def test_mock_provider_error_handling(dummy_metadata, dummy_frames):
    """Testa tratamento e disparo controlado de exceções de provedor."""
    cfg = AIAnalysisConfig()

    with pytest.raises(AIAuthenticationError):
        MockMultimodalProvider(force_auth_error=True).analyze_frames(dummy_metadata, dummy_frames, cfg)

    with pytest.raises(AIRateLimitError):
        MockMultimodalProvider(force_rate_limit=True).analyze_frames(dummy_metadata, dummy_frames, cfg)

    with pytest.raises(AITimeoutError):
        MockMultimodalProvider(force_timeout=True).analyze_frames(dummy_metadata, dummy_frames, cfg)

    with pytest.raises(AIUnavailableError):
        MockMultimodalProvider(force_unavailable=True).analyze_frames(dummy_metadata, dummy_frames, cfg)

    with pytest.raises(AIResponseValidationError):
        MockMultimodalProvider(force_validation_error=True).analyze_frames(dummy_metadata, dummy_frames, cfg)
