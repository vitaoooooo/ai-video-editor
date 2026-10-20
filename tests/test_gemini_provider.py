"""Testes unitários para o GeminiMultimodalProvider sem chamadas de rede."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata, VideoStreamInfo
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.providers.gemini import (
    GeminiMultimodalProvider,
    resolve_gemini_api_key,
)
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.intelligence.schema import AIAnalysisConfig


def test_resolve_gemini_api_key_from_env(monkeypatch):
    """Verifica resolução da chave a partir de variável de ambiente."""
    monkeypatch.setenv("GEMINI_API_KEY", "chave_simulada_teste_123")
    key = resolve_gemini_api_key()
    assert key == "chave_simulada_teste_123"


def test_resolve_gemini_api_key_missing_raises_auth_error(monkeypatch, tmp_path):
    """Chave ausente deve lançar AIAuthenticationError de forma segura."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Garante que não há .env na pasta atual
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        with pytest.raises(AIAuthenticationError):
            resolve_gemini_api_key()


def test_gemini_provider_success_mocked(tmp_path: Path):
    """Testa resposta estruturada do Gemini com mock no client do SDK google-genai."""
    # Cria frame dummy
    img_path = tmp_path / "frame.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")  # cabeçalho JPEG mínimo

    metadata = VideoMetadata(
        source_path=tmp_path / "vid.mp4",
        duration_sec=4.0,
        size_bytes=1000,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )
    frame = ExtractedFrame(
        frame_index=0,
        target_timestamp_sec=0.0,
        image_path=img_path,
        width=640,
        height=360,
        size_bytes=len(img_path.read_bytes()),
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "provider_used": "gemini",
        "model_used": "gemini-3.8-flash",
        "summary": "Cena urbana diurna.",
        "chronological_timeline": [
            {
                "frame_index": 0,
                "target_timestamp_sec": 0.0,
                "observable_content": "Cruzamento de vias com semáforo visível.",
                "inferences": "Horário de pico matutino.",
                "visual_quality_notes": "Boa nitidez.",
            }
        ],
        "scene_transitions": [],
        "limitations_and_blindspots": ["Amostra única."],
    })
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiMultimodalProvider(api_key="chave_fake")
    with patch.object(provider, "_get_client", return_value=mock_client):
        # Patch PIL.Image.open para não tentar decodificar o JPEG dummy incompleto
        with patch("PIL.Image.open", return_value=MagicMock()):
            cfg = AIAnalysisConfig(model_id="gemini-3.8-flash")
            res = provider.analyze_frames(metadata, [frame], cfg)

            http = mock_client.models.generate_content.call_args.kwargs["config"].http_options
            assert http.timeout == int(cfg.timeout_sec * 1000)
            assert res.provider_used == "gemini"
            assert res.model_used == "gemini-3.8-flash"
            assert len(res.chronological_timeline) == 1
            assert res.chronological_timeline[0].frame_index == 0


def test_gemini_provider_error_mapping(tmp_path: Path):
    """Testa mapeamento seguro de exceções do SDK para tipos padronizados."""
    metadata = VideoMetadata(
        source_path=tmp_path / "vid.mp4",
        duration_sec=2.0,
        size_bytes=500,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )
    frame = ExtractedFrame(
        frame_index=0,
        target_timestamp_sec=0.0,
        image_path=tmp_path / "f.jpg",
        width=640,
        height=360,
        size_bytes=10,
    )

    provider = GeminiMultimodalProvider(api_key="chave_fake", retry=RetryPolicy(max_retries=0))
    cfg = AIAnalysisConfig()

    # 1. Simula erro 403 / permissão
    mock_client_auth = MagicMock()
    mock_client_auth.models.generate_content.side_effect = Exception("HTTP 403: API_KEY_INVALID")
    with patch.object(provider, "_get_client", return_value=mock_client_auth):
        with patch("PIL.Image.open", return_value=MagicMock()):
            with pytest.raises(AIAuthenticationError):
                provider.analyze_frames(metadata, [frame], cfg)

    # 2. Simula erro 429 / quota
    mock_client_quota = MagicMock()
    mock_client_quota.models.generate_content.side_effect = Exception("HTTP 429: Resource has been exhausted (rate limit)")
    with patch.object(provider, "_get_client", return_value=mock_client_quota):
        with patch("PIL.Image.open", return_value=MagicMock()):
            with pytest.raises(AIRateLimitError):
                provider.analyze_frames(metadata, [frame], cfg)

    # 3. Simula timeout
    mock_client_timeout = MagicMock()
    mock_client_timeout.models.generate_content.side_effect = Exception("Request timed out after 60s")
    with patch.object(provider, "_get_client", return_value=mock_client_timeout):
        with patch("PIL.Image.open", return_value=MagicMock()):
            with pytest.raises(AITimeoutError):
                provider.analyze_frames(metadata, [frame], cfg)

    # 4. Simula indisponibilidade 503
    mock_client_unavail = MagicMock()
    mock_client_unavail.models.generate_content.side_effect = Exception("HTTP 503: Service Unavailable")
    with patch.object(provider, "_get_client", return_value=mock_client_unavail):
        with patch("PIL.Image.open", return_value=MagicMock()):
            with pytest.raises(AIUnavailableError):
                provider.analyze_frames(metadata, [frame], cfg)


def _gemini_fixture(tmp_path: Path):
    metadata = VideoMetadata(
        source_path=tmp_path / "vid.mp4",
        duration_sec=2.0,
        size_bytes=500,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )
    frame = ExtractedFrame(
        frame_index=0, target_timestamp_sec=0.0, image_path=tmp_path / "f.jpg",
        width=640, height=360, size_bytes=10,
    )
    payload = json.dumps({
        "provider_used": "gemini", "model_used": "x", "summary": "ok",
        "chronological_timeline": [], "scene_transitions": [], "limitations_and_blindspots": [],
    })
    return metadata, frame, payload


def test_gemini_reads_usage_metadata(tmp_path: Path):
    """Tokens são lidos de response.usage_metadata quando disponíveis."""
    metadata, frame, payload = _gemini_fixture(tmp_path)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text=payload,
        usage_metadata=MagicMock(prompt_token_count=1234, candidates_token_count=56),
    )
    provider = GeminiMultimodalProvider(api_key="chave_fake")
    with patch.object(provider, "_get_client", return_value=mock_client):
        with patch("PIL.Image.open", return_value=MagicMock()):
            provider.analyze_frames(metadata, [frame], AIAnalysisConfig())
    assert provider.last_input_tokens == 1234
    assert provider.last_output_tokens == 56


def test_gemini_missing_usage_metadata_gives_none(tmp_path: Path):
    """Sem usage_metadata, tokens ficam None sem quebrar."""
    metadata, frame, payload = _gemini_fixture(tmp_path)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text=payload, usage_metadata=None)
    provider = GeminiMultimodalProvider(api_key="chave_fake")
    with patch.object(provider, "_get_client", return_value=mock_client):
        with patch("PIL.Image.open", return_value=MagicMock()):
            provider.analyze_frames(metadata, [frame], AIAnalysisConfig())
    assert provider.last_input_tokens is None
    assert provider.last_output_tokens is None
