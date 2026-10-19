"""Testes unitários para os schemas da camada de inteligência e IA multimodal."""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    CompleteAIReport,
    SceneTransitionHypothesis,
    VisualAnalysisResponse,
    VisualObservation,
)


def test_visual_observation_valid():
    """Valida a criação de uma observação visual factual com timestamps."""
    obs = VisualObservation(
        frame_index=0,
        target_timestamp_sec=2.5,
        observable_content="Veículo vermelho transitando em rodovia pavimentada sob luz solar direta.",
        inferences="Possível viagem rodoviária diurna.",
        visual_quality_notes="Foco nítido, sem desfoque de movimento.",
    )
    assert obs.frame_index == 0
    assert obs.target_timestamp_sec == 2.5
    assert "Veículo vermelho" in obs.observable_content


def test_visual_observation_negative_timestamp():
    """Timestamp negativo deve ser rejeitado."""
    with pytest.raises(ValidationError):
        VisualObservation(
            frame_index=0,
            target_timestamp_sec=-1.0,
            observable_content="Conteúdo inválido",
        )


def test_scene_transition_defaults():
    """Hipótese de transição deve possuir is_confirmed_exact=False por padrão."""
    tr = SceneTransitionHypothesis(
        from_timestamp_sec=1.0,
        to_timestamp_sec=3.0,
        evidence_description="Mudança de plano aberto para plano detalhe.",
    )
    assert tr.transition_type == "cut"
    assert tr.is_confirmed_exact is False


def test_ai_analysis_config_limits():
    """Valida os tetos e limites de custo da configuração de IA."""
    cfg = AIAnalysisConfig()
    assert cfg.provider == "gemini"
    assert cfg.model_id == "gemini-3.8-flash"
    assert cfg.max_images_per_request == 15
    assert cfg.max_image_dimension == 1024
    assert cfg.image_quality == 80

    # max_images_per_request zero ou negativo deve falhar
    with pytest.raises(ValidationError):
        AIAnalysisConfig(max_images_per_request=0)

    # image_quality fora do intervalo 20-100 deve falhar
    with pytest.raises(ValidationError):
        AIAnalysisConfig(image_quality=15)


def test_visual_analysis_response_json_roundtrip():
    """Valida a serialização e desserialização completa de VisualAnalysisResponse."""
    resp = VisualAnalysisResponse(
        provider_used="gemini",
        model_used="gemini-3.8-flash",
        summary="Apresentação institucional em ambiente de escritório corporativo.",
        chronological_timeline=[
            VisualObservation(
                frame_index=0,
                target_timestamp_sec=0.0,
                observable_content="Pessoa em pé discursando em frente a uma tela de projeção.",
            )
        ],
        scene_transitions=[
            SceneTransitionHypothesis(
                from_timestamp_sec=0.0,
                to_timestamp_sec=2.0,
                evidence_description="Alteração do ângulo de câmera.",
                is_confirmed_exact=False,
            )
        ],
        limitations_and_blindspots=["Amostragem discreta de frames."],
    )

    json_str = resp.model_dump_json(indent=2)
    reconstructed = VisualAnalysisResponse.model_validate_json(json_str)

    assert reconstructed.provider_used == "gemini"
    assert len(reconstructed.chronological_timeline) == 1
    assert len(reconstructed.scene_transitions) == 1
