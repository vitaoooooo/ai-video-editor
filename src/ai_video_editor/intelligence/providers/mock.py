"""Provedor simulado (Mock) para testes determinísticos sem dependência de rede ou API keys."""

from typing import Optional

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.provider import MultimodalAIProvider
from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    SceneTransitionHypothesis,
    VisualAnalysisResponse,
    VisualObservation,
)


class MockMultimodalProvider(MultimodalAIProvider):
    """Provedor simulado que gera respostas estruturadas sem chamadas de rede."""

    def __init__(
        self,
        force_auth_error: bool = False,
        force_rate_limit: bool = False,
        force_timeout: bool = False,
        force_unavailable: bool = False,
        force_validation_error: bool = False,
        custom_response: Optional[VisualAnalysisResponse] = None,
    ):
        self.force_auth_error = force_auth_error
        self.force_rate_limit = force_rate_limit
        self.force_timeout = force_timeout
        self.force_unavailable = force_unavailable
        self.force_validation_error = force_validation_error
        self.custom_response = custom_response

    @property
    def provider_name(self) -> str:
        return "mock"

    def analyze_frames(
        self,
        metadata: VideoMetadata,
        frames: list[ExtractedFrame],
        config: AIAnalysisConfig,
    ) -> VisualAnalysisResponse:
        """Retorna resposta estruturada simulada ou dispara erro configurado para teste."""
        if self.force_auth_error:
            raise AIAuthenticationError("mock", details="Chave de API simulada ausente ou rejeitada.")
        if self.force_rate_limit:
            raise AIRateLimitError("mock", details="Cota de requisições por minuto atingida.")
        if self.force_timeout:
            raise AITimeoutError("mock", timeout_sec=config.timeout_sec)
        if self.force_unavailable:
            raise AIUnavailableError("mock", details="Serviço simulado em manutenção.")
        if self.force_validation_error:
            raise AIResponseValidationError("mock", details="Campo obrigatório ausente no payload simulado.")

        if self.custom_response:
            return self.custom_response

        # Gera linha do tempo determinística vinculada estritamente aos frames recebidos
        observations: list[VisualObservation] = []
        for f in frames:
            observations.append(
                VisualObservation(
                    frame_index=f.frame_index,
                    target_timestamp_sec=f.target_timestamp_sec,
                    observable_content=(
                        f"Padrão de teste geométrico exibindo escala de cores e contador temporal em {f.target_timestamp_sec:.2f}s."
                    ),
                    inferences="Ambiente controlado de calibração técnica de sinal de vídeo.",
                    visual_quality_notes=f"Nitidez alta, resolução nominal {f.width}x{f.height}, sem ruído de compressão perceptível.",
                )
            )

        # Cria hipótese de transição de cena se houver mais de um frame
        transitions: list[SceneTransitionHypothesis] = []
        if len(frames) >= 2:
            transitions.append(
                SceneTransitionHypothesis(
                    from_timestamp_sec=frames[0].target_timestamp_sec,
                    to_timestamp_sec=frames[1].target_timestamp_sec,
                    transition_type="content_shift",
                    evidence_description="Progressão temporal contínua dos marcadores e numeração entre os frames amostrados.",
                    is_confirmed_exact=False,
                )
            )

        return VisualAnalysisResponse(
            provider_used="mock",
            model_used=config.model_id,
            summary=f"Vídeo de teste sintético de {metadata.duration_sec:.2f}s com {len(frames)} frames amostrados.",
            chronological_timeline=observations,
            scene_transitions=transitions,
            limitations_and_blindspots=[
                "Amostragem discreta de frames; eventos que ocorreram entre os timestamps amostrados não foram observados.",
                "Instantes exatos de cortes e transições são hipóteses baseadas em diferenças entre amostras e não fatos comprovados continuamente.",
            ],
        )
