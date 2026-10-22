"""Módulo de inteligência artificial multimodal para análise visual de vídeos.

LEGADO: a análise visual (`MultimodalAIProvider`, `report-ai`) será substituída por `visual/`
(tomadas, medidas e descrição por IA via `llm/` com imagens). Continua aqui porque `analyze` e
`report-ai` ainda a usam; as exceções (`exceptions.py`) e as novas tentativas (`retry.py`)
seguem sendo a base comum da camada `llm/`.
"""

from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.image_proxy import create_image_proxies
from ai_video_editor.intelligence.provider import MultimodalAIProvider
from ai_video_editor.intelligence.providers.gemini import GeminiMultimodalProvider
from ai_video_editor.intelligence.providers.mock import MockMultimodalProvider
from ai_video_editor.intelligence.reporter import MarkdownReportGenerator
from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    CompleteAIReport,
    SceneTransitionHypothesis,
    VisualAnalysisResponse,
    VisualObservation,
)
from ai_video_editor.intelligence.service import (
    MultimodalAnalysisService,
    select_representative_frames,
)

__all__ = [
    "AIAnalysisConfig",
    "AIAuthenticationError",
    "AIProviderError",
    "AIRateLimitError",
    "AIResponseValidationError",
    "AITimeoutError",
    "AIUnavailableError",
    "CompleteAIReport",
    "GeminiMultimodalProvider",
    "MarkdownReportGenerator",
    "MockMultimodalProvider",
    "MultimodalAIProvider",
    "MultimodalAnalysisService",
    "SceneTransitionHypothesis",
    "VisualAnalysisResponse",
    "VisualObservation",
    "create_image_proxies",
    "select_representative_frames",
]
