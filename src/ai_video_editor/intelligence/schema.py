"""Schemas Pydantic para análise visual multimodal e configurações de IA."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class VisualObservation(BaseModel):
    """Observação visual factual vinculada a um frame e timestamp específicos."""

    frame_index: int = Field(..., ge=0, description="Índice sequencial do frame enviado")
    target_timestamp_sec: float = Field(..., ge=0.0, description="Timestamp original do frame em segundos")
    observable_content: str = Field(..., description="Descrição puramente factual do que é visível na imagem")
    inferences: Optional[str] = Field(
        default=None, description="Hipóteses, interpretações ou contexto inferido não comprovado visualmente"
    )
    visual_quality_notes: Optional[str] = Field(
        default=None, description="Observações sobre nitidez, foco, iluminação e estabilidade do frame"
    )


class SceneTransitionHypothesis(BaseModel):
    """Hipótese de mudança de cena ou evento visual entre frames amostrados."""

    from_timestamp_sec: float = Field(..., ge=0.0, description="Timestamp do frame anterior à transição")
    to_timestamp_sec: float = Field(..., ge=0.0, description="Timestamp do frame posterior à transição")
    transition_type: str = Field(
        default="cut",
        description="Tipo aproximado observado: corte, transição suave, mudança de enquadramento ou shift de conteúdo",
    )
    evidence_description: str = Field(
        ..., description="Evidências visuais que sustentam a hipótese de transição entre os dois frames"
    )
    is_confirmed_exact: bool = Field(
        default=False,
        description="Falso por padrão em amostragens esparsas, indicando que o instante exato do corte não foi amostrado",
    )


class VisualAnalysisResponse(BaseModel):
    """Resposta estruturada da interpretação visual produzida pelo modelo multimodal."""

    provider_used: str = Field(..., description="Nome do provedor de IA utilizado")
    model_used: str = Field(..., description="Identificador do modelo utilizado")
    summary: str = Field(..., description="Síntese geral do conteúdo visual do clipe")
    chronological_timeline: list[VisualObservation] = Field(
        default_factory=list, description="Lista cronológica de observações frame a frame"
    )
    scene_transitions: list[SceneTransitionHypothesis] = Field(
        default_factory=list, description="Hipóteses de transição de cena detectadas"
    )
    limitations_and_blindspots: list[str] = Field(
        default_factory=list,
        description="Ressalvas explícitas sobre limitações decorrentes da amostragem discreta de frames",
    )


class AIAnalysisConfig(BaseModel):
    """Configurações de execução e custos da análise de inteligência multimodal."""

    provider: str = Field(default="gemini", description="Identificador do provedor (ex: 'gemini', 'mock')")
    model_id: str = Field(
        default="gemini-3.8-flash",
        description="Identificador configurável do modelo (ex: 'gemini-3.8-flash', 'gemini-3.8-pro')",
    )
    max_images_per_request: int = Field(
        default=15,
        gt=0,
        le=100,
        description="Limite conservador de quantidade máxima de frames enviados por requisição",
    )
    max_image_dimension: int = Field(
        default=1024,
        gt=128,
        le=3840,
        description="Dimensão máxima em pixels para downscaling de imagens proxy",
    )
    image_quality: int = Field(
        default=80,
        ge=20,
        le=100,
        description="Qualidade de compressão JPEG para os proxies de envio",
    )
    timeout_sec: float = Field(default=60.0, gt=5.0, description="Tempo limite em segundos para a chamada de API")
    temperature: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Temperatura de amostragem baixa para respostas factuais"
    )


class AIRunMetrics(BaseModel):
    """Métricas de execução de uma chamada de análise de IA."""

    provider: str = Field(..., description="Provedor de IA utilizado")
    model_id: str = Field(..., description="Identificador do modelo usado na chamada")
    frames_sent: int = Field(..., ge=0, description="Quantidade de frames enviados ao provedor")
    latency_sec: float = Field(..., ge=0.0, description="Latência da chamada analyze_frames em segundos")
    input_tokens: Optional[int] = Field(default=None, description="Tokens de entrada, se informados pelo provedor")
    output_tokens: Optional[int] = Field(default=None, description="Tokens de saída, se informados pelo provedor")
    api_attempts: int = Field(
        default=1, ge=1, description="Chamadas feitas à API (maior que 1 se houve nova tentativa por falha temporária)"
    )
    success: bool = Field(..., description="Indica se a chamada foi concluída com sucesso")
    error_type: Optional[str] = Field(default=None, description="Nome da classe da exceção em caso de falha")


class CompleteAIReport(BaseModel):
    """Registro unificado contendo o resultado da análise de IA e o caminho do relatório Markdown."""

    analysis_response: VisualAnalysisResponse = Field(..., description="Dados estruturados da IA")
    markdown_path: Path = Field(..., description="Caminho do arquivo de relatório Markdown gerado")
    json_path: Path = Field(..., description="Caminho do arquivo de dados JSON da IA")
    metrics: AIRunMetrics = Field(..., description="Métricas de execução da chamada de IA")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp UTC de geração do relatório",
    )
