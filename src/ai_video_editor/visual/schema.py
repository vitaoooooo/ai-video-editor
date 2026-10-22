"""Schemas Pydantic da análise visual (`visual/visual.json`) e da resposta da IA por lote."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_video_editor.catalog.schema import Role, SpeechStatus, StageTime
from ai_video_editor.llm.schema import ImageDetail, ModelSwitch

# Sobe quando o formato do visual.json muda.
VISUAL_SCHEMA_VERSION = 1

# Como a tomada começa: início do clipe, mudança de cena detectada ou janela de uma tomada longa fatiada.
ShotBoundary = Literal["inicio", "cena", "janela"]

# analisada: a IA descreveu; sem_analise: a IA falhou (ou respondeu errado) depois das tentativas;
# nao_pedida: só medidas (`--visual-provider none` ou `--so-estimar`).
AnalysisStatus = Literal["analisada", "sem_analise", "nao_pedida"]

Issue = Literal["obstruida", "desfocada", "tremida", "escura", "estourada", "borrao_movimento", "nenhuma"]
Usable = Literal["sim", "nao", "parcial"]

ISSUE_LABELS: dict[str, str] = {
    "obstruida": "obstruída/parede",
    "desfocada": "desfocada",
    "tremida": "tremida",
    "escura": "escura",
    "estourada": "estourada",
    "borrao_movimento": "borrão de movimento",
    "nenhuma": "nenhuma",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ShotFrame(BaseModel):
    """Miniatura de uma tomada. O tempo fica só aqui: a IA vê apenas o ID."""

    id: str = Field(description="P007.T03.F1")
    position: int = Field(ge=1, le=3, description="1 = início+10%, 2 = meio, 3 = fim−10%")
    time_sec: float = Field(ge=0.0, description="Tempo no clipe original")
    file: str = Field(description="Relativo à pasta visual/")


class ShotMeasures(BaseModel):
    """Medidas determinísticas (FFmpeg) dos quadros amostrados da tomada. São dicas, não conclusões."""

    samples: int = Field(ge=0, description="Quadros amostrados")
    blur: Optional[float] = Field(default=None, description="Mediana do blurdetect (maior = mais borrado); None sem bordas")
    brightness: float = Field(ge=0.0, le=1.0, description="Luma média 0–1")
    motion: float = Field(ge=0.0, le=1.0, description="Mediana da diferença de luma entre amostras, 0–1")
    luma_spread: float = Field(ge=0.0, le=1.0, description="Mediana de (YHIGH − YLOW), 0–1")
    blurry: bool = False
    shaky: bool = False
    uniform: bool = False
    dark: bool = False
    bright: bool = False

    @property
    def hints(self) -> list[str]:
        """Dicas em português para o prompt e o relatório."""
        hints = []
        if self.uniform:
            hints.append("tela uniforme")
        if self.blurry:
            hints.append("borrão alto")
        if self.shaky:
            hints.append("movimento alto")
        if self.dark:
            hints.append("escura")
        if self.bright:
            hints.append("estourada")
        return hints


class ShotAnalysis(BaseModel):
    """Descrição da tomada pela IA (validada pelo código)."""

    description: str = Field(description="Uma frase em português descrevendo a imagem")
    tags: list[str] = Field(
        default_factory=list,
        description="Lista curta: paisagem, lago, praia, trilha, cidade, estrada, veículo, pessoa, rosto, animal, "
        "detalhe, céu, água, árvore, interior...",
    )
    face_to_camera: bool = Field(description="Há uma pessoa falando para a câmera")
    people_count: int = Field(ge=0, description="Número aproximado de pessoas visíveis")
    issues: list[Issue] = Field(description="Problemas de imagem; [\"nenhuma\"] se não houver")
    usable: Usable = Field(description="Se a tomada serve como imagem: sim, nao ou parcial")
    best_frames: list[str] = Field(
        description="IDs das miniaturas desta tomada (F1..F3, em ordem) que delimitam o melhor trecho; "
        "vazio se nada serve"
    )
    beauty: int = Field(ge=1, le=5, description="Valor como imagem de cobertura: 1 (feia/inútil) a 5 (linda)")
    summary_for_editor: str = Field(description="Nota curta para o editor, em português")


class ShotDescription(ShotAnalysis):
    """Um item da resposta da IA: a análise com o ID da tomada."""

    shot_id: str = Field(description="ID da tomada, ex: P007.T03")


class VisualBatchResponse(BaseModel):
    """Resposta da IA para um lote: exatamente uma entrada por tomada pedida."""

    shots: list[ShotDescription]


class Shot(BaseModel):
    id: str = Field(description="P007.T03")
    index: int = Field(ge=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    boundary: ShotBoundary
    frames: list[ShotFrame] = Field(default_factory=list)
    measures: Optional[ShotMeasures] = None
    has_speech: Optional[bool] = Field(default=None, description="Sobrepõe fala detectada; None se desconhecido")
    status: AnalysisStatus = "nao_pedida"
    analysis: Optional[ShotAnalysis] = None
    note: str = Field(default="", description="Motivo de `sem_analise`")

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


class ClipVisual(BaseModel):
    clip_id: str
    role: Role
    relative_path: str
    duration_sec: float
    speech_status: Optional[SpeechStatus] = None
    shots: list[Shot] = Field(default_factory=list)
    cached: bool = Field(default=False, description="Tomadas e medidas reaproveitadas do cache")
    error: Optional[str] = None


class VisualAIRun(BaseModel):
    """Como a IA foi chamada nesta execução (somando os lotes)."""

    provider: str
    model_chain: list[str] = Field(default_factory=list, description="Cadeia pedida")
    effective_model: Optional[str] = Field(default=None, description="`provedor:modelo` da última resposta")
    thinking: Optional[str] = None
    image_detail: ImageDetail = "auto"
    batches: int = 0
    api_calls: int = 0
    images_sent: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_shots: int = Field(default=0, description="Análises reaproveitadas do cache")
    model_switches: list[ModelSwitch] = Field(default_factory=list)


class ProjectVisual(BaseModel):
    schema_version: int = VISUAL_SCHEMA_VERSION
    project_dir: Path
    source_dir: Path
    created_at: datetime = Field(default_factory=_now)
    roles: list[Role] = Field(default_factory=list, description="Papéis analisados")
    clips: list[ClipVisual] = Field(default_factory=list)
    ai: Optional[VisualAIRun] = None
    warnings: list[str] = Field(default_factory=list)
    motion_threshold: Optional[float] = Field(
        default=None, description="Movimento acima disto = tremida (percentil do projeto, com piso)"
    )
    stage_times: list[StageTime] = Field(default_factory=list)
    total_sec: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectVisual":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def all_shots(self) -> list[Shot]:
        return [shot for clip in self.clips for shot in clip.shots]
