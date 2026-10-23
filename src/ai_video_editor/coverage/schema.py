"""Schemas Pydantic do Diretor de cobertura: material visual, janelas obrigatórias, decisão e registro.

A IA vê só IDs criados pelo código (frases `S012`, tomadas `D001.T05`, janelas `O1`) e devolve
coberturas (B-roll por cima de um trecho da fala) e respiros (um plano curto sem fala inserido
entre duas frases). O código valida, converte IDs em tempos e monta a Timeline.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_video_editor.intelligence.schema import AIRunMetrics
from ai_video_editor.llm.schema import ModelSpec, ModelSwitch, ThinkingLevel

# Valores do `visual.json` (`visual/schema.py`: AnalysisStatus, Usable, Issue; papel = `catalog` Role),
# repetidos aqui porque importar os pacotes `visual`/`catalog` ao carregar o pipeline cria um ciclo de
# imports (catalog → pipeline → coverage → catalog).
ShotStatus = Literal["analisada", "sem_analise", "nao_pedida"]
ShotUsable = Literal["sim", "nao", "parcial"]

# Tamanho de um respiro; o código converte em segundos (`BREATH_SECONDS`).
BreathSize = Literal["short", "medium", "long"]
BREATH_SIZE_LABELS: dict[str, str] = {"short": "curto", "medium": "médio", "long": "longo"}

# ia: tudo da resposta validada; ia+fallback: resposta da IA completada pelo código (janelas obrigatórias
# que ficaram descobertas); fallback: só as janelas obrigatórias, pelo código.
CoverageOrigin = Literal["ia", "ia+fallback", "fallback"]
ItemOrigin = Literal["ia", "fallback"]


def sentence_label(sentence_id: int) -> str:
    """ID de frase que a IA vê: `S012`."""
    return f"S{sentence_id:03d}"


class VisualShot(BaseModel):
    """O que a cobertura usa de uma tomada do `visual.json` (+ caminho e áudio do catálogo)."""

    id: str = Field(description="D001.T05")
    clip_id: str
    role: str = Field(description="Papel do catálogo: principal, drone, pov, broll, outro")
    source_path: Path
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    last_in_clip: bool = Field(default=False, description="Última tomada do clipe (o fim pode passar do último quadro)")
    frame_times: dict[str, float] = Field(default_factory=dict, description="ID da miniatura → tempo no clipe")
    has_audio: bool = True
    has_speech: Optional[bool] = None
    status: ShotStatus = "nao_pedida"
    uniform: bool = Field(default=False, description="Medida `tela uniforme`")
    usable: Optional[ShotUsable] = None
    issues: list[str] = Field(default_factory=list, description="Problemas da IA (`obstruida`, `borrao_movimento`...)")
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    beauty: Optional[int] = None
    face_to_camera: Optional[bool] = None
    best_frames: list[str] = Field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


class VisualMaterial(BaseModel):
    """Conteúdo do `coverage_shots.json`: as tomadas do projeto, como a etapa `coverage` as usa."""

    visual_path: Path
    roles: list[str] = Field(default_factory=list, description="Papéis analisados pelo `visual`")
    shots: list[VisualShot] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def by_id(self) -> dict[str, VisualShot]:
        return {s.id: s for s in self.shots}


class CoverageSentence(BaseModel):
    """Frase mantida, na ordem do vídeo, com a posição na espinha (frames, sem respiros)."""

    id: str = Field(description="S012")
    sentence_id: int
    clip_id: str
    text: str
    source_start_sec: float = Field(description="Início na origem (tempo do clipe), já aparado pelos cortes")
    source_end_sec: float
    start_frame: int = Field(description="Primeiro frame na espinha")
    end_frame: int = Field(description="Frame final exclusivo na espinha")
    kept: list[tuple[float, float]] = Field(default_factory=list, description="Pedaços da frase que ficaram (tempo do clipe)")

    def duration_sec(self, fps: int) -> float:
        return (self.end_frame - self.start_frame) / fps


class MandatoryWindow(BaseModel):
    """Trecho da fala mantida com imagem ruim no principal: precisa de cobertura."""

    id: str = Field(description="O1")
    clip_id: str
    sentence_ids: list[str] = Field(min_length=1, description="Frases cobertas, em ordem")
    shot_ids: list[str] = Field(default_factory=list, description="Tomadas ruins do principal")
    reasons: list[str] = Field(default_factory=list, description="Ex.: tela uniforme, obstruída/parede")

    @property
    def span_label(self) -> str:
        first, last = self.sentence_ids[0], self.sentence_ids[-1]
        return first if first == last else f"{first}–{last}"


class CoverageContext(BaseModel):
    """Tudo o que o Diretor de cobertura recebe (o prompt é montado a partir daqui)."""

    project_name: str
    fps: int
    spine_duration_sec: float
    sentences: list[CoverageSentence] = Field(default_factory=list)
    windows: list[MandatoryWindow] = Field(default_factory=list)
    candidates: list[VisualShot] = Field(default_factory=list, description="Tomadas que podem ser cobertura")
    instructions: list[str] = Field(default_factory=list, description="Instruções do briefing e pedido do usuário")
    roles: list[str] = Field(default_factory=list, description="Papéis permitidos como cobertura")
    previous_error: Optional[str] = None


# --- resposta da IA ---


class CoverageChoice(BaseModel):
    """Uma cobertura: a imagem da tomada substitui a do principal enquanto as frases são ditas."""

    from_sentence: str = Field(description="ID da primeira frase coberta, ex: S012")
    to_sentence: str = Field(description="ID da última frase coberta (igual à primeira para uma frase só)")
    shot_id: str = Field(description="ID da tomada usada como imagem, ex: D001.T05")
    reason: str = Field(description="Por que esta imagem combina com a fala, em português")
    windows: list[str] = Field(default_factory=list, description="IDs das janelas obrigatórias que ela atende (ex: O1)")


class BreathChoice(BaseModel):
    """Um respiro: plano sem fala inserido entre duas frases (o vídeo fica mais longo)."""

    after_sentence: str = Field(description="ID da frase depois da qual o respiro entra")
    shot_id: str = Field(description="ID da tomada mostrada (com o som dela)")
    size: BreathSize = Field(description="Tamanho: short, medium ou long")
    reason: str = Field(description="Por que um respiro aqui, em português")


class CoverageDecision(BaseModel):
    """Resposta estruturada do Diretor de cobertura."""

    coverages: list[CoverageChoice] = Field(default_factory=list)
    breaths: list[BreathChoice] = Field(default_factory=list)
    summary: str = Field(description="Resumo das escolhas, em português")


class AcceptedCoverage(CoverageChoice):
    origin: ItemOrigin = "ia"


class AcceptedBreath(BreathChoice):
    origin: ItemOrigin = "ia"


# --- configuração ---


class CoverageParams(BaseModel):
    """IA da cobertura (`--coverage-*`); sem opções, a do Diretor. `none` = só as janelas obrigatórias."""

    provider: str = Field(default="none", description="`none`, `mock` ou um provedor registrado")
    model: Optional[str] = None
    fallback: Optional[str] = None
    thinking: ThinkingLevel = "low"
    timeout_sec: float = Field(default=120.0, gt=5.0)
    api_retries: int = Field(default=4, ge=0)

    @classmethod
    def inherit(
        cls,
        director: str,
        director_model: Optional[str] = None,
        director_fallback: Optional[str] = None,
        director_thinking: ThinkingLevel = "low",
        director_timeout_sec: float = 120.0,
        api_retries: int = 4,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        fallback: Optional[str] = None,
        thinking: Optional[ThinkingLevel] = None,
        timeout_sec: Optional[float] = None,
    ) -> "CoverageParams":
        """Opções da cobertura; o que não foi dito vem do Diretor (como no revisor).

        Modelo e reservas só são herdados quando o provedor é o mesmo do Diretor.
        """
        provider = provider or director
        same = provider == director and model is None
        return cls(
            provider=provider,
            model=model or (director_model if same else None),
            fallback=fallback if fallback is not None or not same else director_fallback,
            thinking=thinking or director_thinking,
            timeout_sec=timeout_sec or director_timeout_sec,
            api_retries=api_retries,
        )

    def chain(self) -> list["ModelSpec"]:
        """Cadeia pedida (só para provedores reais)."""
        from ai_video_editor.llm.chain import resolve_chain

        return resolve_chain(self.provider, self.model, self.fallback)


# --- resultado (coverage_decision.json) ---


class OverlayPiece(BaseModel):
    """Um item de `video_overlay`: um trecho da tomada numa janela da timeline final."""

    shot_id: str
    clip_id: str
    source_in: float
    source_out: float
    timeline_start: float
    timeline_end: float
    chained: bool = Field(default=False, description="Tomada extra encadeada porque a escolhida era curta")


class PlacedCoverage(BaseModel):
    id: str = Field(description="C1")
    from_sentence: str
    to_sentence: str
    shot_id: str
    reason: str
    origin: ItemOrigin
    windows: list[str] = Field(default_factory=list, description="Janelas obrigatórias que esta cobertura toca")
    timeline_start: float
    timeline_end: float
    pieces: list[OverlayPiece] = Field(default_factory=list)


class PlacedBreath(BaseModel):
    id: str = Field(description="R1")
    after_sentence: str
    shot_id: str
    clip_id: str
    size: BreathSize
    reason: str
    origin: ItemOrigin
    source_in: float
    source_out: float
    timeline_start: float
    duration_sec: float
    audio: bool = Field(description="Usa o som da tomada (False: clipe sem áudio ou com fala)")


class CoverageRun(BaseModel):
    """Conteúdo do `coverage_decision.json`: contexto, decisão aceita, onde cada item foi parar e métricas."""

    provider: str
    model_id: str
    thinking: Optional[str] = None
    model_chain: list[str] = Field(default_factory=list)
    effective_provider: Optional[str] = None
    effective_model_id: Optional[str] = None
    model_switches: list[ModelSwitch] = Field(default_factory=list)
    attempts: list[AIRunMetrics] = Field(default_factory=list)
    origin: CoverageOrigin = "fallback"
    origin_reason: str = Field(default="", description="Por que o código completou ou substituiu a IA")
    instructions: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    candidates: int = 0
    windows: list[MandatoryWindow] = Field(default_factory=list)
    window_cover: dict[str, list[str]] = Field(default_factory=dict, description="Janela → coberturas que a atendem")
    summary: str = ""
    coverages: list[AcceptedCoverage] = Field(default_factory=list, description="Decisão aceita (IDs)")
    breaths: list[AcceptedBreath] = Field(default_factory=list)
    placed_coverages: list[PlacedCoverage] = Field(default_factory=list)
    placed_breaths: list[PlacedBreath] = Field(default_factory=list)
    spine_duration_sec: float = 0.0
    final_duration_sec: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    @property
    def uncovered_windows(self) -> list[str]:
        return [w.id for w in self.windows if not self.window_cover.get(w.id)]
