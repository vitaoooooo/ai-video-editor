"""Schemas Pydantic do Diretor de cobertura: material visual, janelas com imagem ruim, decisão e registro.

A IA vê só IDs criados pelo código (frases `S012`, tomadas `D001.T05`, janelas `O1`) e devolve
coberturas (B-roll por cima de um trecho da fala, com uma ou mais tomadas em ordem), respiros (um
plano curto sem fala inserido entre duas frases) e, por janela com imagem ruim, cobrir ou manter. O
código valida item a item, decide os segundos, converte IDs em tempos e monta a Timeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

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

# ia: itens da IA (talvez ajustados ou com alguns descartados); ia+fallback: a IA mais itens do código
# (janelas que ficaram sem decisão); fallback: só o código (IA desligada ou nada aproveitável).
CoverageOrigin = Literal["ia", "ia+fallback", "fallback"]
# ia-ajustada: item da IA que o código mudou (duração, tomada inválida retirada...), com os ajustes no item.
ItemOrigin = Literal["ia", "ia-ajustada", "fallback"]
WindowAction = Literal["cover", "keep"]
WindowStatus = Literal["coberta", "mantida", "descoberta"]


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
    blurry: bool = Field(default=False, description="Medida `borrão alto`")
    usable: Optional[ShotUsable] = None
    issues: list[str] = Field(default_factory=list, description="Problemas da IA (`obstruida`, `borrao_movimento`...)")
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    beauty: Optional[int] = None
    face_to_camera: Optional[bool] = None
    best_frames: list[str] = Field(default_factory=list)
    recorded_at: Optional[datetime] = Field(default=None, description="Gravação do clipe (catálogo, com o ajuste de relógio)")
    date_reliable: bool = Field(
        default=False, description="A data veio da gravação (a ordem do papel é a da data), não do nome do arquivo"
    )

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
    screen: list[str] = Field(
        default_factory=list, description="Tomadas do clipe que fala por baixo da frase (o que está na tela), mais tempo primeiro"
    )

    def duration_sec(self, fps: int) -> float:
        return (self.end_frame - self.start_frame) / fps


class MandatoryWindow(BaseModel):
    """Trecho da fala mantida com imagem ruim no principal: a IA decide cobrir (com tomada relacionada) ou
    manter; com `required`, só cobrir."""

    id: str = Field(description="O1")
    clip_id: str
    sentence_ids: list[str] = Field(min_length=1, description="Frases cobertas, em ordem")
    shot_ids: list[str] = Field(default_factory=list, description="Tomadas ruins do principal")
    reasons: list[str] = Field(default_factory=list, description="Ex.: tela uniforme, obstruída/parede")
    required: bool = Field(
        default=False,
        description="Exige cobertura: toda tomada ruim da janela é grave (tela uniforme, inutilizável ou borrão alto medido)",
    )

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
    screen_shots: list[VisualShot] = Field(
        default_factory=list, description="Tomadas do principal por baixo da fala (`CoverageSentence.screen`)"
    )
    instructions: list[str] = Field(default_factory=list, description="Instruções do briefing e pedido do usuário")
    roles: list[str] = Field(default_factory=list, description="Papéis permitidos como cobertura")
    previous_error: Optional[str] = Field(default=None, description="Nova tentativa: só os problemas da resposta anterior")
    accepted_items: list[str] = Field(
        default_factory=list, description="Nova tentativa: itens já aceitos (a IA não os repete; tomadas e frases ocupadas)"
    )


# --- resposta da IA ---


class CoverageChoice(BaseModel):
    """Uma cobertura: as imagens das tomadas, em ordem, substituem a do principal enquanto as frases são ditas."""

    from_sentence: str = Field(description="ID da primeira frase coberta, ex: S012")
    to_sentence: str = Field(description="ID da última frase coberta (igual à primeira para uma frase só)")
    shot_ids: list[str] = Field(
        min_length=1,
        description="IDs das tomadas, em ordem (ex: [D001.T05, B002.T01]); cada uma mostra no máximo ~8 s, "
        "então um trecho longo precisa de várias",
    )
    reason: str = Field(description="Por que estas imagens combinam com a fala e com o que está na tela, em português")
    windows: list[str] = Field(default_factory=list, description="IDs das janelas com imagem ruim que ela cobre (ex: O1)")

    @model_validator(mode="before")
    @classmethod
    def _single_shot(cls, data: Any) -> Any:
        """Aceita a forma antiga, com uma tomada só (`shot_id`)."""
        if isinstance(data, dict) and "shot_id" in data and not data.get("shot_ids"):
            data = {k: v for k, v in data.items() if k != "shot_id"} | {"shot_ids": [data["shot_id"]]}
        return data

    @property
    def shot_id(self) -> str:
        """Primeira tomada (a que a IA escolheu primeiro)."""
        return self.shot_ids[0]


class BreathChoice(BaseModel):
    """Um respiro: plano sem fala inserido entre duas frases (o vídeo fica mais longo)."""

    after_sentence: str = Field(description="ID da frase depois da qual o respiro entra")
    shot_id: str = Field(description="ID da tomada mostrada (com o som dela)")
    size: BreathSize = Field(description="Tamanho: short, medium ou long")
    reason: str = Field(description="Por que um respiro aqui, em português")


class WindowDecision(BaseModel):
    """O que fazer com uma janela de imagem ruim: cobrir (com a cobertura listada em `coverages`) ou manter."""

    window: str = Field(description="ID da janela, ex: O1")
    action: WindowAction = Field(
        description="cover: coberta por uma cobertura de 'coverages' com tomada relacionada; keep: a imagem original fica"
    )
    reason: str = Field(description="Motivo, em português (ex.: o tour pelo quarto é o conteúdo)")


class CoverageDecision(BaseModel):
    """Resposta estruturada do Diretor de cobertura."""

    coverages: list[CoverageChoice] = Field(default_factory=list)
    breaths: list[BreathChoice] = Field(default_factory=list)
    window_decisions: list[WindowDecision] = Field(
        default_factory=list, description="Uma decisão por janela com imagem ruim (O1, O2...)"
    )
    summary: str = Field(description="Resumo das escolhas, em português")


class ItemNote(BaseModel):
    """Um ajuste ou descarte feito pelo código, com uma categoria curta para o resumo."""

    kind: str = Field(description="Categoria: duração, clipe que fala, tomada repetida, sobreposição...")
    message: str


class AcceptedCoverage(CoverageChoice):
    """Cobertura aceita: `shot_ids` é a lista final (inclui as tomadas acrescentadas pelo código)."""

    origin: ItemOrigin = "ia"
    added_shots: list[str] = Field(default_factory=list, description="Tomadas acrescentadas pelo código (trecho longo)")
    adjustments: list[ItemNote] = Field(default_factory=list)


class AcceptedBreath(BreathChoice):
    origin: ItemOrigin = "ia"
    adjustments: list[ItemNote] = Field(default_factory=list)


class DiscardedItem(ItemNote):
    """Item da resposta da IA descartado na validação."""

    attempt: int
    item: str = Field(description="Ex.: cobertura 19 (S133–S134)")


class WindowOutcome(BaseModel):
    """Como uma janela de imagem ruim ficou no vídeo."""

    window: str
    status: WindowStatus
    by: list[str] = Field(default_factory=list, description="Coberturas (C1...) que a atendem")
    reason: str = ""
    origin: Optional[ItemOrigin] = None


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
    """Um item de `video_overlay`: um trecho da tomada num pedaço da cobertura na timeline final."""

    shot_id: str
    clip_id: str
    source_in: float
    source_out: float
    timeline_start: float
    timeline_end: float
    chained: bool = Field(default=False, description="Tomada acrescentada pelo código (as escolhidas não bastavam)")


class PlacedCoverage(BaseModel):
    id: str = Field(description="C1")
    from_sentence: str
    to_sentence: str
    shot_id: str = Field(description="Primeira tomada")
    shot_ids: list[str] = Field(default_factory=list, description="Tomadas, em ordem (as da IA e as acrescentadas)")
    reason: str
    origin: ItemOrigin
    windows: list[str] = Field(default_factory=list, description="Janelas com imagem ruim que esta cobertura toca")
    adjustments: list[ItemNote] = Field(default_factory=list)
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
    responses: list[Optional[CoverageDecision]] = Field(
        default_factory=list, description="Resposta crua da IA em cada tentativa (None: não interpretável)"
    )
    origin: CoverageOrigin = "fallback"
    origin_reason: str = Field(default="", description="Por que o código completou ou substituiu a IA")
    ai_items: int = Field(default=0, description="Itens recebidos da IA (todas as tentativas)")
    discarded: list[DiscardedItem] = Field(default_factory=list)
    kept_windows: dict[str, WindowOutcome] = Field(default_factory=dict, description="Janelas mantidas (IA ou código)")
    window_outcomes: list[WindowOutcome] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    candidates: int = 0
    windows: list[MandatoryWindow] = Field(default_factory=list)
    window_cover: dict[str, list[str]] = Field(default_factory=dict, description="Janela → coberturas que a tocam")
    summary: str = ""
    coverages: list[AcceptedCoverage] = Field(default_factory=list, description="Decisão aceita (IDs)")
    breaths: list[AcceptedBreath] = Field(default_factory=list)
    placed_coverages: list[PlacedCoverage] = Field(default_factory=list)
    placed_breaths: list[PlacedBreath] = Field(default_factory=list)
    spine_duration_sec: float = 0.0
    final_duration_sec: float = 0.0
    warnings: list[str] = Field(default_factory=list, description="Resumo para o terminal")
    notes: list[str] = Field(default_factory=list, description="Detalhes (relatório): ajustes, descartes, encurtamentos")

    @property
    def uncovered_windows(self) -> list[str]:
        """Janelas sem cobertura completa nem decisão de manter."""
        return [o.window for o in self.window_outcomes if o.status == "descoberta"]

    @property
    def kept_window_ids(self) -> list[str]:
        return [o.window for o in self.window_outcomes if o.status == "mantida"]

    @property
    def windows_label(self) -> str:
        """Ex.: `7 cobertas, 3 mantidas de 10` (e as sem cobertura, se houver)."""
        status = [o.status for o in self.window_outcomes]
        text = f"{status.count('coberta')} cobertas, {status.count('mantida')} mantidas"
        if status.count("descoberta"):
            text += f", {status.count('descoberta')} sem cobertura"
        return f"{text} de {len(self.windows)}"

    @property
    def origin_label(self) -> str:
        """Ex.: `ia (3 ajustadas, 2 descartadas)`."""
        items = [*self.coverages, *self.breaths]
        parts = []
        adjusted = sum(1 for x in items if x.origin == "ia-ajustada")
        if adjusted:
            parts.append(f"{adjusted} ajustada{'s' if adjusted > 1 else ''}")
        if self.discarded:
            parts.append(f"{len(self.discarded)} descartada{'s' if len(self.discarded) > 1 else ''}")
        fallback = sum(1 for x in items if x.origin == "fallback")
        if fallback and self.origin != "fallback":
            parts.append(f"{fallback} do código")
        return self.origin + (f" ({', '.join(parts)})" if parts else "")
