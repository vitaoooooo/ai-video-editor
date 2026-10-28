"""Schemas Pydantic do Diretor de cobertura: material visual, mapa de assuntos, trechos com imagem ruim,
decisão com evidências e registro.

A IA vê só IDs criados pelo código (frases `S012`, tomadas `D001.T05`, trechos `O1`) e devolve, em duas
chamadas, o mapa de assuntos (blocos de frases seguidas) e as coberturas (B-roll por cima de um trecho da
fala, com uma ou mais tomadas em ordem e as evidências de fala e de imagem) e respiros (um plano curto sem
fala entre dois blocos). O código confere as evidências, valida item a item, decide os segundos, converte
IDs em tempos e monta a Timeline.
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
# (tela uniforme que ficou descoberta); fallback: só o código (IA desligada ou nada aproveitável).
CoverageOrigin = Literal["ia", "ia+fallback", "fallback"]
# ia-ajustada: item da IA que o código mudou (duração, tomada sem a evidência retirada...), com os ajustes no item.
ItemOrigin = Literal["ia", "ia-ajustada", "fallback"]
# conteudo: a imagem mostra o que é falado (com evidência); tela_uniforme: tapa de parede/tampa com tomada
# relacionada, sem evidência de fala, só pelo tempo da tela uniforme.
CoverageKind = Literal["conteudo", "tela_uniforme"]
BadSpanStatus = Literal["coberto", "parcial", "mantido"]


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
    to_camera: bool = Field(
        default=False, description="A pessoa fala para a câmera (tomada principal da frase com rosto e utilizável)"
    )

    def duration_sec(self, fps: int) -> float:
        return (self.end_frame - self.start_frame) / fps


class BadSpan(BaseModel):
    """Trecho da fala mantida com imagem ruim de verdade no principal, em frames da espinha (o código mede:
    parte mantida das frases ∩ tomadas graves + `BAD_MARGIN_SEC`). Não obriga cobertura: a IA cobre se
    houver imagem que combine; sem ela, a imagem original fica, com aviso."""

    id: str = Field(description="O1")
    clip_id: str
    sentence_ids: list[str] = Field(min_length=1, description="Frases que o trecho toca, em ordem")
    shot_ids: list[str] = Field(default_factory=list, description="Tomadas ruins do principal")
    reasons: list[str] = Field(default_factory=list, description="Ex.: tela uniforme, imagem inutilizável")
    uniform: bool = Field(
        default=False,
        description="Tela uniforme (parede, tampa): pode ser tapada sem evidência de fala, com tomada relacionada",
    )
    start_frame: int
    end_frame: int = Field(description="Frame final exclusivo")

    def duration_sec(self, fps: int) -> float:
        return (self.end_frame - self.start_frame) / fps

    @property
    def span_label(self) -> str:
        first, last = self.sentence_ids[0], self.sentence_ids[-1]
        return first if first == last else f"{first}–{last}"


class TopicBlock(BaseModel):
    """Bloco de assunto: frases seguidas que falam da mesma coisa, no mesmo lugar/momento."""

    from_sentence: str = Field(description="Primeira frase do bloco, ex: S000")
    to_sentence: str = Field(description="Última frase do bloco (igual à primeira para uma frase só)")
    topic: str = Field(description="Assunto em poucas palavras, em português (ex.: trilha até o lago)")
    place: str = Field(default="", description="Lugar ou objeto principal falado (ex.: lago, ônibus); vazio se nenhum")


class TopicMap(BaseModel):
    """Resposta da IA na 1ª chamada: do que o vídeo trata e os blocos de assunto, em ordem."""

    summary: str = Field(default="", description="Do que o vídeo trata, em uma frase, em português")
    blocks: list[TopicBlock] = Field(default_factory=list, description="Blocos contíguos, cobrindo todas as frases")


class Topic(TopicBlock):
    """Bloco validado: contíguo, sem buraco nem sobreposição, com os frames da espinha."""

    id: str = Field(description="A1")
    start_frame: int = 0
    end_frame: int = 0

    @property
    def span_label(self) -> str:
        return self.from_sentence if self.from_sentence == self.to_sentence else f"{self.from_sentence}–{self.to_sentence}"


class CoverageContext(BaseModel):
    """Tudo o que o Diretor de cobertura recebe (os prompts são montados a partir daqui)."""

    project_name: str
    fps: int
    spine_duration_sec: float
    sentences: list[CoverageSentence] = Field(default_factory=list)
    bad_spans: list[BadSpan] = Field(default_factory=list)
    candidates: list[VisualShot] = Field(default_factory=list, description="Tomadas que podem ser cobertura")
    screen_shots: list[VisualShot] = Field(
        default_factory=list, description="Tomadas do principal por baixo da fala (`CoverageSentence.screen`)"
    )
    instructions: list[str] = Field(default_factory=list, description="Instruções do briefing e pedido do usuário")
    roles: list[str] = Field(default_factory=list, description="Papéis permitidos como cobertura")
    topics: list[Topic] = Field(default_factory=list, description="Mapa de assuntos validado (vai para a 2ª chamada)")
    video_summary: str = ""
    previous_error: Optional[str] = Field(default=None, description="Nova tentativa: só os problemas da resposta anterior")
    accepted_items: list[str] = Field(
        default_factory=list, description="Nova tentativa: itens já aceitos (a IA não os repete; tomadas e frases ocupadas)"
    )


# --- resposta da IA ---


class CoverageChoice(BaseModel):
    """Uma cobertura: as imagens das tomadas, em ordem, substituem a do principal enquanto as frases são ditas,
    com a evidência de que mostram o que é falado."""

    from_sentence: str = Field(description="ID da primeira frase coberta, ex: S012")
    to_sentence: str = Field(description="ID da última frase coberta (igual à primeira para uma frase só)")
    shot_ids: list[str] = Field(
        min_length=1,
        description="IDs das tomadas, em ordem (ex: [D001.T05, B002.T01]); cada uma mostra no máximo ~8 s",
    )
    speech_evidence: str = Field(
        default="", description="Trecho copiado literalmente do texto das frases cobertas que fala do que a imagem mostra"
    )
    image_evidence: str = Field(
        default="", description="Tag ou trecho copiado da descrição de cada tomada listada que mostra o que é falado"
    )
    reason: str = Field(description="Por que estas imagens mostram o que é falado, em português")

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
    """Um respiro: plano sem fala inserido entre dois blocos de assunto (o vídeo fica mais longo)."""

    after_sentence: str = Field(description="ID da última frase de um bloco de assunto (o respiro entra depois dela)")
    shot_id: str = Field(description="ID da tomada mostrada (com o som dela)")
    size: BreathSize = Field(description="Tamanho: short, medium ou long")
    image_evidence: str = Field(
        default="", description="Tag ou trecho da descrição da tomada que mostra o lugar/assunto do bloco anterior ou seguinte"
    )
    reason: str = Field(description="Por que um respiro aqui, em português")


class CoverageDecision(BaseModel):
    """Resposta estruturada do Diretor de cobertura (2ª chamada)."""

    coverages: list[CoverageChoice] = Field(default_factory=list)
    breaths: list[BreathChoice] = Field(default_factory=list)
    summary: str = Field(description="Resumo das escolhas, em português")


class ItemNote(BaseModel):
    """Um ajuste ou descarte feito pelo código, com uma categoria curta para o resumo."""

    kind: str = Field(description="Categoria: evidência, duração, clipe que fala, tomada repetida, sobreposição...")
    message: str


class Evidence(BaseModel):
    """Evidência conferida pelo código."""

    speech: str = Field(default="", description="Como a IA escreveu (vazio: tapa de tela uniforme)")
    image: str = ""
    shared: list[str] = Field(default_factory=list, description="Palavras em comum entre a fala e a imagem")
    strong: bool = Field(default=False, description="Uma palavra em comum é tag da tomada (assunto principal da imagem)")
    relation: str = Field(default="", description="Tapa de tela uniforme: por que a tomada é relacionada")


class AcceptedCoverage(CoverageChoice):
    """Cobertura aceita: `shot_ids` é a lista final e os frames (da espinha) são os do código."""

    kind: CoverageKind = "conteudo"
    origin: ItemOrigin = "ia"
    added_shots: list[str] = Field(default_factory=list, description="Tomadas acrescentadas pelo código (só tela uniforme)")
    adjustments: list[ItemNote] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
    start_frame: int = Field(default=0, description="Início na espinha (decidido pelo código)")
    end_frame: int = Field(default=0, description="Fim exclusivo na espinha")
    bad_spans: list[str] = Field(default_factory=list, description="Trechos com imagem ruim que ela cobre")


class AcceptedBreath(BreathChoice):
    origin: ItemOrigin = "ia"
    adjustments: list[ItemNote] = Field(default_factory=list)
    relation: str = Field(default="", description="Por que a tomada tem a ver com o bloco vizinho")


class DiscardedItem(ItemNote):
    """Item da resposta da IA descartado na validação (ou pelo limite de cobertura)."""

    attempt: int
    item: str = Field(description="Ex.: cobertura 19 (S133–S134)")


class BadSpanOutcome(BaseModel):
    """Como um trecho com imagem ruim ficou no vídeo."""

    span: str
    status: BadSpanStatus
    by: list[str] = Field(default_factory=list, description="Coberturas (C1...) que o tocam")
    reason: str = ""


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
    chained: bool = Field(default=False, description="Tomada acrescentada pelo código (tela uniforme)")


class PlacedCoverage(BaseModel):
    id: str = Field(description="C1")
    from_sentence: str
    to_sentence: str
    shot_id: str = Field(description="Primeira tomada")
    shot_ids: list[str] = Field(default_factory=list, description="Tomadas, em ordem (as da IA e as acrescentadas)")
    reason: str
    kind: CoverageKind = "conteudo"
    origin: ItemOrigin
    evidence: Evidence = Field(default_factory=Evidence)
    bad_spans: list[str] = Field(default_factory=list, description="Trechos com imagem ruim que esta cobertura toca")
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
    relation: str = ""
    source_in: float
    source_out: float
    timeline_start: float
    duration_sec: float
    audio: bool = Field(description="Usa o som da tomada (False: clipe sem áudio ou com fala)")


class CoverageRun(BaseModel):
    """Conteúdo do `coverage_decision.json`: mapa de assuntos, contexto, decisão aceita, onde cada item foi
    parar e métricas das duas chamadas."""

    provider: str
    model_id: str
    thinking: Optional[str] = None
    model_chain: list[str] = Field(default_factory=list)
    effective_provider: Optional[str] = None
    effective_model_id: Optional[str] = None
    model_switches: list[ModelSwitch] = Field(default_factory=list)
    topic_attempts: list[AIRunMetrics] = Field(default_factory=list, description="1ª chamada: mapa de assuntos")
    topic_responses: list[Optional[TopicMap]] = Field(
        default_factory=list, description="Resposta crua do mapa de assuntos em cada tentativa (None: não interpretável)"
    )
    topics_origin: Literal["ia", "fallback"] = "fallback"
    topics_reason: str = Field(default="", description="Por que o mapa é do código (IA desligada, mapa inválido...)")
    video_summary: str = ""
    topics: list[Topic] = Field(default_factory=list)
    attempts: list[AIRunMetrics] = Field(default_factory=list, description="2ª chamada: cobertura")
    responses: list[Optional[CoverageDecision]] = Field(
        default_factory=list, description="Resposta crua da cobertura em cada tentativa (None: não interpretável)"
    )
    origin: CoverageOrigin = "fallback"
    origin_reason: str = Field(default="", description="Por que o código completou ou substituiu a IA")
    ai_items: int = Field(default=0, description="Itens recebidos da IA (todas as tentativas)")
    discarded: list[DiscardedItem] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    candidates: int = 0
    bad_spans: list[BadSpan] = Field(default_factory=list)
    span_outcomes: list[BadSpanOutcome] = Field(default_factory=list)
    summary: str = ""
    coverages: list[AcceptedCoverage] = Field(default_factory=list, description="Decisão aceita (IDs e frames da espinha)")
    breaths: list[AcceptedBreath] = Field(default_factory=list)
    placed_coverages: list[PlacedCoverage] = Field(default_factory=list)
    placed_breaths: list[PlacedBreath] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list, description="Uma linha por bloco de assunto (terminal e relatório)")
    covered_sec: float = Field(default=0.0, description="Segundos de cobertura de conteúdo (o limite vale sobre eles)")
    fps: int = Field(default=30, description="Da espinha (frames dos trechos com imagem ruim e das coberturas)")
    spine_duration_sec: float = 0.0
    final_duration_sec: float = 0.0
    warnings: list[str] = Field(default_factory=list, description="Resumo para o terminal")
    notes: list[str] = Field(default_factory=list, description="Detalhes (relatório): ajustes, descartes, encurtamentos")

    @property
    def bad_label(self) -> str:
        """Ex.: `1 coberto, 1 mantido de 2` (trechos com imagem ruim)."""
        status = [o.status for o in self.span_outcomes]
        if not status:
            return "nenhum"
        parts = [f"{status.count('coberto')} coberto(s)"]
        if status.count("parcial"):
            parts.append(f"{status.count('parcial')} parcial(is)")
        parts.append(f"{status.count('mantido')} mantido(s)")
        return f"{', '.join(parts)} de {len(status)}"

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
