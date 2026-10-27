"""Schemas Pydantic do Editor de IA: as respostas das três passadas (só IDs) e o que o código registra.

As respostas são tolerantes de propósito (campos opcionais, listas cortadas no limite): um item ruim é
descartado sozinho pela validação do código (`editor/validate.py`), em vez de derrubar a resposta inteira.
Os nomes dos campos das respostas são os do pedido à IA (em português); os do código, em inglês.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ai_video_editor.llm.schema import ModelSwitch

# Sobe quando o formato do editor_run.json / editor_plano.json muda.
EDITOR_SCHEMA_VERSION = 1

MAX_HIGHLIGHTS = 3
MAX_PROBLEMS = 3

# ia: resposta validada do modelo; cache: a mesma resposta, de uma execução anterior; fallback: o código
# montou o resultado neutro porque a IA falhou de vez (aviso no relatório).
Origin = Literal["ia", "cache", "fallback"]


def _text(value) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _id(value) -> Optional[str]:
    """`s012` / ` S012 ` → `S012`; vazio → None (a validação diz que falta)."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _optional_number(value):
    """Número da IA ou None ("", "auto", texto): duração opcional não derruba o item."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ".").replace("s", "").strip())
    except ValueError:
        return None


# --- E9: resumo de um clipe ---


class SummaryNote(BaseModel):
    """Destaque ou problema de um clipe: IDs de frases/momentos e o porquê."""

    ids: list[str] = Field(default_factory=list, description="IDs de frases (S012) ou momentos (P007.M04)")
    motivo: str = Field(default="", description="Por que é forte (destaque) ou o que está ruim (problema)")

    @field_validator("ids", mode="before")
    @classmethod
    def _ids(cls, value):
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, list):
            return value
        return [i for i in (_id(v) for v in value) if i]

    @field_validator("motivo", mode="before")
    @classmethod
    def _motivo(cls, value):
        return _text(value)


class ClipSummaryAnswer(BaseModel):
    """Resposta da IA para um clipe (ou uma parte de um clipe grande)."""

    resumo: str = Field(..., min_length=1, description="2 a 4 frases: do que trata, onde, quem")
    destaques: list[SummaryNote] = Field(default_factory=list, description=f"Até {MAX_HIGHLIGHTS} trechos fortes")
    problemas: list[SummaryNote] = Field(default_factory=list, description=f"Até {MAX_PROBLEMS} problemas")
    tom: str = Field(default="", description="Curto, ex.: divertido, informativo, contemplativo")
    tem_fala_para_camera: bool = Field(default=False, description="Alguém fala olhando para a câmera")

    @field_validator("resumo", "tom", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)

    @field_validator("destaques", mode="after")
    @classmethod
    def _highlights(cls, value):
        return value[:MAX_HIGHLIGHTS]

    @field_validator("problemas", mode="after")
    @classmethod
    def _problems(cls, value):
        return value[:MAX_PROBLEMS]


class ClipSummary(BaseModel):
    """Resumo de um clipe como o código guarda: a resposta (juntada, se o clipe foi em partes) e de onde veio."""

    clip_id: str
    answer: Optional[ClipSummaryAnswer] = None
    origin: Literal["ia", "cache", "falha"] = "ia"
    parts: int = Field(default=1, ge=1, description="Chamadas em que o clipe foi dividido (seção maior que o orçamento)")
    cached_parts: int = 0
    error: str = ""
    warnings: list[str] = Field(default_factory=list)


# --- E10: passada 1 (estrutura) ---


class StructureBlock(BaseModel):
    id: str = Field(default="", description="B01, B02... (o código renumera na ordem final)")
    titulo: str = ""
    assunto: str = ""
    clipes: list[str] = Field(default_factory=list, description="IDs de clipes, em ordem")
    duracao_alvo_sec: Optional[float] = None

    @field_validator("clipes", mode="before")
    @classmethod
    def _clips(cls, value):
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, list):
            return value
        return [i for i in (_id(v) for v in value) if i]

    @field_validator("titulo", "assunto", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)

    @field_validator("duracao_alvo_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)


class OutClip(BaseModel):
    clipe: str = ""
    motivo: str = ""

    @field_validator("clipe", mode="before")
    @classmethod
    def _clip(cls, value):
        return _id(value) or ""

    @field_validator("motivo", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


class StructureAnswer(BaseModel):
    """Resposta da passada 1: blocos na ordem final, clipes fora e a duração alvo."""

    blocos: list[StructureBlock] = Field(default_factory=list)
    fora: list[OutClip] = Field(default_factory=list)
    duracao_total_alvo_sec: Optional[float] = None
    notas: str = ""

    @field_validator("duracao_total_alvo_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)

    @field_validator("notas", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


# --- E11: passada 2 (detalhe de um bloco) ---


class KeepAnswer(BaseModel):
    """Item da faixa principal: faixa de frases (`de`/`ate`) OU um momento sem fala (`momento`)."""

    de: Optional[str] = None
    ate: Optional[str] = None
    momento: Optional[str] = None
    duracao_sec: Optional[float] = None

    @field_validator("de", "ate", "momento", mode="before")
    @classmethod
    def _ids(cls, value):
        return _id(value)

    @field_validator("duracao_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)


class CoverageAnswer(BaseModel):
    de: Optional[str] = None
    ate: Optional[str] = None
    momentos: list[str] = Field(default_factory=list)
    evidencia_fala: str = ""
    evidencia_imagem: str = ""
    motivo: str = ""

    @field_validator("de", "ate", mode="before")
    @classmethod
    def _ids(cls, value):
        return _id(value)

    @field_validator("momentos", mode="before")
    @classmethod
    def _moments(cls, value):
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, list):
            return value
        return [i for i in (_id(v) for v in value) if i]

    @field_validator("evidencia_fala", "evidencia_imagem", "motivo", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


class BreathAnswer(BaseModel):
    depois_de: Optional[str] = None
    momento: Optional[str] = None
    duracao_sec: Optional[float] = None
    motivo: str = ""

    @field_validator("depois_de", "momento", mode="before")
    @classmethod
    def _ids(cls, value):
        return _id(value)

    @field_validator("duracao_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)

    @field_validator("motivo", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


class BlockAnswer(BaseModel):
    """Resposta da passada 2 para um bloco (ou sub-bloco)."""

    manter: list[KeepAnswer] = Field(default_factory=list)
    coberturas: list[CoverageAnswer] = Field(default_factory=list)
    respiros: list[BreathAnswer] = Field(default_factory=list)
    notas: str = ""

    @field_validator("notas", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


# --- o que o código registra ---


class CallStats(BaseModel):
    """Chamadas e tokens de uma etapa (somando novas tentativas e correções)."""

    calls: int = 0
    cached: int = 0
    api_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    seconds: float = 0.0

    def add(self, other: "CallStats") -> None:
        self.calls += other.calls
        self.cached += other.cached
        self.api_calls += other.api_calls
        for name in ("input_tokens", "output_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, (getattr(self, name) or 0) + value)
        self.seconds = round(self.seconds + other.seconds, 3)


class Discarded(BaseModel):
    """Item da resposta recusado pela validação (só segurança), com o motivo."""

    kind: str = Field(..., description="bloco, fora, manter, cobertura, respiro")
    item: str = Field(..., description="O item como a IA mandou, resumido")
    reason: str


class Block(BaseModel):
    """Bloco da estrutura já validado (IDs que existem, cada clipe uma vez)."""

    id: str = Field(..., description="B01, B02... na ordem final")
    title: str = ""
    subject: str = ""
    clip_ids: list[str]
    target_sec: Optional[float] = None


class StructureRun(BaseModel):
    origin: Origin
    blocks: list[Block] = Field(default_factory=list)
    out: list[OutClip] = Field(default_factory=list)
    library_clip_ids: list[str] = Field(default_factory=list, description="Acervo: clipes nem em bloco nem fora")
    target_total_sec: Optional[float] = None
    target_origin: Literal["pedido", "ia", "nenhuma"] = "nenhuma"
    notes: str = ""
    discarded: list[Discarded] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list, description="Erro de cada tentativa recusada")
    prompt_mode: Literal["completo", "compacto"] = "completo"
    warnings: list[str] = Field(default_factory=list)
    stats: CallStats = Field(default_factory=CallStats)


class KeepRange(BaseModel):
    """Item aceito da faixa principal (frases de um clipe, em ordem, ou um momento)."""

    kind: Literal["frases", "momento"]
    clip_id: str
    from_sentence: Optional[str] = None
    to_sentence: Optional[str] = None
    sentence_ids: list[str] = Field(default_factory=list)
    moment_id: Optional[str] = None
    duration_sec: Optional[float] = Field(default=None, description="Pedida pela IA (momento); None = inteiro")

    @property
    def key(self) -> str:
        """ID do item (o que `depois_de` cita): a frase final ou o momento."""
        return self.to_sentence if self.kind == "frases" else self.moment_id


class CoverageChoice(BaseModel):
    from_sentence: str
    to_sentence: str
    sentence_ids: list[str]
    clip_id: str = Field(..., description="Clipe que está falando")
    moment_ids: list[str]
    speech_evidence: str
    image_evidence: str
    reason: str = ""
    matched_speech: list[str] = Field(default_factory=list, description="Palavras da evidência achadas na fala")
    matched_image: dict[str, list[str]] = Field(default_factory=dict, description="Por momento, palavras achadas")


class BreathChoice(BaseModel):
    after: str = Field(..., description="Frase (S015) ou momento da faixa principal depois do qual entra")
    moment_id: str
    duration_sec: Optional[float] = None
    reason: str = ""


class DetailRun(BaseModel):
    """Passada 2 de um bloco ou sub-bloco: o que a IA pediu, o que foi aceito e o que foi descartado."""

    unit_id: str = Field(..., description="B01, ou B01a/B01b quando o bloco foi dividido para caber")
    block_id: str
    clip_ids: list[str]
    moment_ids: Optional[list[str]] = Field(default=None, description="Só os momentos deste pedaço (clipe gigante)")
    origin: Origin
    keep: list[KeepRange] = Field(default_factory=list)
    coverages: list[CoverageChoice] = Field(default_factory=list)
    breaths: list[BreathChoice] = Field(default_factory=list)
    notes: str = ""
    discarded: list[Discarded] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list)
    library_offered: list[str] = Field(default_factory=list, description="Momentos do acervo pré-selecionados")
    used_before_sec: float = Field(default=0.0, description="Duração já usada pelos blocos anteriores (estimada)")
    estimated_tokens: int = 0
    warnings: list[str] = Field(default_factory=list)
    stats: CallStats = Field(default_factory=CallStats)


class BudgetInfo(BaseModel):
    """Como o orçamento de contexto foi calculado (`editor/budget.py`)."""

    num_ctx: int
    source: str = Field(..., description="De onde veio o num_ctx (ollama, --editor-ctx, padrão da nuvem)")
    summary_output: int
    structure_output: int
    detail_output: int
    margin: int
    token_factor: float


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EditorRun(BaseModel):
    """Conteúdo do `editor_run.json`: as três passadas, avisos, chamadas e tokens."""

    schema_version: int = EDITOR_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=_now)
    request: str = ""
    target_sec: Optional[float] = None
    provider: str
    model_chain: list[str] = Field(default_factory=list)
    effective_model: Optional[str] = None
    model_switches: list[ModelSwitch] = Field(default_factory=list)
    budget: BudgetInfo
    summaries: list[ClipSummary] = Field(default_factory=list)
    structure: Optional[StructureRun] = None
    details: list[DetailRun] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary_stats: CallStats = Field(default_factory=CallStats)
    detail_stats: CallStats = Field(default_factory=CallStats)

    @property
    def total_stats(self) -> CallStats:
        total = CallStats()
        for stats in (self.summary_stats, self.structure.stats if self.structure else CallStats(), self.detail_stats):
            total.add(stats)
        return total

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EditorRun":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# --- E12: o plano em IDs convertido em tempos (`editor_plano.json`) ---


class PlanPiece(BaseModel):
    """Um item da Timeline (base ou overlay) e de onde ele veio."""

    kind: Literal["frases", "momento", "respiro", "cobertura"]
    clip_id: str
    ids: list[str] = Field(default_factory=list, description="Frases ou momento de origem")
    source_in: float
    source_out: float
    timeline_start: float
    timeline_end: float


class PlanCoverage(BaseModel):
    from_sentence: str
    to_sentence: str
    moment_ids: list[str]
    speech_evidence: str
    image_evidence: str
    reason: str = ""
    timeline_start: float
    timeline_end: float
    pieces: list[PlanPiece] = Field(default_factory=list)
    uncovered_sec: float = 0.0


class PlanBlock(BaseModel):
    id: str
    title: str = ""
    clip_ids: list[str]
    origin: str = Field(..., description="ia, cache, fallback (por sub-bloco: a pior)")
    timeline_start: float = 0.0
    timeline_end: float = 0.0
    pieces: list[PlanPiece] = Field(default_factory=list)
    coverages: list[PlanCoverage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EditorPlan(BaseModel):
    """Conteúdo do `editor_plano.json`: blocos na ordem, cada item com os IDs e os tempos."""

    schema_version: int = EDITOR_SCHEMA_VERSION
    project_name: str
    fps: int
    resolution: tuple[int, int]
    duration_sec: float
    target_sec: Optional[float] = None
    blocks: list[PlanBlock] = Field(default_factory=list)
    out: list[OutClip] = Field(default_factory=list)
    library_clip_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EditorPlan":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
