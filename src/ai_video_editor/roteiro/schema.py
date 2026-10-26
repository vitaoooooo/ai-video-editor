"""Schemas Pydantic do roteiro unificado (`roteiro/roteiro.json`) e da resposta da IA de visão."""

import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ai_video_editor.catalog.schema import StageTime
from ai_video_editor.visual.schema import ShotMeasures

# Sobe quando o formato do roteiro.json muda.
ROTEIRO_SCHEMA_VERSION = 1

Volume = Literal["sussurro", "baixo", "normal", "alto"]
Speed = Literal["lenta", "normal", "rápida"]
# intervalo: foto da grade fixa; frase: início de frase depois de pausa; cena: troca de cena.
PhotoReason = Literal["intervalo", "frase", "cena"]
PhotoIssue = Literal["nenhum", "borrada", "tremida", "obstruída", "escura", "clara", "tela uniforme"]
PHOTO_ISSUES: tuple[str, ...] = ("nenhum", "borrada", "tremida", "obstruída", "escura", "clara", "tela uniforme")
# descrita: tem descrição; sem_descricao: a IA falhou (pedida de novo na próxima execução);
# pendente: a IA não foi chamada (--so-estimar ou --visao-provider none).
PhotoStatus = Literal["descrita", "sem_descricao", "pendente"]

MAX_TAGS = 6


def _plain(text: str) -> str:
    """Minúsculas, sem acentos e sem espaços nas pontas (para comparar respostas da IA)."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Respostas comuns da IA que querem dizer um dos problemas aceitos.
_ISSUE_SYNONYMS = {
    **{_plain(issue): issue for issue in PHOTO_ISSUES},
    "nenhuma": "nenhum", "none": "nenhum", "": "nenhum",
    "desfocada": "borrada", "borrao": "borrada", "blur": "borrada", "blurry": "borrada",
    "obstruida": "obstruída", "obstruction": "obstruída", "tapada": "obstruída",
    "escuro": "escura", "dark": "escura", "clara demais": "clara", "estourada": "clara", "bright": "clara",
    "tela_uniforme": "tela uniforme", "uniforme": "tela uniforme", "shaky": "tremida", "tremido": "tremida",
}


class PhotoDescription(BaseModel):
    """O que a IA de visão diz de uma foto (em português)."""

    what: str = Field(..., min_length=1, description="O que aparece na foto, em uma frase curta")
    place: str = Field(default="", description="Lugar (ex.: praia, quarto de hotel, trilha); vazio se não der para saber")
    people: int = Field(default=0, ge=0, description="Quantas pessoas aparecem")
    talking_to_camera: bool = Field(default=False, description="Alguém olha e fala para a câmera")
    action: str = Field(default="", description="O que está acontecendo, em poucas palavras")
    tags: list[str] = Field(..., min_length=1, max_length=MAX_TAGS, description="1 a 6 palavras curtas, minúsculas")
    issue: PhotoIssue = Field(
        default="nenhum",
        description="Problema da imagem: nenhum, borrada, tremida, obstruída, escura, clara ou tela uniforme",
    )

    @field_validator("what", "place", "action", mode="before")
    @classmethod
    def _strip(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value):
        """Minúsculas, sem repetição e sem vazias; mais de 6 são cortadas (não vale uma nova chamada)."""
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list):
            return value
        tags: list[str] = []
        for item in value:
            tag = " ".join(str(item).strip().lower().split())
            if tag and tag not in tags:
                tags.append(tag)
        return tags[:MAX_TAGS]

    @field_validator("issue", mode="before")
    @classmethod
    def _issue(cls, value):
        if isinstance(value, list):  # a IA às vezes devolve uma lista: vale o primeiro problema real
            value = next((v for v in value if _plain(str(v)) not in ("nenhum", "nenhuma", "")), "nenhum")
        if isinstance(value, str):
            return _ISSUE_SYNONYMS.get(_plain(value), value)
        return value


class SentenceRhythm(BaseModel):
    """Ritmo de uma frase, medido pelo programa. Os valores medidos ficam junto das classes."""

    volume: Optional[Volume] = Field(default=None, description="Relativo à mediana das frases do clipe; None sem medida")
    level_db: Optional[float] = Field(default=None, description="Nível da frase (percentil dos picos de 50 ms), dBFS")
    relative_db: Optional[float] = Field(default=None, description="level_db − mediana das frases do clipe")
    speed: Optional[Speed] = Field(default=None, description="None em frase curta demais para medir")
    words: int = 0
    words_per_sec: Optional[float] = None
    pause_before_sec: float = 0.0
    pause_after_sec: float = 0.0


class RoteiroSentence(BaseModel):
    id: str = Field(..., description="S000, S001... (globais, na ordem do catálogo)")
    index: int = Field(..., ge=0)
    clip_id: str
    start_sec: float
    end_sec: float
    text: str
    rhythm: SentenceRhythm = Field(default_factory=SentenceRhythm)
    moment_ids: list[str] = Field(default_factory=list, description="Momentos que a frase atravessa")


class RoteiroPhoto(BaseModel):
    """Foto mantida (as quase iguais à anterior são descartadas e só contadas)."""

    id: str = Field(..., description="P007.F023")
    clip_id: str
    time_sec: float
    reasons: list[PhotoReason]
    file: str = Field(..., description="Relativo à pasta roteiro/")
    image_hash: str
    diff: Optional[float] = Field(default=None, description="Diferença para a foto mantida anterior (0–255)")
    speech: str = Field(default="", description="Palavras do Whisper até a próxima foto; vazio = sem fala")
    status: PhotoStatus = "pendente"
    description: Optional[PhotoDescription] = None
    note: str = ""


class DiscardedPhoto(BaseModel):
    time_sec: float
    reasons: list[PhotoReason]
    similar_to: str = Field(..., description="ID da foto mantida anterior")
    diff: float


class Moment(BaseModel):
    """Fotos seguidas do mesmo clipe com descrições parecidas, sem troca de cena entre elas."""

    id: str = Field(..., description="P007.M04")
    clip_id: str
    start_sec: float
    end_sec: float
    photo_ids: list[str] = Field(default_factory=list)
    description: str = ""
    place: str = ""
    people: int = 0
    talking_to_camera: bool = False
    tags: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list, description="Da IA e das medidas (borrão, tremida...)")
    measures: Optional[ShotMeasures] = None
    scene_change: bool = Field(default=False, description="Começa numa troca de cena")
    sentence_ids: list[str] = Field(default_factory=list, description="Frases que atravessam o momento")

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


class RoteiroClip(BaseModel):
    id: str
    role: str
    relative_path: str
    order: int = Field(..., ge=1, description="Posição na ordem do catálogo")
    duration_sec: float
    recorded_at: Optional[datetime] = Field(default=None, description="Só quando a data de gravação é confiável")
    has_audio: bool = True
    speech_status: Optional[str] = Field(default=None, description="Do catálogo: analisada, desconhecido, sem_audio")
    has_speech: bool = Field(default=False, description="Tem frases transcritas")
    transcribed: bool = False
    language: Optional[str] = None
    silence_total_sec: Optional[float] = None
    scene_changes: int = 0
    photos_before: int = Field(default=0, description="Fotos extraídas, antes do descarte")
    photos_kept: int = 0
    discarded: list[DiscardedPhoto] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, description="Tamanho da seção do clipe no roteiro.md (caracteres/4)")
    warnings: list[str] = Field(default_factory=list)


class VisionRun(BaseModel):
    """O que a IA de visão fez nesta execução."""

    provider: str
    model_chain: list[str] = Field(default_factory=list)
    effective_model: Optional[str] = None
    parallel: int = 1
    photos: int = 0
    cached: int = 0
    described: int = Field(default=0, description="Descritas agora (fora do cache)")
    failed: int = 0
    api_calls: int = 0
    corrections: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    wall_sec: float = 0.0
    sec_per_photo: Optional[float] = Field(default=None, description="Média das fotos descritas agora")


class RoteiroEstimate(BaseModel):
    """`--so-estimar`: volume de trabalho da IA de visão, sem chamá-la."""

    photos_before: int
    photos_kept: int
    cached: int
    to_describe: int
    calls: int
    input_tokens: int
    output_tokens: int
    seconds: float


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectRoteiro(BaseModel):
    schema_version: int = ROTEIRO_SCHEMA_VERSION
    project_dir: Path
    source: Path = Field(..., description="Arquivo ou pasta de clipes")
    kind: Literal["arquivo", "pasta"]
    created_at: datetime = Field(default_factory=_now)
    interval_sec: float
    rules: dict[str, float] = Field(default_factory=dict, description="Constantes usadas (estimativas a calibrar)")
    clips: list[RoteiroClip] = Field(default_factory=list)
    moments: list[Moment] = Field(default_factory=list)
    sentences: list[RoteiroSentence] = Field(default_factory=list)
    photos: list[RoteiroPhoto] = Field(default_factory=list)
    vision: Optional[VisionRun] = None
    motion_threshold: Optional[float] = None
    estimated_tokens: int = Field(default=0, description="Tamanho do roteiro.md (caracteres/4)")
    warnings: list[str] = Field(default_factory=list)
    stage_times: list[StageTime] = Field(default_factory=list)
    total_sec: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectRoteiro":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
