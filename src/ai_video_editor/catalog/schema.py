"""Schemas Pydantic do catálogo de um projeto com clipes (`catalog.json`)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

# Sobe quando o formato do catalog.json muda.
CATALOG_SCHEMA_VERSION = 2

Role = Literal["principal", "drone", "pov", "broll", "outro"]
ROLES: tuple[Role, ...] = ("principal", "drone", "pov", "broll", "outro")
ROLE_PREFIX: dict[Role, str] = {"principal": "P", "drone": "D", "pov": "V", "broll": "B", "outro": "X"}

# padrão: nome da pasta igual a um sinônimo; contém: sinônimo como palavra do nome;
# --papel: definido pelo usuário; raiz: arquivos soltos na pasta do projeto.
RoleSource = Literal["padrão", "contém", "--papel", "raiz"]

# data: ordem pela data de gravação; nome: datas não confiáveis (parecem de exportação), ordem natural do nome.
OrderSource = Literal["data", "nome"]

# Suspeita de perfil de cor log (imagem "lavada"): só quando contraste E saturação estão baixos ao
# mesmo tempo. Valores normalizados 0–1 sobre o pico da profundidade de bits do clipe.
# - Faixa de luma = YHIGH − YLOW do signalstats (percentis 90 e 10).
# - Saturação = SATAVG.
# Estimativa inicial (a calibrar com o material real; o catalog.md mostra os dois valores por clipe):
# vídeo comum ≈0,35–0,75 de faixa e ≈0,06–0,20 de saturação; log ≈0,2–0,3 e ≈0,01–0,04. Os limites
# antigos (0,45 e 0,08) pegavam vídeo comum: cena clara ou pouco colorida já ficava abaixo dos dois.
LOG_MAX_LUMA_SPREAD = 0.30
LOG_MAX_SATURATION = 0.04

# container: tag do container (format); stream: tag da trilha de vídeo; quicktime: com.apple.quicktime.creationdate;
# filename: padrão de data no nome do arquivo; mtime: data de modificação (pouco confiável).
DateSource = Literal["quicktime", "container", "stream", "filename", "mtime"]

# analisada: VAD rodou; desconhecido: VAD indisponível ou falhou; sem_audio: o clipe não tem áudio.
SpeechStatus = Literal["analisada", "desconhecido", "sem_audio"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ClipMedia(BaseModel):
    """Dados técnicos do clipe (ffprobe). Largura/altura já com a rotação aplicada."""

    duration_sec: float = Field(ge=0.0)
    width: int
    height: int
    fps: float
    codec: str
    pix_fmt: Optional[str] = None
    bit_depth: Optional[int] = None
    rotation: int = 0
    has_audio: bool
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = Field(default=None, description="Bitrate total do arquivo, bits/s")

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


class RecordingDate(BaseModel):
    """Data de gravação e de onde ela veio. `value` em UTC (com fuso)."""

    value: datetime
    source: DateSource
    raw: str = Field(description="Texto original (tag, nome do arquivo ou mtime)")
    reliable: bool = True


class SpeechSegment(BaseModel):
    start_sec: float
    end_sec: float


class SpeechDetection(BaseModel):
    status: SpeechStatus
    ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Fração da duração com fala")
    speech_sec: Optional[float] = None
    segments: list[SpeechSegment] = Field(default_factory=list)
    detector: str = Field(default="", description="Detector que produziu o resultado (entra no cache)")
    note: str = ""


class ColorSignals(BaseModel):
    """Sinais simples de cor (signalstats em poucos frames). São suspeitas, não conclusões."""

    frames: int
    luma_spread: float = Field(description="(YHIGH − YLOW) normalizado 0–1, média dos frames")
    saturation: float = Field(description="SATAVG normalizado 0–1, média dos frames")

    @computed_field
    @property
    def log_suspect(self) -> bool:
        """Calculado dos valores (não fica no cache): mudar os limites vale para clipes já analisados."""
        return self.luma_spread < LOG_MAX_LUMA_SPREAD and self.saturation < LOG_MAX_SATURATION


class ClipAnalysis(BaseModel):
    """O que é calculado por clipe e fica no cache (independe de papel, ID e ajuste de relógio)."""

    format_version: int
    path: Path
    size_bytes: int
    mtime_ns: int
    media: Optional[ClipMedia] = None
    error: Optional[str] = Field(default=None, description="Arquivo ilegível ou sem vídeo")
    tag_date: Optional[RecordingDate] = None
    color: Optional[ColorSignals] = None
    thumbnails: list[str] = Field(default_factory=list, description="Arquivos na pasta do cache do clipe")
    speech: Optional[SpeechDetection] = None
    warnings: list[str] = Field(default_factory=list)


class CatalogClip(BaseModel):
    id: Optional[str] = Field(default=None, description="P001, D001...; None em arquivo com problema")
    role: Role
    folder: str = Field(description="Subpasta de primeiro nível (vazio = raiz)")
    relative_path: str
    size_bytes: int
    media: Optional[ClipMedia] = None
    recorded_at: Optional[RecordingDate] = None
    clock_offset_sec: float = 0.0
    adjusted_at: Optional[datetime] = Field(default=None, description="recorded_at + ajuste de relógio do papel")
    speech: Optional[SpeechDetection] = None
    color: Optional[ColorSignals] = None
    thumbnails: list[str] = Field(default_factory=list, description="Relativas à pasta catalog/")
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    order_source: OrderSource = Field(default="data", description="O que definiu a posição do clipe no papel")
    cached: bool = Field(default=False, description="Análise reaproveitada do cache nesta execução")


class RoleAssignment(BaseModel):
    folder: str
    role: Role
    source: RoleSource


class StageTime(BaseModel):
    name: str
    elapsed_sec: float


class ProjectCatalog(BaseModel):
    schema_version: int = CATALOG_SCHEMA_VERSION
    source_dir: Path
    project_dir: Path
    listing_hash: str
    created_at: datetime = Field(default_factory=_now)
    roles: list[RoleAssignment] = Field(default_factory=list)
    clock_adjustments: dict[str, float] = Field(default_factory=dict, description="papel → segundos")
    clips: list[CatalogClip] = Field(default_factory=list)
    ignored_files: int = 0
    warnings: list[str] = Field(default_factory=list)
    contact_sheets: dict[str, str] = Field(default_factory=dict, description="papel → arquivo")
    stage_times: list[StageTime] = Field(default_factory=list)
    total_sec: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectCatalog":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def clips_of(self, role: str) -> list[CatalogClip]:
        return [c for c in self.clips if c.role == role and c.error is None]
