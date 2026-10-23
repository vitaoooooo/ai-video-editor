"""Schemas Pydantic do pipeline de edição e do manifesto do projeto."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ai_video_editor.core.schema import LoudnessPreset
from ai_video_editor.director.schema import DirectorThinking, SentenceParams, UserBrief
from ai_video_editor.llm.chain import resolve_chain
from ai_video_editor.llm.schema import ModelSpec
from ai_video_editor.planning.schema import PausePreset, SilenceCutParams, apply_pause_preset
from ai_video_editor.subtitles.schema import SubtitleParams
from ai_video_editor.transcription.schema import TranscriptionConfig

StageName = Literal["probe", "silences", "transcription", "director", "plan", "render", "subtitles", "report"]
STAGES: tuple[StageName, ...] = (
    "probe",
    "silences",
    "transcription",
    "director",
    "plan",
    "render",
    "subtitles",
    "report",
)

StageStatus = Literal["pending", "done", "skipped", "failed"]
# Resultado de uma etapa nesta execução: executada, reaproveitada do cache ou não pedida.
StageOutcome = Literal["ran", "cached", "skipped"]

MetricValue = Union[int, float, str]


class EditPipelineParams(BaseModel):
    """Todos os parâmetros que influenciam o resultado do comando `edit`."""

    threshold_db: Union[Literal["auto"], float] = Field(
        default="auto", description="Limiar de silêncio em dB, ou `auto` para medir o ruído de cada vídeo"
    )
    min_silence_sec: float = Field(default=0.5, gt=0.0, description="Duração mínima do silêncio")
    cut: SilenceCutParams = Field(default_factory=SilenceCutParams)
    pacing: PausePreset = Field(
        default="normal", description="Ritmo das pausas aplicado sobre `cut`: natural, normal ou tight"
    )
    transcribe: bool = Field(default=False, description="Gera transcript.json (necessário para o Diretor)")
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    director: str = Field(
        default="none",
        description="Diretor de IA que remove frases: `none`, `mock` ou um provedor registrado (`gemini`, `openai`)",
    )
    director_model: Optional[str] = Field(default=None, description="Modelo do Diretor (padrão: o do provedor)")
    director_fallback: Optional[str] = Field(
        default=None,
        description="Modelos de reserva `provedor:modelo,...`; `none` desliga; vazio usa a reserva padrão do provedor",
    )
    director_thinking: DirectorThinking = Field(
        default="low", description="Nível de raciocínio do Diretor (off/low/high)"
    )
    director_timeout_sec: float = Field(default=120.0, gt=5.0, description="Tempo limite da chamada ao Diretor")
    api_retries: int = Field(
        default=4, ge=0, description="Novas tentativas da API em falhas temporárias (0 desliga)"
    )
    sentences: SentenceParams = Field(default_factory=SentenceParams)
    brief: UserBrief = Field(default_factory=UserBrief)
    subtitles: bool = Field(default=True, description="Gera o .srt quando há transcrição")
    subtitle: SubtitleParams = Field(default_factory=SubtitleParams)
    transcript_min_pause_sec: float = Field(
        default=1.0,
        ge=0.0,
        description="Pausas removidas a partir desta duração ganham marcador na transcrição comparativa",
    )
    loudness: LoudnessPreset = Field(
        default="voz", description="Normalização de volume do vídeo final (EBU R128): voz/youtube/broadcast/off"
    )

    @model_validator(mode="after")
    def _check_director(self) -> "EditPipelineParams":
        if self.director not in ("none", "mock"):
            self.director_chain()  # valida provedor e reservas
        return self

    def director_chain(self) -> list[ModelSpec]:
        """Cadeia pedida do Diretor: o modelo principal seguido das reservas."""
        return resolve_chain(self.director, self.director_model, self.director_fallback)

    @property
    def effective_cut(self) -> SilenceCutParams:
        """`cut` com o ritmo das pausas aplicado (`normal` = o próprio `cut`)."""
        return apply_pause_preset(self.cut, self.pacing)

    def fingerprint_view(self) -> dict:
        """Os parâmetros como entram no fingerprint do relatório. O ritmo `normal` fica de fora: os
        projetos de antes do ritmo no `edit` continuam com o mesmo fingerprint."""
        return self.model_dump(mode="json", exclude={"pacing"} if self.pacing == "normal" else None)


# Origem de um valor efetivo do `edit`: opção explícita da CLI, briefing do Produtor ou padrão do código.
SettingOrigin = Literal["cli", "briefing", "padrao"]
# gerado: o Produtor rodou nesta execução; reaproveitado: `briefing.json` atual (mesmo pedido e material);
# desligado: `--sem-briefing`; sem_briefing: sem pedido e sem `briefing.json` atual.
BriefingUse = Literal["gerado", "reaproveitado", "desligado", "sem_briefing"]

SETTING_ORIGIN_LABELS: dict[str, str] = {"cli": "opção da CLI", "briefing": "briefing", "padrao": "padrão"}


class EditSetting(BaseModel):
    """Um valor efetivo do `edit` e de onde ele veio."""

    value: Optional[Union[float, str]] = None
    origin: SettingOrigin

    @property
    def origin_label(self) -> str:
        return SETTING_ORIGIN_LABELS[self.origin]


class EditSettings(BaseModel):
    """Valores efetivos do `edit` e a origem de cada um (opção da CLI > briefing > padrão)."""

    briefing: BriefingUse = "sem_briefing"
    briefing_origin: Optional[str] = Field(default=None, description="`ia` ou `fallback` (do briefing)")
    request: Optional[str] = Field(default=None, description="Pedido que gerou o briefing")
    understood: Optional[str] = Field(default=None, description="O \"Entendi: ...\" do briefing")
    recipe: EditSetting
    intensity: EditSetting
    pace: EditSetting
    instructions: EditSetting = Field(description="Texto que chega ao Diretor como pedido do usuário")
    target_duration_sec: EditSetting
    notes: list[str] = Field(default_factory=list, description="Avisos sobre o briefing (ignorado, refeito...)")

    @property
    def uses_briefing(self) -> bool:
        return self.briefing in ("gerado", "reaproveitado")

    def fingerprint_view(self) -> dict:
        """O que o relatório mostra: sem o estado do briefing (gerado/reaproveitado) nem os avisos, para um
        briefing igual não refazer o relatório."""
        return self.model_dump(mode="json", exclude={"briefing", "notes"})


def settings_fingerprint(settings: Optional[EditSettings]) -> tuple:
    """Parte do fingerprint do relatório: os valores e origens só quando há briefing (é quando o relatório
    os mostra); sem briefing, o fingerprint de antes."""
    return (settings.fingerprint_view(),) if settings is not None and settings.uses_briefing else ()


class StageRecord(BaseModel):
    """Estado persistido de uma etapa no manifesto."""

    status: StageStatus = "pending"
    fingerprint: Optional[str] = Field(
        default=None, description="Hash das entradas e parâmetros que afetam a etapa"
    )
    artifact: Optional[str] = Field(default=None, description="Artefato relativo à pasta do projeto")
    updated_at: Optional[datetime] = None
    elapsed_sec: Optional[float] = None
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    error: Optional[str] = None


class StageTiming(BaseModel):
    """Tempo de relógio de uma etapa em uma execução (sem tempo se veio do cache ou foi pulada)."""

    name: str = Field(..., description="Nome da etapa (`probe`, `render`, `reviser`...)")
    status: StageOutcome
    elapsed_sec: Optional[float] = Field(default=None, description="Só em etapas executadas")


class StageProgress(BaseModel):
    """Andamento de uma etapa em curso, para exibição ao vivo (não é gravado no projeto).

    `done`/`total` em segundos de mídia (`unit="sec"`, transcrição), em itens
    (`unit="items"`, trechos do render) ou em clipes (`unit="clips"`, comando `catalog`). `note` é um texto livre, ex: modelo e tentativa do Diretor.
    """

    done: Optional[float] = None
    total: Optional[float] = None
    unit: Literal["sec", "items", "clips"] = "sec"
    note: str = ""

    @property
    def fraction(self) -> Optional[float]:
        """Fração concluída (0–1) ou None sem progresso mensurável."""
        if self.done is None or not self.total or self.total <= 0:
            return None
        return min(1.0, max(0.0, self.done / self.total))


class RunTimings(BaseModel):
    """Tempos de uma execução do `edit` ou do `revise`."""

    stages: list[StageTiming] = Field(default_factory=list)
    total_sec: float = Field(..., description="Tempo de relógio da execução inteira")
    video_duration_sec: Optional[float] = Field(default=None, description="Duração do vídeo original")


class VersionRecord(BaseModel):
    """Uma versão do projeto (`versions/vN/`): o resultado do `edit` (v1) ou de uma revisão."""

    number: int = Field(..., ge=1)
    kind: Literal["edit", "revision"]
    request: str = Field(..., description="Pedido que originou a versão")
    parent: Optional[int] = Field(default=None, description="Versão em que a revisão se baseou")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = Field(default=None, description="Resumo do revisor (só em revisões)")
    cut: SilenceCutParams = Field(
        default_factory=SilenceCutParams, description="Parâmetros de corte do `edit` (antes do preset)"
    )
    pacing: PausePreset = "normal"
    loudness: Optional[LoudnessPreset] = Field(
        default=None, description="Normalização de volume da versão (None: versão anterior à normalização)"
    )
    duration_sec: float
    cuts: int = Field(..., description="Número de trechos removidos do vídeo original")
    segments: int
    removed_sentences: int = 0
    video: str
    subtitles: Optional[str] = None
    edit_fp: Optional[str] = Field(
        default=None, description="Fingerprint do render/legendas do `edit` que gerou a versão"
    )
    timings: Optional[RunTimings] = Field(
        default=None, description="Tempos da execução que gerou a versão"
    )


class ProjectManifest(BaseModel):
    """Conteúdo do `project.json`."""

    video_path: Path
    video_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    params: EditPipelineParams
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {name: StageRecord() for name in STAGES}
    )
    versions: list[VersionRecord] = Field(default_factory=list)
    current_version: Optional[int] = Field(default=None, description="Número da versão atual")
    last_run: Optional[RunTimings] = Field(
        default=None, description="Tempos da última execução de `edit` ou `revise`"
    )
    settings: Optional[EditSettings] = Field(
        default=None, description="Valores efetivos do último `edit` pela CLI e a origem de cada um"
    )

    @field_validator("stages")
    @classmethod
    def _fill_missing_stages(cls, stages: dict[StageName, StageRecord]) -> dict[StageName, StageRecord]:
        """Manifestos antigos não têm as etapas novas: entram como pendentes."""
        return {name: stages.get(name, StageRecord()) for name in STAGES}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ProjectManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class EditPipelineResult(BaseModel):
    """Resumo de uma execução do pipeline."""

    project_dir: Path
    manifest_path: Path
    output_path: Path
    report_path: Path
    subtitles_path: Optional[Path] = Field(
        default=None, description="Legenda .srt gerada ou reaproveitada nesta execução"
    )
    transcript_path: Optional[Path] = Field(
        default=None, description="Transcrição comparativa (`edit_transcript.md`), quando há transcrição"
    )
    version: Optional[int] = Field(default=None, description="Versão do projeto que corresponde ao resultado")
    original_duration_sec: float
    final_duration_sec: float
    outcomes: dict[str, StageOutcome] = Field(..., description="Resultado de cada etapa da receita")
    timings: Optional[RunTimings] = None
    warnings: list[str] = Field(default_factory=list)


# --- Transcrição comparativa (`edit_diff.json`) ---

DiffEntryType = Literal["sentence", "pause"]
# kept: frase sem pedido de remoção. As demais são o status de aplicação de uma remoção pedida:
# applied (>= 90% saiu), partial (saiu uma parte) ou not_applied (a frase continua no vídeo).
SentenceDiffStatus = Literal["kept", "applied", "partial", "not_applied"]
RemovalOrigin = Literal["director", "revision"]
RevisionChange = Literal["restored", "removed"]

_ORIGIN_LABELS: dict[str, str] = {"director": "Diretor", "revision": "revisão"}


def format_seconds(seconds: float, digits: int = 2) -> str:
    """`0.19` -> `0,19s`."""
    return f"{seconds:.{digits}f}".replace(".", ",") + "s"


class DiffEntry(BaseModel):
    """Uma frase ou pausa removida da transcrição comparativa, nos tempos do original e do editado."""

    type: DiffEntryType
    original_start_sec: float = Field(..., ge=0.0)
    original_end_sec: float = Field(..., ge=0.0)
    edited_start_sec: Optional[float] = Field(
        default=None, description="Onde o item começa no vídeo editado; None se nada dele ficou"
    )
    edited_end_sec: Optional[float] = Field(
        default=None, description="Onde o item termina no vídeo editado; em pausas, igual ao início"
    )
    sentence_id: Optional[int] = None
    text: Optional[str] = None
    status: Optional[SentenceDiffStatus] = Field(default=None, description="Só em frases")
    removed_sec: float = Field(default=0.0, ge=0.0, description="Quanto do intervalo saiu do vídeo")
    removed_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    origin: Optional[RemovalOrigin] = Field(default=None, description="Quem pediu a remoção")
    category: Optional[str] = None
    reason: Optional[str] = Field(default=None, description="Motivo da remoção")
    application_note: Optional[str] = Field(
        default=None, description="Por que a remoção não foi aplicada ou foi só parcial"
    )
    revision_change: Optional[RevisionChange] = Field(
        default=None, description="Mudança desta revisão: frase restaurada ou removida agora"
    )
    restoration_reason: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "DiffEntry":
        if self.original_end_sec < self.original_start_sec:
            raise ValueError("original_end_sec deve ser maior ou igual a original_start_sec")
        if self.type == "sentence" and (self.sentence_id is None or self.text is None or self.status is None):
            raise ValueError("frases precisam de sentence_id, text e status")
        return self

    @property
    def duration_sec(self) -> float:
        return round(self.original_end_sec - self.original_start_sec, 3)

    @property
    def removal_requested(self) -> bool:
        return self.type == "sentence" and self.status != "kept"

    @property
    def origin_label(self) -> str:
        return _ORIGIN_LABELS.get(self.origin or "director", "Diretor")

    def application_label(self) -> str:
        """`aplicada`, `parcial (0,19s de 0,74s, 26%)` ou `não aplicada: motivo`; vazio se mantida."""
        if self.status == "applied":
            return "aplicada"
        if self.status == "partial":
            label = (
                f"parcial ({format_seconds(self.removed_sec)} de {format_seconds(self.duration_sec)}, "
                f"{self.removed_pct:.0f}%)"
            )
            return f"{label}: {self.application_note}" if self.application_note else label
        if self.status == "not_applied":
            return f"não aplicada: {self.application_note}" if self.application_note else "não aplicada"
        return ""


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


class EditDiff(BaseModel):
    """Conteúdo do `edit_diff.json`: a fala original inteira com o que saiu, o que ficou e por quê."""

    video_name: str
    version: Optional[int] = Field(default=None, description="Versão de revisão; None no `edit`")
    base_version: Optional[int] = Field(default=None, description="Versão em que a revisão se baseou")
    request: Optional[str] = Field(default=None, description="Pedido da revisão")
    original_duration_sec: float = Field(..., ge=0.0)
    final_duration_sec: float = Field(..., ge=0.0)
    removed_pause_sec: float = Field(
        default=0.0, ge=0.0, description="Tempo removido fora das frases marcadas para remoção"
    )
    min_pause_marker_sec: float = Field(
        default=1.0, ge=0.0, description="Pausas a partir desta duração aparecem como item"
    )
    entries: list[DiffEntry] = Field(default_factory=list, description="Itens em ordem cronológica")

    @property
    def sentences(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.type == "sentence"]

    def count(self, status: SentenceDiffStatus) -> int:
        return sum(1 for e in self.sentences if e.status == status)

    def entry_for(self, sentence_id: int) -> Optional[DiffEntry]:
        return next((e for e in self.sentences if e.sentence_id == sentence_id), None)

    def application_counts(self) -> str:
        """`5 aplicadas, 1 parcial, 1 não aplicada`."""
        return ", ".join(
            [
                _plural(self.count("applied"), "aplicada", "aplicadas"),
                _plural(self.count("partial"), "parcial", "parciais"),
                _plural(self.count("not_applied"), "não aplicada", "não aplicadas"),
            ]
        )

    def removal_counts(self) -> str:
        """`7 remoções pedidas, 5 aplicadas, 1 parcial, 1 não aplicada`."""
        requested = sum(1 for e in self.sentences if e.removal_requested)
        return f"{_plural(requested, 'remoção pedida', 'remoções pedidas')}, {self.application_counts()}"
