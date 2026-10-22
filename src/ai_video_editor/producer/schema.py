"""Schemas do Produtor: resumo do material, receitas oferecidas e o briefing (`briefing.json`)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_video_editor.catalog.schema import Role
from ai_video_editor.director.schema import Intensity
from ai_video_editor.intelligence.schema import AIRunMetrics
from ai_video_editor.llm.schema import ModelSwitch
from ai_video_editor.planning.schema import PausePreset

# Sobe quando o formato do briefing.json muda.
BRIEFING_SCHEMA_VERSION = 1

# ia: resposta validada do modelo; fallback: regra determinística do código (com o motivo).
BriefingOrigin = Literal["ia", "fallback"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RoleSummary(BaseModel):
    """Resumo de um papel do catálogo (sem lista de clipes nem caminhos)."""

    role: Role
    clips: int
    duration_sec: float
    speech_share: Optional[float] = Field(
        default=None, description="Fração do tempo com fala nos clipes analisados; None se desconhecida"
    )
    clips_with_speech: int = 0
    speech_unknown: int = Field(default=0, description="Clipes com áudio cuja fala não foi analisada")
    speech_clips_duration_sec: float = Field(
        default=0.0, description="Duração dos clipes com fala (ou fala desconhecida): o que seria transcrito"
    )
    formats: list[str] = Field(default_factory=list, description="Ex.: 1920x1080 30fps h264")
    log_suspects: int = 0


class MaterialSummary(BaseModel):
    """O que a IA vê do material: papéis com quantidades, durações, fala e formatos."""

    kind: Literal["arquivo", "pasta"]
    name: str = Field(description="Nome do arquivo ou da pasta (sem o caminho)")
    clips: int
    duration_sec: float
    roles: list[RoleSummary] = Field(default_factory=list)

    @property
    def role_names(self) -> list[str]:
        return [r.role for r in self.roles]

    def role(self, name: str) -> Optional[RoleSummary]:
        return next((r for r in self.roles if r.role == name), None)

    @property
    def clips_with_speech(self) -> int:
        return sum(r.clips_with_speech for r in self.roles)

    @property
    def speech_unknown(self) -> int:
        return sum(r.speech_unknown for r in self.roles)


class RecipeInfo(BaseModel):
    """Receita registrada, como o Produtor a enxerga (lida do registro em tempo de execução)."""

    name: str
    description: str
    steps: list[str]
    multi_clip: bool = Field(description="Edita vários clipes; False = 1 vídeo")
    needs_speech: bool = Field(description="Precisa de fala (transcrição)")


class BriefingContent(BaseModel):
    """O que o Produtor decide. É o schema pedido à IA (o código valida depois)."""

    video_type: str = Field(description="Tipo de vídeo, curto, ex.: 'vlog de viagem', 'aula'")
    recipe: str = Field(description="Nome de uma das receitas disponíveis")
    understood: str = Field(description="Uma frase em português começando com 'Entendi: '")
    target_duration_sec: Optional[float] = Field(
        default=None, description="Duração final desejada em segundos, só se o usuário pediu"
    )
    intensity: Intensity = Field(default="light", description="Quanto cortar da fala: light, medium ou aggressive")
    pace: PausePreset = Field(default="normal", description="Ritmo das pausas: natural, normal ou tight")
    director_instructions: str = Field(default="", description="Instruções para o Diretor de cortes da fala")
    coverage_instructions: str = Field(
        default="", description="Instruções para a cobertura (B-roll, drone, POV, som ambiente)"
    )
    roles_to_use: list[Role] = Field(default_factory=list, description="Papéis do catálogo a usar")
    warnings: list[str] = Field(default_factory=list, description="Dúvidas ou limitações, em português")


class WorkEstimate(BaseModel):
    """Estimativa de trabalho da edição (sem preços)."""

    clips: int = Field(description="Clipes nos papéis a usar")
    clips_to_transcribe: int
    audio_to_transcribe_sec: float = Field(description="Áudio a transcrever (clipes com fala nos papéis a usar)")
    ai_steps: list[str] = Field(default_factory=list, description="Etapas de IA da receita")
    ai_calls: int = Field(description="Chamadas de IA previstas na edição, sem novas tentativas")
    ai_calls_max: int = Field(description="Com uma nova tentativa por etapa de IA")
    producer_calls: int = Field(default=0, description="Chamadas já feitas pelo Produtor (com novas tentativas)")


class Briefing(BriefingContent):
    """Briefing validado + de onde veio + estimativa; salvo em `briefing.json` na pasta do projeto."""

    schema_version: int = BRIEFING_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=_now)
    request: str = Field(default="", description="Pedido do usuário, como foi escrito")
    forced_recipe: Optional[str] = Field(default=None, description="--receita")
    origin: BriefingOrigin
    origin_reason: str = Field(default="", description="Por que o fallback foi usado (vazio quando ia)")
    material: MaterialSummary
    estimate: Optional[WorkEstimate] = None
    provider: Optional[str] = None
    model_chain: list[str] = Field(default_factory=list)
    effective_model: Optional[str] = Field(default=None, description="`provedor:modelo` que respondeu")
    model_switches: list[ModelSwitch] = Field(default_factory=list)
    attempts: list[AIRunMetrics] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Briefing":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class ProducerContext(BaseModel):
    """Além do pedido e do material: receita forçada e o erro da resposta anterior (nova tentativa)."""

    forced_recipe: Optional[str] = None
    previous_error: Optional[str] = None
