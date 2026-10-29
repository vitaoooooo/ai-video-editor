"""Interface da IA do Editor (resumo, estrutura e as três perguntas da passada 2) e os pedidos que ela recebe.

Cada pedido leva o texto já montado (`prompt`, o que um modelo de verdade lê) e os dados de onde ele saiu
(o mock decide a partir deles, sem ler o texto).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.prompt import EditorContext
from ai_video_editor.editor.schema import (
    Candidate,
    ClipSummary,
    ClipSummaryAnswer,
    CoveragesAnswer,
    CutsAnswer,
    InsertsAnswer,
    StructureAdditionsAnswer,
    StructureAnswer,
    Trecho,
)
from ai_video_editor.llm.schema import ModelSwitch


@dataclass
class SummaryRequest:
    clip_id: str
    prompt: str
    material: EditorMaterial
    moment_ids: Optional[list[str]] = None  # pedaço de um clipe grande (None = o clipe inteiro)
    part: int = 1
    parts: int = 1


@dataclass
class StructureRequest:
    prompt: str
    material: EditorMaterial
    context: EditorContext
    summaries: dict[str, ClipSummary] = field(default_factory=dict)


@dataclass
class AdditionsRequest:
    """Pergunta de acréscimo da estrutura (falta de duração): os clipes com fala não usados oferecidos."""

    prompt: str
    material: EditorMaterial
    context: EditorContext
    offered: list[str] = field(default_factory=list)  # clipes oferecidos, em ordem de gravação
    missing_sec: float = 0.0


@dataclass
class CutsRequest:
    """2a: um clipe (ou parte dele)."""

    unit_id: str
    prompt: str
    material: EditorMaterial
    clip_id: str
    sentence_ids: list[str]  # frases livres do clipe (ou parte), em ordem
    target_sec: Optional[float] = None


@dataclass
class InsertsRequest:
    """2b: um bloco."""

    unit_id: str
    prompt: str
    material: EditorMaterial
    trechos: list[Trecho]
    candidates: list[Candidate]
    block_target_sec: Optional[float] = None
    current_sec: float = 0.0
    spare_sec: Optional[float] = None  # meta do bloco − fala mantida (None = sem meta)
    breath_positions: list[int] = field(default_factory=list)  # posições de respiro oferecidas
    own_positions: list[int] = field(default_factory=list)  # posições para as imagens do bloco


@dataclass
class CoverageRequest:
    """2c: um bloco (ou parte dos trechos dele)."""

    unit_id: str
    prompt: str
    material: EditorMaterial
    trechos: list[Trecho]
    questions: dict[int, list[Candidate]]  # trecho → candidatos com letra
    max_output_tokens: Optional[int] = None  # limite da resposta (`budget.coverage_output_limit`)


class EditorAI(ABC):
    """Contrato da IA do Editor: só devolve IDs, letras e números (a validação e os tempos são do código).

    Raises (em todos os métodos): AIResponseValidationError se a resposta não puder ser lida; outros
    AIProviderError da camada `llm`.
    """

    last_input_tokens: Optional[int] = None
    last_output_tokens: Optional[int] = None
    last_api_attempts: int = 1
    thinking_label: Optional[str] = None
    model_chain: list[str] = []
    effective_provider: Optional[str] = None
    effective_model_id: Optional[str] = None
    model_switches: list[ModelSwitch] = []

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def summarize(self, request: SummaryRequest) -> ClipSummaryAnswer: ...

    @abstractmethod
    def structure(self, request: StructureRequest) -> StructureAnswer: ...

    @abstractmethod
    def additions(self, request: AdditionsRequest) -> StructureAdditionsAnswer: ...

    @abstractmethod
    def cuts(self, request: CutsRequest) -> CutsAnswer: ...

    @abstractmethod
    def inserts(self, request: InsertsRequest) -> InsertsAnswer: ...

    @abstractmethod
    def coverages(self, request: CoverageRequest) -> CoveragesAnswer: ...
