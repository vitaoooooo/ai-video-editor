"""Interface da IA do Editor (três passadas) e os pedidos que ela recebe.

Cada pedido leva o texto já montado (`prompt`, o que um modelo de verdade lê) e os dados de onde ele saiu
(o mock decide a partir deles, sem ler o texto).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.prompt import EditorContext
from ai_video_editor.editor.schema import BlockAnswer, ClipSummary, ClipSummaryAnswer, StructureAnswer
from ai_video_editor.llm.schema import ModelSwitch
from ai_video_editor.roteiro.schema import Moment


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
class DetailRequest:
    unit_id: str
    prompt: str
    material: EditorMaterial
    context: EditorContext
    clip_ids: list[str]
    sentence_ids: list[str]  # frases do pedaço ainda não usadas no vídeo, em ordem
    moments: list[Moment]  # momentos do pedaço ainda não usados, em ordem
    library: list[Moment]  # pré-seleção do acervo
    target_sec: Optional[float] = None
    block_target_sec: Optional[float] = None  # alvo do pedaço (o do bloco; proporcional se dividido)
    silent: list[Moment] = field(default_factory=list)  # momentos sem fala livres do pedaço (os de `incluir`)


class EditorAI(ABC):
    """Contrato da IA do Editor: só devolve IDs (a validação e os tempos são do código).

    Raises (nos três métodos): AIResponseValidationError se a resposta não puder ser lida; outros
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
    def detail(self, request: DetailRequest) -> BlockAnswer: ...
