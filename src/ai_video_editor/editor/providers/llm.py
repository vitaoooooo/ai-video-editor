"""IA do Editor sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.editor.provider import (
    CoverageRequest,
    CutsRequest,
    EditorAI,
    InsertsRequest,
    StructureRequest,
    SummaryRequest,
)
from ai_video_editor.editor.schema import ClipSummaryAnswer, CoveragesAnswer, CutsAnswer, InsertsAnswer, StructureAnswer
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.stage import ChainBackedStage


class LLMEditorAI(ChainBackedStage, EditorAI):
    """Manda o pedido já montado e pede o schema da passada à cadeia de modelos."""

    def __init__(self, chain: ModelChain):
        ChainBackedStage.__init__(self, chain)

    def summarize(self, request: SummaryRequest) -> ClipSummaryAnswer:
        return self._generate(request.prompt, ClipSummaryAnswer)

    def structure(self, request: StructureRequest) -> StructureAnswer:
        return self._generate(request.prompt, StructureAnswer)

    def cuts(self, request: CutsRequest) -> CutsAnswer:
        return self._generate(request.prompt, CutsAnswer)

    def inserts(self, request: InsertsRequest) -> InsertsAnswer:
        return self._generate(request.prompt, InsertsAnswer)

    def coverages(self, request: CoverageRequest) -> CoveragesAnswer:
        return self._generate(request.prompt, CoveragesAnswer)
