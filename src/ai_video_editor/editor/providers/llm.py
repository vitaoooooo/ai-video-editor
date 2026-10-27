"""IA do Editor sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.editor.provider import DetailRequest, EditorAI, StructureRequest, SummaryRequest
from ai_video_editor.editor.schema import BlockAnswer, ClipSummaryAnswer, StructureAnswer
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

    def detail(self, request: DetailRequest) -> BlockAnswer:
        return self._generate(request.prompt, BlockAnswer)
