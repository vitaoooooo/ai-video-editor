"""Produtor de IA sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.stage import ChainBackedStage
from ai_video_editor.producer.prompt import build_producer_prompt
from ai_video_editor.producer.provider import BriefingProducer
from ai_video_editor.producer.schema import BriefingContent, MaterialSummary, ProducerContext, RecipeInfo


class LLMProducer(ChainBackedStage, BriefingProducer):
    """Produtor que monta o prompt e pede `BriefingContent` à cadeia de modelos."""

    def __init__(self, chain: ModelChain):
        ChainBackedStage.__init__(self, chain)

    def propose(
        self, request: str, material: MaterialSummary, recipes: list[RecipeInfo], context: ProducerContext
    ) -> BriefingContent:
        return self._generate(build_producer_prompt(request, material, recipes, context), BriefingContent)
