"""Diretor de cobertura sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.coverage.prompt import build_coverage_prompt, build_topic_prompt
from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import CoverageContext, CoverageDecision, TopicMap
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.stage import ChainBackedStage


class LLMCoverageDirector(ChainBackedStage, CoverageDirector):
    """Monta os prompts e pede `TopicMap` e `CoverageDecision` à cadeia de modelos."""

    def __init__(self, chain: ModelChain):
        ChainBackedStage.__init__(self, chain)

    def map_topics(self, context: CoverageContext) -> TopicMap:
        return self._generate(build_topic_prompt(context), TopicMap)

    def decide(self, context: CoverageContext) -> CoverageDecision:
        return self._generate(build_coverage_prompt(context), CoverageDecision)
