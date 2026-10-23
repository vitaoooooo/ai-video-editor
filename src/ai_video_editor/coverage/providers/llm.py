"""Diretor de cobertura sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.coverage.prompt import build_coverage_prompt
from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import CoverageContext, CoverageDecision
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.stage import ChainBackedStage


class LLMCoverageDirector(ChainBackedStage, CoverageDirector):
    """Monta o prompt e pede `CoverageDecision` à cadeia de modelos."""

    def __init__(self, chain: ModelChain):
        ChainBackedStage.__init__(self, chain)

    def decide(self, context: CoverageContext) -> CoverageDecision:
        return self._generate(build_coverage_prompt(context), CoverageDecision)
