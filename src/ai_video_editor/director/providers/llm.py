"""Diretor de IA sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.director.prompt import build_director_prompt
from ai_video_editor.director.provider import EditDirector
from ai_video_editor.director.schema import DirectorContext, DirectorDecision, Sentence, UserBrief
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.stage import ChainBackedStage


class LLMDirector(ChainBackedStage, EditDirector):
    """Diretor que monta o prompt e pede `DirectorDecision` à cadeia de modelos."""

    def __init__(self, chain: ModelChain):
        ChainBackedStage.__init__(self, chain)

    def decide(
        self, sentences: list[Sentence], brief: UserBrief, context: DirectorContext
    ) -> DirectorDecision:
        return self._generate(build_director_prompt(sentences, brief, context), DirectorDecision)
