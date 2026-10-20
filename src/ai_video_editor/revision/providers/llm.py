"""Revisor de IA sobre a camada genérica `llm` (qualquer provedor registrado, com cadeia de modelos)."""

from ai_video_editor.director.schema import UserBrief
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.stage import ChainBackedStage
from ai_video_editor.revision.prompt import build_reviser_prompt
from ai_video_editor.revision.provider import EditReviser
from ai_video_editor.revision.schema import RevisionContext, RevisionDecision, SentenceState


class LLMReviser(ChainBackedStage, EditReviser):
    """Revisor que monta o prompt e pede `RevisionDecision` à cadeia de modelos."""

    def __init__(self, chain: ModelChain):
        ChainBackedStage.__init__(self, chain)

    def revise(
        self, states: list[SentenceState], brief: UserBrief, context: RevisionContext
    ) -> RevisionDecision:
        return self._generate(build_reviser_prompt(states, brief, context), RevisionDecision)
