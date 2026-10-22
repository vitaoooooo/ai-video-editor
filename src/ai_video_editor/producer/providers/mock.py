"""Produtor simulado, determinístico e sem dependências externas."""

from typing import Optional, Union

from ai_video_editor.producer.provider import BriefingProducer
from ai_video_editor.producer.schema import BriefingContent, MaterialSummary, ProducerContext, RecipeInfo
from ai_video_editor.producer.service import fallback_content

MockResponse = Union[BriefingContent, dict, Exception]


class MockProducer(BriefingProducer):
    """Devolve uma sequência de respostas (uma por chamada; exceção na lista é lançada) ou a regra do fallback.

    A última resposta se repete quando a lista acaba. `calls` guarda o contexto de cada chamada
    (com `previous_error` na nova tentativa).
    """

    def __init__(self, responses: Optional[list[MockResponse]] = None):
        self.responses = list(responses) if responses is not None else None
        self.calls: list[ProducerContext] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock"

    def propose(
        self, request: str, material: MaterialSummary, recipes: list[RecipeInfo], context: ProducerContext
    ) -> BriefingContent:
        self.calls.append(context)
        if self.responses is None:
            return fallback_content(request, material, recipes, context.forced_recipe)
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, BriefingContent) else BriefingContent.model_validate(response)
