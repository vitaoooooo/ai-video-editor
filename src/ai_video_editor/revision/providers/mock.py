"""Revisor simulado, determinístico e sem dependências externas."""

from typing import Optional, Union

from ai_video_editor.director.schema import SentenceRemoval, UserBrief
from ai_video_editor.revision.provider import EditReviser
from ai_video_editor.revision.schema import (
    RevisionContext,
    RevisionDecision,
    SentenceRestoration,
    SentenceState,
)

MockResponse = Union[RevisionDecision, Exception]


class MockReviser(EditReviser):
    """Devolve uma decisão fixa, uma sequência de respostas, ou aplica regras simples ao pedido.

    - `decision`: devolvida em toda chamada.
    - `responses`: uma por chamada, na ordem (uma exceção na lista é lançada); útil para
      testar a nova tentativa.
    - Sem nenhum dos dois: mantém a remoção atual, restaura tudo se o pedido contém
      "restaur" e ajusta o ritmo por "tight"/"natural" no pedido.
    """

    def __init__(
        self,
        decision: Optional[RevisionDecision] = None,
        responses: Optional[list[MockResponse]] = None,
    ):
        self.decision = decision
        self.responses = list(responses) if responses is not None else None
        self.calls: list[tuple[list[SentenceState], RevisionContext]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock"

    def revise(
        self, states: list[SentenceState], brief: UserBrief, context: RevisionContext
    ) -> RevisionDecision:
        self.calls.append((states, context))
        if self.responses is not None:
            response = self.responses[min(len(self.calls), len(self.responses)) - 1]
            if isinstance(response, Exception):
                raise response
            return response
        if self.decision is not None:
            return self.decision

        request = context.request.lower()
        pacing = "tight" if "tight" in request else "natural" if "natural" in request else None
        if "restaur" in request:
            restorations = [
                SentenceRestoration(sentence_id=s.sentence.id, reason="Restaurada a pedido.")
                for s in states
                if s.removed
            ]
            return RevisionDecision(
                restorations=restorations, pacing=pacing, summary=f"{len(restorations)} frase(s) restaurada(s)."
            )
        removals = [
            SentenceRemoval(sentence_id=s.sentence.id, category=s.category or "other", reason=s.reason or "Mantida.")
            for s in states
            if s.removed
        ]
        return RevisionDecision(removals=removals, pacing=pacing, summary="Remoção atual mantida.")
