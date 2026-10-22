"""Diretor simulado, determinístico e sem dependências externas."""

import re
from typing import Optional, Union

from ai_video_editor.director.provider import EditDirector
from ai_video_editor.director.schema import (
    DirectorContext,
    DirectorDecision,
    Sentence,
    SentenceRemoval,
    UserBrief,
)

MockResponse = Union[DirectorDecision, Exception]


def _normalize(text: str) -> str:
    return re.sub(r"[^\w ]+", "", text.lower()).strip()


class MockDirector(EditDirector):
    """Devolve remoções fixas, uma sequência de respostas, ou remove frases repetidas.

    - `removals`: devolvidas em toda chamada.
    - `responses`: uma por chamada, na ordem (uma exceção na lista é lançada); útil para
      testar a nova tentativa.
    - Sem nenhum dos dois: remove frases cujo texto repete o de uma frase anterior.
    """

    def __init__(
        self,
        removals: Optional[list[SentenceRemoval]] = None,
        responses: Optional[list[MockResponse]] = None,
    ):
        self.removals = removals
        self.responses = list(responses) if responses is not None else None
        self.calls: list[DirectorContext] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock"

    def decide(
        self, sentences: list[Sentence], brief: UserBrief, context: DirectorContext
    ) -> DirectorDecision:
        self.calls.append(context)
        if self.responses is not None:
            response = self.responses[min(len(self.calls), len(self.responses)) - 1]
            if isinstance(response, Exception):
                raise response
            return response
        if self.removals is not None:
            return DirectorDecision(
                removals=list(self.removals), summary=f"{len(self.removals)} frase(s) removida(s)."
            )

        if any(s.clip_id is not None for s in sentences):
            return self._remove_retakes(sentences)
        removals: list[SentenceRemoval] = []
        first_seen: dict[str, int] = {}
        for sentence in sentences:
            key = _normalize(sentence.text)
            if key in first_seen:
                removals.append(
                    SentenceRemoval(
                        sentence_id=sentence.id,
                        category="repetition",
                        reason=f"Repete a frase {first_seen[key]}.",
                    )
                )
            else:
                first_seen[key] = sentence.id
        return DirectorDecision(
            removals=removals, summary=f"{len(removals)} frase(s) repetida(s) removida(s)."
        )

    @staticmethod
    def _remove_retakes(sentences: list[Sentence]) -> DirectorDecision:
        """Vários clipes: de cada fala repetida fica a última tomada; as anteriores saem como `retake`."""
        removals: list[SentenceRemoval] = []
        last_seen: dict[str, int] = {}
        for sentence in reversed(sentences):
            key = _normalize(sentence.text)
            if key in last_seen:
                removals.append(
                    SentenceRemoval(
                        sentence_id=sentence.id,
                        category="retake",
                        reason=f"Tomada repetida; fica a frase {last_seen[key]}.",
                    )
                )
            else:
                last_seen[key] = sentence.id
        removals.reverse()
        return DirectorDecision(
            removals=removals, summary=f"{len(removals)} tomada(s) repetida(s) removida(s)."
        )
