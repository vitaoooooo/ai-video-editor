"""Diretor de cobertura simulado, determinístico e sem rede."""

import re
from typing import Optional, Union

from ai_video_editor.coverage import evidence as ev
from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.related import related_shots
from ai_video_editor.coverage.schema import (
    BreathChoice,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    TopicBlock,
    TopicMap,
    VisualShot,
)
from ai_video_editor.coverage.service import MIN_COVERAGE_SEC, max_breaths, shot_length
from ai_video_editor.coverage.topics import topics_by_clip

MockResponse = Union[CoverageDecision, Exception]
MockTopics = Union[TopicMap, Exception]
# Tamanho dos respiros do mock e o mínimo de tomada para ele usar.
MOCK_BREATH_SIZE = "medium"
MOCK_BREATH_MIN_SEC = 2.0
MOCK_MAX_BREATHS = 5

_WORD = re.compile(r"\w+")


def _matching_tag(word: str, shot: VisualShot) -> Optional[str]:
    """Tag da tomada que fala da mesma coisa que a palavra (com plural e sinônimos), se houver."""
    for tag in shot.tags:
        if any(ev.same_word(w, c) for w in ev.concepts(word) for c in ev.concepts(tag)):
            return tag
    return None


class MockCoverageDirector(CoverageDirector):
    """Devolve sequências de respostas ou, sem elas, decide como um editor literal:

    - **mapa:** um bloco por trecho seguido do mesmo clipe;
    - **coberturas:** em cada frase, a primeira palavra de conteúdo que é tag de uma tomada livre (fora do
      clipe que fala) vira uma cobertura daquela frase com essa tomada (a de maior beleza), com a palavra
      como evidência de fala e a tag como evidência de imagem;
    - **respiros:** nas trocas de bloco (até `MOCK_MAX_BREATHS` e o máximo permitido), com uma tomada
      relacionada à tela do bloco seguinte.

    `responses`/`topics`: uma resposta por chamada, na ordem (uma exceção na lista é lançada).
    """

    def __init__(self, responses: Optional[list[MockResponse]] = None, topics: Optional[list[MockTopics]] = None):
        self.responses = list(responses) if responses is not None else None
        self.topics = list(topics) if topics is not None else None
        self.calls: list[CoverageContext] = []
        self.topic_calls: list[CoverageContext] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock"

    def map_topics(self, context: CoverageContext) -> TopicMap:
        self.topic_calls.append(context)
        if self.topics is not None:
            answer = self.topics[min(len(self.topic_calls), len(self.topics)) - 1]
            if isinstance(answer, Exception):
                raise answer
            return answer
        blocks = [
            TopicBlock(from_sentence=t.from_sentence, to_sentence=t.to_sentence, topic=t.topic, place="")
            for t in topics_by_clip(context.sentences)
        ]
        return TopicMap(summary=f"vlog '{context.project_name}'", blocks=blocks)

    def decide(self, context: CoverageContext) -> CoverageDecision:
        self.calls.append(context)
        if self.responses is not None:
            response = self.responses[min(len(self.calls), len(self.responses)) - 1]
            if isinstance(response, Exception):
                raise response
            return response

        used: set[str] = set()
        coverages: list[CoverageChoice] = []
        for sentence in context.sentences:
            free = [
                c for c in context.candidates
                if c.id not in used and c.clip_id != sentence.clip_id and shot_length(c) >= MIN_COVERAGE_SEC
            ]
            found = None
            for match in _WORD.finditer(sentence.text):
                word = match.group(0)
                if not ev.concepts(word):
                    continue
                hits = [(shot, tag) for shot in free if (tag := _matching_tag(word, shot))]
                if hits:
                    found = (word, *max(hits, key=lambda h: (h[0].beauty or 0, shot_length(h[0]))))
                    break
            if found is None:
                continue
            word, shot, tag = found
            used.add(shot.id)
            coverages.append(
                CoverageChoice(
                    from_sentence=sentence.id, to_sentence=sentence.id, shot_ids=[shot.id], speech_evidence=word,
                    image_evidence=tag, reason=f"A fala cita '{word}' e a tomada mostra '{tag}'.",
                )
            )

        breaths: list[BreathChoice] = []
        by_id = {s.id: s for s in context.screen_shots}
        order = {s.id: i for i, s in enumerate(context.sentences)}
        for prev, nxt in zip(context.topics, context.topics[1:]):
            if len(breaths) >= min(MOCK_MAX_BREATHS, max_breaths(context)):
                break
            i, j = order[nxt.from_sentence], order[nxt.to_sentence]
            screens = [by_id[x] for k in range(i, j + 1) for x in context.sentences[k].screen if x in by_id]
            pool = [c for c in context.candidates if c.id not in used and shot_length(c) >= MOCK_BREATH_MIN_SEC]
            related = related_shots(pool, screens)
            if not related:
                continue
            shot, why = related[0]
            used.add(shot.id)
            breaths.append(
                BreathChoice(
                    after_sentence=prev.to_sentence, shot_id=shot.id, size=MOCK_BREATH_SIZE,
                    reason=f"Troca de {prev.id} para {nxt.id}; a tomada tem a ver com {nxt.id} ({why.label}).",
                )
            )
        return CoverageDecision(
            coverages=coverages, breaths=breaths,
            summary=f"{len(coverages)} cobertura(s) onde a fala cita uma tag da tomada e {len(breaths)} respiro(s) nas trocas de assunto.",
        )
