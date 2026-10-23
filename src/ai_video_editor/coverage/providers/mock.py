"""Diretor de cobertura simulado, determinístico e sem rede."""

from typing import Optional, Union

from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import (
    BreathChoice,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    WindowDecision,
)
from ai_video_editor.coverage.service import CoverageState, best_free_shot, fill_windows, max_breaths, shot_length

MockResponse = Union[CoverageDecision, Exception]
# Tamanho dos respiros do mock e o mínimo de tomada para ele usar.
MOCK_BREATH_SIZE = "medium"
MOCK_BREATH_MIN_SEC = 2.0
MOCK_MAX_BREATHS = 5


class MockCoverageDirector(CoverageDirector):
    """Devolve uma sequência de respostas ou, sem ela, decide as janelas como o fallback (tomada
    relacionada ou manter; numa janela que exige cobertura sem relacionada, a tomada livre de maior
    beleza) e põe um respiro nas primeiras trocas de clipe (até `MOCK_MAX_BREATHS` e o máximo permitido).

    - `responses`: uma por chamada, na ordem (uma exceção na lista é lançada); útil para a nova tentativa.
    """

    def __init__(self, responses: Optional[list[MockResponse]] = None):
        self.responses = list(responses) if responses is not None else None
        self.calls: list[CoverageContext] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock"

    def decide(self, context: CoverageContext) -> CoverageDecision:
        self.calls.append(context)
        if self.responses is not None:
            response = self.responses[min(len(self.calls), len(self.responses)) - 1]
            if isinstance(response, Exception):
                raise response
            return response

        state = CoverageState(context)
        fill_windows(state, context.windows, origin="ia")
        coverages = [CoverageChoice(**c.model_dump(include={"from_sentence", "to_sentence", "shot_ids", "reason", "windows"})) for c in state.coverages]
        used = set(state.used)
        decisions = []
        for window in context.windows:
            kept = state.kept.get(window.id)
            if kept is None:
                decisions.append(WindowDecision(window=window.id, action="cover", reason="tomada relacionada"))
            elif not window.required:
                decisions.append(WindowDecision(window=window.id, action="keep", reason=kept.reason))
            else:
                speaking = {window.clip_id}
                need = sum(s.duration_sec(context.fps) for s in context.sentences if s.id in window.sentence_ids)
                shot = best_free_shot(context.candidates, need, speaking, used)
                if shot is not None:
                    used.add(shot.id)
                    coverages.append(
                        CoverageChoice(
                            from_sentence=window.sentence_ids[0], to_sentence=window.sentence_ids[-1], shot_ids=[shot.id],
                            reason="tomada livre de maior beleza", windows=[window.id],
                        )
                    )
        order = {s.id: i for i, s in enumerate(context.sentences)}
        coverages.sort(key=lambda c: order[c.from_sentence])
        inside = {k for c in coverages for k in range(order[c.from_sentence], order[c.to_sentence])}
        breaths: list[BreathChoice] = []
        sentences = context.sentences
        for k in range(len(sentences) - 1):
            if len(breaths) >= min(MOCK_MAX_BREATHS, max_breaths(context)):
                break
            if sentences[k].clip_id == sentences[k + 1].clip_id or k in inside:
                continue
            shot = best_free_shot(
                [c for c in context.candidates if shot_length(c) >= MOCK_BREATH_MIN_SEC], MOCK_BREATH_MIN_SEC, set(), used
            )
            if shot is None:
                break
            used.add(shot.id)
            breaths.append(
                BreathChoice(
                    after_sentence=sentences[k].id, shot_id=shot.id, size=MOCK_BREATH_SIZE,
                    reason=f"Troca do clipe {sentences[k].clip_id} para {sentences[k + 1].clip_id}.",
                )
            )
        return CoverageDecision(
            coverages=coverages, breaths=breaths, window_decisions=decisions,
            summary=f"{len(coverages)} cobertura(s) nas janelas com imagem ruim e {len(breaths)} respiro(s) nas trocas de clipe.",
        )
