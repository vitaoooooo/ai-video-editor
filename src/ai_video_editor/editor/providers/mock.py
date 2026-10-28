"""IA do Editor simulada: determinística, sem rede, sempre com IDs válidos tirados do pedido.

- Resumo: frases e momentos contados, a descrição do 1º momento, o 1º trecho como destaque e os momentos
  com problema.
- Estrutura: um bloco por clipe com fala, na ordem do catálogo; os sem fala ficam no acervo (sem nenhum
  clipe com fala: um bloco só, com todos).
- 2a cortes: nada cortado (todas as frases livres ficam).
- 2b inclusões: bloco sem fala = todas as imagens do bloco, em ordem; com fala, as imagens do bloco no fim
  e um respiro de até 2 s (e do que sobra da sobra do bloco) na 1ª posição de respiro oferecida, com a 1ª
  imagem do acervo que NÃO divide palavra com a fala (as que dividem ficam para a cobertura).
- 2c coberturas: o 1º trecho perguntado com candidato "mostra o que é dito" ganha esse candidato (a 1ª
  palavra em comum); os outros, "nenhuma".

`responses` = respostas fixas por passada (`"summary"`, `"structure"`, `"cuts"`, `"inserts"`, `"coverages"`),
uma por chamada, na ordem; um item pode ser a resposta (modelo ou dict), uma exceção (lançada) ou uma
função do pedido. Acabada a lista daquela passada, volta à resposta determinística.
"""

from typing import Any, Callable, Optional, Union

from ai_video_editor.editor.material import common_words, moment_visible_text
from ai_video_editor.editor.provider import (
    CoverageRequest,
    CutsRequest,
    EditorAI,
    InsertsRequest,
    StructureRequest,
    SummaryRequest,
)
from ai_video_editor.editor.schema import (
    COVERAGE_SAYS,
    ClipSummaryAnswer,
    CoveragesAnswer,
    CutsAnswer,
    InsertsAnswer,
    StructureAnswer,
)

MockEditorResponse = Union[dict, Exception, Callable[[Any], Any], Any]
MOCK_BREATH_SEC = 2.0


class MockEditorAI(EditorAI):
    def __init__(self, responses: Optional[dict[str, list[MockEditorResponse]]] = None):
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: dict[str, list] = {"summary": [], "structure": [], "cuts": [], "inserts": [], "coverages": []}

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock"

    @property
    def model_chain(self) -> list[str]:  # type: ignore[override]
        return ["mock:mock"]

    def _fixed(self, kind: str, request, schema):
        self.calls[kind].append(request)
        queue = self.responses.get(kind)
        if not queue:
            return None
        response = queue.pop(0)
        if callable(response) and not isinstance(response, Exception):
            response = response(request)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, schema) else schema.model_validate(response)

    def summarize(self, request: SummaryRequest) -> ClipSummaryAnswer:
        fixed = self._fixed("summary", request, ClipSummaryAnswer)
        if fixed is not None:
            return fixed
        material = request.material
        clip = material.clips[request.clip_id]
        moments = [
            m for m in material.clip_moments.get(clip.id, [])
            if request.moment_ids is None or m.id in request.moment_ids
        ]
        sentences = material.moment_sentences([m.id for m in moments]) if request.moment_ids else material.clip_sentences.get(clip.id, [])
        first = moments[0].description if moments and moments[0].description else "sem descrição"
        highlight = sentences[0].id if sentences else (moments[0].id if moments else None)
        problems = [{"ids": [m.id], "motivo": ", ".join(m.issues)} for m in moments if m.issues][:3]
        return ClipSummaryAnswer(
            resumo=f"Clipe {clip.id} ({clip.role}): {len(sentences)} frase(s) e {len(moments)} momento(s). {first}",
            destaques=[{"ids": [highlight], "motivo": "primeiro trecho"}] if highlight else [],
            problemas=problems,
            tom="informativo" if sentences else "contemplativo",
            tem_fala_para_camera=any(m.talking_to_camera for m in moments),
        )

    def structure(self, request: StructureRequest) -> StructureAnswer:
        fixed = self._fixed("structure", request, StructureAnswer)
        if fixed is not None:
            return fixed
        material = request.material
        speaking = [c for c in material.clip_ids if material.has_speech(c)]
        groups = [[c] for c in speaking] if speaking else [material.clip_ids]
        blocks = []
        for n, clip_ids in enumerate(groups, start=1):
            seconds = sum(s.end_sec - s.start_sec for c in clip_ids for s in material.clip_sentences.get(c, []))
            if seconds:
                seconds += MOCK_BREATH_SEC  # o detalhe simulado põe um respiro por bloco com fala
            else:
                seconds = sum(material.clips[c].duration_sec for c in clip_ids)
            blocks.append({"id": f"B{n:02d}", "titulo": f"Parte {n}", "clipes": clip_ids, "duracao_alvo_sec": round(seconds, 1)})
        total = request.context.target_sec or round(sum(b["duracao_alvo_sec"] for b in blocks), 1)
        return StructureAnswer(blocos=blocks, fora=[], duracao_total_alvo_sec=total, notas="estrutura simulada")

    def cuts(self, request: CutsRequest) -> CutsAnswer:
        fixed = self._fixed("cuts", request, CutsAnswer)
        return fixed if fixed is not None else CutsAnswer(notas="cortes simulados: nada cortado")

    def inserts(self, request: InsertsRequest) -> InsertsAnswer:
        fixed = self._fixed("inserts", request, InsertsAnswer)
        if fixed is not None:
            return fixed
        material = request.material
        own = [c for c in request.candidates if c.source == "bloco"]
        if not request.trechos:
            items = [{"depois_do_trecho": 0, "letra": c.letter} for c in own]
            return InsertsAnswer.model_validate({"inserir": items, "notas": "inclusões simuladas"})
        last = len(request.trechos)
        items = [{"depois_do_trecho": last, "letra": c.letter} for c in own]
        speech = " ".join(material.sentences[s].text for t in request.trechos for s in t.sentence_ids)
        room = (request.spare_sec or 0.0) - sum(c.duration_sec for c in own)
        for candidate in request.candidates:
            if not request.breath_positions or room <= 0:
                break
            moment = material.moments[candidate.moment_id]
            if candidate.source != "bloco" and not common_words(speech, moment_visible_text(moment)):
                items.append({
                    "depois_do_trecho": request.breath_positions[0], "letra": candidate.letter,
                    "duracao_sec": round(min(MOCK_BREATH_SEC, candidate.duration_sec, room), 2),
                })
                break
        return InsertsAnswer.model_validate({"inserir": items, "notas": "inclusões simuladas"})

    def coverages(self, request: CoverageRequest) -> CoveragesAnswer:
        fixed = self._fixed("coverages", request, CoveragesAnswer)
        if fixed is not None:
            return fixed
        items, done = [], False
        for number, candidates in request.questions.items():
            says = next((c for c in candidates if c.kind == COVERAGE_SAYS), None)
            if says is not None and not done:
                items.append({"trecho": number, "letra": says.letter, "palavra": says.words[0]})
                done = True
            else:
                items.append({"trecho": number, "letra": "nenhuma", "palavra": ""})
        return CoveragesAnswer.model_validate({"coberturas": items, "notas": "coberturas simuladas"})
