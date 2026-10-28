"""IA do Editor simulada: determinística, sem rede, sempre com IDs válidos tirados do pedido.

- Resumo: frases e momentos contados, a descrição do 1º momento, o 1º trecho como destaque e os momentos
  com problema.
- Estrutura: um bloco por clipe com fala, na ordem do catálogo; os sem fala ficam no acervo (sem nenhum
  clipe com fala: um bloco só, com todos).
- Detalhe: nada cortado (todas as frases livres ficam); um clipe sem fala entra com todos os momentos sem
  fala livres (`incluir`, no lugar do clipe). A 1ª faixa ganha uma cobertura "mostra o que é dito" com o
  primeiro momento do acervo que divide uma palavra com a fala (essa palavra é as duas evidências) e, depois
  dela, um respiro com o momento seguinte do acervo, de até 2 s.

`responses` = respostas fixas por passada (`"summary"`, `"structure"`, `"detail"`), uma por chamada, na
ordem; um item pode ser a resposta (modelo ou dict), uma exceção (lançada) ou uma função do pedido.
Acabada a lista daquela passada, volta à resposta determinística.
"""

from typing import Any, Callable, Optional, Union

from ai_video_editor.editor.material import matched_words, moment_text
from ai_video_editor.editor.provider import DetailRequest, EditorAI, StructureRequest, SummaryRequest
from ai_video_editor.editor.schema import COVERAGE_SAYS, BlockAnswer, ClipSummaryAnswer, StructureAnswer

MockEditorResponse = Union[dict, Exception, Callable[[Any], Any], Any]
MOCK_BREATH_SEC = 2.0


class MockEditorAI(EditorAI):
    def __init__(self, responses: Optional[dict[str, list[MockEditorResponse]]] = None):
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: dict[str, list] = {"summary": [], "structure": [], "detail": []}

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

    def detail(self, request: DetailRequest) -> BlockAnswer:
        fixed = self._fixed("detail", request, BlockAnswer)
        if fixed is not None:
            return fixed
        material = request.material
        free = set(request.sentence_ids)
        includes: list[dict] = []
        faixas: list[list[str]] = []
        anchor = None
        for clip_id in request.clip_ids:
            sentences = [s.id for s in material.clip_sentences.get(clip_id, []) if s.id in free]
            if sentences:
                faixas.append(sentences)
                anchor = sentences[-1]
            elif not material.clip_sentences.get(clip_id):
                for moment in request.silent:
                    if moment.clip_id == clip_id:
                        includes.append({"momento": moment.id, "depois_de": anchor})
                        anchor = moment.id
        coverages, breaths = [], []
        library = list(request.library)
        if faixas:
            first = faixas[0]
            speech = " ".join(material.sentences[s].text for s in first)
            for moment in library:
                found = matched_words(moment_text(moment), speech)
                if found:
                    coverages.append({
                        "de": first[0], "ate": first[-1], "momentos": [moment.id], "tipo": COVERAGE_SAYS,
                        "evidencia_fala": found[0],
                        "evidencia_imagem": found[0], "motivo": "a imagem mostra o que é dito",
                    })
                    library.remove(moment)
                    break
            free_breath = next((m for m in library if not m.sentence_ids), None)
            if free_breath is not None:
                breaths.append({
                    "depois_de": first[-1], "momento": free_breath.id,
                    "duracao_sec": round(min(MOCK_BREATH_SEC, free_breath.duration), 2), "motivo": "respiro simulado",
                })
        return BlockAnswer.model_validate({
            "incluir": includes, "coberturas": coverages, "respiros": breaths, "notas": "detalhe simulado",
        })
