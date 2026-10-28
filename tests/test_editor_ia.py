"""Testes do Editor de IA (receita `editor_ia`): schemas, orçamento e divisão, validação item a item,
nova tentativa e fallbacks neutros, caches por passada, montagem (IDs → Timeline) e ponta a ponta com
clipes sintéticos (render curto marcado `slow`). Sem rede: IA simulada (`MockEditorAI`)."""

import re
from pathlib import Path
from typing import Optional

import pytest

from ai_video_editor.audio.schema import AudioAnalysisParams, AudioAnalysisResult, SilenceInterval
from ai_video_editor.catalog.schema import SpeechDetection
from ai_video_editor.catalog.speech import SpeechDetector, merge_segments, speech_result
from ai_video_editor.cli import build_parser, handle_edit
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.editor import EditorContext, EditorMaterial
from ai_video_editor.editor import service as editor_service
from ai_video_editor.editor.assemble import ClipSource, assemble_plan, faixa_ranges
from ai_video_editor.editor.budget import Budget, resolve_budget, tokens
from ai_video_editor.editor.material import library_ranking, matched_words
from ai_video_editor.editor.prompt import DetailPromptInput, build_detail_prompt, build_structure_prompt, build_summary_prompt
from ai_video_editor.editor.providers.llm import LLMEditorAI
from ai_video_editor.editor.providers.mock import MockEditorAI
from ai_video_editor.editor.report import build_editor_report
from ai_video_editor.editor.provider import SummaryRequest
from ai_video_editor.editor.schema import (
    Block,
    BlockAnswer,
    BreathChoice,
    ClipSummaryAnswer,
    CoverageChoice,
    DetailRun,
    EditorPlan,
    EditorRun,
    CutAnswer,
    IncludeAnswer,
    KeepRange,
    StructureAnswer,
    StructureRun,
)
from ai_video_editor.editor.service import CallCache, EditorService
from ai_video_editor.editor.validate import (
    Unit,
    Usage,
    neutral_detail,
    place_forgotten,
    validate_detail,
    validate_structure,
)
from ai_video_editor.intelligence.exceptions import AIAuthenticationError, AIProviderError
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.schema import ModelSpec
from ai_video_editor.pipeline import EditPipelineParams
from ai_video_editor.pipeline.editor_ia import EditorParams, EditorPipeline
from ai_video_editor.planning import SilenceCutParams
from ai_video_editor.producer import registered_recipes
from ai_video_editor.recipes import get_recipe
from ai_video_editor.roteiro.schema import Moment, ProjectRoteiro, RoteiroClip, RoteiroSentence
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig, TranscriptSegment, Word
from tests.test_roteiro import make_video, mock_describer

# --- roteiro montado à mão (sem FFmpeg) ---


def _moment(clip, n, start, end, desc, place="", tags=(), sentences=(), issues=(), talking=False) -> Moment:
    return Moment(
        id=f"{clip}.M{n:02d}", clip_id=clip, start_sec=start, end_sec=end, description=desc, place=place,
        tags=list(tags), sentence_ids=list(sentences), issues=list(issues), talking_to_camera=talking,
    )


def _sentence(i, clip, start, end, text, moment) -> RoteiroSentence:
    return RoteiroSentence(
        id=f"S{i:03d}", index=i, clip_id=clip, start_sec=start, end_sec=end, text=text, moment_ids=[moment]
    )


def _clip(clip_id, role, order, duration, speech, audio=True) -> RoteiroClip:
    return RoteiroClip(
        id=clip_id, role=role, relative_path=f"{role}/{clip_id}.mp4", order=order, duration_sec=duration,
        has_audio=audio, has_speech=speech, transcribed=speech,
    )


def make_roteiro(tmp_path: Path) -> ProjectRoteiro:
    """P001 (fala, praia e trilha, com retake), P002 (fala, restaurante), D001 (drone sem áudio), B001 (b-roll)."""
    clips = [
        _clip("P001", "principal", 1, 12.0, True), _clip("P002", "principal", 2, 6.0, True),
        _clip("D001", "drone", 3, 8.0, False, audio=False), _clip("B001", "broll", 4, 3.0, False),
    ]
    moments = [
        _moment("P001", 1, 0.0, 6.0, "pessoa falando na praia", "praia", ["praia", "mar", "pessoa"], ["S000", "S001"], talking=True),
        _moment("P001", 2, 6.0, 12.0, "pessoa caminhando na trilha", "trilha", ["trilha", "mata"], ["S002", "S003"], ["tremida"]),
        _moment("P002", 1, 0.0, 6.0, "pessoa no restaurante", "restaurante", ["restaurante", "comida"], ["S004", "S005"]),
        _moment("D001", 1, 0.0, 4.0, "vista aérea do mar azul", "praia", ["mar", "praia", "aérea"]),
        _moment("D001", 2, 4.0, 8.0, "vista aérea da mata fechada", "trilha", ["mata", "floresta", "aérea"]),
        _moment("B001", 1, 0.0, 1.0, "prato de comida na mesa", "restaurante", ["comida", "prato"]),
        _moment("B001", 2, 1.0, 3.0, "garçom servindo a mesa", "restaurante", ["restaurante", "garçom"]),
    ]
    sentences = [
        _sentence(0, "P001", 0.5, 2.5, "Chegamos na praia hoje.", "P001.M01"),
        _sentence(1, "P001", 3.0, 5.5, "O mar está muito azul.", "P001.M01"),
        _sentence(2, "P001", 6.5, 9.0, "Agora vamos pela trilha.", "P001.M02"),
        _sentence(3, "P001", 9.5, 11.5, "Agora vamos pela trilha da mata.", "P001.M02"),
        _sentence(4, "P002", 0.5, 2.5, "A comida chegou.", "P002.M01"),
        _sentence(5, "P002", 3.0, 5.0, "Muito boa essa comida.", "P002.M01"),
    ]
    return ProjectRoteiro(
        project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0, clips=clips, moments=moments,
        sentences=sentences,
    )


@pytest.fixture
def material(tmp_path) -> EditorMaterial:
    return EditorMaterial.from_roteiro(make_roteiro(tmp_path))


def _block(*clips, block_id="B01") -> Block:
    return Block(id=block_id, clip_ids=list(clips))


def _unit(*clips) -> Unit:
    return Unit(unit_id="B01", block=_block(*clips), clip_ids=list(clips))


def _service(material, tmp_path, ai=None, num_ctx=32768, context=None) -> EditorService:
    return EditorService(
        ai or MockEditorAI(), material, context or EditorContext(request="vlog da viagem"),
        Budget(num_ctx=num_ctx, source="teste"), CallCache(tmp_path / "cache"),
    )


# --- schemas e receita ---


def test_answers_are_tolerant_and_trimmed():
    cut = CutAnswer.model_validate({"de": " s012 ", "ate": "s015", "motivo": " recomeço "})
    assert (cut.de, cut.ate, cut.motivo) == ("S012", "S015", "recomeço")
    assert (CutAnswer.model_validate("s020").de, CutAnswer.model_validate("s020").ate) == ("S020", "S020")
    assert CutAnswer.model_validate({"frase": "S021"}).de == "S021"
    include = IncludeAnswer.model_validate({"momento": "p001.m02", "depois_de": "início", "duracao_sec": "3,5s"})
    assert (include.momento, include.depois_de, include.duracao_sec) == ("P001.M02", None, 3.5)
    assert IncludeAnswer.model_validate({"momento": "p001.m02", "duracao_sec": "auto"}).duracao_sec is None
    answer = BlockAnswer.model_validate({"ordem": "p002; p001", "coberturas": [{"tipo": "Mostra o lugar"}, {"tipo": "x"}]})
    assert answer.ordem == ["P002", "P001"]
    assert [c.tipo for c in answer.coberturas] == ["mostra o lugar", None]
    summary = ClipSummaryAnswer.model_validate({
        "resumo": "  Um clipe.  ", "destaques": [{"ids": "S001, S002", "motivo": "x"}] * 5, "problemas": [],
    })
    assert summary.resumo == "Um clipe." and len(summary.destaques) == 3 and summary.destaques[0].ids == ["S001", "S002"]
    structure = StructureAnswer.model_validate({"blocos": [{"clipes": "p001;d001", "duracao_alvo_sec": "60"}]})
    assert structure.blocos[0].clipes == ["P001", "D001"] and structure.blocos[0].duracao_alvo_sec == 60.0


def test_recipe_editor_ia_accepts_any_material_and_is_not_offered_to_the_producer():
    recipe = get_recipe("editor_ia")
    assert recipe.steps == ["catalog", "roteiro", "editor", "timeline", "render", "subtitles", "report"]
    assert recipe.accepts(True) and recipe.accepts(False)
    assert not get_recipe("vlog").accepts(False) and not get_recipe("fala").accepts(True)
    # O roteiro faz silêncios e transcrição por dentro: legenda e timeline não pedem essas etapas na receita.
    assert "silences" in get_recipe("editor_ia").dependencies("timeline")
    assert "editor_ia" not in [r.name for r in registered_recipes()]


# --- evidência e acervo ---


def test_evidence_words_ignore_accents_case_stopwords_and_accept_the_same_stem():
    assert matched_words("Praias lindas", "Chegamos na PRAIA hoje") == ["praias"]
    assert matched_words("água", "a agua do mar") == ["agua"]
    assert matched_words("de la que", "de la que") == []  # só palavras vazias: não é evidência
    assert matched_words("montanha", "Chegamos na praia") == []


def test_library_preselection_ranks_by_common_tags_and_place(material):
    candidates = [m for c in ("D001", "B001") for m in material.clip_moments[c]]
    ranked = [m.id for m in library_ranking(material, ["P001"], candidates)]
    assert ranked == ["D001.M01", "D001.M02", "B001.M01", "B001.M02"]
    ranked = [m.id for m in library_ranking(material, ["P002"], candidates)]
    assert ranked[:2] == ["B001.M01", "B001.M02"]


# --- validação: estrutura ---


def test_structure_validation_discards_unknown_and_repeated_clips(material):
    answer = StructureAnswer.model_validate({
        "blocos": [
            {"id": "X", "clipes": ["P001", "Z009"]},
            {"id": "Y", "clipes": ["P001"]},  # repetido: bloco vazio, descartado
            {"id": "Z", "clipes": ["D001", "P002"], "duracao_alvo_sec": -3},
        ],
        "fora": [{"clipe": "P002", "motivo": "repetido"}, {"clipe": "B001", "motivo": "ruim"}, {"clipe": "Q1"}],
    })
    check = validate_structure(answer, material)
    assert [(b.id, b.clip_ids, b.target_sec) for b in check.blocks] == [("B01", ["P001"], None), ("B02", ["D001", "P002"], None)]
    assert [o.clipe for o in check.out] == ["B001"]
    reasons = " | ".join(check.errors)
    assert "Z009 não existe" in reasons and "P001 já usado" in reasons and "nenhum clipe válido" in reasons
    assert "P002 já está num bloco" in reasons and "Q1 não existe" in reasons


# --- validação: detalhe, item a item ---


def test_detail_validation_item_by_item(material):
    usage = Usage()
    usage.sentences.add("S005")  # já no vídeo (bloco anterior)
    usage.moments.add("B001.M02")
    answer = BlockAnswer.model_validate({
        "cortar": [
            {"de": "S002", "ate": "S002", "motivo": "recomeço"},  # válido
            {"de": "S999", "ate": "S999"},  # ID inexistente
            {"de": "S004", "ate": "S004"},  # fora do bloco (P002)
            {"de": "S003", "ate": "S002"},  # fora de ordem
        ],
        "incluir": [
            {"momento": "D001.M02", "depois_de": "S001", "duracao_sec": 2},  # válido: sem fala, depois de S001
            {"momento": "P001.M02"},  # tem fala: recusado
            {"momento": "D001.M01", "duracao_sec": 99},  # duração fora do momento
            {"momento": "D001.M01", "depois_de": "S002"},  # S002 foi cortada
        ],
        "respiros": [
            {"depois_de": "S001", "momento": "B001.M02"},  # momento repetido (já usado)
            {"depois_de": "S002", "momento": "B001.M01"},  # S002 foi cortada
            {"depois_de": "S000", "momento": "B001.M01", "duracao_sec": 0.5},  # válido
        ],
    })
    check = validate_detail(answer, _unit("P001", "D001"), material, ["B001.M01", "B001.M02"], usage)
    joined = " | ".join(f"{d.kind}: {d.reason}" for d in check.discarded)
    assert "frase S999 não existe" in joined
    assert "S004 é do clipe P002, fora deste bloco" in joined
    assert "S003 vem depois de S002" in joined
    assert "momento P001.M02 tem fala (S002, S003)" in joined
    assert "duracao_sec 99 maior que o momento D001.M01" in joined
    assert "depois_de S002 não é uma frase que ficou" in joined
    assert "momento B001.M02 já usado no vídeo" in joined
    assert "respiro: depois_de S002 não é uma frase que ficou" in joined
    assert len(check.discarded) == 8
    # Fica tudo menos S002, na ordem do roteiro; o incluído quebra a faixa depois de S001.
    assert [(k.kind, k.key, k.sentence_ids) for k in check.keep] == [
        ("frases", "S001", ["S000", "S001"]), ("momento", "D001.M02", []), ("frases", "S003", ["S003"]),
    ]
    assert [(c.sentence_ids, c.reason) for c in check.cuts] == [(["S002"], "recomeço")]
    assert [(b.after, b.moment_id) for b in check.breaths] == [("S000", "B001.M01")]
    # Duração resultante: frases que ficaram (2 + 2,5 + 2) + incluído (2) + respiro (0,5).
    assert check.result_sec == pytest.approx(9.0)
    # O uso original não muda; o do resultado tem o que foi aceito.
    assert usage.sentences == {"S005"} and {"S000", "S001", "S003", "S005"} <= check.usage.sentences


def test_cut_by_range_and_by_id_keeps_the_rest_in_order_and_optional_clip_order(material):
    answer = BlockAnswer.model_validate({"cortar": ["S001", {"de": "S002", "ate": "S003"}], "ordem": ["P002", "P001"]})
    check = validate_detail(answer, _unit("P001", "P002"), material, [], Usage())
    assert check.clip_order == ["P002", "P001"] and not check.discarded
    assert [(k.clip_id, k.sentence_ids) for k in check.keep] == [("P002", ["S004", "S005"]), ("P001", ["S000"])]
    assert check.usage.kept_order == ["S004", "S005", "S000"]
    # Sem ordem (ou com clipe de fora do bloco): a ordem da estrutura.
    bad = validate_detail(BlockAnswer.model_validate({"ordem": ["D001"]}), _unit("P001", "P002"), material, [], Usage())
    assert bad.clip_order == ["P001", "P002"] and "clipes fora deste bloco: D001" in bad.errors[0]
    assert [k.sentence_ids for k in bad.keep] == [["S000", "S001", "S002", "S003"], ["S004", "S005"]]


def test_include_only_moments_without_speech_at_the_start_or_after_another_include(material):
    answer = BlockAnswer.model_validate({"incluir": [
        {"momento": "D001.M01"}, {"momento": "D001.M02", "depois_de": "D001.M01", "duracao_sec": 1.5},
        {"momento": "P001.M01", "depois_de": "S003"},  # tem fala: recusado
    ]})
    check = validate_detail(answer, _unit("D001", "P001"), material, [], Usage())
    assert [k.key for k in check.keep] == ["D001.M01", "D001.M02", "S003"]
    assert check.keep[1].duration_sec == 1.5
    assert [d.reason for d in check.discarded] == ["momento P001.M01 tem fala (S000, S001): só entram momentos sem fala"]


def test_coverage_inside_a_respiro_and_half_invalid_answer_is_refused(material):
    answer = BlockAnswer.model_validate({
        "cortar": ["S003"],
        "respiros": [{"depois_de": "S000", "momento": "B001.M02"}],
        "coberturas": [{"de": "S000", "ate": "S001", "momentos": ["D001.M01"], "tipo": "mostra o que é dito",
                        "evidencia_fala": "mar", "evidencia_imagem": "mar"}],
    })
    check = validate_detail(answer, _unit("P001"), material, ["D001.M01", "B001.M02"], Usage())
    assert "há um respiro no meio das frases cobertas" in check.discarded[0].reason
    assert check.refused is None  # 1 de 3 inválido
    bad = BlockAnswer.model_validate({"cortar": ["S000", "S999", {"de": "S004"}], "incluir": [{"momento": "X.M01"}]})
    check = validate_detail(bad, _unit("P001"), material, [], Usage())
    assert "3 de 4 itens inválidos" in check.refused and check.unusable is None  # 1 válido: aproveitável
    assert validate_detail(BlockAnswer.model_validate({"cortar": ["S999"]}), _unit("P001"), material, [], Usage()).unusable == "nenhum item válido"
    everything = BlockAnswer.model_validate({"cortar": [{"de": "S000", "ate": "S003"}]})
    assert "nada fica no bloco" in validate_detail(everything, _unit("P001"), material, [], Usage()).refused
    # Resposta vazia = nada cortado: aceita.
    assert validate_detail(BlockAnswer(), _unit("P001"), material, [], Usage()).refused is None


def test_neutral_detail_cuts_nothing_and_includes_the_moments_of_silent_clips(material):
    usage = Usage()
    usage.sentences.add("S001")
    answer = neutral_detail(_unit("P001", "D001"), material, usage)
    assert not answer.cortar and not answer.coberturas and not answer.respiros
    assert [(i.momento, i.depois_de) for i in answer.incluir] == [("D001.M01", "S003"), ("D001.M02", "D001.M01")]
    check = validate_detail(answer, _unit("P001", "D001"), material, [], usage)
    assert [k.key for k in check.keep] == ["S000", "S003", "D001.M01", "D001.M02"]


# --- cobertura: tipo e evidência ---


def make_evidence_material(tmp_path: Path) -> EditorMaterial:
    """P001 fala no terminal (ómnibus, cachorros); D001 = drone com ônibus, cachorro e a fachada do terminal."""
    clips = [_clip("P001", "principal", 1, 12.0, True), _clip("D001", "drone", 2, 12.0, False, audio=False),
             _clip("P002", "principal", 3, 6.0, True)]
    moments = [
        _moment("P001", 1, 0.0, 12.0, "pessoa falando no saguão", "terminal de ônibus", ["pessoa"], ["S000", "S001"]),
        _moment("D001", 1, 0.0, 4.0, "ônibus na estrada", "estrada", ["ônibus", "estrada"]),
        _moment("D001", 2, 4.0, 8.0, "cachorro na praia", "praia", ["cachorro", "areia"]),
        _moment("D001", 3, 8.0, 12.0, "fachada do terminal rodoviário", "terminal rodoviário", ["fachada"]),
        _moment("P002", 1, 0.0, 6.0, "rua com carros e montanhas", "rua", ["rua", "carros"], ["S002"]),
    ]
    sentences = [
        _sentence(0, "P001", 0.5, 4.0, "Pegamos o ómnibus para Bariloche.", "P001.M01"),
        _sentence(1, "P001", 5.0, 8.0, "Olha os cachorros correndo.", "P001.M01"),
        _sentence(2, "P002", 0.5, 3.0, "Chegamos na cidade.", "P002.M01"),
    ]
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0, clips=clips,
                             moments=moments, sentences=sentences)
    return EditorMaterial.from_roteiro(roteiro)


def _cover(material, **coverage):
    answer = BlockAnswer.model_validate({"coberturas": [coverage]})
    return validate_detail(answer, _unit("P001", "P002"), material, ["D001.M01", "D001.M02", "D001.M03"], Usage())


def test_coverage_shows_what_is_said_needs_a_common_word_with_accents_and_plural(tmp_path):
    material = make_evidence_material(tmp_path)
    says = "mostra o que é dito"
    bus = _cover(material, de="S000", ate="S000", momentos=["D001.M01"], tipo=says,
                 evidencia_fala="ómnibus para Bariloche", evidencia_imagem="ônibus na estrada")
    [coverage] = bus.coverages
    assert coverage.kind == says and coverage.common_words == ["omnibus"] and coverage.matched_image == {"D001.M01": ["onibus", "estrada"]}
    dogs = _cover(material, de="S001", ate="S001", momentos=["D001.M02"], tipo=says,
                  evidencia_fala="cachorros correndo", evidencia_imagem="cachorro na praia")
    assert dogs.coverages[0].common_words == ["cachorros"]
    # Sem palavra em comum entre as duas evidências: recusada (a imagem não mostra o que é dito).
    nothing = _cover(material, de="S000", ate="S000", momentos=["D001.M02"], tipo=says,
                     evidencia_fala="ómnibus", evidencia_imagem="cachorro na praia")
    assert not nothing.coverages and "sem palavra em comum" in nothing.discarded[0].reason
    # Sem tipo: recusada.
    untyped = _cover(material, de="S000", ate="S000", momentos=["D001.M01"], evidencia_fala="ómnibus", evidencia_imagem="ônibus")
    assert "tipo ausente" in untyped.discarded[0].reason
    # Evidência da fala com texto da imagem (não está nas frases cobertas): recusada.
    wrong = _cover(material, de="S000", ate="S000", momentos=["D001.M01"], tipo=says,
                   evidencia_fala="ómnibus na estrada", evidencia_imagem="ônibus na estrada")
    assert "evidencia_fala cita palavras que não estão em S000–S000: estrada" in wrong.discarded[0].reason


def test_coverage_image_evidence_must_come_from_the_moment_used(tmp_path):
    material = make_evidence_material(tmp_path)
    # "rua", "carros" são da imagem de quem fala (P002.M01), não do drone: o código não aceita.
    check = _cover(material, de="S000", ate="S000", momentos=["D001.M01"], tipo="mostra o que é dito",
                   evidencia_fala="ómnibus", evidencia_imagem="ônibus rua carros")
    assert not check.coverages
    reason = check.discarded[0].reason
    assert "não estão nos momentos usados (D001.M01): rua, carros" in reason and "P002.M01" in reason


def test_coverage_shows_the_place_needs_the_same_or_similar_place(tmp_path):
    material = make_evidence_material(tmp_path)
    place = "mostra o lugar"
    ok = _cover(material, de="S000", ate="S001", momentos=["D001.M03"], tipo=place,
                evidencia_fala="Bariloche", evidencia_imagem="fachada do terminal")
    [coverage] = ok.coverages
    assert coverage.kind == place and coverage.common_words == ["terminal"]
    beach = _cover(material, de="S000", ate="S000", momentos=["D001.M02"], tipo=place,
                   evidencia_fala="Bariloche", evidencia_imagem="cachorro na praia")
    assert not beach.coverages and "nenhum momento válido" in beach.discarded[0].reason
    assert any("lugar 'praia' diferente do lugar da fala ('terminal de ônibus')" in w for w in beach.warnings)


# --- passadas: nova tentativa, fallbacks, cache ---


def test_full_run_with_mock_produces_blocks_coverage_and_breath(material, tmp_path):
    run = _service(material, tmp_path).run()
    assert [s.origin for s in run.summaries] == ["ia"] * 4
    assert run.structure.origin == "ia" and [b.clip_ids for b in run.structure.blocks] == [["P001"], ["P002"]]
    assert run.structure.library_clip_ids == ["D001", "B001"]
    first, second = run.details
    assert [k.key for k in first.keep] == ["S003"] and first.keep[0].sentence_ids == ["S000", "S001", "S002", "S003"]
    assert first.coverages[0].moment_ids == ["D001.M01"]  # "mar" na fala e no momento
    assert first.breaths and first.breaths[0].moment_id not in first.coverages[0].moment_ids
    assert second.coverages and second.coverages[0].moment_ids == ["B001.M01"]  # "comida"
    assert run.summary_stats.calls == 4 and run.structure.stats.calls == 1 and run.detail_stats.calls == 2
    assert not any(d.duration_off for d in run.details)


def test_retry_with_the_errors_and_neutral_fallbacks(material, tmp_path):
    ai = MockEditorAI({
        "structure": [{"blocos": []}, {"blocos": [{"clipes": ["X999"]}]}],
        "detail": [
            {"cortar": ["S999"]}, {"cortar": ["S000"]},  # bloco 1: nenhum item válido → nova tentativa aceita
            {"cortar": [{"de": "S004", "ate": "S005"}]}, {"cortar": ["S998"]},  # bloco 2: nada fica; nenhum válido
        ],
    })
    run = _service(material, tmp_path, ai).run()
    # Estrutura: 2 respostas sem bloco válido → fallback neutro (um bloco por clipe com fala), com aviso grande.
    assert run.structure.origin == "fallback" and [b.clip_ids for b in run.structure.blocks] == [["P001"], ["P002"]]
    assert "nenhum bloco" in run.structure.attempts[0] and "X999 não existe" in run.structure.attempts[1]
    assert any("ESTRUTURA PELO FALLBACK NEUTRO" in w for w in run.warnings)
    second_prompt = ai.calls["structure"][1].prompt
    assert "Sua resposta anterior foi recusada" in second_prompt
    # Bloco 1: recusado e refeito com a lista de erros; bloco 2: inaproveitável duas vezes → nada cortado.
    first, second = run.details
    assert first.origin == "ia" and first.attempts and "S999 não existe" in ai.calls["detail"][1].prompt
    assert [c.sentence_ids for c in first.cuts] == [["S000"]]
    assert second.origin == "fallback" and [k.sentence_ids for k in second.keep] == [["S004", "S005"]]
    assert not second.coverages and not second.breaths and not second.cuts
    assert any("BLOCO B02 PELO FALLBACK NEUTRO" in w for w in run.warnings)


def test_valid_items_are_accepted_after_the_retry(material, tmp_path):
    ai = MockEditorAI({
        "structure": [{"blocos": [{"clipes": ["P001"]}, {"clipes": ["P002"]}]}],
        "detail": [
            {"cortar": ["S999", "S998", "S001"]},  # 2 de 3 inválidos: nova tentativa
            {"cortar": ["S997", "S002"], "notas": "segunda"},  # 1 de 2 inválido: ainda recusada, mas aproveitável
        ],
    })
    run = _service(material, tmp_path, ai).run()
    first = run.details[0]
    assert len(ai.calls["detail"]) == 3 and first.origin == "ia" and first.notes == "segunda"
    assert [c.sentence_ids for c in first.cuts] == [["S002"]]
    assert [k.sentence_ids for k in first.keep] == [["S000", "S001"], ["S003"]]
    assert any("aceitos os itens válidos" in w for w in first.warnings)
    assert any("frase S997 não existe" in w for w in first.warnings)


def test_duration_out_of_the_target_retries_once_with_the_numbers_and_is_registered(material, tmp_path):
    ai = MockEditorAI({
        "structure": [{"blocos": [{"clipes": ["P001"], "duracao_alvo_sec": 4}], "fora": [{"clipe": "P002", "motivo": "repetido"}]}],
        "detail": [{}, {"notas": "mantive tudo"}],  # nada cortado: 9 s de frases contra 4 s de alvo
    })
    run = _service(material, tmp_path, ai).run()
    [detail] = run.details
    assert len(ai.calls["detail"]) == 2
    assert "ficou 9 s, o alvo deste bloco é 4 s (aceito de 3 a 5 s); ajuste os cortes: corte mais" in ai.calls["detail"][1].prompt
    assert "Duração total das frases deste bloco: 0:09 (9 s); alvo deste bloco: 0:04 (4 s)" in ai.calls["detail"][0].prompt
    assert detail.origin == "ia" and detail.duration_off and detail.notes == "mantive tudo"
    assert (detail.available_sec, detail.result_sec, detail.target_sec) == (9.0, 9.0, 4.0)
    assert any("fora do alvo depois da nova tentativa" in w for w in run.warnings)
    # Dentro do alvo ± 35%: sem nova tentativa.
    ai = MockEditorAI({
        "structure": [{"blocos": [{"clipes": ["P001"], "duracao_alvo_sec": 7}], "fora": [{"clipe": "P002", "motivo": "x"}]}],
        "detail": [{"cortar": ["S003"]}],
    })
    run = _service(material, tmp_path / "b", ai).run()
    assert len(ai.calls["detail"]) == 1 and not run.details[0].duration_off and run.details[0].result_sec == 7.0


def test_structure_completeness_forgotten_speaking_clip_retry_then_own_block(material, tmp_path):
    forgot = {"blocos": [{"clipes": ["P002"]}], "fora": [{"clipe": "B001"}]}  # P001 esquecido; fora sem motivo
    ai = MockEditorAI({"structure": [forgot, forgot]})
    run = _service(material, tmp_path, ai).run()
    retry = ai.calls["structure"][1].prompt
    assert "clipes com fala esquecidos" in retry and "P001" in retry and "em fora sem motivo" in retry
    # Persistiu: P001 vai para um bloco próprio, na ordem do catálogo (antes do bloco de P002), com aviso.
    assert run.structure.origin == "ia" and [b.clip_ids for b in run.structure.blocks] == [["P001"], ["P002"]]
    assert [b.id for b in run.structure.blocks] == ["B01", "B02"]
    assert any("esquecido(s) pela IA (P001)" in w for w in run.warnings)
    # Clipes sem fala podem ficar no acervo; B001 (fora sem motivo) também.
    assert run.structure.library_clip_ids == ["D001", "B001"]


def test_forgotten_clip_goes_after_the_block_with_the_previous_clip_in_the_catalog(material):
    blocks = [Block(id="B01", clip_ids=["P001"]), Block(id="B02", clip_ids=["B001"])]
    placed = place_forgotten(blocks, ["D001"], material)
    assert [(b.id, b.clip_ids) for b in placed] == [("B01", ["P001"]), ("B02", ["D001"]), ("B03", ["B001"])]
    assert [b.clip_ids for b in place_forgotten([Block(id="B01", clip_ids=["B001"])], ["P002"], material)] == [["P002"], ["B001"]]


# Qualquer coisa com cara de ID (letras + dígitos, com ou sem ".M…"/".S…"): S012, P007.M04, X01.S900, B01.
ANY_ID = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,2}\d+(?:\.[A-Z]\d+)?(?![A-Za-z0-9])")


def test_prompts_of_passes_1_and_2_have_no_example_ids():
    context = EditorContext(request="vlog", target_sec=300)
    structure = build_structure_prompt(context, [], block_tokens=4000)
    detail = build_detail_prompt(DetailPromptInput(
        unit_id="B01", block_id="B01", title="t", subject="s", context=context, target_total_sec=300,
        block_target_sec=60, used_before_sec=0.0, part=1, parts=2, available_sec=90, unit_target_sec=30,
    ))
    assert ANY_ID.findall(structure) == []
    assert set(ANY_ID.findall(detail)) == {"B01"}  # só o bloco que está sendo detalhado (código do programa)
    # O formato vem descrito em palavras.
    assert "frase = só o código S… que aparece antes do texto da frase, sem o clipe na frente" in detail
    assert "momento = o código completo que abre o título do momento" in detail
    assert "copiando os códigos exatamente como aparecem na entrada" in structure
    # O padrão pega os IDs que vazaram nos planos reais.
    assert ANY_ID.findall("S012–S015, D002.M05, X01.S900, P009.S025, P007") == [
        "S012", "S015", "D002.M05", "X01.S900", "P009.S025", "P007",
    ]


def test_prompts_with_real_material_only_show_ids_of_the_input(material, tmp_path):
    ai = MockEditorAI()
    _service(material, tmp_path, ai).run()
    known = set(material.clips) | set(material.sentences) | set(material.moments)
    prompts = [r.prompt for r in ai.calls["structure"] + ai.calls["detail"]]
    for prompt in prompts:
        found = set(ANY_ID.findall(prompt))
        assert found - known <= {"B01", "B02"}, found - known
        assert not re.search(r"[A-Z]\d+\.S\d+", prompt)  # nenhuma frase com o clipe na frente


# --- IDs como a IA escreve (editor/ids.py) ---


def test_sentence_with_the_prefix_of_its_own_clip_is_accepted(material):
    answer = BlockAnswer.model_validate({
        "cortar": ["P001.S002", {"de": "p001 . s1", "ate": "p001.s001"}],
        "respiros": [{"depois_de": "P001.S003", "momento": "b1.m1"}],
    })
    check = validate_detail(answer, _unit("P001"), material, ["B001.M01"], Usage())
    assert not check.discarded
    assert [c.sentence_ids for c in check.cuts] == [["S002"], ["S001"]]
    assert [(b.after, b.moment_id) for b in check.breaths] == [("S003", "B001.M01")]
    assert check.normalized == [
        "P001.S002 → S002", "P001 . S1 → S001", "P001.S001 → S001", "P001.S003 → S003",
        "B1.M1 → B001.M01",
    ]


def test_short_number_is_padded_to_the_roteiro_format(material):
    check = validate_detail(BlockAnswer.model_validate({"cortar": ["S2", "s0003"]}), _unit("P001"), material, [], Usage())
    assert [c.sentence_ids for c in check.cuts] == [["S002"], ["S003"]]
    assert check.normalized == ["S2 → S002", "S0003 → S003"]


def test_prefix_of_another_clip_or_unknown_clip_is_refused_with_the_right_format(material):
    answer = BlockAnswer.model_validate({"cortar": ["P002.S002", "X018.S003", "S999"], "incluir": [{"momento": "M01"}]})
    check = validate_detail(answer, _unit("P001", "D001"), material, [], Usage())
    reasons = [d.reason for d in check.discarded]
    assert reasons[0] == "frase P002.S002 não existe (use S002, não P002.S002: S002 é do clipe P001)"
    assert reasons[1] == "frase X018.S003 não existe (use S003, não X018.S003: S003 é do clipe P001)"
    assert "frase S999 não existe (frase = só o código S…" in reasons[2]
    assert reasons[3].startswith("momento M01 não existe (momento = o código completo")
    assert not check.cuts and check.normalized == []


def test_moments_are_only_accepted_if_they_exist(material):
    answer = BlockAnswer.model_validate({"incluir": [
        {"momento": "D001.M09"}, {"momento": "Z001.M01"}, {"momento": "S001"}, {"momento": "d001.m2"},
    ]})
    check = validate_detail(answer, _unit("D001"), material, [], Usage())
    reasons = [d.reason for d in check.discarded]
    assert reasons[0].startswith("momento D001.M09 não existe (momento = o código completo")
    assert reasons[1].startswith("momento Z001.M01 não existe (o clipe Z001 não existe; momento = o código completo")
    assert reasons[2].startswith("momento S001 não existe (S001 é uma frase, não um momento; momento =")
    assert [k.moment_id for k in check.keep] == ["D001.M02"] and check.normalized == ["D001.M2 → D001.M02"]


def test_range_with_one_side_prefixed_or_in_a_single_field(material):
    answer = BlockAnswer.model_validate({
        "cortar": [{"de": "P001.S002", "ate": "S003"}],
        "coberturas": [{"de": "P001.S000–P001.S001", "momentos": ["D001.M01"], "tipo": "mostra o que é dito",
                        "evidencia_fala": "mar", "evidencia_imagem": "mar"}],
    })
    check = validate_detail(answer, _unit("P001"), material, ["D001.M01"], Usage())
    assert not check.discarded
    assert [c.sentence_ids for c in check.cuts] == [["S002", "S003"]]
    assert [(c.from_sentence, c.to_sentence) for c in check.coverages] == [("S000", "S001")]
    assert check.normalized == ["P001.S002 → S002", "P001.S000–P001.S001 → S000–S001"]
    # Um lado de outro clipe: recusado como hoje, com o formato certo.
    bad = BlockAnswer.model_validate({"cortar": [{"de": "P002.S002", "ate": "S003"}]})
    assert "use S002, não P002.S002" in validate_detail(bad, _unit("P001"), material, [], Usage()).discarded[0].reason


def test_structure_clips_are_normalized_and_unknown_get_the_format(material):
    answer = StructureAnswer.model_validate({
        "blocos": [{"clipes": ["p1", "P002", "Q7"]}], "fora": [{"clipe": "d 1", "motivo": "x"}],
    })
    check = validate_structure(answer, material)
    assert check.blocks[0].clip_ids == ["P001", "P002"] and check.out[0].clipe == "D001"
    assert check.normalized == ["P1 → P001", "D 1 → D001"]
    assert "clipe Q7 não existe (clipe = o código que abre a linha do clipe" in check.errors[0]


def test_retry_shows_the_right_format_of_each_refused_id(material, tmp_path):
    ai = MockEditorAI({
        "structure": [{"blocos": [{"clipes": ["P001"]}, {"clipes": ["P002"]}]}],
        "detail": [{"cortar": ["X018.S002", "P002.S003"]}, {"cortar": ["S002"]}],
    })
    run = _service(material, tmp_path, ai).run()
    retry = ai.calls["detail"][1].prompt
    assert "use S002, não X018.S002" in retry and "use S003, não P002.S003" in retry
    assert [c.sentence_ids for c in run.details[0].cuts] == [["S002"]]


def test_report_counts_normalized_ids_per_pass(material, tmp_path):
    ai = MockEditorAI({
        "summary": [{"resumo": "praia", "destaques": [{"ids": ["P001.S000", "p001.m1"], "motivo": "bom"}]}],
        "structure": [{"blocos": [{"clipes": ["p1"]}, {"clipes": ["P002"]}]}],
        "detail": [{"cortar": [{"de": "P001.S002", "ate": "P001.S003"}]}, {"cortar": ["P002.S005"]}],
    })
    run = _service(material, tmp_path, ai).run()
    assert run.summaries[0].normalized_ids == ["P001.S000 → S000", "P001.M1 → P001.M01"]
    assert run.summaries[0].answer.destaques[0].ids == ["S000", "P001.M01"]
    assert run.structure.normalized_ids == ["P1 → P001"]
    assert [len(d.normalized_ids) for d in run.details] == [2, 1]
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    rows = {
        line.split(" | ")[0]: line.rstrip(" |").rsplit(" | ", 1)[-1]
        for line in report.splitlines() if line.startswith("| ") and "(E" in line
    }
    assert rows == {"| Resumos (E9)": "2", "| Estrutura (E10)": "1", "| Detalhe (E11)": "3"}
    assert "IDs normalizados: 2 (P001.S002 → S002; P001.S003 → S003)" in report


def test_stops_calling_after_consecutive_api_errors_and_auth_error_stops_everything(material, tmp_path):
    ai = MockEditorAI({"summary": [AIProviderError("servidor caiu")] * 10})
    run = _service(material, tmp_path, ai).run()
    assert len(ai.calls["summary"]) == 3 and not ai.calls["structure"] and not ai.calls["detail"]
    assert all(s.origin == "falha" for s in run.summaries)
    assert run.structure.origin == "fallback" and all(d.origin == "fallback" for d in run.details)
    with pytest.raises(AIAuthenticationError):
        _service(material, tmp_path / "b", MockEditorAI({"summary": [AIAuthenticationError("mock")]})).run()


def test_cache_per_pass_changing_the_detail_prompt_keeps_summaries(material, tmp_path, monkeypatch):
    first = MockEditorAI()
    _service(material, tmp_path, first).run()
    again = MockEditorAI()
    run = _service(material, tmp_path, again).run()
    assert not any(again.calls.values())  # tudo do cache
    assert {s.origin for s in run.summaries} == {"cache"} and run.structure.origin == "cache"
    assert {d.origin for d in run.details} == {"cache"}

    monkeypatch.setattr(editor_service, "DETAIL_PROMPT_VERSION", "detalhe-teste")
    third = MockEditorAI()
    _service(material, tmp_path, third).run()
    assert not third.calls["summary"] and not third.calls["structure"] and len(third.calls["detail"]) == 2

    monkeypatch.setattr(editor_service, "SUMMARY_PROMPT_VERSION", "resumo-teste")
    fourth = MockEditorAI()
    _service(material, tmp_path, fourth).run()
    assert len(fourth.calls["summary"]) == 4  # mudar o pedido do resumo refaz os resumos


def test_budget_splits_a_big_block_by_clips_and_a_giant_clip_by_moments(material, tmp_path):
    block = _block("P001", "P002")
    probe = _service(material, tmp_path, num_ctx=100_000)
    fixed = 100_000 - probe._content_budget(block)  # tudo o que não é seção de clipe
    size = {c: tokens(material.section(c) + "\n") for c in ("P001", "P002")}

    by_clips = _service(material, tmp_path, num_ctx=fixed + size["P001"] + 5)
    units = by_clips._units(block)
    assert [(u.unit_id, u.clip_ids, u.moment_ids) for u in units] == [("B01a", ["P001"], None), ("B01b", ["P002"], None)]

    by_moments = _service(material, tmp_path, num_ctx=fixed + size["P001"] - 5)
    units = by_moments._units(block)
    assert [(u.unit_id, u.moment_ids) for u in units][:2] == [("B01a", ["P001.M01"]), ("B01b", ["P001.M02"])]
    assert all(u.parts == len(units) for u in units)

    # Rodando com o contexto pequeno: cada pedido de detalhe cabe no disponível e o aviso aparece.
    ai = MockEditorAI({"structure": [{"blocos": [{"clipes": ["P001", "P002"]}]}]})
    service = _service(material, tmp_path / "pequeno", ai, num_ctx=fixed + size["P001"] + 5)
    run = service.run()
    assert [d.unit_id for d in run.details] == ["B01a", "B01b"]
    assert all(tokens(r.prompt) <= service.budget.detail_available for r in ai.calls["detail"])
    assert any("dividido em 2 pedaços" in w for w in run.warnings)
    assert "Pedaço 1 de 2 do bloco" in ai.calls["detail"][0].prompt


def test_huge_clip_summary_is_split_in_parts_and_joined(material, tmp_path):
    whole = tokens(build_summary_prompt(material.section("P001")))
    service = _service(material, tmp_path, num_ctx=whole + 1200 - 10)  # 1200 = saída do resumo + margem
    assert service._summary_parts("P001") == [["P001.M01"], ["P001.M02"]]
    ai = MockEditorAI()
    run = _service(material, tmp_path, ai, num_ctx=whole + 1200 - 10).run()
    p001 = run.summaries[0]
    assert p001.parts == 2 and p001.answer.resumo.count("Clipe P001") == 2
    assert "pedaço 1 de 2 do clipe" in ai.calls["summary"][0].prompt


def test_resolve_budget_uses_override_ollama_num_ctx_or_cloud(monkeypatch):
    assert resolve_budget(["gemini"], 5000).num_ctx == 5000
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "4096")
    assert resolve_budget(["ollama", "gemini"]).num_ctx == 4096
    assert resolve_budget(["gemini"]).source == "padrão da nuvem"


def test_llm_editor_reads_the_schema_through_the_llm_layer(material):
    client = MockLLMClient(responses=[{"resumo": "Um clipe na praia.", "tom": "leve"}])
    chain = ModelChain([ModelSpec(provider="mock", model_id="mock")], thinking="off", clients={"mock": client})
    answer = LLMEditorAI(chain).summarize(SummaryRequest(clip_id="P001", prompt="resuma", material=material))
    assert answer.resumo == "Um clipe na praia." and client.calls[0][0] == "resuma"


# --- montagem ---


def _audio(duration: float, speech: list[tuple[float, float]]) -> AudioAnalysisResult:
    silences, cursor = [], 0.0
    for a, b in speech:
        if a > cursor:
            silences.append(SilenceInterval(start_sec=cursor, end_sec=a, duration=round(a - cursor, 3)))
        cursor = b
    if cursor < duration:
        silences.append(SilenceInterval(start_sec=cursor, end_sec=duration, duration=round(duration - cursor, 3)))
    return AudioAnalysisResult(
        source_path=Path("x.mp4"), duration_sec=duration, params=AudioAnalysisParams(threshold_db=-35, min_silence_sec=0.3),
        silences=silences, speech=[SilenceInterval(start_sec=a, end_sec=b, duration=round(b - a, 3)) for a, b in speech],
    )


def _sources(material) -> dict[str, ClipSource]:
    speech = {"P001": [(0.5, 2.5), (3.0, 5.5), (6.5, 9.0), (9.5, 11.5)], "P002": [(0.5, 2.5), (3.0, 5.0)]}
    result = {}
    for clip in material.clips.values():
        result[clip.id] = ClipSource(
            clip_id=clip.id, path=Path(f"{clip.id}.mp4"), fps=25.0, duration_sec=clip.duration_sec, has_audio=clip.has_audio,
            audio=_audio(clip.duration_sec, speech[clip.id]) if clip.id in speech else None,
        )
    return result


def _run_with(details: list[DetailRun], blocks: list[Block], library=("D001", "B001")) -> EditorRun:
    from ai_video_editor.editor.schema import BudgetInfo

    return EditorRun(
        provider="mock",
        budget=BudgetInfo(num_ctx=1, source="t", summary_output=1, structure_output=1, detail_output=1, margin=1, token_factor=1),
        structure=StructureRun(origin="ia", blocks=blocks, library_clip_ids=list(library)),
        details=details,
    )


def test_faixa_cut_on_word_borders_without_the_removed_retake(material):
    sources = _sources(material)
    keep = KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S002",
                     sentence_ids=["S000", "S001", "S002"])
    ranges = [(r.start_sec, r.end_sec) for r in faixa_ranges(keep, material, sources["P001"], SilenceCutParams())]
    # Silêncio cortado entre S001 e S002; S003 (retake) fora, cortado dentro do silêncio antes dela.
    assert ranges == [(0.35, 5.7), (6.35, 9.2)]
    assert ranges[0][0] <= 0.5 and ranges[-1][1] >= 9.0 and ranges[-1][1] <= 9.5


def test_assembly_breath_in_the_middle_coverage_split_between_moments_and_short_moment(material, tmp_path):
    detail = DetailRun(
        unit_id="B01", block_id="B01", clip_ids=["P001"], origin="ia",
        keep=[KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S002",
                        sentence_ids=["S000", "S001", "S002"])],
        breaths=[BreathChoice(after="S000", moment_id="B001.M02", duration_sec=1.5)],
        coverages=[CoverageChoice(
            from_sentence="S001", to_sentence="S002", sentence_ids=["S001", "S002"], clip_id="P001",
            moment_ids=["D001.M01", "B001.M01"], speech_evidence="mar", image_evidence="mar",
        )],
    )
    assembled = assemble_plan(_run_with([detail], [_block("P001")]), material, _sources(material), SilenceCutParams(),
                              tmp_path / "out.mp4", "teste")
    timeline = assembled.timeline
    main = [(i.label, i.source_in, i.source_out) for i in timeline.video_main]
    assert main == [
        ("P001 S000–S002", 0.35, 2.75), ("respiro B001.M02", 1.0, 2.5), ("P001 S000–S002", 2.75, 5.7),
        ("P001 S000–S002", 6.35, 9.2),
    ]
    # Cobertura de S001 (no 3º item) até o fim de S002: D001.M01 inteiro (4 s) e depois B001.M01 (1 s), em ordem;
    # o que faltou fica sem cobertura (nunca estica nem repete).
    overlays = [(o.clip_id, o.source_in, round(o.source_out, 3)) for o in timeline.video_overlay]
    assert overlays == [("D001", 0.0, 4.0), ("B001", 0.0, 1.0)]
    # S001 começa 0,25 s depois da borda do item: a cobertura encosta na borda (sem pedaço de poucos frames).
    assert timeline.video_overlay[0].timeline_start == pytest.approx(2.4 + 1.5, abs=0.05)
    assert any("ficam sem cobertura" in w for w in assembled.warnings)
    # Ambiente baixinho só do momento com áudio (o drone não tem).
    assert [a.clip_id for a in timeline.audio_ambient] == ["B001"]
    pieces = [(p.kind, p.ids) for p in assembled.plan.blocks[0].pieces]
    assert pieces[:2] == [("frases", ["S000"]), ("respiro", ["B001.M02"])]
    assert assembled.plan.blocks[0].coverages[0].uncovered_sec > 0


def test_assembly_moment_item_uses_requested_duration_and_whole_moment_without_it(material, tmp_path):
    detail = DetailRun(
        unit_id="B01", block_id="B01", clip_ids=["D001"], origin="ia",
        keep=[KeepRange(kind="momento", clip_id="D001", moment_id="D001.M02", duration_sec=2.5),
              KeepRange(kind="momento", clip_id="D001", moment_id="D001.M01")],
    )
    timeline = assemble_plan(_run_with([detail], [_block("D001")], library=()), material, _sources(material),
                             SilenceCutParams(), tmp_path / "o.mp4", "t").timeline
    assert [(i.label, i.source_in, i.source_out, i.use_audio) for i in timeline.video_main] == [
        ("momento D001.M02", 4.0, 6.5, False), ("momento D001.M01", 0.0, 4.0, False),
    ]


# --- ponta a ponta (FFmpeg) ---

SCRIPTS = {
    "fala": [(0.1, "Chegamos na praia bonita."), (3.6, "Olha o campo verde lá embaixo.")],
}
SPEECH = {"fala": [(0.0, 2.0), (3.5, 6.0)]}


class ScriptTranscriber(TranscriptionProvider):
    """Transcrição simulada pelo nome do arquivo (cada palavra 0,3 s)."""

    @property
    def provider_name(self) -> str:
        return "mock-editor"

    def transcribe(self, media_path: Path, config: TranscriptionConfig, on_progress=None) -> Transcript:
        segments = []
        for i, (start, text) in enumerate(SCRIPTS[Path(media_path).stem]):
            words = [
                Word(text=t, start_sec=round(start + k * 0.3, 3), end_sec=round(start + (k + 1) * 0.3 - 0.02, 3))
                for k, t in enumerate(text.split())
            ]
            segments.append(TranscriptSegment(id=i, start_sec=start, end_sec=words[-1].end_sec, text=text, words=words))
        return Transcript(
            source_path=media_path, language="pt", duration_sec=segments[-1].end_sec, model_used="mock", segments=segments
        )


class ScriptDetector(SpeechDetector):
    @property
    def name(self) -> str:
        return "teste-editor"

    def detect(self, path: Path, duration_sec: float) -> SpeechDetection:
        return speech_result(merge_segments(SPEECH.get(path.stem, [])), duration_sec, self.name)


def _speech_video(path: Path) -> Path:
    return make_video(path, [("red", 3), ("blue", 3)], [("tone", 2.0), ("silence", 1.5), ("tone", 2.5)])


@pytest.fixture(scope="session")
def editor_folder(tmp_path_factory) -> Path:
    """Somente leitura: principal com fala, b-roll com som e sem fala, drone sem áudio (verde)."""
    root = tmp_path_factory.mktemp("editor") / "viagem"
    _speech_video(root / "principal" / "fala.mp4")
    make_video(root / "broll" / "rua.mp4", [("yellow", 3)], [("tone", 3.0)])
    make_video(root / "drone" / "voo.mp4", [("green", 4)])
    return root


@pytest.fixture(scope="session")
def drone_folder(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("so_drone") / "voos"
    make_video(root / "drone" / "voo.mp4", [("green", 3), ("blue", 3)])
    make_video(root / "drone" / "mar.mp4", [("blue", 3)])
    return root


def _pipeline(tmp_path: Path, editor_ai=None) -> EditorPipeline:
    describer, _ = mock_describer()
    return EditorPipeline(
        projects_root=tmp_path / "projects", transcription_provider=ScriptTranscriber(), speech_detector=ScriptDetector(),
        describer=describer, editor_ai=editor_ai,
    )


def _params() -> EditPipelineParams:
    return EditPipelineParams(transcribe=True)


@pytest.mark.slow
def test_end_to_end_folder_renders_a_valid_timeline_with_coverage(editor_folder, tmp_path):
    result = _pipeline(tmp_path).run(editor_folder, _params(), EditorParams(provider="mock"), EditorContext(request="vlog"))
    assert result.output_path is not None and result.output_path.is_file() and result.subtitles_path.is_file()
    timeline = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    assert {i.clip_id for i in timeline.video_main if not (i.label or "").startswith("respiro")} == {"P001"}
    # O drone verde cobre "campo verde"; o b-roll (sem fala) fica no acervo e entra como respiro.
    assert [o.clip_id for o in timeline.video_overlay] == ["D001"]
    assert any((i.label or "").startswith("respiro B001") for i in timeline.video_main)
    assert result.final_duration_sec == pytest.approx(timeline.duration_sec, abs=0.1)
    report = result.report_path.read_text(encoding="utf-8")
    assert "## Estrutura (passada 1)" in report and "cobertura S000–S001 ← D001.M01" in report
    assert "fala: \"verde\"" in report and "## Chamadas e tokens" in report
    assert result.plan_md_path.is_file() and "## B01" in result.plan_md_path.read_text(encoding="utf-8")
    # Mesmo material: tudo do cache (nada de IA, render reaproveitado).
    again = _pipeline(tmp_path).run(editor_folder, _params(), EditorParams(provider="mock"), EditorContext(request="vlog"))
    assert again.outcomes["editor"] == "cached" and again.outcomes["render"] == "cached"


@pytest.mark.slow
def test_plan_only_does_not_render_and_single_video_is_one_clip(editor_folder, tmp_path):
    result = _pipeline(tmp_path).run(editor_folder, _params(), EditorParams(provider="mock", plan_only=True))
    assert result.outcomes["render"] == "skipped" and result.outcomes["subtitles"] == "skipped"
    assert result.output_path is None and not list(result.work_dir.glob("*.mp4"))
    assert result.plan_path.is_file() and result.timeline_path.is_file()
    assert "sem render (`--so-plano`)" in result.report_path.read_text(encoding="utf-8")

    video = _speech_video(tmp_path / "solo" / "fala.mp4")
    single = _pipeline(tmp_path).run(video, _params(), EditorParams(provider="mock", plan_only=True))
    assert single.outcomes["catalog"] == "skipped" and single.blocks == 1
    timeline = Timeline.model_validate_json(single.timeline_path.read_text(encoding="utf-8"))
    assert {i.clip_id for i in timeline.video_main} == {"P001"}


@pytest.mark.slow
def test_folder_without_speech_is_edited_by_moments(drone_folder, tmp_path):
    result = _pipeline(tmp_path).run(drone_folder, _params(), EditorParams(provider="mock", plan_only=True))
    timeline = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    assert timeline.video_main and all((i.label or "").startswith("momento ") for i in timeline.video_main)
    assert {i.clip_id for i in timeline.video_main} == {"D001", "D002"}


@pytest.mark.slow
def test_cli_editor_ia_plan_only(editor_folder, tmp_path, capsys, monkeypatch):
    original = EditorPipeline.__init__

    def init(self, *args, **kwargs):
        kwargs.update(transcription_provider=ScriptTranscriber(), speech_detector=ScriptDetector())
        original(self, *args, **kwargs)

    monkeypatch.setattr(EditorPipeline, "__init__", init)
    args = build_parser().parse_args([
        "edit", str(editor_folder), "--receita", "editor_ia", "--editor-provider", "mock", "--visao-provider", "mock",
        "--so-plano", "--sem-briefing", "--no-live", "--projects-dir", str(tmp_path / "projects"),
        "--pedido", "um vlog curto",
    ])
    assert handle_edit(args) == 0
    out = capsys.readouterr().out
    assert "Plano do Editor de IA pronto (sem render)." in out and "[editor] mock:mock" in out
    assert "Estrutura: 1 bloco(s)" in out and "editor_plano.md" in out
    assert not list((tmp_path / "projects").glob("*/editor_ia/*.mp4"))
