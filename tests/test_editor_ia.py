"""Testes do Editor de IA (receita `editor_ia`): schemas, calibração do orçamento e divisão, as três perguntas
da passada 2 (2a cortes, 2b inclusões e respiros, 2c coberturas) com validação item a item, nova tentativa e
fallbacks neutros, caches por passada, montagem (IDs → Timeline) e ponta a ponta com clipes sintéticos
(render curto marcado `slow`). Sem rede: IA simulada (`MockEditorAI`)."""

import math
import re
from pathlib import Path
from typing import Optional

import pytest

from ai_video_editor.audio.schema import AudioAnalysisParams, AudioAnalysisResult, SilenceInterval
from ai_video_editor.catalog.schema import SpeechDetection
from ai_video_editor.catalog.speech import SpeechDetector, merge_segments, speech_result
from ai_video_editor.cli import build_parser, handle_edit
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.coverage.assemble import BREATH_VOLUME
from ai_video_editor.editor import EditorContext, EditorMaterial
from ai_video_editor.editor import budget as editor_budget
from ai_video_editor.editor import service as editor_service
from ai_video_editor.editor.assemble import ClipSource, assemble_plan, faixa_ranges
from ai_video_editor.editor.budget import Budget, resolve_budget, tokens
from ai_video_editor.editor.ids import IdResolver
from ai_video_editor.editor.material import coverage_matches, library_ranking, matched_words
from ai_video_editor.editor.prompt import (
    BlockHeader,
    CoveragePromptInput,
    CutsPromptInput,
    InsertsPromptInput,
    build_coverage_prompt,
    build_cuts_prompt,
    build_inserts_prompt,
    build_structure_prompt,
    build_summary_prompt,
)
from ai_video_editor.editor.provider import SummaryRequest
from ai_video_editor.editor.providers.llm import LLMEditorAI
from ai_video_editor.editor.providers.mock import MockEditorAI
from ai_video_editor.editor.report import build_editor_report
from ai_video_editor.editor.schema import (
    COVERAGE_PLACE,
    COVERAGE_SAYS,
    Block,
    Candidate,
    ClipSummaryAnswer,
    CoverageChoice,
    CoverageItem,
    CoveragesAnswer,
    CutItem,
    CutsAnswer,
    DetailRun,
    EditorPlan,
    EditorRun,
    InsertItem,
    InsertsAnswer,
    KeepRange,
    StructureAnswer,
    StructureRun,
)
from ai_video_editor.editor.service import CallCache, EditorService
from ai_video_editor.editor.validate import (
    Usage,
    build_trechos,
    place_forgotten,
    take_sentences,
    validate_coverages,
    validate_cuts,
    validate_inserts,
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


def _block(*clips, block_id="B01", target=None) -> Block:
    return Block(id=block_id, clip_ids=list(clips), target_sec=target)


def _service(material, tmp_path, ai=None, num_ctx=32768, context=None) -> EditorService:
    return EditorService(
        ai or MockEditorAI(), material, context or EditorContext(request="vlog da viagem"),
        Budget(num_ctx=num_ctx, source="teste"), CallCache(tmp_path / "cache"),
    )


def _structure(*blocks, fora=()) -> dict:
    """Resposta fixa da estrutura: cada bloco = (clipes, alvo opcional)."""
    result = {"blocos": [], "fora": [{"clipe": c, "motivo": "teste"} for c in fora]}
    for block in blocks:
        clips, target = (block, None) if isinstance(block[0], str) else block
        result["blocos"].append({"clipes": list(clips), "duracao_alvo_sec": target})
    return result


# --- schemas e receita ---


def test_answers_are_tolerant_and_trimmed():
    assert CutItem.model_validate(" s012 ").frase == "S012"
    assert CutItem.model_validate({"id": "s3", "motivo": " recomeço "}).model_dump() == {"frase": "S3", "motivo": "recomeço"}
    assert CutItem.model_validate({"de": "S020"}).frase == "S020"
    item = InsertItem.model_validate({"depois_do_trecho": "trecho 2", "letra": "**b**", "duracao_sec": "3,5s"})
    assert (item.depois_do_trecho, item.letra, item.duracao_sec) == (2, "B", 3.5)
    assert InsertItem.model_validate({"depois_do_trecho": "início", "letra": "letra C"}).model_dump()["depois_do_trecho"] == 0
    assert InsertItem.model_validate({"letra": "A)", "duracao_sec": "auto"}).letra == "A"
    assert CoverageItem.model_validate({"trecho": "3", "letra": "Nenhuma.", "palavra": ""}).none
    assert not CoverageItem.model_validate({"trecho": 3, "letra": "c", "palavra": " lago "}).none
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


# --- orçamento: calibração medida ---

# (passada, caracteres, tokens reais) medidos no vlog real com o qwen2.5:7b (prompt_eval_count do Ollama):
# o maior real/estimado de cada passada e o B07 que estourou o contexto (7932 reais, ~5160 caracteres/4).
MEASURED = [
    ("resumo P025", 3291, 1378), ("resumo P009", 4307, 1671), ("resumo P019", 4472, 1719),
    ("estrutura v3 (compacta)", 13857, 5331), ("estrutura v4", 9100, 3429),
    ("detalhe B01", 12069, 4307), ("detalhe B07 (erro real)", 5160 * 4, 7932),
    ("2a cortes B20.P025", 4100, 1663), ("2b inclusões B01", 6456, 2455), ("2c coberturas B06", 2442, 873),
]


@pytest.mark.parametrize("name, chars, real", MEASURED, ids=[m[0] for m in MEASURED])
def test_token_estimate_is_calibrated_above_the_measured_real_tokens(name, chars, real):
    estimated = tokens("x" * chars)
    assert estimated >= real, f"{name}: estimado {estimated} < real {real}"
    assert estimated <= real * 1.30  # e não superestima demais (pedidos divididos sem necessidade)


def test_old_factor_underestimated_the_real_tokens():
    # Com o fator antigo (1,25) o B07 estimava ~6450 tokens; o Ollama contou 7932 e o pedido estourou.
    assert math.ceil(5160 * 1.25) < 7932 <= tokens("x" * 5160 * 4)
    assert editor_budget.TOKEN_FACTOR == pytest.approx(1.70)


def make_big_material(tmp_path: Path, clips: int = 1, sentences: int = 120, words: int = 14) -> EditorMaterial:
    """Clipes longos com fala (frases e momentos repetidos), para o orçamento."""
    roteiro_clips, moments, roteiro_sentences = [], [], []
    index = 0
    for c in range(1, clips + 1):
        clip_id = f"P{c:03d}"
        roteiro_clips.append(_clip(clip_id, "principal", c, sentences * 4.0, True))
        for m in range(sentences // 4):
            ids = [f"S{index + k:03d}" for k in range(4)]
            moments.append(_moment(clip_id, m + 1, m * 16.0, (m + 1) * 16.0, f"pessoa caminhando na trilha {m}",
                                   "trilha", ["trilha"], ids, ["tremida"]))
            for k in range(4):
                start = m * 16.0 + k * 4.0
                text = " ".join(f"palavra{(index + w) % 97}" for w in range(words))
                roteiro_sentences.append(_sentence(index, clip_id, start + 0.2, start + 3.6, text, f"{clip_id}.M{m + 1:02d}"))
                index += 1
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0,
                             clips=roteiro_clips, moments=moments, sentences=roteiro_sentences)
    return EditorMaterial.from_roteiro(roteiro)


def test_b07_case_fits_after_the_calibration_when_split(tmp_path):
    """Um clipe cuja 2a passa do contexto (como o B07, ~20 mil caracteres) é dividido por momentos; cada parte,
    com o PIOR real/estimado medido (1,674), cabe em num_ctx = 8192 com a saída reservada."""
    big = make_big_material(tmp_path, sentences=120)
    assert len(big.section("P001")) > 5160 * 4
    service = _service(big, tmp_path, num_ctx=8192)
    header = BlockHeader(block_id="B07", title="", subject="", context=EditorContext(request="vlog"))
    ids = [s.id for s in big.clip_sentences["P001"]]
    parts = service._cut_parts(header, "P001", ids, 120.0, 1)
    assert len(parts) >= 2 and [s for _, p in parts for s in p] == ids
    for moment_ids, part_ids in parts:
        prompt = build_cuts_prompt(CutsPromptInput(
            header=header, clip_id="P001", section=big.section("P001", moment_ids), available_sec=0.0,
            target_sec=60.0, part=1, parts=len(parts),
        ))
        assert tokens(prompt) <= service.budget.cuts_available
        worst_real = math.ceil(len(prompt) / 4 * 1.674)
        assert worst_real + editor_budget.CUTS_OUTPUT_TOKENS <= 8192
        retry = build_cuts_prompt(CutsPromptInput(
            header=header, clip_id="P001", section=big.section("P001", moment_ids), available_sec=0.0,
            target_sec=60.0, part=1, parts=len(parts),
        ), ["x" * 400] * 20)  # a nova tentativa com a lista de erros mais longa possível também cabe
        assert tokens(retry) <= service.budget.cuts_available + editor_budget.RETRY_TOKENS


def test_resolve_budget_uses_override_ollama_num_ctx_or_cloud(monkeypatch):
    assert resolve_budget(["gemini"], 5000).num_ctx == 5000
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "4096")
    assert resolve_budget(["ollama", "gemini"]).num_ctx == 4096
    assert resolve_budget(["gemini"]).source == "padrão da nuvem"


def test_huge_clip_summary_is_split_in_parts_and_joined(material, tmp_path):
    whole = tokens(build_summary_prompt(material.section("P001")))
    reserved = editor_budget.SUMMARY_OUTPUT_TOKENS + editor_budget.MARGIN_TOKENS + editor_budget.RETRY_TOKENS
    service = _service(material, tmp_path, num_ctx=whole + reserved - 10)
    assert service._summary_parts("P001") == [["P001.M01"], ["P001.M02"]]
    ai = MockEditorAI()
    run = _service(material, tmp_path, ai, num_ctx=whole + reserved - 10).run()
    p001 = run.summaries[0]
    assert p001.parts == 2 and p001.answer.resumo.count("Clipe P001") == 2
    assert "pedaço 1 de 2 do clipe" in ai.calls["summary"][0].prompt


# --- passada 1 (estrutura) ---


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
    assert run.structure.library_clip_ids == ["D001", "B001"]


def test_forgotten_clip_goes_after_the_block_with_the_previous_clip_in_the_catalog(material):
    blocks = [Block(id="B01", clip_ids=["P001"]), Block(id="B02", clip_ids=["B001"])]
    placed = place_forgotten(blocks, ["D001"], material)
    assert [(b.id, b.clip_ids) for b in placed] == [("B01", ["P001"]), ("B02", ["D001"]), ("B03", ["B001"])]
    assert [b.clip_ids for b in place_forgotten([Block(id="B01", clip_ids=["B001"])], ["P002"], material)] == [["P002"], ["B001"]]


def make_vlog_material(tmp_path: Path, clips: int = 41) -> EditorMaterial:
    """41 clipes como o vlog real (28 com fala), cada um com 2 frases e 2 momentos."""
    roteiro_clips, moments, sentences = [], [], []
    for n in range(1, clips + 1):
        speech = n <= 28
        clip_id = f"P{n:03d}" if speech else f"B{n - 28:03d}"
        roteiro_clips.append(_clip(clip_id, "principal" if speech else "broll", n, 30.0, speech))
        ids = [f"S{2 * n:03d}", f"S{2 * n + 1:03d}"] if speech else []
        moments.append(_moment(clip_id, 1, 0.0, 15.0, "pessoa na trilha do lago", "lago", ["lago"], ids))
        moments.append(_moment(clip_id, 2, 15.0, 30.0, "vista do lago com montanhas", "lago", ["montanha"]))
        if speech:
            sentences.append(_sentence(2 * n, clip_id, 1.0, 6.0, "Chegamos no lago depois da trilha.", f"{clip_id}.M01"))
            sentences.append(_sentence(2 * n + 1, clip_id, 7.0, 12.0, "A água está gelada demais.", f"{clip_id}.M01"))
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0,
                             clips=roteiro_clips, moments=moments, sentences=sentences)
    return EditorMaterial.from_roteiro(roteiro)


def test_structure_prompt_lists_all_41_clips_in_one_line_each_without_shortening(tmp_path):
    vlog = make_vlog_material(tmp_path)
    long_summary = "Ele caminha pela trilha até o lago e mostra a água gelada. " * 5  # 60 palavras
    ai = MockEditorAI({"summary": [{"resumo": long_summary, "tom": "leve"}] * 41})
    run = _service(vlog, tmp_path, ai, num_ctx=8192).run()
    prompt = ai.calls["structure"][0].prompt
    lines = [line for line in prompt.splitlines() if line.startswith("- ") and " — " in line]
    assert [line.split(" · ")[0][2:] for line in lines] == vlog.clip_ids  # todos, na ordem
    assert all(len(line.split(" — ", 1)[1].split()) == 25 for line in lines)  # resumo em 25 palavras
    assert "P001 · principal · 0:30 · fala sim (10 s de frases) — Ele caminha pela trilha" in prompt
    assert "B001 · broll · 0:30 · fala não — " in prompt
    assert tokens(prompt) <= Budget(8192, "t").structure_available
    assert not any("encurtad" in w for w in run.warnings) and run.structure.origin == "ia"


def test_structure_prompt_without_room_goes_to_the_neutral_fallback(material, tmp_path):
    run = _service(material, tmp_path, num_ctx=3000).run()  # nem a estrutura de 4 clipes cabe
    assert run.structure.origin == "fallback" and "não cabe no contexto" in run.structure.attempts[0]


# --- IDs como a IA escreve (editor/ids.py) ---


def test_markdown_and_punctuation_around_ids_are_removed(material):
    ids = IdResolver(material)
    assert ids.moment("### P001.M02") == "P001.M02"
    assert ids.sentence("**S002**") == "S002"
    assert ids.sentence('"P001.S003".') == "S003"
    assert ids.clip("[p1]") == "P001"
    assert ids.fixed == ["### P001.M02 → P001.M02", "**S002** → S002", '"P001.S003". → S003', "[p1] → P001"]
    check = validate_cuts(CutsAnswer.model_validate({"cortar": ["### S001", {"frase": "**s2**"}]}),
                          ["S000", "S001", "S002", "S003"], material)
    assert [c.sentence_id for c in check.cuts] == ["S001", "S002"] and not check.discarded


def test_sentence_with_the_prefix_of_its_own_clip_is_accepted_and_another_clip_is_refused(material):
    answer = CutsAnswer.model_validate({"cortar": ["P001.S002", "p001 . s1", "S3", "P002.S000", "X018.S000", "S999"]})
    check = validate_cuts(answer, ["S000", "S001", "S002", "S003"], material)
    assert [c.sentence_id for c in check.cuts] == ["S001", "S002", "S003"]
    assert check.normalized == ["P001.S002 → S002", "P001 . S1 → S001", "S3 → S003"]
    reasons = [d.reason for d in check.discarded]
    assert reasons[0] == "frase P002.S000 não existe (use S000, não P002.S000: S000 é do clipe P001)"
    assert reasons[1] == "frase X018.S000 não existe (use S000, não X018.S000: S000 é do clipe P001)"
    assert "frase S999 não existe (frase = só o código S…" in reasons[2]


def test_structure_clips_are_normalized_and_unknown_get_the_format(material):
    answer = StructureAnswer.model_validate({
        "blocos": [{"clipes": ["p1", "### P002", "Q7"]}], "fora": [{"clipe": "d 1", "motivo": "x"}],
    })
    check = validate_structure(answer, material)
    assert check.blocks[0].clip_ids == ["P001", "P002"] and check.out[0].clipe == "D001"
    assert check.normalized == ["P1 → P001", "### P002 → P002", "D 1 → D001"]
    assert "clipe Q7 não existe (clipe = o código que abre a linha do clipe" in check.errors[0]


# --- 2a: cortes ---


def test_cuts_one_sentence_per_item_from_this_clip_only(material):
    answer = CutsAnswer.model_validate({"cortar": [
        {"frase": "S002", "motivo": "recomeço"},
        "S000–S001",  # faixa: uma frase por item
        "S001, S003",  # várias frases num item
        "S004",  # de outro clipe
        "S002",  # repetida
        {"motivo": "sem frase"},
    ]})
    check = validate_cuts(answer, ["S000", "S001", "S002", "S003"], material)
    assert [(c.sentence_id, c.reason) for c in check.cuts] == [("S002", "recomeço")]
    assert check.kept == ["S000", "S001", "S003"] and check.result_sec == pytest.approx(6.5)
    reasons = " | ".join(d.reason for d in check.discarded)
    assert "S000–S001: uma frase por item" in reasons and "S001, S003: uma frase por item" in reasons
    assert "frase S004 não é deste clipe nem desta parte (é do clipe P002)" in reasons
    assert "frase S002 repetida" in reasons and "item sem frase" in reasons
    assert check.refused == "5 de 6 itens inválidos" and check.unusable is None
    everything = validate_cuts(CutsAnswer.model_validate({"cortar": ["S000", "S001"]}), ["S000", "S001"], material)
    assert "todas as frases do clipe foram cortadas" in everything.refused and everything.unusable is None
    assert validate_cuts(CutsAnswer(), ["S000"], material).refused is None  # nada cortado: aceita


def test_cuts_target_is_split_by_the_speech_of_each_clip_and_retried_by_duration(material, tmp_path):
    ai = MockEditorAI({
        "structure": [_structure((("P001", "P002"), 6.5))],
        "cuts": [
            {}, {"cortar": ["S002", "S003"], "notas": "retake"},  # P001: 9 s × meta 4,5 → nova tentativa → 4,5 s
            {}, {"notas": "mantive"},  # P002: 4 s × meta 2 → fora; persiste e é aceito
        ],
    })
    run = _service(material, tmp_path, ai).run()
    [detail] = run.details
    first, second = detail.cut_runs
    # Meta do clipe = alvo do bloco × fala do clipe / fala do bloco: 6,5 × 9/13 e 6,5 × 4/13.
    assert (first.unit_id, first.target_sec, second.target_sec) == ("B01.P001", 4.5, 2.0)
    assert "Meta: ficar com cerca de 0:04 (4 s) de frases" in ai.calls["cuts"][0].prompt
    assert "corte cerca de 4 s de frases" in ai.calls["cuts"][0].prompt
    assert "ficou 9 s, a meta deste clipe é 4 s (aceito de 3 a 6 s); corte mais frases" in ai.calls["cuts"][1].prompt
    assert (first.result_sec, first.duration_off, first.notes) == (4.5, False, "retake")
    assert (second.result_sec, second.duration_off, second.origin) == (4.0, True, "ia")
    assert any("fora da meta depois da nova tentativa" in w for w in run.warnings)
    assert len(ai.calls["cuts"]) == 4
    # O que não foi cortado fica, em trechos na ordem do roteiro.
    assert [(t.number, t.sentence_ids) for t in detail.trechos] == [(1, ["S000", "S001"]), (2, ["S004", "S005"])]


def test_cuts_of_a_big_clip_are_asked_in_parts_with_proportional_targets(tmp_path):
    big = make_big_material(tmp_path, sentences=120)
    ai = MockEditorAI({"structure": [_structure((("P001",), 200))]})
    run = _service(big, tmp_path, ai, num_ctx=8192).run()
    runs = run.details[0].cut_runs
    assert len(runs) >= 2 and [r.unit_id for r in runs][:2] == ["B01.P001a", "B01.P001b"]
    assert sum(r.target_sec for r in runs) == pytest.approx(200, abs=0.5)
    assert all(tokens(r.prompt) <= Budget(8192, "t").cuts_available for r in ai.calls["cuts"])
    assert "parte 1 de" in ai.calls["cuts"][0].prompt


def test_cuts_api_error_falls_back_to_nothing_cut(material, tmp_path):
    ai = MockEditorAI({"structure": [_structure(("P001",), ("P002",))],
                       "cuts": [AIProviderError("caiu"), AIProviderError("caiu de novo")]})
    run = _service(material, tmp_path, ai).run()
    first = run.details[0].cut_runs[0]
    assert first.origin == "fallback" and first.kept_ids == ["S000", "S001", "S002", "S003"]
    assert run.details[0].origin == "fallback" and any("CORTES B01.P001 PELO FALLBACK NEUTRO" in w for w in run.warnings)


# --- 2b: inclusões e respiros ---


def _candidates(*items) -> list[Candidate]:
    return [Candidate(letter=letter, moment_id=mid, clip_id=mid.split(".")[0], duration_sec=dur, source=source)
            for letter, mid, dur, source in items]


def test_inserts_letters_and_trecho_numbers_become_ids(material):
    trechos = build_trechos(["P001"], {"S000", "S001", "S003"}, material)
    assert [(t.number, t.sentence_ids) for t in trechos] == [(1, ["S000", "S001"]), (2, ["S003"])]
    candidates = _candidates(("A", "D001.M01", 4.0, "bloco"), ("B", "B001.M02", 2.0, "acervo"), ("C", "D001.M02", 4.0, "bloco"))
    answer = InsertsAnswer.model_validate({"inserir": [
        {"depois_do_trecho": 0, "letra": "A"},  # começo do bloco
        {"depois_do_trecho": 1, "letra": "b", "duracao_sec": 1.5},  # respiro do acervo
        {"depois_do_trecho": 2, "letra": "C", "duracao_sec": 99},  # fim; duração maior que a imagem = inteira
        {"depois_do_trecho": 1, "letra": "A"},  # letra repetida
        {"depois_do_trecho": 3, "letra": "B"},  # trecho que não existe
        {"depois_do_trecho": 1, "letra": "Z"},  # letra fora da lista
        {"depois_do_trecho": 1.5, "letra": "B"},  # número quebrado
    ]})
    check = validate_inserts(answer, trechos, candidates, material, Usage())
    assert [(c.after_trecho, c.letter, c.moment_id, c.duration_sec, c.breath) for c in check.inserts] == [
        (0, "A", "D001.M01", None, False), (1, "B", "B001.M02", 1.5, True), (2, "C", "D001.M02", None, False),
    ]
    reasons = [d.reason for d in check.discarded]
    assert reasons == [
        "letra A repetida (cada letra no máximo uma vez)",
        "depois_do_trecho 3 não existe (de 0 = começo a 2 = fim)",
        "letra Z não está na lista (letras: A, B, C)",
        "depois_do_trecho deve ser um número inteiro de 0 a 2",
    ]
    assert any("duracao_sec 99 maior que a imagem" in w for w in check.warnings)
    assert check.added_sec == pytest.approx(4.0 + 1.5 + 4.0) and check.refused == "4 de 7 itens inválidos"


def test_inserts_of_a_block_without_speech_must_pick_something_or_use_all_its_images(material):
    candidates = _candidates(("A", "D001.M01", 4.0, "bloco"), ("B", "D001.M02", 4.0, "bloco"), ("C", "B001.M01", 1.0, "acervo"))
    empty = validate_inserts(InsertsAnswer(), [], candidates, material, Usage())
    assert "o bloco não tem fala e nenhuma imagem foi escolhida" in empty.refused and empty.unusable == "bloco vazio"
    from ai_video_editor.editor.validate import neutral_inserts

    neutral = validate_inserts(neutral_inserts([], candidates), [], candidates, material, Usage())
    assert [c.moment_id for c in neutral.inserts] == ["D001.M01", "D001.M02"]


def test_inserts_build_the_main_track_in_the_roteiro_order(material, tmp_path):
    def answer(request):
        letter = {c.moment_id: c.letter for c in request.candidates}
        return {"inserir": [
            {"depois_do_trecho": 2, "letra": letter["D001.M02"]},  # fim do bloco
            {"depois_do_trecho": 0, "letra": letter["D001.M01"], "duracao_sec": 2},  # abertura
            {"depois_do_trecho": 1, "letra": letter["B001.M02"]},  # respiro entre os trechos
        ]}

    ai = MockEditorAI({
        "structure": [_structure(("D001", "P001"), fora=("P002",))],
        "cuts": [{"cortar": ["S002"]}], "inserts": [answer], "coverages": [{"coberturas": []}],
    })
    run = _service(material, tmp_path, ai).run()
    [detail] = run.details
    request = ai.calls["inserts"][0]
    assert [c.source for c in request.candidates] == ["bloco", "bloco", "acervo", "acervo"]
    assert "1: S000–S001 (P001, 4 s): Chegamos na praia hoje. O mar está muito azul." in request.prompt
    assert re.search(r"\nA: vista aérea do mar azul · lugar: praia · 4,0s · D001 \(do bloco\)", request.prompt)
    assert [(k.kind, k.key, k.breath) for k in detail.keep] == [
        ("momento", "D001.M01", False), ("frases", "S001", False), ("momento", "B001.M02", True),
        ("frases", "S003", False), ("momento", "D001.M02", False),
    ]
    assert detail.result_sec == pytest.approx(4.5 + 2.0 + 2.0 + 2.0 + 4.0)


# --- 2c: coberturas ---


def make_evidence_material(tmp_path: Path) -> EditorMaterial:
    """P001 fala no terminal (ómnibus, cachorros); D001 = drone com ônibus, cachorro e a fachada do terminal."""
    clips = [_clip("P001", "principal", 1, 12.0, True), _clip("D001", "drone", 2, 12.0, False, audio=False),
             _clip("P002", "principal", 3, 6.0, True)]
    moments = [
        _moment("P001", 1, 0.0, 12.0, "pessoa falando no saguão", "terminal de ônibus", ["pessoa"], ["S000", "S001"],
                ["tremida"]),
        _moment("D001", 1, 0.0, 4.0, "ônibus na estrada", "estrada", ["ônibus", "estrada"]),
        _moment("D001", 2, 4.0, 8.0, "cachorro na praia", "praia", ["cachorro", "areia"]),
        _moment("D001", 3, 8.0, 12.0, "fachada do terminal rodoviário", "terminal rodoviário", ["fachada"]),
        _moment("P002", 1, 0.0, 6.0, "rua com carros e montanhas", "rua", ["rua", "carros"], ["S002"]),
    ]
    sentences = [
        _sentence(0, "P001", 0.5, 4.0, "Pegamos o ómnibus para Bariloche.", "P001.M01"),
        _sentence(1, "P001", 5.0, 8.0, "Olha os cachorros correndo.", "P001.M01"),
        _sentence(2, "P002", 0.5, 3.0, "Chegamos na cidade com o ônibus.", "P002.M01"),
    ]
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0, clips=clips,
                             moments=moments, sentences=sentences)
    return EditorMaterial.from_roteiro(roteiro)


def test_coverage_candidates_are_measured_by_common_word_or_same_place(tmp_path):
    material = make_evidence_material(tmp_path)
    speech = "Pegamos o ómnibus para Bariloche. Olha os cachorros correndo."
    pool = [material.moments[m] for m in ("D001.M01", "D001.M02", "D001.M03", "P002.M01")]
    matches = coverage_matches(speech, ["terminal de ônibus"], pool, 4, COVERAGE_SAYS, COVERAGE_PLACE)
    assert [(m.moment.id, m.kind, m.words) for m in matches] == [
        ("D001.M01", COVERAGE_SAYS, ["omnibus"]),  # ómnibus × ônibus (uma letra a mais)
        ("D001.M02", COVERAGE_SAYS, ["cachorros"]),  # plural × singular
        ("D001.M03", COVERAGE_PLACE, ["terminal"]),  # lugar × lugar
    ]
    assert coverage_matches(speech, [], pool, 1, COVERAGE_SAYS, COVERAGE_PLACE)[0].moment.id == "D001.M01"


def test_coverage_never_offers_the_speaking_clip_and_skips_trechos_without_candidates(tmp_path):
    material = make_evidence_material(tmp_path)
    ai = MockEditorAI({"structure": [_structure(("P001", "P002"))], "inserts": [{"inserir": []}]})
    run = _service(material, tmp_path, ai).run()
    request = ai.calls["coverages"][0]
    offered = {n: [c.moment_id for c in cs] for n, cs in request.questions.items()}
    assert offered[1] == ["D001.M01", "D001.M02", "D001.M03"]  # nunca P001 (quem fala)
    assert offered[2] == ["D001.M01"]  # P002 diz "ônibus": o drone; P001 está no vídeo (conflito), nunca P002
    assert "Trecho 1 (S000–S001 · P001 · 6 s; imagem de quem fala: pessoa falando no saguão [tremida])" in request.prompt
    assert "— mostra o que é dito; em comum: omnibus" in request.prompt
    # O mock cobre o 1º trecho com o 1º candidato "o que é dito": da frase com a palavra, na duração do momento.
    [coverage] = run.details[0].coverages
    assert (coverage.trecho, coverage.letter, coverage.moment_ids, coverage.word) == (1, "A", ["D001.M01"], "omnibus")
    assert coverage.sentence_ids == ["S000"] and run.details[0].coverage.nones == [2]


def _questions(material):
    trechos = build_trechos(["P001", "P002"], {"S000", "S001", "S002"}, material)
    letters = {"D001.M01": "A", "D001.M02": "B", "D001.M03": "C"}
    questions = {}
    for trecho in trechos:
        speech = " ".join(material.sentences[s].text for s in trecho.sentence_ids)
        places = ["terminal de ônibus"] if trecho.clip_id == "P001" else ["rua"]
        pool = [material.moments[m] for m in letters]
        questions[trecho.number] = [
            Candidate(letter=letters[m.moment.id], moment_id=m.moment.id, clip_id="D001", duration_sec=4.0,
                      source="acervo", kind=m.kind, words=m.words)
            for m in coverage_matches(speech, places, pool, 4, COVERAGE_SAYS, COVERAGE_PLACE)
        ]
    return trechos, questions


def test_coverage_answer_letters_none_and_evidence(tmp_path):
    material = make_evidence_material(tmp_path)
    trechos, questions = _questions(material)
    usage = Usage()
    take_sentences(usage, ["S000", "S001", "S002"], material)
    answer = CoveragesAnswer.model_validate({"coberturas": [
        {"trecho": 1, "letra": "B", "palavra": "cachorros"},  # válida: começa em S001, a frase com a palavra
        {"trecho": 1, "letra": "A", "palavra": "ómnibus"},  # trecho repetido
        {"trecho": 2, "letra": "A", "palavra": "estrada"},  # palavra que não está na fala
        {"trecho": 7, "letra": "A", "palavra": "ônibus"},  # trecho não perguntado
    ]})
    check = validate_coverages(answer, questions, trechos, material, usage)
    [coverage] = check.coverages
    assert (coverage.trecho, coverage.moment_ids, coverage.sentence_ids, coverage.kind) == (1, ["D001.M02"], ["S001"], COVERAGE_SAYS)
    reasons = [d.reason for d in check.discarded]
    assert reasons[0] == "trecho 1 repetido (um item por trecho)"
    assert reasons[1] == "palavra 'estrada' não está na fala do trecho 2: estrada"
    assert reasons[2].startswith("trecho 7 não foi perguntado")
    # "nenhuma", letra de outro trecho, letra já usada e a palavra que não está na imagem.
    answer = CoveragesAnswer.model_validate({"coberturas": [
        {"trecho": 1, "letra": "C", "palavra": "terminal"},  # lugar: válida
        {"trecho": 2, "letra": "nenhuma", "palavra": ""},
    ]})
    check = validate_coverages(answer, questions, trechos, material, usage)
    assert [(c.trecho, c.kind, c.sentence_ids) for c in check.coverages] == [(1, COVERAGE_PLACE, ["S000"])]
    assert check.nones == [2]
    wrong = validate_coverages(CoveragesAnswer.model_validate({"coberturas": [
        {"trecho": 2, "letra": "B", "palavra": "cachorro"}, {"trecho": 1, "letra": "A", "palavra": "Bariloche"},
    ]}), questions, trechos, material, usage)
    reasons = [d.reason for d in wrong.discarded]
    assert reasons[0] == 'letra B não é candidata do trecho 2 (letras: A ou "nenhuma")'
    assert reasons[1] == "palavra 'Bariloche' não está na descrição/lugar/tags da imagem A: bariloche"


def test_coverage_span_follows_the_sentences_that_fit_in_the_moment(material):
    trechos = build_trechos(["P001"], {"S000", "S001", "S002", "S003"}, material)
    long_moment = Candidate(letter="A", moment_id="D001.M02", clip_id="D001", duration_sec=4.0, kind=COVERAGE_SAYS, words=["trilha"])
    from ai_video_editor.editor.validate import coverage_span

    # "trilha" aparece primeiro em S002 (6,5–9,0); S003 termina 11,5 − 6,5 = 5 s > 4 s + 0,5: fica fora.
    assert coverage_span(trechos[0], long_moment, "trilha", material) == ["S002"]
    assert coverage_span(trechos[0], long_moment.model_copy(update={"moment_id": "D001.M01"}), "mar", material) == ["S001"]


# --- passadas: ponta a ponta com o mock, fallbacks e cache ---


def test_full_run_with_mock_builds_the_plan_from_2a_2b_2c(material, tmp_path):
    ai = MockEditorAI()
    run = _service(material, tmp_path, ai).run()
    assert [s.origin for s in run.summaries] == ["ia"] * 4
    assert run.structure.origin == "ia" and [b.clip_ids for b in run.structure.blocks] == [["P001"], ["P002"]]
    assert run.structure.library_clip_ids == ["D001", "B001"]
    first, second = run.details
    assert [(k.kind, k.key, k.breath) for k in first.keep] == [("frases", "S003", False), ("momento", "B001.M01", True)]
    assert first.coverages[0].moment_ids == ["D001.M01"] and first.coverages[0].word == "praia"
    assert [(k.kind, k.key) for k in second.keep] == [("frases", "S005"), ("momento", "B001.M02")]
    assert not second.coverages and second.coverage.requests == 0  # nenhum candidato: nem pergunta
    counts = {k: len(v) for k, v in ai.calls.items()}
    assert counts == {"summary": 4, "structure": 1, "cuts": 2, "inserts": 2, "coverages": 1}
    assert run.cuts_stats.calls == 2 and run.inserts_stats.calls == 2 and run.coverage_stats.calls == 1
    assert [r.kind for r in run.requests].count("cortes") == 2 and all(r.estimated_tokens > 0 for r in run.requests)


def test_retry_with_the_errors_and_neutral_fallbacks(material, tmp_path):
    ai = MockEditorAI({
        "structure": [{"blocos": []}, {"blocos": [{"clipes": ["X999"]}]}],
        "cuts": [{"cortar": ["S999"]}, {"cortar": ["S000"]}],  # B01: nenhum válido → nova tentativa aceita
        "inserts": [{"inserir": [{"depois_do_trecho": 9, "letra": "Q"}]}] * 2,  # B01: nenhum válido duas vezes
    })
    run = _service(material, tmp_path, ai).run()
    assert run.structure.origin == "fallback" and [b.clip_ids for b in run.structure.blocks] == [["P001"], ["P002"]]
    assert "nenhum bloco" in run.structure.attempts[0] and "X999 não existe" in run.structure.attempts[1]
    assert any("ESTRUTURA PELO FALLBACK NEUTRO" in w for w in run.warnings)
    assert "Sua resposta anterior foi recusada" in ai.calls["structure"][1].prompt
    first = run.details[0]
    assert first.cut_runs[0].origin == "ia" and "S999 não existe" in ai.calls["cuts"][1].prompt
    assert [c.sentence_id for c in first.cut_runs[0].cuts] == ["S000"]
    assert first.inserts.origin == "fallback" and not first.inserts.inserts and first.origin == "fallback"
    assert any("INCLUSÕES B01 PELO FALLBACK NEUTRO" in w for w in run.warnings)


def test_stops_calling_after_consecutive_api_errors_and_auth_error_stops_everything(material, tmp_path):
    ai = MockEditorAI({"summary": [AIProviderError("servidor caiu")] * 10})
    run = _service(material, tmp_path, ai).run()
    assert len(ai.calls["summary"]) == 3 and not ai.calls["structure"] and not ai.calls["cuts"]
    assert all(s.origin == "falha" for s in run.summaries)
    assert run.structure.origin == "fallback" and all(d.origin == "fallback" for d in run.details)
    assert [k.key for k in run.details[0].keep] == ["S003"]  # nada cortado
    with pytest.raises(AIAuthenticationError):
        _service(material, tmp_path / "b", MockEditorAI({"summary": [AIAuthenticationError("mock")]})).run()


def test_cache_per_pass_changing_the_cuts_prompt_keeps_the_others(material, tmp_path, monkeypatch):
    _service(material, tmp_path, MockEditorAI()).run()
    again = MockEditorAI()
    run = _service(material, tmp_path, again).run()
    assert not any(again.calls.values())  # tudo do cache
    assert {s.origin for s in run.summaries} == {"cache"} and run.structure.origin == "cache"
    assert {d.origin for d in run.details} == {"cache"} and all(r.cached for r in run.requests)

    monkeypatch.setattr(editor_service, "CUTS_PROMPT_VERSION", "cortes-teste")
    third = MockEditorAI()
    _service(material, tmp_path, third).run()
    # Mesmos cortes → mesmos trechos → 2b e 2c com os mesmos pedidos (do cache).
    assert {k: len(v) for k, v in third.calls.items()} == {"summary": 0, "structure": 0, "cuts": 2, "inserts": 0, "coverages": 0}

    monkeypatch.setattr(editor_service, "SUMMARY_PROMPT_VERSION", "resumo-teste")
    fourth = MockEditorAI()
    _service(material, tmp_path, fourth).run()
    assert len(fourth.calls["summary"]) == 4  # mudar o pedido do resumo refaz os resumos


# Qualquer coisa com cara de ID (letras + dígitos, com ou sem ".M…"/".S…"): S012, P007.M04, X01.S900, B01.
ANY_ID = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,2}\d+(?:\.[A-Z]\d+)?(?![A-Za-z0-9])")


def test_prompts_have_no_example_ids():
    context = EditorContext(request="vlog", target_sec=300)
    header = BlockHeader(block_id="B01", title="t", subject="s", context=context, target_total_sec=300, block_target_sec=60)
    prompts = [
        build_structure_prompt(context, []),
        build_cuts_prompt(CutsPromptInput(header=header, clip_id="", section="", available_sec=90, target_sec=30)),
        build_inserts_prompt(InsertsPromptInput(header=header, trecho_lines=[], candidate_lines=[], current_sec=10)),
        build_coverage_prompt(CoveragePromptInput(header=header, questions=[])),
    ]
    assert ANY_ID.findall(prompts[0]) == []
    for prompt in prompts[1:]:
        assert set(ANY_ID.findall(prompt)) == {"B01"}  # só o bloco que está sendo montado (código do programa)
    assert "frase = só o código S… que aparece antes do texto da frase, sem o clipe na frente" in prompts[1]
    assert "Uma frase por item" in prompts[1]
    assert "copiando os códigos exatamente como aparecem na entrada" in prompts[0]
    assert ANY_ID.findall("S012–S015, D002.M05, X01.S900, P009.S025, P007") == [
        "S012", "S015", "D002.M05", "X01.S900", "P009.S025", "P007",
    ]


def test_prompts_with_real_material_only_show_ids_of_the_input(material, tmp_path):
    ai = MockEditorAI()
    _service(material, tmp_path, ai).run()
    known = set(material.clips) | set(material.sentences) | set(material.moments)
    prompts = [r.prompt for kind in ("structure", "cuts", "inserts", "coverages") for r in ai.calls[kind]]
    for prompt in prompts:
        found = set(ANY_ID.findall(prompt))
        assert found - known <= {"B01", "B02"}, found - known
        assert not re.search(r"[A-Z]\d+\.S\d+", prompt)  # nenhuma frase com o clipe na frente


class CountingAI(MockEditorAI):
    """Mock que informa tokens reais (como o provedor de verdade), para a tabela de calibração."""

    def _fixed(self, kind, request, schema):
        self.last_input_tokens, self.last_output_tokens = len(request.prompt) // 3, 50
        return super()._fixed(kind, request, schema)


def test_report_per_block_subpasses_and_estimated_vs_real_tokens(material, tmp_path):
    ai = CountingAI({
        "summary": [{"resumo": "praia", "destaques": [{"ids": ["P001.S000", "p001.m1"], "motivo": "bom"}]}],
        "structure": [{"blocos": [{"clipes": ["p1"]}, {"clipes": ["P002"]}]}],
        "cuts": [{"cortar": [{"frase": "P001.S002", "motivo": "recomeço"}, "P001.S003"]}, {"cortar": []}],
    })
    run = _service(material, tmp_path, ai).run()
    assert run.summaries[0].normalized_ids == ["P001.S000 → S000", "P001.M1 → P001.M01"]
    assert run.structure.normalized_ids == ["P1 → P001"]
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "- B01.P001 (IA, ~" in report and "· 2 corte(s)" in report
    assert "  - cortou S002 — Agora vamos pela trilha. · motivo: recomeço" in report
    assert "IDs normalizados: 2 (P001.S002 → S002; P001.S003 → S003)" in report
    # Sem S002–S003 (cortadas), a mata já não divide palavra com a fala: o mock a usa como respiro.
    assert "- B → D001.M02 (respiro (acervo), 2s) depois do trecho 1" in report
    assert re.search(r"- trecho 1 → A → D001\.M01 · mostra o que é dito · palavra \"praia\" · cobre S000–S000", report)
    rows = {line.split(" | ")[0]: line for line in report.splitlines() if line.startswith("| ") and "(E" in line}
    assert set(rows) == {"| Resumos (E9)", "| Estrutura (E10)", "| 2a Cortes (E11)", "| 2b Inclusões (E11)", "| 2c Coberturas (E11)"}
    assert rows["| 2a Cortes (E11)"].endswith("| 2 |")  # IDs normalizados na 2a
    assert "## Tokens estimados × reais por pedido" in report and "| cortes | B01.P001 | 1 |" in report
    assert "| cortes | 2 |" in report  # razão real/estimado por passada


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
    return EditorRun(
        provider="mock", budget=Budget(num_ctx=1, source="t").info(),
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


def test_assembly_breath_between_trechos_coverage_split_between_moments_and_short_moment(material, tmp_path):
    detail = DetailRun(
        unit_id="B01", block_id="B01", clip_ids=["P001"], origin="ia",
        keep=[
            KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S000", sentence_ids=["S000"]),
            KeepRange(kind="momento", clip_id="B001", moment_id="B001.M02", duration_sec=1.5, breath=True),
            KeepRange(kind="frases", clip_id="P001", from_sentence="S001", to_sentence="S002", sentence_ids=["S001", "S002"]),
        ],
        coverages=[CoverageChoice(
            from_sentence="S001", to_sentence="S002", sentence_ids=["S001", "S002"], clip_id="P001",
            moment_ids=["D001.M01", "B001.M01"], speech_evidence="mar", image_evidence="mar",
        )],
    )
    assembled = assemble_plan(_run_with([detail], [_block("P001")]), material, _sources(material), SilenceCutParams(),
                              tmp_path / "out.mp4", "teste")
    timeline = assembled.timeline
    labels = [i.label for i in timeline.video_main]
    assert labels[0] == "P001 S000–S000" and labels[1] == "respiro B001.M02"
    assert set(labels[2:]) == {"P001 S001–S002"}
    breath = timeline.video_main[1]
    assert (breath.source_in, breath.source_out, breath.volume) == (1.0, 2.5, BREATH_VOLUME)
    # S000 sozinha: o corte do `edit` sem S001 (as outras frases do clipe saem), dentro da janela da faixa.
    first = timeline.video_main[0]
    assert first.source_in <= 0.5 and 2.5 <= first.source_out <= 3.0
    # Cobertura de S001 até o fim de S002: D001.M01 inteiro (4 s) e depois B001.M01 (1 s), em ordem; o que
    # faltou fica sem cobertura (nunca estica nem repete).
    overlays = [(o.clip_id, o.source_in, round(o.source_out, 3)) for o in timeline.video_overlay]
    assert overlays == [("D001", 0.0, 4.0), ("B001", 0.0, 1.0)]
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
    assert "## Estrutura (passada 1)" in report and "→ D001.M01 · mostra o que é dito · palavra \"verde\"" in report
    assert "**2a cortes**" in report and "## Chamadas e tokens" in report
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
