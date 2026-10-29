"""Testes do Editor de IA (receita `editor_ia`): schemas, calibração do orçamento e divisão, as três perguntas
da passada 2 (2a cortes, 2b inclusões e respiros, 2c coberturas) com validação item a item, nova tentativa e
fallbacks neutros, caches por passada, montagem (IDs → Timeline) e ponta a ponta com clipes sintéticos
(render curto marcado `slow`). Sem rede: IA simulada (`MockEditorAI`)."""

import json
import math
import re
from datetime import datetime, timezone
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
from ai_video_editor.editor.assemble import (
    INTERNAL_PAUSE_MIN_SEC,
    MIN_PIECE_SEC,
    ClipSource,
    assemble_plan,
    faixa_ranges,
    merge_pieces,
    piece_sentences,
)
from ai_video_editor.editor.budget import Budget, coverage_output_limit, resolve_budget, tokens
from ai_video_editor.editor.ids import IdResolver
from ai_video_editor.editor.material import (
    FREE_ORDER_PHRASES,
    GENERIC_WORDS,
    STRUCTURE_SUMMARY_CHARS,
    asks_free_order,
    asks_landscape_opening,
    coverage_matches,
    library_ranking,
    matched_words,
    place_key,
)
from ai_video_editor.editor.prompt import (
    BlockHeader,
    CoveragePromptInput,
    CutsPromptInput,
    InsertsPromptInput,
    StructureShortfall,
    build_additions_prompt,
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
    ClipSummary,
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
    Trecho,
)
from ai_video_editor.editor.service import (
    CallCache,
    EditorService,
    _add_to_neighbors,
    _join_parts,
    cover_refusal,
    split_parts,
    cover_words,
)
from ai_video_editor.editor.steady import steady_range, steady_window
from ai_video_editor.editor.validate import (
    RESPIRO_MAX_SEC,
    Usage,
    breath_span,
    build_trechos,
    moment_span,
    insert_positions,
    speech_continues,
    take_sentences,
    talking_sentences,
    validate_coverages,
    validate_cuts,
    validate_inserts,
    validate_structure,
)
from ai_video_editor.intelligence.exceptions import AIAuthenticationError, AIProviderError, AIResponseValidationError
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.schema import ModelSpec
from ai_video_editor.pipeline import EditPipelineParams
from ai_video_editor.pipeline.editor_ia import EditorParams, EditorPipeline
from ai_video_editor.planning import SilenceCutParams, TimeRange
from ai_video_editor.producer import registered_recipes
from ai_video_editor.recipes import get_recipe
from ai_video_editor.roteiro.report import clock
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


def _clip(clip_id, role, order, duration, speech, audio=True, recorded_at=None) -> RoteiroClip:
    return RoteiroClip(
        id=clip_id, role=role, relative_path=f"{role}/{clip_id}.mp4", order=order, duration_sec=duration,
        has_audio=audio, has_speech=speech, transcribed=speech, recorded_at=recorded_at,
    )


def _at(hhmm: str) -> datetime:
    """Data de gravação confiável no dia da viagem de teste."""
    hour, minute = hhmm.split(":")
    return datetime(2026, 5, 7, int(hour), int(minute), tzinfo=timezone.utc)


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


@pytest.fixture(autouse=True)
def _cuts_on_short_clips(monkeypatch):
    """Os clipes do material de teste são curtos (≤ 12 s) e as metas dos testes cabem na fala: sem isto, nenhum
    chegaria à pergunta de cortes; e o vídeo de teste tem ~15 s, então o limite de cobertura (15%) tiraria toda
    cobertura. Os testes de `CUT_MIN_SPEECH_SEC`, `CUT_EXCESS_RATIO` e `COVER_MAX_RATIO` voltam os valores de
    produção."""
    monkeypatch.setattr(editor_service, "CUT_MIN_SPEECH_SEC", 0.0)
    monkeypatch.setattr(editor_service, "CUT_EXCESS_RATIO", 0.0)
    monkeypatch.setattr(editor_service, "COVER_MAX_RATIO", 1.0)


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


def test_speaking_clip_not_placed_counts_as_out_without_retry_or_extra_block(material, tmp_path):
    forgot = {"blocos": [{"clipes": ["P002"], "duracao_alvo_sec": 4}], "fora": [{"clipe": "B001", "motivo": "ruim"}]}
    ai = MockEditorAI({"structure": [forgot]})
    run = _service(material, tmp_path, ai).run()
    assert len(ai.calls["structure"]) == 1  # sem nova tentativa
    assert [b.clip_ids for b in run.structure.blocks] == [["P002"]]  # sem bloco extra
    assert [(o.clipe, o.motivo) for o in run.structure.out] == [("B001", "ruim"), ("P001", "não escolhido pela IA")]
    assert run.structure.library_clip_ids == ["D001"] and not run.structure.attempts
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "- P001: não escolhido pela IA" in report


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


def test_structure_prompt_lists_all_41_clips_in_one_lean_line_each_in_one_request(tmp_path):
    vlog = make_vlog_material(tmp_path)
    long_summary = "O clipe mostra que ele caminha pela trilha até o lago e mostra a água gelada. " * 5
    ai = MockEditorAI({"summary": [{"resumo": long_summary, "tom": "leve"}] * 41})
    run = _service(vlog, tmp_path, ai, num_ctx=8192).run()
    [request] = ai.calls["structure"]  # a lista inteira cabe: um pedido só, sem parte
    prompt = request.prompt
    lines = [line for line in prompt.splitlines() if line.startswith("- ") and " — " in line]
    assert [line.split(" · ")[0][2:] for line in lines] == vlog.clip_ids  # todos, na ordem
    summaries = [line.split(" — ", 1)[1] for line in lines]
    assert all(len(text) <= STRUCTURE_SUMMARY_CHARS and text.endswith("…") for text in summaries)
    # Sem a abertura repetida ("O clipe mostra"), sem a duração de clipe com fala, com o lugar.
    assert "- P001 · principal · fala 10 s · lago — Que ele caminha pela trilha até o lago" in prompt
    assert "- B001 · broll · sem fala 0:30 · lago — " in prompt
    assert "parte 1 de" not in prompt and request.clip_ids == vlog.clip_ids
    assert len(run.structure.parts) == 1 and run.structure.parts[0].estimated_tokens == tokens(prompt)
    assert tokens(prompt) <= Budget(8192, "t").structure_available
    assert run.structure.origin == "ia"


def test_structure_prompt_without_room_goes_to_the_fallback(material, tmp_path):
    run = _service(material, tmp_path, num_ctx=3000).run()  # nem o pedido sem a lista cabe: dividir não resolve
    assert len(run.structure.parts) == 1
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
    assert everything.refused == "você cortou todas as frases; mantenha as que contam o assunto ou responda fora: true"
    assert everything.unusable is None and everything.removed
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
    assert "Este clipe tem 9 s de fala; a meta é 4 s; corte cerca de 4 s de frases" in ai.calls["cuts"][0].prompt
    retry = ai.calls["cuts"][1].prompt
    assert "ficou 9 s, a meta deste clipe é 4 s (aceito de 3 a 6 s): corte cerca de 4 s" in retry
    assert "frases que ficaram (escolha quais cortar):" in retry and "- mantida S003 (2 s): Agora vamos" in retry
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
    available = Budget(8192, "t").cuts_available
    assert all(tokens(r.prompt) <= available + editor_budget.RETRY_TOKENS for r in ai.calls["cuts"])
    assert all(tokens(r.prompt) <= available for r in ai.calls["cuts"] if "Sua resposta anterior" not in r.prompt)
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
        "structure": [_structure((("D001", "P001"), 20), fora=("P002",))],
        "cuts": [{"cortar": ["S002"]}] * 2,  # abaixo da meta: nova tentativa, a mesma resposta é aceita
        "inserts": [answer], "coverages": [{"coberturas": []}],
    })
    run = _service(material, tmp_path, ai).run()
    [detail] = run.details
    request = ai.calls["inserts"][0]
    assert [c.source for c in request.candidates] == ["bloco", "bloco", "acervo", "acervo"]
    assert "1: S000–S001 (P001, 4 s): Chegamos na praia hoje. O mar está muito azul." in request.prompt
    assert re.search(r"\n- A: vista aérea do mar azul · lugar: praia · 4,0s · D001 \(do bloco\)", request.prompt)
    # 1º bloco: sem respiro no começo; entre os trechos o lugar muda (praia → trilha): posição 1 oferecida.
    assert request.breath_positions == [1] and request.own_positions == [0, 1, 2]
    assert "Sobram 14 s neste bloco" in request.prompt and "1 (troca de lugar: praia → trilha)" in request.prompt
    assert [(k.kind, k.key, k.breath) for k in detail.keep] == [
        ("momento", "D001.M01", False), ("frases", "S001", False), ("momento", "B001.M02", True),
        ("frases", "S003", False), ("momento", "D001.M02", False),
    ]
    assert detail.result_sec == pytest.approx(4.5 + 2.0 + 2.0 + 2.0 + 4.0)


# --- 2c: coberturas ---


def make_evidence_material(tmp_path: Path) -> EditorMaterial:
    """P001 fala no terminal (ómnibus, cachorros); D001 = drone com ônibus, cachorro e a fachada do terminal."""
    clips = [_clip("P001", "principal", 1, 12.0, True, recorded_at=_at("10:00")),
             _clip("D001", "drone", 2, 12.0, False, audio=False, recorded_at=_at("10:05")),
             _clip("P002", "principal", 3, 6.0, True, recorded_at=_at("10:10"))]
    moments = [
        _moment("P001", 1, 0.0, 12.0, "pessoa falando no saguão", "terminal de ônibus", ["pessoa"], ["S000", "S001"],
                ["tremida"], talking=True),
        _moment("D001", 1, 0.0, 4.0, "ônibus na estrada", "estrada", ["ônibus", "estrada"]),
        _moment("D001", 2, 4.0, 8.0, "cachorro na praia", "praia", ["cachorro", "areia"]),
        _moment("D001", 3, 8.0, 12.0, "fachada do terminal rodoviário", "terminal rodoviário", ["fachada"]),
        _moment("P002", 1, 0.0, 6.0, "rua com carros e montanhas", "rua", ["rua", "carros"], ["S002"], talking=True),
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
    # Nunca P001 (quem fala); D001.M03 (fachada do terminal) só tem o lugar em comum: a palavra precisa estar na
    # descrição da imagem e não estar na imagem de quem fala.
    assert offered[1] == ["D001.M01", "D001.M02"]
    assert offered[2] == ["D001.M01"]  # P002 diz "ônibus": o drone; P001 está no vídeo (conflito), nunca P002
    assert "Trecho 1 (S000–S001 · P001 · 6 s; imagem de quem fala: pessoa falando no saguão [tremida], fala para a câmera)" in request.prompt
    assert "— em comum: omnibus" in request.prompt
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
        {"trecho": 7, "letra": "A", "palavra": "ônibus"},  # trecho não perguntado: ignorado, não inválido
        {"trecho": 9, "letra": "nenhuma", "palavra": ""},
    ]})
    check = validate_coverages(answer, questions, trechos, material, usage)
    [coverage] = check.coverages
    assert (coverage.trecho, coverage.moment_ids, coverage.sentence_ids, coverage.kind) == (1, ["D001.M02"], ["S001"], COVERAGE_SAYS)
    reasons = [d.reason for d in check.discarded]
    assert reasons == ["trecho 1 repetido (um item por trecho)", "palavra 'estrada' não está na fala do trecho 2: estrada"]
    assert check.ignored == ["7", "9"] and check.items == 3
    # "nenhuma" e o lugar (a descrição de quem fala não mostra o terminal); "saguão" já está na imagem de quem fala.
    answer = CoveragesAnswer.model_validate({"coberturas": [
        {"trecho": 1, "letra": "C", "palavra": "terminal"},
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
    assert reasons[1] == "palavra 'Bariloche' não está na descrição/lugar da imagem A: bariloche"
    saguao = trechos[0].model_copy(update={"sentence_ids": ["S000"]})
    material.sentences["S000"].text = "Pegamos o ómnibus no saguão."
    hall = [Candidate(letter="A", moment_id="D001.M03", clip_id="D001", duration_sec=4.0, kind=COVERAGE_SAYS, words=["saguao"])]
    material.moments["D001.M03"].description = "saguão do terminal rodoviário"
    check = validate_coverages(CoveragesAnswer.model_validate({"coberturas": [{"trecho": 1, "letra": "A", "palavra": "saguão"}]}),
                               {1: hall}, [saguao], material, Usage())
    assert check.discarded[0].reason.startswith("palavra 'saguão' já aparece na imagem de quem fala")


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
    # 1º bloco: um trecho só e o começo do vídeo: nenhuma posição de respiro, nenhuma imagem do bloco → sem 2b.
    assert [(k.kind, k.key, k.breath) for k in first.keep] == [("frases", "S003", False)]
    assert first.inserts.requests == 0 and first.inserts.spare_sec == pytest.approx(2.0)
    # "praia" está na imagem de quem fala (pessoa falando na praia): a palavra nova é "mar".
    assert first.coverages[0].moment_ids == ["D001.M01"] and first.coverages[0].word == "mar"
    # 2º bloco: respiro no começo (troca de bloco), com a imagem do acervo do mesmo lugar (restaurante).
    assert [(k.kind, k.key, k.breath) for k in second.keep] == [("momento", "B001.M02", True), ("frases", "S005", False)]
    assert second.inserts.breath_positions == [0] and second.inserts.candidates[0].moment_id == "B001.M01"
    assert not second.coverages and second.coverage.requests == 0  # a fala do P002 não é para a câmera
    assert [(k.number, k.reason) for k in second.coverage.skipped] == [(1, "imagem não é fala para câmera")]
    counts = {k: len(v) for k, v in ai.calls.items()}
    assert counts == {"summary": 4, "structure": 1, "additions": 0, "cuts": 2, "inserts": 1, "coverages": 1}
    assert run.cuts_stats.calls == 2 and run.inserts_stats.calls == 1 and run.coverage_stats.calls == 1
    assert [r.kind for r in run.requests].count("cortes") == 2 and all(r.estimated_tokens > 0 for r in run.requests)


def test_retry_with_the_errors_and_neutral_fallbacks(material, tmp_path):
    ai = MockEditorAI({
        "structure": [{"blocos": []}, {"blocos": [{"clipes": ["X999"]}]}],
        "cuts": [{"cortar": ["S999"]}, {"cortar": ["S000"]}],  # B01: nenhum válido → nova tentativa aceita
    })
    run = _service(material, tmp_path, ai).run()
    assert run.structure.origin == "fallback" and [b.clip_ids for b in run.structure.blocks] == [["P001"], ["P002"]]
    assert "nenhum bloco" in run.structure.attempts[0] and "X999 não existe" in run.structure.attempts[1]
    assert any("ESTRUTURA PELO FALLBACK (a IA falhou de vez)" in w for w in run.warnings)
    assert "Sua resposta anterior foi recusada" in ai.calls["structure"][1].prompt
    first = run.details[0]
    assert first.cut_runs[0].origin == "ia" and "S999 não existe" in ai.calls["cuts"][1].prompt
    assert [c.sentence_id for c in first.cut_runs[0].cuts] == ["S000"]
    # Estrutura do fallback não tem alvo por bloco: sem sobra medida, sem 2b.
    assert first.inserts.requests == 0 and first.inserts.no_question == "sem alvo do bloco: sem sobra medida"
    ai = MockEditorAI({
        "structure": [_structure((("P001",), 12), (("P002",), 10))],
        "inserts": [{"inserir": [{"depois_do_trecho": 9, "letra": "Q"}]}] * 2,  # B02: nenhum válido duas vezes
    })
    run = _service(material, tmp_path / "b", ai).run()
    second = run.details[1]
    assert second.inserts.origin == "fallback" and not second.inserts.inserts and second.origin == "fallback"
    assert any("INCLUSÕES B02 PELO FALLBACK NEUTRO" in w for w in run.warnings)


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
    assert {k: len(v) for k, v in third.calls.items()} == {"summary": 0, "structure": 0, "additions": 0, "cuts": 2, "inserts": 0, "coverages": 0}

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
        "structure": [{"blocos": [{"clipes": ["p1"], "duracao_alvo_sec": 5}, {"clipes": ["P002"], "duracao_alvo_sec": 6}]}],
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
    # B01: o vídeo começa nele e tem um trecho só: nenhuma posição de respiro (sem pergunta).
    assert "- sobra (alvo do bloco − fala que ficou): 0,5s" in report
    assert "- posições de respiro recusadas: 0 (começo do vídeo (o pedido não pede abertura com paisagem))" in report
    # B02: respiro no começo do bloco com a imagem do acervo do mesmo lugar que não divide palavra com a fala.
    assert "- B → B001.M02 (respiro (acervo), 2s) depois do trecho 0" in report
    assert "- trechos sem pergunta: 1 (imagem não é fala para câmera)" in report
    assert "| Respiros | 0,0s (0% do vídeo) |" in report
    assert re.search(r"- trecho 1 → A → D001\.M01 · mostra o que é dito · palavra \"mar\" · cobre S001–S001", report)
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
    # Silêncio de 1,0 s entre S001 e S002 (≥ INTERNAL_PAUSE_MIN_SEC): cortado; S003 (retake) fora, cortado
    # dentro do silêncio antes dela.
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
    assert labels[2:] == ["P001 S001–S001", "P001 S002–S002"]  # cada pedaço rotulado só com as frases dele
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


# --- filtros: sobra, posições e candidatos de respiro (2b), coberturas só para quem fala à câmera (2c),
# picote, clipe tirado pela IA e o pedido no topo ---


def make_trip_material(tmp_path: Path) -> EditorMaterial:
    """P001 no hotel (quarto → rua → lago, sem falar para a câmera), P002 que acaba no meio da fala, P003 na
    praça falando para a câmera; acervo: D001 (lago, 20 s) e B001 (8 imagens da rua)."""
    clips = [
        _clip("P002", "principal", 1, 6.0, True), _clip("P001", "principal", 2, 30.0, True),
        _clip("P003", "principal", 3, 10.0, True), _clip("D001", "drone", 4, 20.0, False, audio=False),
        _clip("B001", "broll", 5, 16.0, False),
    ]
    clips[2].recorded_at, clips[3].recorded_at = _at("17:30"), _at("17:20")  # praça e drone do lago: 10 min
    moments = [
        _moment("P002", 1, 0.0, 6.0, "pessoa no ônibus", "interior de ônibus", ["ônibus"], ["S000"]),
        _moment("P001", 1, 0.0, 6.0, "quarto de hotel com cama grande", "quarto de hotel", ["quarto"], ["S001", "S002"]),
        _moment("P001", 2, 6.0, 12.0, "rua com árvores vista da janela", "rua", ["rua"], ["S003", "S004"]),
        _moment("P001", 3, 12.0, 18.0, "lago com montanhas", "lago", ["lago"], ["S005", "S006"]),
        _moment("P003", 1, 0.0, 10.0, "pessoa falando na praça", "praça", ["praça", "rio"], ["S007"], talking=True),
        _moment("D001", 1, 0.0, 20.0, "lago azul com montanhas ao fundo", "lago", ["lago", "aérea"]),
    ]
    moments += [
        _moment("B001", n, 2.0 * (n - 1), 2.0 * n, f"carros passando na rua {n}", "rua", ["rua"]) for n in range(1, 9)
    ]
    sentences = [
        _sentence(0, "P002", 0.5, 5.6, "Vamos entrar na província.", "P002.M01"),
        _sentence(1, "P001", 0.5, 3.0, "Chegamos no hotel agora.", "P001.M01"),
        _sentence(2, "P001", 3.5, 5.5, "Esse é o nosso quarto e a vista...", "P001.M01"),
        _sentence(3, "P001", 6.5, 9.0, "a rua lá fora com as árvores", "P001.M02"),
        _sentence(4, "P001", 9.5, 11.5, "Que tarde bonita.", "P001.M02"),
        _sentence(5, "P001", 12.5, 15.0, "Agora estamos no lago.", "P001.M03"),
        _sentence(6, "P001", 15.5, 17.5, "O lago é lindo.", "P001.M03"),
        _sentence(7, "P003", 0.5, 4.0, "Terminamos o passeio na praça perto do lago.", "P003.M01"),
    ]
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0, clips=clips,
                             moments=moments, sentences=sentences)
    return EditorMaterial.from_roteiro(roteiro)


def _single_trechos(material, *ids) -> list[Trecho]:
    """Um trecho por frase (como se as vizinhas tivessem sido cortadas)."""
    return [
        Trecho(number=n, clip_id=material.sentences[sid].clip_id, sentence_ids=[sid],
               seconds=material.sentences[sid].end_sec - material.sentences[sid].start_sec)
        for n, sid in enumerate(ids, start=1)
    ]


def test_breath_positions_only_at_measured_changes_never_where_the_speech_continues(tmp_path):
    material = make_trip_material(tmp_path)
    trechos = _single_trechos(material, "S000", "S001", "S002", "S003", "S004", "S005", "S006")
    positions = insert_positions(trechos, material, first_block=False, opening=False)
    # 0 = começo do bloco; 5 = rua → lago (troca de lugar; "Que tarde bonita." termina a frase).
    assert positions.breath == [0, 5]
    refused = {k.number: k.reason for k in positions.refused}
    assert refused == {
        1: "fala continua (o clipe acaba no meio da fala)",  # S000 termina a 0,4 s do fim do P002
        2: "mesmo lugar (quarto de hotel)",
        3: "fala continua (termina em reticências)",  # "...e a vista..."
        4: "fala continua (sem pontuação final)",  # "a rua lá fora com as árvores"
        6: "mesmo lugar (lago)",
    }
    assert 7 not in positions.breath  # nem no fim do bloco
    assert speech_continues(trechos[3], material) == "fala continua (sem pontuação final)"
    # Imagem do próprio bloco: qualquer posição, menos onde a fala continua.
    assert positions.own == [0, 2, 5, 6, 7] and positions.notes[5] == "troca de lugar: rua → lago"


def test_first_block_has_no_breath_at_0_unless_the_request_asks_for_a_landscape_opening(tmp_path):
    material = make_trip_material(tmp_path)
    trechos = _single_trechos(material, "S005")
    first = insert_positions(trechos, material, first_block=True, opening=False)
    assert first.breath == [] and first.refused[0].number == 0 and first.own == [0, 1]
    assert insert_positions(trechos, material, first_block=True, opening=True).breath == [0]
    assert asks_landscape_opening("Abra o vídeo com uma paisagem de drone, depois a fala")
    assert asks_landscape_opening("começar com a vista aérea do lago")
    assert not asks_landscape_opening("vlog da trilha, uns 8 minutos")
    assert not asks_landscape_opening("um vídeo com paisagens bonitas")  # paisagem sem pedir abertura


def _breath_candidates(material) -> list[Candidate]:
    return [
        Candidate(letter="A", moment_id="D001.M01", clip_id="D001", duration_sec=6.0, source="acervo"),
        Candidate(letter="B", moment_id="B001.M01", clip_id="B001", duration_sec=2.0, source="acervo"),
        Candidate(letter="C", moment_id="B001.M02", clip_id="B001", duration_sec=2.0, source="acervo"),
    ]


def test_breath_of_20_s_is_cut_to_6_s_in_the_middle_of_the_moment(tmp_path):
    material = make_trip_material(tmp_path)
    trechos = _single_trechos(material, "S004", "S005")
    answer = InsertsAnswer.model_validate({"inserir": [{"depois_do_trecho": 1, "letra": "A"}]})
    check = validate_inserts(answer, trechos, _breath_candidates(material), material, Usage(), spare_sec=30,
                             breath_positions=[1], own_positions=[0, 1, 2])
    [choice] = check.inserts
    assert RESPIRO_MAX_SEC == 6.0
    assert (choice.start_sec, choice.duration_sec, choice.breath) == (7.0, 6.0, True)  # 20 s → 7–13 s
    assert check.breath_sec == pytest.approx(6.0) and any("cortado para 6,0s" in w.replace(".", ",") for w in check.warnings)
    # Na montagem, o respiro começa no centro escolhido pela validação.
    detail = DetailRun(unit_id="B01", block_id="B01", clip_ids=["P001"], origin="ia", keep=[
        KeepRange(kind="frases", clip_id="P001", from_sentence="S004", to_sentence="S004", sentence_ids=["S004"]),
        KeepRange(kind="momento", clip_id="D001", moment_id="D001.M01", duration_sec=choice.duration_sec,
                  start_sec=choice.start_sec, breath=True),
    ])
    sources = {c: ClipSource(clip_id=c, path=Path(f"{c}.mp4"), fps=25.0, duration_sec=material.clips[c].duration_sec,
                             has_audio=material.clips[c].has_audio) for c in material.clips}
    timeline = assemble_plan(_run_with([detail], [_block("P001")]), material, sources, SilenceCutParams(),
                             tmp_path / "o.mp4", "t").timeline
    breath = timeline.video_main[-1]
    assert (breath.label, breath.source_in, breath.source_out) == ("respiro D001.M01", 7.0, 13.0)


def test_breaths_above_the_spare_are_refused_from_the_last_and_never_two_in_a_row(tmp_path):
    material = make_trip_material(tmp_path)
    trechos = _single_trechos(material, "S004", "S005")
    candidates = _breath_candidates(material)
    answer = InsertsAnswer.model_validate({"inserir": [
        {"depois_do_trecho": 0, "letra": "A"},  # 6 s
        {"depois_do_trecho": 0, "letra": "B"},  # logo depois de outro respiro: recusado
        {"depois_do_trecho": 1, "letra": "C"},  # 2 s: passa da sobra (7 s) → recusado (o último)
        {"depois_do_trecho": 2, "letra": "B"},  # fim do bloco: não está na lista de respiro
    ]})
    check = validate_inserts(answer, trechos, candidates, material, Usage(), spare_sec=7, breath_positions=[0, 1],
                             own_positions=[0, 1, 2])
    assert [(c.after_trecho, c.letter) for c in check.inserts] == [(0, "A")]
    reasons = [d.reason for d in check.discarded]
    assert reasons == [
        "dois respiros seguidos depois do trecho 0",
        "posição 2 fora da lista para respiro (imagem do acervo) (posições: 0, 1)",
        "respiro C (2.0s) passa da sobra do bloco (7 s)",
    ]
    # Sobra menor: os dois saem, do último para o primeiro.
    both = InsertsAnswer.model_validate({"inserir": [{"depois_do_trecho": 0, "letra": "A"}, {"depois_do_trecho": 1, "letra": "C"}]})
    check = validate_inserts(both, trechos, candidates, material, Usage(), spare_sec=3, breath_positions=[0, 1])
    assert not check.inserts and len(check.discarded) == 2 and check.refused


def test_empty_answers_of_2b_and_2c_are_valid(tmp_path):
    material = make_trip_material(tmp_path)
    trechos = _single_trechos(material, "S004", "S005")
    inserts = validate_inserts(InsertsAnswer(), trechos, _breath_candidates(material), material, Usage(), spare_sec=10,
                               breath_positions=[1])
    assert inserts.refused is None and inserts.unusable is None and not inserts.inserts
    coverages = validate_coverages(CoveragesAnswer(), {1: _breath_candidates(material)[:1]}, trechos, material, Usage())
    assert coverages.refused is None and coverages.unusable is None and not coverages.coverages


def _trip_service(material, tmp_path, ai, request="vlog da viagem"):
    return _service(material, tmp_path, ai, context=EditorContext(request=request))


def test_block_without_spare_has_no_breath_question(tmp_path):
    material = make_trip_material(tmp_path)
    # B02 (P001): alvo 10 s, 14,5 s de fala mantida → sobra negativa: sem 2b.
    ai = MockEditorAI({"structure": [_structure((("P002",), 6), (("P001",), 10), fora=("P003",))],
                       "cuts": [{}, {}, {}]})
    run = _trip_service(material, tmp_path, ai).run()
    second = run.details[1]
    assert second.inserts.spare_sec == pytest.approx(10 - 13.5) and second.inserts.requests == 0
    assert second.inserts.no_question == "sem sobra para respiro" and not second.inserts.candidates
    assert not [k for k in second.keep if k.breath] and len(ai.calls["inserts"]) == 0
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "- sem pergunta: sem sobra para respiro" in report


def test_breath_candidates_are_at_most_6_with_description_and_place_block_ones_first(tmp_path):
    material = make_trip_material(tmp_path)
    ai = MockEditorAI({"structure": [_structure((("P002",), 6), (("P001",), 40), fora=("P003",))],
                       "inserts": [{"inserir": []}]})
    run = _trip_service(material, tmp_path, ai).run()
    request = ai.calls["inserts"][0]
    assert request.unit_id == "B02" and request.breath_positions == [0] and request.spare_sec == pytest.approx(26.5)
    candidates = request.candidates
    assert len(candidates) == 6 and all(c.description and c.place for c in candidates)
    # O acervo com lugar/palavra em comum com o trecho vizinho primeiro (lago, rua), e as imagens com "words".
    assert candidates[0].moment_id == "D001.M01" and candidates[0].words == ["lago"]
    assert [c.moment_id for c in candidates[1:]] == [f"B001.M0{n}" for n in range(1, 6)]  # "rua" na fala
    lines = [line for line in request.prompt.splitlines() if re.match(r"- [A-F]: ", line)]
    assert len(lines) == 6 and lines[0].startswith("- A: lago azul com montanhas ao fundo · lugar: lago · 6,0s · D001")
    # "nenhum" é a primeira opção; a resposta vazia foi aceita sem nova tentativa.
    prompt = request.prompt
    assert prompt.index("- nenhum: não incluir nada") < prompt.index("- A: ")
    assert run.details[1].inserts.origin == "ia" and not run.details[1].inserts.attempts
    # No editor_run.json, nenhum candidato leva a lista vazia de palavras.
    dumped = run.model_dump_json()
    assert '"words":[]' not in dumped.replace(" ", "")


def test_library_moment_used_in_a_block_is_not_offered_again(tmp_path):
    material = make_trip_material(tmp_path)

    def take_first(request):
        return {"inserir": [{"depois_do_trecho": request.breath_positions[0], "letra": request.candidates[0].letter}]}

    ai = MockEditorAI({"structure": [_structure((("P002",), 6), (("P001",), 40), (("P003",), 20))],
                       "inserts": [take_first, {"inserir": []}]})
    run = _trip_service(material, tmp_path, ai).run()
    first, second = ai.calls["inserts"]
    used = run.details[1].inserts.inserts[0].moment_id
    assert used == "D001.M01" and run.details[1].keep[0].breath
    assert used in [c.moment_id for c in first.candidates] and used not in [c.moment_id for c in second.candidates]
    assert run.details[1].breath_sec == pytest.approx(6.0)


def test_coverage_is_not_asked_when_the_speech_image_is_not_talking_to_the_camera(tmp_path):
    material = make_trip_material(tmp_path)
    ai = MockEditorAI({"structure": [_structure((("P001",), 40), (("P003",), 10), fora=("P002",))],
                       "inserts": [{"inserir": []}] * 3, "coverages": [{"coberturas": []}]})
    run = _trip_service(material, tmp_path, ai).run()
    hotel, square = run.details
    # P001 mostra o próprio assunto (quarto, rua, lago): nenhum trecho perguntado.
    assert hotel.coverage.requests == 0 and hotel.coverage.no_question == "nenhum trecho dito para a câmera"
    assert [k.reason for k in hotel.coverage.skipped] == ["imagem não é fala para câmera"]
    # P003 fala para a câmera na praça: perguntado; o lago do drone tem "lago" na descrição. A resposta vazia vale.
    [request] = ai.calls["coverages"]
    assert list(request.questions) == [1] and request.questions[1][0].moment_id == "D001.M01"
    assert square.coverage.origin == "ia" and not square.coverage.attempts and not square.coverages
    assert request.prompt.index("  nenhuma: a imagem de quem fala continua") < request.prompt.index("  A: ")


def test_generic_words_and_words_only_in_tags_do_not_count_as_common(tmp_path):
    material = make_trip_material(tmp_path)
    beach = _moment("X001", 1, 0.0, 4.0, "praia com pessoas", "praia", ["pessoas"])
    square = _moment("X001", 2, 4.0, 8.0, "praça com bancos", "praça", ["rio", "árvores"])
    river = _moment("X001", 3, 8.0, 12.0, "rio Limay com pedras", "margem do rio", ["rio"])
    speech = "Quarto duplo com duas pessoas, dois casais; quando cruzarmos o rio Limay."
    matches = coverage_matches(speech, [], [beach, square, river], 4, COVERAGE_SAYS, COVERAGE_PLACE)
    assert [(m.moment.id, m.words) for m in matches] == [("X001.M03", ["rio", "limay"])]
    assert GENERIC_WORDS >= {"pessoas", "gente", "pessoa", "coisa", "lugar", "dia", "vista", "paisagem", "natureza",
                             "aqui", "ali", "personas", "paisaje", "naturaleza"}
    assert matched_words("pessoas", "duas pessoas") == []
    # Na validação, "pessoas" não é palavra em comum e "rio" só nas tags não é o que a imagem mostra.
    trechos = _single_trechos(material, "S007")
    questions = {1: [Candidate(letter="A", moment_id="P001.M02", clip_id="P001", duration_sec=6.0, kind=COVERAGE_SAYS,
                               words=["rua"])]}
    answer = CoveragesAnswer.model_validate({"coberturas": [{"trecho": 1, "letra": "A", "palavra": "pessoas"}]})
    check = validate_coverages(answer, questions, trechos, material, Usage())
    assert check.discarded[0].reason.startswith("sem a palavra em comum")


def test_internal_pauses_below_1_s_stay_and_short_pieces_are_joined():
    def ranges(*pairs):
        return [TimeRange(start_sec=a, end_sec=b) for a, b in pairs]

    def pairs(result):
        return [(r.start_sec, r.end_sec) for r in result]

    assert INTERNAL_PAUSE_MIN_SEC == 1.0 and MIN_PIECE_SEC == 0.6
    assert pairs(merge_pieces(ranges((0.0, 2.0), (2.5, 4.0)))) == [(0.0, 4.0)]  # pausa de 0,5 s: fica
    assert pairs(merge_pieces(ranges((0.0, 2.0), (3.2, 5.0)))) == [(0.0, 2.0), (3.2, 5.0)]  # 1,2 s: cortada
    # Com as folgas do corte (0,35 s), 0,8 s entre os pedaços já é pausa de 1,15 s: cortada.
    assert len(merge_pieces(ranges((0.0, 2.0), (2.8, 5.0)), padding_sec=0.35)) == 2
    assert len(merge_pieces(ranges((0.0, 2.0), (2.5, 5.0)), padding_sec=0.35)) == 1
    # Pedaço de 0,3 s: unido ao vizinho mais perto (a pausa entre eles volta).
    assert pairs(merge_pieces(ranges((0.0, 2.0), (3.5, 3.8), (5.5, 8.0)))) == [(0.0, 3.8), (5.5, 8.0)]


def test_faixa_keeps_a_short_internal_pause_that_the_edit_cut_would_remove(material):
    sources = _sources(material)
    source = sources["P001"]
    source.audio = _audio(12.0, [(0.5, 2.5), (3.0, 5.7), (6.5, 9.0), (9.5, 11.5)])  # 0,8 s de silêncio entre S001 e S002
    keep = KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S002",
                     sentence_ids=["S000", "S001", "S002"])
    ranges = [(r.start_sec, r.end_sec) for r in faixa_ranges(keep, material, source, SilenceCutParams())]
    assert ranges == [(0.35, 9.2)]  # o `edit` cortaria a pausa; dentro da faixa, ela fica


def test_piece_label_lists_only_the_sentences_inside_the_piece(material):
    keep = KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S003",
                     sentence_ids=["S000", "S001", "S002", "S003"])
    assert piece_sentences(keep, TimeRange(start_sec=0.35, end_sec=5.7), material) == ["S000", "S001"]
    assert piece_sentences(keep, TimeRange(start_sec=9.45, end_sec=11.7), material) == ["S003"]
    assert piece_sentences(keep, TimeRange(start_sec=5.52, end_sec=6.45), material) == ["S001"]  # no silêncio: a mais perto
    detail = DetailRun(unit_id="B01", block_id="B01", clip_ids=["P001"], origin="ia", keep=[keep])
    timeline = assemble_plan(_run_with([detail], [_block("P001")]), material, _sources(material), SilenceCutParams(),
                             Path("nao-renderiza.mp4"), "t").timeline
    # Pausa de 1,0 s entre S001 e S002 cortada; as de 0,5 s ficam: dois pedaços, cada um com as suas frases.
    assert [i.label for i in timeline.video_main] == ["P001 S000–S001", "P001 S002–S003"]


def test_every_sentence_cut_twice_without_out_keeps_the_clip_whole_and_explicit_out_removes_it(material, tmp_path):
    everything = {"cortar": ["S000", "S001", "S002", "S003"]}
    ai = MockEditorAI({
        "structure": [_structure((("P001",), 9), (("P002",), 4))],
        "cuts": [everything, everything, {"fora": True, "notas": "fora do assunto"}],
    })
    run = _service(material, tmp_path, ai).run()
    first, second = run.details
    retry = ai.calls["cuts"][1].prompt
    assert "você cortou todas as frases; mantenha as que contam o assunto ou responda fora: true" in retry
    # Só listas que cortam tudo: tirar o clipe é da estrutura ou de um fora: true; o clipe entra inteiro.
    p001 = first.cut_runs[0]
    assert not p001.removed and p001.kept_whole and p001.kept_ids == ["S000", "S001", "S002", "S003"]
    assert first.removed_clip_ids == [] and first.speech_sec == pytest.approx(9.0)
    assert any("IA cortou tudo sem dizer fora; mantido inteiro" in w for w in run.warnings)
    # fora: true explícito na 1ª resposta: sai sem nova tentativa.
    assert second.cut_runs[0].removed and len(ai.calls["cuts"]) == 3 and second.removed_clip_ids == ["P002"]
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "**IA cortou tudo sem dizer fora; mantido inteiro**" in report and "IA tirou do bloco: P002" in report
    assert "tentativas: 1: 0,0s (cortou tudo) · 2: 0,0s (cortou tudo) → usado o clipe inteiro (sem cortes)" in report


def test_the_request_opens_every_question_with_the_respect_rule(material, tmp_path):
    request = "vlog da praia, curto"
    ai = MockEditorAI({"structure": [_structure((("P001",), 9), (("P002",), 6))]})
    _service(material, tmp_path, ai, context=EditorContext(request=request)).run()
    prompts = [ai.calls[k][0].prompt for k in ("structure", "cuts", "inserts", "coverages")]
    for prompt in prompts:
        first, second = prompt.splitlines()[:2]
        assert first == f'Pedido do usuário: "{request}"'
        assert second == "Faça o que o pedido pede; não acrescente nada que ele não peça; na dúvida, não acrescente."
    assert ai.calls["summary"][0].prompt.startswith("Você ajuda um editor")  # o resumo (E9) continua igual
    # Sem exemplo de respiro ou cobertura escolhida: o formato mostra a lista vazia / "nenhuma".
    assert '"inserir": [], "notas"' in prompts[2]
    assert '{"trecho": <número>, "letra": "nenhuma", "palavra": "", "mostra": "", "porque": ""}' in prompts[3]
    assert "TODO clipe" not in prompts[0] and "Não precisa usar todos" in prompts[0]


# --- equilíbrio: falta de duração na estrutura e na 2a, pares contínuos ---


def _texts(tmp_path: Path, **texts: str) -> EditorMaterial:
    """O roteiro de teste com o texto de algumas frases trocado (S000="...")."""
    roteiro = make_roteiro(tmp_path)
    for sentence in roteiro.sentences:
        sentence.text = texts.get(sentence.id, sentence.text)
    return EditorMaterial.from_roteiro(roteiro)


def test_structure_below_85_percent_asks_only_for_additions_and_an_empty_answer_keeps_the_first(material, tmp_path):
    only_p002 = _structure((("P002",), 20))
    ai = MockEditorAI({"structure": [only_p002]})  # o acréscimo simulado não acrescenta nada
    context = EditorContext(request="vlog, uns 20 segundos", target_sec=20)
    run = _service(material, tmp_path, ai, context=context).run()
    assert len(ai.calls["structure"]) == 1 and "faltam" not in ai.calls["structure"][0].prompt
    [request] = ai.calls["additions"]
    ask = request.prompt
    assert request.offered == ["P001"]  # só clipes com fala: o acervo não é oferecido
    assert "Duração: o pedido é de 20 s; os clipes escolhidos têm 4 s de fala; faltam 16 s." in ask
    assert "- P001 · 9 s de fala — Clipe P001" in ask and "D001" not in ask and "B001" not in ask
    assert "Faltam 16 s; quais destes clipes acrescentar?" in ask and "nenhum também vale (lista vazia)" in ask
    assert '{"acrescentar": ["código do clipe"], "notas": "..."}' in ask and '"blocos"' not in ask
    structure = run.structure
    assert [b.clip_ids for b in structure.blocks] == [["P002"]] and not structure.attempts
    assert (structure.speech_sec_first, structure.speech_sec, structure.added_clip_ids) == (4.0, 4.0, [])
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    # O pedido (20 s) passa da fala do material (13 s): o alvo vira 13 s, com aviso.
    assert "Fala dos clipes escolhidos: 0:04 (4s) (31% do alvo) → acréscimo (faltam 16 s" in report
    assert "Clipes acrescentados (→ bloco em que entraram): nenhum (a IA não acrescentou clipes)" in report

    ai = MockEditorAI({"structure": [only_p002], "additions": [{"acrescentar": ["P001"]}]})
    run = _service(material, tmp_path / "b", ai, context=context).run()
    assert run.structure.added_clip_ids == ["P001"] and run.structure.speech_sec == 13.0
    assert [b.clip_ids for b in run.structure.blocks] == [["P001", "P002"]]
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "→ 0:13 (13s) (100% do alvo)" in report
    assert "Clipes acrescentados (→ bloco em que entraram): P001 → B01" in report


def test_structure_at_85_percent_or_without_requested_duration_has_no_retry(material, tmp_path):
    both = _structure((("P001", "P002"), 14))
    ai = MockEditorAI({"structure": [both]})
    run = _service(material, tmp_path, ai, context=EditorContext(request="vlog", target_sec=15)).run()
    assert len(ai.calls["structure"]) == 1 and run.structure.speech_sec_first is None  # 13 s ≥ 85% de 15 s
    ai = MockEditorAI({"structure": [{**_structure((("P002",), 4)), "duracao_total_alvo_sec": 60}]})
    run = _service(material, tmp_path / "b", ai).run()  # a duração escolhida pela IA não pede nova tentativa
    assert len(ai.calls["structure"]) == 1 and run.structure.speech_sec == 4.0


def test_whole_library_out_is_reported(material, tmp_path):
    ai = MockEditorAI({"structure": [_structure((("P001", "P002"), 13), fora=("D001", "B001"))]})
    run = _service(material, tmp_path, ai).run()
    assert run.structure.library_clip_ids == []
    assert any("todo o acervo foi para \"fora\" (D001, B001)" in w for w in run.warnings)


@pytest.mark.parametrize("texts, cut, expected_cuts, joined", [
    # S000 sem pontuação final: S000+S001 é uma unidade; cortou só S000 (2 s < 2,5 s) → mantém as duas.
    ({"S000": "Chegamos na praia hoje"}, ["S000"], [], "par contínuo S000+S001 mantido junto"),
    # S000 com reticências: cortou só S001 (2,5 s > 2 s) → corta as duas.
    ({"S000": "Chegamos na praia hoje..."}, ["S001"], ["S000", "S001"], "par contínuo S000+S001 cortado junto"),
    # S003 começa com minúscula: continua S002; cortou só S002 (2,5 s > 2 s) → corta as duas.
    ({"S003": "agora vamos pela trilha da mata."}, ["S002"], ["S002", "S003"], "par contínuo S002+S003 cortado junto"),
])
def test_continuous_pair_cut_in_half_is_cut_or_kept_together_by_the_bigger_part(tmp_path, texts, cut, expected_cuts, joined):
    material = _texts(tmp_path, **texts)
    check = validate_cuts(CutsAnswer.model_validate({"cortar": cut}), ["S000", "S001", "S002", "S003"], material)
    assert [c.sentence_id for c in check.cuts] == expected_cuts and check.joined == [joined]
    assert not check.discarded and check.refused is None
    whole = validate_cuts(CutsAnswer.model_validate({"cortar": ["S000", "S001"]}), ["S000", "S001", "S002", "S003"],
                          _texts(tmp_path, S000="Chegamos na praia hoje"))
    assert not whole.joined  # o par inteiro cortado: nada a ajustar


def test_cuts_prompt_marks_continuous_pairs_and_the_report_lists_the_adjustment(tmp_path):
    material = _texts(tmp_path, S000="Chegamos na praia hoje", S003="agora vamos pela trilha da mata.")
    ai = MockEditorAI({"structure": [_structure((("P001",), 9))], "cuts": [{"cortar": ["S000"]}]})
    run = _service(material, tmp_path, ai).run()
    prompt = ai.calls["cuts"][0].prompt
    assert "- S000 — Chegamos na praia hoje (continua na próxima)" in prompt
    assert "- S002 — Agora vamos pela trilha. (continua na próxima)" in prompt  # S003 começa com minúscula
    assert "- S001 — O mar está muito azul.\n" in prompt and "S003 — agora vamos pela trilha da mata.\n" in prompt
    assert "corte as duas ou mantenha as duas" in prompt
    cut_run = run.details[0].cut_runs[0]
    assert cut_run.cuts == [] and cut_run.joined == ["par contínuo S000+S001 mantido junto"]
    assert run.details[0].trechos[0].sentence_ids[0] == "S000"  # o bloco não começa pela metade da fala
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    assert "  - par contínuo S000+S001 mantido junto" in build_editor_report("t", run, plan, material, "editor_ia", None, None)


def test_cuts_below_the_target_retry_listing_the_cut_sentences_even_when_skipped(material, tmp_path):
    ai = MockEditorAI({
        "structure": [_structure((("P001",), 9))],
        "cuts": [{"cortar": ["S000", "S002"]}, {"cortar": ["S002"]}],  # salteado: 4,5 s × meta 9 s (aceito de 6 a 12)
    })
    run = _service(material, tmp_path, ai).run()
    first, retry = (r.prompt for r in ai.calls["cuts"])
    assert "cortada" not in first
    assert "ficou 4 s, a meta deste clipe é 9 s (aceito de 6 a 12 s): devolva cerca de 4 s (tire frases" in retry
    assert "frases que você cortou (para devolver uma, tire-a da lista cortar):" in retry
    assert "- cortada S000 (2 s): Chegamos na praia hoje." in retry and "- cortada S002 (" in retry
    cut_run = run.details[0].cut_runs[0]
    assert [c.sentence_id for c in cut_run.cuts] == ["S002"] and cut_run.result_sec == pytest.approx(6.5)
    assert cut_run.attempts == [
        "duração: ficou 4 s, a meta deste clipe é 9 s (aceito de 6 a 12 s): devolva cerca de 4 s (tire frases da lista "
        "cortar); se devolver frases prejudica a fala, ajuste só o que não prejudica e explique em limite_meta"
    ]


def test_clip_with_one_sentence_or_short_speech_has_no_cuts_question_and_enters_whole(tmp_path, monkeypatch):
    roteiro = make_roteiro(tmp_path)
    roteiro.sentences = [s for s in roteiro.sentences if s.id != "S005"]  # P002 com uma frase só
    material = EditorMaterial.from_roteiro(roteiro)
    ai = MockEditorAI({"structure": [_structure((("P001", "P002"), 8))]})
    run = _service(material, tmp_path, ai).run()
    assert {r.clip_id for r in ai.calls["cuts"]} == {"P001"}  # 1 frase: sem pedido
    p001, p002 = run.details[0].cut_runs
    assert (p002.no_question, p002.kept_ids, p002.result_sec) == ("1 frase", ["S004"], 2.0)
    assert p001.target_sec == 6.0  # a meta do bloco menos a fala do clipe que entra inteiro
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "- B01.P002: sem pergunta de cortes (1 frase): entra inteiro" in report

    monkeypatch.setattr(editor_service, "CUT_MIN_SPEECH_SEC", 12.0)  # o valor de produção
    material = EditorMaterial.from_roteiro(make_roteiro(tmp_path / "b"))
    ai = MockEditorAI({"structure": [_structure((("P001",), 4), (("P002",), 2))]})
    run = _service(material, tmp_path / "b", ai).run()
    assert not ai.calls["cuts"]  # 9 s e 4 s de fala: nenhum pedido 2a
    assert [(r.no_question, r.kept_ids) for d in run.details for r in d.cut_runs] == [
        ("fala curta", ["S000", "S001", "S002", "S003"]), ("fala curta", ["S004", "S005"]),
    ]


def _check(result_sec, kept=True, out=False):
    from types import SimpleNamespace

    return SimpleNamespace(result_sec=result_sec, kept=["S"] if kept else [], out=out)


def test_the_attempt_closest_to_the_target_is_used_and_cutting_everything_never_wins():
    closest = editor_service._closest_cut
    assert closest([_check(67.0), _check(2.6)], 48.0) == 0  # P019: 67 s × 2,6 s para meta 48 s
    assert closest([_check(47.0), _check(0.0, kept=False)], 34.6) == 0  # P007: a que corta tudo nunca ganha
    assert closest([_check(30.0), _check(40.0)], 35.0) == 0  # empate: a primeira
    assert closest([_check(20.0), _check(0.0, kept=False, out=True)], 34.6) == 1  # fora: true explícito vale


def test_cuts_keep_the_attempt_closest_to_the_target_and_report_each_one(material, tmp_path):
    ai = MockEditorAI({
        "structure": [_structure((("P001",), 4.8))],
        "cuts": [{"cortar": ["S000"]}, {"cortar": ["S000", "S001", "S002"]}],  # 7 s (acima) → 2 s (abaixo)
    })
    run = _service(material, tmp_path, ai).run()
    cut_run = run.details[0].cut_runs[0]
    assert [(t.attempt, t.result_sec) for t in cut_run.tries] == [(1, 7.0), (2, 2.0)]
    assert cut_run.chosen_attempt == 1 and cut_run.result_sec == 7.0 and cut_run.duration_off
    assert [c.sentence_id for c in cut_run.cuts] == ["S000"]
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "tentativas: 1: 7,0s · 2: 2,0s → usada a 1" in report


def test_target_is_read_from_the_request_text_and_block_targets_are_rescaled_to_it(tmp_path):
    big = make_big_material(tmp_path, clips=2, sentences=80)  # 2 × 272 s de fala
    answer = {**_structure((("P001",), 100), (("P002",), 300)), "duracao_total_alvo_sec": 385}
    ai = MockEditorAI({"structure": [answer]})
    run = _service(big, tmp_path, ai, context=EditorContext(request="vlog da trilha, uns 8 minutos")).run()
    structure = run.structure
    assert (structure.target_total_sec, structure.target_origin) == (480.0, "texto")
    assert structure.target_note == '"8 minutos" no texto do pedido'
    assert [(b.ai_target_sec, b.target_sec) for b in structure.blocks] == [(100, 120.0), (300, 360.0)]
    assert sum(b.target_sec for b in structure.blocks) == 480
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, big, "editor_ia", None, None)
    assert '| Duração alvo | 8:00 (480s) (lida do texto do pedido: "8 minutos" no texto do pedido) |' in report
    assert "| 2:00 (120s) (IA: 100s) |" in report


def test_requested_target_above_the_material_speech_becomes_the_speech_with_a_warning(material, tmp_path):
    run = _service(material, tmp_path, context=EditorContext(request="vlog de 8 minutos")).run()
    assert (run.structure.target_total_sec, run.structure.target_requested_sec) == (13.0, 480.0)
    assert any("alvo: 8:00 pedido, mas o material só tem 0:13 de fala; alvo = 0:13" in w for w in run.warnings)


def test_clip_targets_share_the_block_by_speech_and_never_pass_the_clip_speech():
    targets = editor_service._clip_targets
    assert targets(300.0, {"A": 100.0, "B": 150.0}, set()) == {"A": 100.0, "B": 150.0}
    assert targets(60.0, {"A": 10.0, "B": 100.0, "C": 50.0}, {"A"}) == {"A": 10.0, "B": 33.3, "C": 16.7}
    assert targets(None, {"A": 10.0}, set()) == {}


def test_empty_block_is_dropped_from_the_plan_with_a_warning(material, tmp_path):
    full = DetailRun(
        unit_id="B01", block_id="B01", clip_ids=["P001"], origin="ia",
        keep=[KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S001",
                        sentence_ids=["S000", "S001"])],
    )
    empty = DetailRun(unit_id="B02", block_id="B02", clip_ids=[], origin="ia", removed_clip_ids=["P002"])
    run = _run_with([full, empty], [_block("P001"), _block("P002", block_id="B02")])
    assembled = assemble_plan(run, material, _sources(material), SilenceCutParams(), tmp_path / "out.mp4", "teste")
    assert [b.id for b in assembled.plan.blocks] == ["B01"]
    assert "B02: bloco vazio (nenhum clipe ficou), fora do plano" in assembled.plan.warnings


def test_cuts_prompt_has_no_list_of_example_reasons(material, tmp_path):
    ai = MockEditorAI({"structure": [_structure((("P001",), 9))]})
    _service(material, tmp_path, ai).run()
    prompt = ai.calls["cuts"][0].prompt
    assert '"curto"' not in prompt and "Fala fraca" not in prompt and "hesitação" not in prompt
    assert "escreva com as suas palavras" in prompt and "não há lista de motivos" in prompt
    assert '"motivo": "por que esta frase sai"' in prompt
    long_reason = CutItem.model_validate({"frase": "S000", "motivo": "x " * 200}).motivo
    assert len(long_reason) == 120 and long_reason.endswith("…")


def test_clip_or_block_whose_speech_fits_the_target_has_no_cuts_question(material, tmp_path, monkeypatch):
    monkeypatch.setattr(editor_service, "CUT_EXCESS_RATIO", 1.15)  # o valor de produção
    ai = MockEditorAI({"structure": [_structure((("P001",), 8))]})  # 9 s de fala ≤ 8 × 1,15
    run = _service(material, tmp_path, ai).run()
    assert not ai.calls["cuts"]
    [cut_run] = run.details[0].cut_runs
    assert (cut_run.no_question, cut_run.result_sec, cut_run.target_sec) == ("cabe na meta", 9.0, 8.0)
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "- B01.P001: sem pergunta de cortes (cabe na meta): entra inteiro" in report

    ai = MockEditorAI({"structure": [_structure((("P001", "P002"), 12))]})  # 13 s de fala ≤ 12 × 1,15
    run = _service(material, tmp_path / "b", ai).run()
    assert not ai.calls["cuts"]
    assert [r.no_question for r in run.details[0].cut_runs] == ["o bloco cabe na meta"] * 2

    ai = MockEditorAI({"structure": [_structure((("P001",), 7))]})  # 9 s > 7 × 1,15: perguntado
    _service(material, tmp_path / "c", ai).run()
    assert "Este clipe tem 9 s de fala; a meta é 7 s; corte cerca de 2 s de frases" in ai.calls["cuts"][0].prompt


def test_whole_clip_competes_when_the_answers_are_out_of_the_target(material, tmp_path, monkeypatch):
    monkeypatch.setattr(editor_service, "CUT_EXCESS_RATIO", 1.15)
    almost_all = {"cortar": ["S000", "S001", "S002"]}  # fica 2 s para a meta 7,5 s (como 0,9 s para 27 s)
    ai = MockEditorAI({"structure": [_structure((("P001",), 7.5))], "cuts": [almost_all, almost_all]})
    run = _service(material, tmp_path, ai).run()
    cut_run = run.details[0].cut_runs[0]
    assert [t.result_sec for t in cut_run.tries] == [2.0, 2.0]
    assert cut_run.chosen_attempt == 0 and cut_run.cuts == [] and cut_run.result_sec == 9.0  # 9 s: mais perto
    assert any("clipe inteiro (9,0s) mais perto da meta" in w for w in run.warnings)
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "tentativas: 1: 2,0s · 2: 2,0s → usado o clipe inteiro (sem cortes)" in report


def test_cached_structure_below_85_percent_gets_the_addition_question_once(material, tmp_path):
    only_p002 = _structure((("P002",), 20))
    context = EditorContext(request="vlog", target_sec=20)
    ai = MockEditorAI({"structure": [only_p002], "additions": [AIProviderError("caiu")]})  # o acréscimo falha
    run = _service(material, tmp_path, ai, context=context).run()
    assert len(ai.calls["additions"]) == 1 and run.structure.origin == "ia" and run.structure.added_clip_ids == []
    assert any(a.startswith("acréscimo: erro da IA") for a in run.structure.attempts)
    # Do cache, ainda sem o acréscimo: a pergunta é feita agora (e a resposta vai para o cache).
    ai = MockEditorAI({"additions": [{"acrescentar": ["P001"]}]})
    run = _service(material, tmp_path, ai, context=context).run()
    assert not ai.calls["structure"] and len(ai.calls["additions"]) == 1
    assert "faltam 16 s" in ai.calls["additions"][0].prompt
    assert run.structure.origin == "cache" and run.structure.added_clip_ids == ["P001"]
    ai = MockEditorAI()
    run = _service(material, tmp_path, ai, context=context).run()
    assert not ai.calls["structure"] and not ai.calls["additions"] and run.structure.origin == "cache"
    assert (run.structure.speech_sec_first, run.structure.added_clip_ids) == (4.0, ["P001"])
    assert [b.clip_ids for b in run.structure.blocks] == [["P001", "P002"]]


def test_another_target_invalidates_the_cached_structure(material, tmp_path):
    both = _structure((("P001", "P002"), 13))
    _service(material, tmp_path, MockEditorAI({"structure": [both]}), context=EditorContext(request="vlog", target_sec=14)).run()
    same = MockEditorAI()
    _service(material, tmp_path, same, context=EditorContext(request="vlog", target_sec=14)).run()
    assert not same.calls["structure"]
    other = MockEditorAI({"structure": [both]})
    _service(material, tmp_path, other, context=EditorContext(request="vlog", target_sec=15)).run()
    assert len(other.calls["structure"]) == 1


def test_structure_prompt_asks_for_narrative_function_continuity_and_flexible_order(material, tmp_path):
    ai = MockEditorAI()
    _service(material, tmp_path, ai).run()
    prompt = ai.calls["structure"][0].prompt
    assert "Cada bloco tem uma função na história" in prompt and "continua o anterior" in prompt
    assert "evite repetir num bloco o que outro já contou" in prompt
    assert "o vídeo segue a ordem de gravação, que é a ordem da lista abaixo" in prompt
    assert "Você escolhe onde começa cada bloco" in prompt and "é dividido em partes" in prompt
    ai = MockEditorAI()
    _service(material, tmp_path / "t", ai, context=EditorContext(request="vlog montado por tema")).run()
    prompt = ai.calls["structure"][0].prompt
    assert "o pedido permite outra ordem que não a de gravação" in prompt and "ordem de gravação, que é" not in prompt


def test_cuts_put_meaning_before_the_target_and_the_ai_can_report_the_limit(material, tmp_path, monkeypatch):
    monkeypatch.setattr(editor_service, "CUT_EXCESS_RATIO", 1.15)
    limit = {"cortar": ["S000"], "limite_meta": "cortar mais tira a explicação da trilha"}  # 7 s para meta 4,5 s
    ai = MockEditorAI({"structure": [_structure((("P001",), 4.5))], "cuts": [limit]})
    run = _service(material, tmp_path, ai).run()
    prompt = ai.calls["cuts"][0].prompt
    assert "A meta é uma restrição, mas o sentido da fala vem antes" in prompt
    assert '"limite_meta": ""' in prompt
    assert len(ai.calls["cuts"]) == 1  # a IA já disse que não dá: sem nova tentativa por duração
    cut_run = run.details[0].cut_runs[0]
    assert cut_run.result_sec == 7.0 and cut_run.duration_off and cut_run.chosen_attempt == 1
    assert cut_run.target_limit == "cortar mais tira a explicação da trilha"
    assert any("a IA diz que não dá sem prejudicar a fala" in w for w in run.warnings)
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "IA: a meta não dá sem prejudicar a fala — cortar mais tira a explicação da trilha" in report
    assert CutsAnswer.model_validate({"limite_meta": "nenhum"}).limite_meta == ""


def test_context_of_the_previous_piece_only_for_split_clips(material, tmp_path):
    whole = tokens(build_summary_prompt(material.section("P001")))
    reserved = editor_budget.SUMMARY_OUTPUT_TOKENS + editor_budget.MARGIN_TOKENS + editor_budget.RETRY_TOKENS
    ai = MockEditorAI()
    _service(material, tmp_path, ai, num_ctx=whole + reserved - 10).run()
    first, second = [r.prompt for r in ai.calls["summary"] if r.clip_id == "P001"]
    assert "Contexto do pedaço anterior" not in first
    assert "Contexto do pedaço anterior (só para entender este; não resuma de novo): resumo: Clipe P001" in second
    assert 'última frase: "O mar está muito azul."' in second
    p002 = next(r.prompt for r in ai.calls["summary"] if r.clip_id == "P002")
    assert "Contexto do pedaço anterior" not in p002  # clipe inteiro: o pedido de sempre

    big = make_big_material(tmp_path / "b", sentences=120)
    ai = MockEditorAI({"structure": [_structure((("P001",), 200))]})
    _service(big, tmp_path / "b", ai, num_ctx=8192).run()
    by_unit = {}
    for request in ai.calls["cuts"]:
        by_unit.setdefault(request.unit_id, request.prompt)  # o 1º pedido de cada parte (sem a nova tentativa)
    first_part, second_part = by_unit["B01.P001a"], by_unit["B01.P001b"]
    assert "- logo depois:" in first_part and "- logo antes:" not in first_part
    assert "- logo antes:" in second_part and "não decida sobre elas" in second_part
    ai = MockEditorAI({"structure": [_structure((("P001",), 9))]})
    _service(material, tmp_path / "c", ai).run()
    assert "logo antes" not in ai.calls["cuts"][0].prompt  # clipe num pedido só: sem frases vizinhas


def make_semantic_material(tmp_path: Path) -> EditorMaterial:
    """P001 fala do almoço na mesa para a câmera; B001 (gravado 5 min depois) mostra um prato de comida na mesa."""
    clips = [_clip("P001", "principal", 1, 8.0, True, recorded_at=_at("13:00")),
             _clip("B001", "broll", 2, 6.0, False, recorded_at=_at("13:05"))]
    moments = [
        _moment("P001", 1, 0.0, 8.0, "pessoa falando na varanda", "varanda", ["pessoa"], ["S000", "S001"], talking=True),
        _moment("B001", 1, 0.0, 6.0, "prato de comida na mesa", "restaurante", ["almoço", "comida"]),
    ]
    sentences = [
        _sentence(0, "P001", 0.5, 3.5, "Hoje tivemos um almoço delicioso na mesa.", "P001.M01"),
        _sentence(1, "P001", 4.0, 7.5, "Depois seguimos viagem.", "P001.M01"),
    ]
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0, clips=clips,
                             moments=moments, sentences=sentences)
    return EditorMaterial.from_roteiro(roteiro)


def test_coverage_accepts_an_explained_correspondence_with_visual_evidence(tmp_path):
    material = make_semantic_material(tmp_path)

    def answer(request):
        [(number, [candidate])] = request.questions.items()
        return {"coberturas": [{"trecho": number, "letra": candidate.letter, "palavra": "",
                                "mostra": "um prato de comida na mesa", "porque": "ilustra o almoço que ela conta"}]}

    ai = MockEditorAI({"structure": [_structure((("P001",), 7))], "coverages": [answer]})
    run = _service(material, tmp_path, ai).run()
    prompt = ai.calls["coverages"][0].prompt
    assert "— em comum: mesa" in prompt
    assert "Diga em mostra o que a imagem realmente mostra" in prompt
    [coverage] = run.details[0].coverages
    assert coverage.semantic and coverage.moment_ids == ["B001.M01"] and coverage.sentence_ids[0] == "S000"
    assert (coverage.word, coverage.image_evidence) == ("almoco", "um prato de comida na mesa")
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert 'correspondência explicada, fala "almoco"' in report and "porque: ilustra o almoço" in report

    trechos = run.details[0].trechos
    [[candidate]] = [c for c in [ai.calls["coverages"][0].questions[1]]]
    questions = {1: [candidate]}
    letter = candidate.letter

    def check(item):
        return validate_coverages(CoveragesAnswer.model_validate({"coberturas": [{"trecho": 1, "letra": letter, **item}]}),
                                  questions, trechos, material, Usage())

    # Sem evidência suficiente: o item sai e o trecho fica sem cobertura.
    assert check({"mostra": "montanha nevada", "porque": "fala do almoço"}).discarded[0].reason.startswith(
        "mostra 'montanha nevada' não está na descrição/lugar da imagem")
    assert "não cita nada da fala do trecho 1" in check({"mostra": "prato de comida", "porque": "imagem bonita"}).discarded[0].reason
    assert check({"palavra": ""}).discarded[0].reason.startswith("sem a palavra em comum")


def _first_clip_only(request):
    """Estrutura simulada de uma parte: um bloco só, com o primeiro clipe da parte (o resto não escolhido)."""
    return {"blocos": [{"titulo": f"Começo {request.clip_ids[0]}", "clipes": [request.clip_ids[0]]}]}


LONG_SUMMARY = "extraordinariamente " * 10  # linhas longas: a lista do acréscimo não cabe inteira


def test_addition_after_the_parts_with_the_list_limited_to_the_budget(tmp_path):
    vlog = make_vlog_material(tmp_path)
    ai = MockEditorAI({"summary": [{"resumo": LONG_SUMMARY}] * 41, "structure": [_first_clip_only] * 10})
    run = _service(vlog, tmp_path, ai, num_ctx=4600, context=EditorContext(request="vlog", target_sec=480)).run()
    parts = run.structure.parts
    assert len(parts) > 1 and len(ai.calls["structure"]) == len(parts)
    [request] = ai.calls["additions"]  # uma pergunta só, depois de juntar as partes, sobre o total
    chosen = [p.clip_ids[0] for p in parts]
    assert run.structure.speech_sec_first == pytest.approx(sum(vlog.speech_sec(c) for c in chosen))
    unused = [c for c in vlog.clip_ids if vlog.has_speech(c) and c not in chosen]
    offered = [line for line in request.prompt.splitlines() if line.startswith("- P") and " s de fala" in line]
    hidden = len(unused) - len(request.offered)
    assert len(offered) == len(request.offered) and hidden > 0
    assert f"- (mais {hidden} clipe(s) com fala não usado(s), de menos fala, ficaram fora desta lista)" in request.prompt
    assert tokens(request.prompt) <= Budget(4600, "t").structure_available


def test_addition_list_shows_the_clips_with_most_speech_first(tmp_path):
    material = make_story_material(tmp_path)
    for n, clip_id in enumerate(material.clip_ids):  # P001 4 s ... P005 20 s de fala
        first, last = material.clip_sentences[clip_id]
        last.end_sec = last.start_sec + 4.0 * (n + 1) - (first.end_sec - first.start_sec)
    summaries = {c: ClipSummary(clip_id=c, answer=ClipSummaryAnswer(resumo=LONG_SUMMARY)) for c in material.clip_ids}
    service = _service(material, tmp_path, context=EditorContext(request="vlog", target_sec=100))
    check = editor_service.validate_structure(StructureAnswer.model_validate(_structure(("P001",))), material)
    full = service._shortfall(check, summaries)
    assert full.offered == ["P002", "P003", "P004", "P005"] and full.hidden == 0
    # Espaço para três: ficam os de mais fala (P003 a P005), em ordem de gravação, e o aviso do que ficou fora.
    three = StructureShortfall(target_sec=full.target_sec, speech_sec=full.speech_sec, offers=full.offers[1:],
                               block_lines=full.block_lines, hidden=1)
    room = tokens(build_additions_prompt(service.context, three))
    service.budget = Budget(room + service.budget.num_ctx - service.budget.structure_available, "t")
    limited = service._shortfall(check, summaries)
    assert limited.offered == ["P003", "P004", "P005"] and limited.hidden == 1
    assert "- (mais 1 clipe(s) com fala não usado(s)" in build_additions_prompt(service.context, limited)


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
    # O drone verde não cobre "campo verde": sem data de gravação e de outro lugar ("cenário verde" × o lugar de
    # quem fala), não é do mesmo momento da viagem. Um bloco só, que abre o vídeo, com um trecho de fala: nenhuma
    # posição de respiro (o vídeo começa pela fala).
    assert timeline.video_overlay == []
    assert not any((i.label or "").startswith("respiro") for i in timeline.video_main)
    assert result.final_duration_sec == pytest.approx(timeline.duration_sec, abs=0.1)
    report = result.report_path.read_text(encoding="utf-8")
    assert "## Estrutura (passada 1)" in report and "Ordem dos blocos: de gravação" in report
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


# --- ordem de gravação (estrutura) ---


def make_story_material(tmp_path: Path, dates: Optional[dict] = None) -> EditorMaterial:
    """P001..P005 com fala, na ordem do catálogo; `dates`: clipe → "HH:MM" (data de gravação confiável)."""
    dates = dates or {}
    clips, moments, sentences = [], [], []
    for n in range(1, 6):
        clip_id = f"P{n:03d}"
        clips.append(_clip(clip_id, "principal", n, 10.0, True, recorded_at=_at(dates[clip_id]) if clip_id in dates else None))
        moments.append(_moment(clip_id, 1, 0.0, 10.0, f"pessoa falando {n}", "rua", ["rua"],
                               [f"S{2 * n - 2:03d}", f"S{2 * n - 1:03d}"], talking=True))
        sentences += [_sentence(2 * n - 2, clip_id, 0.5, 4.5, f"Primeira fala do clipe {n}.", f"{clip_id}.M01"),
                      _sentence(2 * n - 1, clip_id, 5.0, 9.0, f"Segunda fala do clipe {n}.", f"{clip_id}.M01")]
    roteiro = ProjectRoteiro(project_dir=tmp_path, source=tmp_path, kind="pasta", interval_sec=2.0, clips=clips,
                             moments=moments, sentences=sentences)
    return EditorMaterial.from_roteiro(roteiro)


VLOG = Path(__file__).parent / "data" / "editor_vlog"


@pytest.fixture(scope="module")
def vlog() -> EditorMaterial:
    """Roteiro real do vlog "Trilha La Islita" (fotos a cada 1 s; movimento por amostra do scan de cada clipe)."""
    return EditorMaterial.from_roteiro(ProjectRoteiro.load(VLOG / "roteiro.json"))


def test_chronology_by_reliable_date_or_by_catalog_order(tmp_path, vlog):
    dated = make_story_material(tmp_path, {"P001": "10:00", "P002": "09:00", "P003": "11:00", "P004": "08:00",
                                           "P005": "12:00"})
    assert dated.chronology() == (["P004", "P002", "P001", "P003", "P005"], "data")
    # Uma data faltando (ou não confiável): a ordem do catálogo, que já ordena cada papel pela melhor data.
    partial = make_story_material(tmp_path, {"P001": "10:00", "P002": "09:00"})
    assert partial.chronology() == (["P001", "P002", "P003", "P004", "P005"], "catálogo")
    assert dated.chronology(["P003", "P001"]) == (["P001", "P003"], "data")
    # Vlog real: só o V001 tem data confiável; os P seguem o catálogo.
    order, how = vlog.chronology([f"P{n:03d}" for n in range(1, 29)])
    assert how == "catálogo" and order == [f"P{n:03d}" for n in range(1, 29)]
    assert [c for c in vlog.clip_ids if vlog.clips[c].recorded_at is not None] == ["V001"]


def test_mixed_block_is_split_into_contiguous_parts_in_recording_order(tmp_path):
    material = make_story_material(tmp_path)
    answer = {"blocos": [
        {"titulo": "Praia", "clipes": ["P005"], "duracao_alvo_sec": 10},
        {"titulo": "Trilha", "clipes": ["P004", "P001"], "duracao_alvo_sec": 20},
        {"titulo": "Hotel", "clipes": ["P003", "P002"], "duracao_alvo_sec": 20},
    ]}
    ai = MockEditorAI({"structure": [answer]})
    run = _service(material, tmp_path, ai, context=EditorContext(request="vlog", target_sec=50)).run()
    blocks = run.structure.blocks
    assert [(b.id, b.title, b.clip_ids) for b in blocks] == [
        ("B01", "Trilha (parte 1/2)", ["P001"]), ("B02", "Hotel", ["P002", "P003"]),
        ("B03", "Trilha (parte 2/2)", ["P004"]), ("B04", "Praia", ["P005"]),
    ]
    # A meta de cada bloco da IA repartida pela fala (o alvo de 50 s vira a fala disponível, 40 s).
    assert [b.target_sec for b in blocks] == [8.0, 16.0, 8.0, 8.0]
    assert blocks[0].ai_block == "B02 Trilha" and blocks[1].ai_block == ""
    assert run.structure.order == "de gravação (ordem do catálogo)"
    assert any('o bloco "Trilha" da IA juntava momentos diferentes da gravação (P001 | P004): dividido em 2 partes' in w
               for w in run.warnings)
    report = build_editor_report("t", run, EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0),
                                 material, "editor_ia", None, None)
    assert "Ordem dos blocos: de gravação (ordem do catálogo)" in report


def test_added_clips_join_the_block_of_the_chronological_neighbor(tmp_path):
    material = make_story_material(tmp_path)
    blocks = [Block(id="B01", title="Chegada", clip_ids=["P002"], target_sec=8.0),
              Block(id="B02", title="Trilha", clip_ids=["P004"], target_sec=8.0)]
    result, warnings = _add_to_neighbors(blocks, ["P005", "P001", "P003"], material)
    # P001 não tem anterior: vai para o bloco do seguinte (P002); P003 → P002; P005 → P004. Nenhum bloco novo;
    # a meta do bloco soma a fala do clipe acrescentado (8 s cada).
    assert [(b.title, b.clip_ids, b.target_sec) for b in result] == [
        ("Chegada", ["P001", "P002", "P003"], 24.0), ("Trilha", ["P004", "P005"], 16.0),
    ]
    assert [b.clip_ids for b in blocks] == [["P002"], ["P004"]]  # os blocos recebidos não mudam
    assert len(warnings) == 3
    assert warnings[2] == 'acréscimo: P005 entrou no bloco de P004 ("Trilha"), o vizinho na ordem de gravação'


_FIRST = {"blocos": [{"titulo": "Começo", "clipes": ["P001", "P002"], "duracao_alvo_sec": 16},
                     {"titulo": "Fim", "clipes": ["P004"], "duracao_alvo_sec": 8}]}


def test_addition_only_adds_to_the_first_structure(tmp_path):
    material = make_story_material(tmp_path)
    ai = MockEditorAI({"structure": [_FIRST], "additions": [{"acrescentar": ["P003"]}]})
    run = _service(material, tmp_path, ai, context=EditorContext(request="vlog", target_sec=40)).run()
    assert len(ai.calls["structure"]) == 1 and ai.calls["additions"][0].offered == ["P003", "P005"]
    structure = run.structure
    # 3 clipes + 1 acrescentado = 4, nenhum removido; o novo entra no bloco do vizinho (P002).
    assert [(b.title, b.clip_ids) for b in structure.blocks] == [("Começo", ["P001", "P002", "P003"]), ("Fim", ["P004"])]
    assert structure.added_clip_ids == ["P003"] and structure.added_blocks == {"P003": "B01 Começo"}
    assert (structure.speech_sec_first, structure.speech_sec) == (24.0, 32.0)
    assert [o.clipe for o in structure.out] == ["P005"]
    plan = EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0)
    report = build_editor_report("t", run, plan, material, "editor_ia", None, None)
    assert "Clipes acrescentados (→ bloco em que entraram): P003 → B01 Começo" in report


def test_addition_ignores_codes_outside_the_offer_with_one_warning(tmp_path):
    material = make_story_material(tmp_path)
    ai = MockEditorAI({"structure": [_FIRST], "additions": [{"acrescentar": ["P002", "X999", "p005", "P005"]}]})
    run = _service(material, tmp_path, ai, context=EditorContext(request="vlog", target_sec=40)).run()
    assert run.structure.added_clip_ids == ["P005"]
    assert [b.clip_ids for b in run.structure.blocks] == [["P001", "P002"], ["P004", "P005"]]
    assert [w for w in run.warnings if "fora da lista oferecida" in w] == [
        "acréscimo: 2 código(s) fora da lista oferecida ignorado(s): P002, X999 (oferecidos: P003, P005)"
    ]


def test_empty_addition_keeps_the_first_structure(tmp_path):
    material = make_story_material(tmp_path)
    for empty in ({"acrescentar": []}, {"acrescentar": "nenhum"}, {}):
        ai = MockEditorAI({"structure": [_FIRST], "additions": [empty]})
        run = _service(material, tmp_path / str(len(str(empty))), ai, context=EditorContext(request="vlog", target_sec=40)).run()
        assert [b.clip_ids for b in run.structure.blocks] == [["P001", "P002"], ["P004"]]
        assert run.structure.added_clip_ids == [] and not run.structure.attempts
        assert not [w for w in run.warnings if "acréscimo" in w]


# --- estrutura em partes (material grande: a lista inteira não cabe no contexto) ---

VLOG99 = Path(__file__).parent / "data" / "editor_vlog99"


@pytest.fixture(scope="module")
def vlog99() -> tuple[EditorMaterial, dict, str]:
    """Caso real de 99 clipes (75 com fala): roteiro reduzido ao que a estrutura usa, resumos e o pedido de 15 min."""
    data = json.loads((VLOG99 / "resumos.json").read_text(encoding="utf-8"))
    summaries = {
        c: ClipSummary(clip_id=c, answer=ClipSummaryAnswer(resumo=r) if r else None, origin="cache")
        for c, r in data["summaries"].items()
    }
    return EditorMaterial.from_roteiro(ProjectRoteiro.load(VLOG99 / "roteiro.json")), summaries, data["request"]


def _structure_only(material, summaries, request, tmp_path, ai, num_ctx=8192):
    service = _service(material, tmp_path, ai, num_ctx=num_ctx, context=EditorContext(request=request))
    return service, service._structure(summaries)


def test_real_99_clips_do_not_fit_and_are_asked_in_contiguous_parts(vlog99, tmp_path):
    material, summaries, request = vlog99
    ai = MockEditorAI()
    service, structure = _structure_only(material, summaries, request, tmp_path, ai)
    lines = service.structure_lines(summaries)
    available = Budget(8192, "t").structure_available
    whole = build_structure_prompt(service.context, list(lines.values()), target_sec=900)
    assert len(lines) == 99 and 6000 < tokens(whole) and tokens(whole) > available  # enxuta, mas não cabe
    parts = structure.parts
    assert [(p.clip_ids[0], p.clip_ids[-1], len(p.clip_ids)) for p in parts] == [("P001", "P049", 49), ("P050", "B010", 50)]
    assert [c for p in parts for c in p.clip_ids] == material.chronology()[0]  # contíguas, na ordem de gravação
    assert all(p.estimated_tokens <= available for p in parts)
    assert [r.clip_ids for r in ai.calls["structure"]] == [p.clip_ids for p in parts]
    # O corte cai numa mudança de lugar (exterior → trilha na mata).
    assert (material.clip_place("P049"), material.clip_place("P050")) == ("exterior", "trilha na mata")
    # Meta da parte = alvo (900 s do texto do pedido) × fala da parte / fala total.
    total = sum(material.speech_sec(c) for c in material.clip_ids)
    for part in parts:
        assert part.target_sec == pytest.approx(900 * part.speech_sec / total, abs=0.1)
    first, second = (r.prompt for r in ai.calls["structure"])
    assert first.startswith(f'Pedido do usuário: "{request}"')
    assert "Esta é a parte 1 de 2: clipes da gravação P001 a P049." in first and "já decididos" not in first
    assert "Esta é a parte 2 de 2: clipes da gravação P050 a B010." in second
    assert "Blocos já decididos nas partes anteriores (o vídeo continua a partir deles): " in second
    assert f"Meta desta parte: {clock(parts[1].target_sec)}" in second
    assert "- P049 · " not in second and "- P050 · " in second and "- P050 · " not in first


def test_split_prefers_a_place_change_near_the_ideal_size():
    costs = [10] * 20
    same = ["trilha"] * 20
    assert split_parts(costs, same, 140) == [(0, 10), (10, 20)]  # sem mudança de lugar: o tamanho igual
    places = ["hotel"] * 8 + ["trilha"] * 12
    assert split_parts(costs, places, 140) == [(0, 8), (8, 20)]  # a mudança de lugar dentro da janela
    far = ["hotel"] * 3 + ["trilha"] * 17
    assert split_parts(costs, far, 140) == [(0, 10), (10, 20)]  # longe demais do tamanho ideal: ignorada
    assert split_parts(costs, same, 1000) == [(0, 20)]
    spans = split_parts([10] * 30, ["x"] * 30, 120)  # 300 tokens em partes de até 108
    assert len(spans) == 3 and all(sum([10] * (b - a)) <= 108 for a, b in spans)
    assert all(b - a >= editor_service.STRUCTURE_PART_MIN_CLIPS for a, b in spans)


def test_parts_join_in_order_and_merge_the_same_place_at_the_border(tmp_path):
    material = make_story_material(tmp_path)  # todos os clipes na "rua"
    first = [Block(id="B01", title="Chegada", clip_ids=["P001"], target_sec=8.0),
             Block(id="B02", title="Rua", clip_ids=["P002"], target_sec=8.0)]
    second = [Block(id="B01", title="Rua de novo", clip_ids=["P003"], target_sec=6.0),
              Block(id="B02", title="Fim", clip_ids=["P004", "P005"], target_sec=None)]
    blocks, joins = _join_parts([first, second], material)
    assert [(b.id, b.title, b.clip_ids, b.target_sec) for b in blocks] == [
        ("B01", "Chegada", ["P001"], 8.0), ("B02", "Rua", ["P002", "P003"], 14.0), ("B03", "Fim", ["P004", "P005"], None),
    ]
    assert joins == ['junção: "Rua" (fim da parte 1) e "Rua de novo" (começo da parte 2) são do mesmo lugar (rua): um bloco só']
    assert [b.clip_ids for b in first] == [["P001"], ["P002"]]  # os blocos recebidos não mudam
    other = make_story_material(tmp_path / "b")
    for moment in other.clip_moments["P003"]:
        moment.place = "praia"
    blocks, joins = _join_parts([first, second], other)
    assert len(blocks) == 4 and joins == []


def test_failed_part_uses_the_grouped_fallback_only_there(vlog99, tmp_path):
    material, summaries, request = vlog99
    broken = AIResponseValidationError("resposta quebrada")
    ai = MockEditorAI({"structure": [_first_clip_only, broken, broken]})
    service, structure = _structure_only(material, summaries, request, tmp_path, ai)
    first, second = structure.parts
    assert (first.origin, second.origin) == ("ia", "fallback") and structure.origin == "ia"
    assert "resposta fora do formato" in second.fallback_reason
    fallback_clips = [c for b in structure.blocks for c in b.clip_ids if c in second.clip_ids]
    speech = sum(material.speech_sec(c) for c in fallback_clips)
    assert 0 < speech <= second.target_sec  # clipes inteiros, em ordem, até a meta da parte
    assert fallback_clips == [c for c in second.clip_ids if material.has_speech(c)][:len(fallback_clips)]
    blocks = [b for b in structure.blocks if b.clip_ids[0] in second.clip_ids]
    assert len(blocks) < len(fallback_clips)  # vizinhos do mesmo lugar juntos, não um bloco por clipe
    for block in blocks:
        assert len({place_key(material.clip_place(c)) for c in block.clip_ids}) == 1
    assert any(w.startswith("ESTRUTURA (PARTE 2/2) PELO FALLBACK") for w in structure.warnings)
    run = EditorRun(request=request, provider="mock", model_chain=["mock:mock"], effective_model="mock",
                    budget=Budget(8192, "t").info(), structure=structure)
    report = build_editor_report("t", run, EditorPlan(project_name="t", fps=30, resolution=(1920, 1080), duration_sec=0.0),
                                 material, "editor_ia", None, None)
    assert "Estrutura em 2 partes (a lista inteira não cabe no contexto):" in report
    assert "| 1 | P001–P049 (49) | ~" in report and "- parte 2 pelo fallback: resposta fora do formato" in report


def test_structure_parts_are_cached_one_by_one(vlog99, tmp_path):
    material, summaries, request = vlog99
    broken = AIResponseValidationError("resposta quebrada")
    ai = MockEditorAI({"structure": [_first_clip_only, broken, broken]})
    _structure_only(material, summaries, request, tmp_path, ai)
    # De novo, com o mesmo material: a parte 1 vem do cache; só a parte 2 (que caiu no fallback) é perguntada.
    ai = MockEditorAI({"structure": [_first_clip_only]})
    _, structure = _structure_only(material, summaries, request, tmp_path, ai)
    assert [r.clip_ids[0] for r in ai.calls["structure"]] == ["P050"]
    assert [p.origin for p in structure.parts] == ["cache", "ia"]
    ai = MockEditorAI()
    _, structure = _structure_only(material, summaries, request, tmp_path, ai)
    assert not ai.calls["structure"] and structure.origin == "cache"


def test_free_order_request_keeps_the_ai_order(tmp_path):
    assert asks_free_order("vlog da trilha, uns 8 minutos") is None
    assert asks_free_order("Montagem temática da viagem") == "tematic"
    assert asks_free_order("pode ser fora de ordem") == "fora de ordem" and asks_free_order("ordem livre") == "ordem livre"
    assert asks_free_order("uma edição sistemática") is None  # só no começo de uma palavra
    assert "por tema" in FREE_ORDER_PHRASES and "nao cronologic" in FREE_ORDER_PHRASES
    material = make_story_material(tmp_path)
    answer = {"blocos": [{"titulo": "Praia", "clipes": ["P005", "P001"]}, {"titulo": "Trilha", "clipes": ["P004", "P002"]}]}
    ai = MockEditorAI({"structure": [answer]})
    run = _service(material, tmp_path, ai, context=EditorContext(request="vlog montado por tema")).run()
    assert [b.clip_ids for b in run.structure.blocks] == [["P005", "P001"], ["P004", "P002"]]
    assert run.structure.order == 'a da IA (o pedido diz "por tema")'


# --- coberturas mais rígidas (2c) ---


def _spoken(material, *ids) -> Trecho:
    return Trecho(number=1, clip_id=material.sentences[ids[0]].clip_id, sentence_ids=list(ids), seconds=0.0)


def test_coverage_refused_by_time_gap_and_by_another_place_without_date(tmp_path):
    material = make_evidence_material(tmp_path)  # P001 10:00, D001 10:05, P002 10:10
    spoken = _spoken(material, "S000", "S001")
    drone = material.moments["D001.M01"]
    assert cover_refusal(drone, spoken, ["P001"], material) is None  # 5 min da fala
    material.clips["D001"].recorded_at = _at("11:00")
    assert cover_refusal(drone, spoken, ["P001"], material) == "gravada a 60 min da fala (máximo 15 min)"
    assert cover_refusal(drone, spoken, ["P001", "D001"], material) is None  # do mesmo bloco
    # Sem data: só do mesmo lugar da fala ("terminal de ônibus").
    material.clips["D001"].recorded_at = None
    assert cover_refusal(drone, spoken, ["P001"], material) == "sem data e de outro lugar (estrada × terminal de ônibus)"
    material.moments["D001.M01"].place = "Terminal de  Ônibus"
    assert cover_refusal(drone, spoken, ["P001"], material) is None


def test_coverage_refused_when_the_word_is_already_in_the_speaker_image(tmp_path):
    material = make_evidence_material(tmp_path)
    spoken = _spoken(material, "S000", "S001")
    assert cover_words(spoken, material.moments["D001.M02"], material) == (["cachorros"], ["cachorros"])
    material.moments["P001.M01"].description = "pessoa falando no saguão com cachorros correndo"
    assert cover_words(spoken, material.moments["D001.M02"], material) == (["cachorros"], [])
    ai = MockEditorAI({"structure": [_structure(("P001", "P002"))], "inserts": [{"inserir": []}]})
    _service(material, tmp_path, ai).run()
    offered = {n: [c.moment_id for c in cs] for n, cs in ai.calls["coverages"][0].questions.items()}
    assert offered[1] == ["D001.M01"]  # o cachorro já aparece na imagem de quem fala


def test_coverage_total_is_limited_to_15_percent_of_the_video(material, tmp_path, monkeypatch):
    monkeypatch.setattr(editor_service, "COVER_MAX_RATIO", 0.15)
    run = _service(material, tmp_path, MockEditorAI()).run()
    first = run.details[0]
    assert not first.coverages and not first.coverage.coverages
    [dropped] = first.coverage.discarded
    assert dropped.item == "S001–S001 ← D001.M01" and "passa do limite de cobertura do vídeo (15% de" in dropped.reason
    assert any(w.startswith("B01: cobertura S001–S001 ← D001.M01 (2.5 s) recusada") for w in run.warnings)


def test_real_vlog_coverages_under_the_new_rules(vlog):
    """As 4 coberturas do plano real (fotos a cada 1 s) e as 3 do vídeo de 516 s que o usuário viu erradas."""
    plan = json.loads((VLOG / "editor_plano.json").read_text(encoding="utf-8"))

    def verdict(moment_id, ids, block):
        spoken = _spoken(vlog, *ids)
        spoken = spoken.model_copy(update={"sentence_ids": talking_sentences(spoken, vlog) or list(ids)})
        refusal = cover_refusal(vlog.moments[moment_id], spoken, block, vlog)
        fresh = cover_words(spoken, vlog.moments[moment_id], vlog)[1]
        return refusal or ("ok" if fresh else "palavra só na imagem de quem fala ou nenhuma")

    found = {}
    for block in plan["blocks"]:
        for c in block["coverages"]:
            ids = [s.id for s in vlog.sentence_range(c["from_sentence"], c["to_sentence"])]
            found[(c["moment_ids"][0], c["from_sentence"])] = verdict(c["moment_ids"][0], ids, block["clip_ids"])
    assert found == {
        ("P010.M01", "S102"): "ok",  # mesmo bloco da fala (P019); "árvores" não está na imagem de quem fala
        ("D001.M08", "S112"): "sem data e de outro lugar (praça × floresta)",
        ("D002.M02", "S127"): "sem data e de outro lugar (praia, lago × exterior, montanha)",
        ("D002.M03", "S169"): "sem data e de outro lugar (praia × exterior)",
    }
    trilha = ["P004", "P013", "P019", "P025"]
    assert verdict("D001.M07", ["S114", "S115", "S116"], trilha) == "sem data e de outro lugar (praça × floresta)"
    assert verdict("B004.M01", ["S088"], trilha) == "sem data e de outro lugar (lago × floresta)"
    praia = [f"P{n:03d}" for n in range(21, 29)]
    for moment_id in ("D002.M16", "D002.M06", "D002.M01"):
        for clip_id in ("P022", "P024", "P028"):
            ids = [s.id for s in vlog.clip_sentences[clip_id]]
            assert verdict(moment_id, ids, praia).startswith("sem data e de outro lugar"), (moment_id, clip_id)


def test_coverage_output_is_limited_and_items_for_unasked_trechos_are_ignored_together(material, tmp_path):
    assert coverage_output_limit(2) == 340 and coverage_output_limit(0) == 220

    def loop(request):  # o trecho perguntado e mais 127 que não existem, como o qwen fez no vlog real
        items = [{"trecho": n, "letra": "nenhuma"} for n in range(1, 129)]
        return {"coberturas": items}

    ai = MockEditorAI({"coverages": [loop]})
    run = _service(material, tmp_path, ai).run()
    [request] = ai.calls["coverages"]  # aceita sem nova tentativa: os itens a mais não contam como inválidos
    assert request.max_output_tokens == coverage_output_limit(len(request.questions)) == 220
    sub = run.details[0].coverage
    assert sub.nones == [1] and not sub.discarded and not sub.attempts
    assert [w for w in sub.warnings if "não perguntados" in w] == [
        "127 itens para trechos não perguntados ignorados (trechos perguntados: 1)"
    ]
    # Só trechos não perguntados: recusada (nova tentativa com o erro).
    ai = MockEditorAI({"coverages": [{"coberturas": [{"trecho": 5, "letra": "nenhuma"}]}, {"coberturas": []}]})
    run = _service(material, tmp_path / "b", ai).run()
    assert len(ai.calls["coverages"]) == 2
    assert run.details[0].coverage.attempts[0].startswith("nenhum item para os trechos perguntados (vieram só trechos não perguntados: 5)")


def test_llm_editor_passes_the_coverage_output_limit_to_the_chain(material):
    seen = {}

    from ai_video_editor.editor.provider import CoverageRequest

    ai = LLMEditorAI(ModelChain([ModelSpec(provider="mock", model_id="m")], thinking="off", clients={"mock": MockLLMClient()}))

    def generate(prompt, schema, images=None, max_output_tokens=None):
        seen["max"] = max_output_tokens
        return schema()

    ai._generate = generate
    ai.coverages(CoverageRequest(unit_id="B01", prompt="p", material=material, trechos=[], questions={}, max_output_tokens=340))
    assert seen["max"] == 340


# --- pontas de câmera mexida e janela estável ---


def _with_motion(material: EditorMaterial, clip_id: str, values: list[float]) -> None:
    """Movimento por amostra (2 por segundo, a 1ª em 0,5 s) do clipe."""
    material.clips[clip_id].motion = [(round(0.5 * (k + 1), 3), v) for k, v in enumerate(values)]


def test_end_with_a_motion_peak_is_trimmed_and_the_steady_window_is_chosen(material):
    # D001.M02 (4–8 s): parada (0,005) com um esbarrão em 5–6 s e a câmera sendo pega no fim (7–8 s).
    values = [0.005] * 16
    for t, v in ((5.5, 0.05), (6.0, 0.04), (7.5, 0.06), (8.0, 0.07)):
        values[int(t / 0.5) - 1] = v
    _with_motion(material, "D001", values)
    moment = material.moments["D001.M02"]
    assert steady_range(material, "D001", moment.start_sec, moment.end_sec) == (4.0, 7.0)  # pico do fim fora
    assert moment_span(moment, None, material) == (4.0, 7.0)  # inclusão do momento inteiro: sem a ponta
    # Janela de 1 s com o menor movimento, fora do esbarrão (centrada seria 5,5–6,5).
    assert steady_window(material, "D001", moment.start_sec, moment.end_sec, 1.0) == (4.0, 5.0)
    assert breath_span(moment, 1.0) == (5.5, 6.5) and breath_span(moment, 1.0, material) == (4.0, 5.0)
    # Todo instável (câmera na mão o tempo todo) ou sobra curta demais: nada sai.
    _with_motion(material, "D001", [0.3] * 16)
    assert steady_range(material, "D001", 4.0, 8.0) == (4.0, 8.0)
    # Sem medidas: o de antes (respiro centrado, inclusão a partir do início).
    material.clips["D001"].motion = []
    assert breath_span(moment, 1.0, material) == (5.5, 6.5) and moment_span(moment, 2.0, material) == (4.0, 6.0)


def test_assembly_uses_the_steady_window_for_coverage_and_trims_speaking_clip_edges(material, tmp_path):
    values = [0.005] * 16
    values[7] = values[6] = 0.08  # D001 3,5–4,0 s: a câmera mexe no fim do D001.M01 (0–4 s)
    _with_motion(material, "D001", values)
    detail = DetailRun(
        unit_id="B01", block_id="B01", clip_ids=["P001"], origin="ia",
        keep=[KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S001", sentence_ids=["S000", "S001"])],
        coverages=[CoverageChoice(from_sentence="S001", to_sentence="S001", sentence_ids=["S001"], clip_id="P001",
                                  moment_ids=["D001.M01"], speech_evidence="mar", image_evidence="mar")],
    )
    assembled = assemble_plan(_run_with([detail], [_block("P001")]), material, _sources(material), SilenceCutParams(),
                              tmp_path / "o.mp4", "t")
    [overlay] = assembled.timeline.video_overlay
    assert overlay.source_in == 0.0 and overlay.source_out <= 3.0 + 1e-6  # nunca o fim mexido (3,0–4,0 s)
    # Clipe com fala: a câmera mexe no começo (0–1 s) e no fim; a 1ª frase começa em 0,5 s e a última termina
    # em 11,5 s: as pontas fora das frases saem, nunca um pedaço de frase.
    _with_motion(material, "P001", [0.09, 0.08] + [0.01] * 19 + [0.09, 0.08, 0.09])
    keep = KeepRange(kind="frases", clip_id="P001", from_sentence="S000", to_sentence="S000", sentence_ids=["S000"])
    source = _sources(material)["P001"]
    source.audio = None
    cut = SilenceCutParams(padding_before_sec=0.5, padding_after_sec=0.5)
    [(start, end)] = [(r.start_sec, r.end_sec) for r in faixa_ranges(keep, material, source, cut)]
    assert start == 0.5  # a frase começa em 0,5 s: o corte para nela (sem a folga dentro da câmera mexida)
    keep = KeepRange(kind="frases", clip_id="P001", from_sentence="S003", to_sentence="S003", sentence_ids=["S003"])
    [(start, end)] = [(r.start_sec, r.end_sec) for r in faixa_ranges(keep, material, source, cut)]
    assert end == 11.5  # a frase termina em 11,5 s; a câmera mexe a partir de 11,0 s: o corte fica no fim dela


def test_real_vlog_camera_picked_up_at_the_end_of_p012_is_left_out(vlog):
    """P012: câmera parada atrás do casal (1–2/255) até 33 s; em 33,5–34,8 s o casal volta e pega a câmera."""
    first, second = vlog.moments["P012.M01"], vlog.moments["P012.M02"]
    assert (second.start_sec, second.end_sec) == (16.0, 34.816)
    assert vlog.roteiro.motion_threshold == pytest.approx(0.1215)  # o pico fica abaixo do limite do projeto
    assert max(m for t, m in vlog.clips["P012"].motion if t > 33.0) < vlog.roteiro.motion_threshold
    assert moment_span(second, None, vlog) == (16.0, 33.0)  # inclusão do momento inteiro: sem a câmera sendo pega
    assert breath_span(first, None, vlog) == (9.0, 15.0)  # respiro longe do esbarrão de 2–3,5 s (centrado: 5–11)
    # P010 (câmera andando junto): o fim mais mexido (5,5–6,5 s, acima do limite do projeto) sai.
    assert moment_span(vlog.moments["P010.M01"], None, vlog) == (0.0, 5.0)
