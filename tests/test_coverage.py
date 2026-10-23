"""Testes do Diretor de cobertura e do render com cobertura (`edit <pasta>`, receita `vlog_cobertura`):
janelas obrigatórias, validação, nova tentativa e fallback, IDs → tempos, respiros deslocando legendas e
relatórios, pipeline e render ponta a ponta."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from ai_video_editor.catalog.schema import CatalogClip, ClipMedia, ProjectCatalog, SpeechDetection
from ai_video_editor.cli import build_parser, default_edit_recipe, handle_edit
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.coverage import (
    BreathChoice,
    CoverageChoice,
    CoverageContext,
    CoverageDecision,
    CoverageError,
    CoverageParams,
    CoverageRun,
    assemble_coverage,
    build_coverage_prompt,
    build_visual_material,
    coverage_candidates,
    coverage_problems,
    coverage_roles,
    coverage_sentences,
    find_mandatory_windows,
    run_coverage,
)
from ai_video_editor.coverage.assemble import shot_segment
from ai_video_editor.coverage.providers.mock import MockCoverageDirector
from ai_video_editor.coverage.schema import AcceptedBreath, AcceptedCoverage, CoverageSentence, MandatoryWindow, VisualShot
from ai_video_editor.coverage.service import excluded_shots
from ai_video_editor.director.schema import Sentence
from ai_video_editor.pipeline import EditPipelineParams, SpinePipeline
from ai_video_editor.pipeline.clips import catalog_location, catalog_path
from ai_video_editor.pipeline.spine_schema import SpineDiff, SpineManifest
from ai_video_editor.planning import TimeRange
from ai_video_editor.planning.timeline_plan import ClipCut, build_spine_timeline, clip_placements, is_breath_item
from ai_video_editor.recipes import get_recipe
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.subtitles import ClipSubtitleSource, build_clip_subtitle_blocks
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment, Word
from ai_video_editor.visual.schema import ClipVisual, ProjectVisual, Shot, ShotAnalysis, ShotFrame, ShotMeasures
from tests.conftest import create_tone_silence_video
from tests.test_spine import PATTERNS, FakeTimelineExecutor, FolderTranscriber, NamedSpeechDetector

FPS = 30


# --- material sintético (sem FFmpeg) ---


def _measures(uniform: bool = False) -> ShotMeasures:
    return ShotMeasures(samples=4, blur=4.0, brightness=0.5, motion=0.02, luma_spread=0.05 if uniform else 0.4, uniform=uniform)


def _shot(
    clip_id: str, n: int, start: float, end: float, has_speech: Optional[bool] = False, usable: str = "sim",
    issues: Optional[list[str]] = None, beauty: int = 3, status: str = "analisada", uniform: bool = False,
) -> Shot:
    shot_id = f"{clip_id}.T{n:02d}"
    times = [start + (end - start) * f for f in (0.1, 0.5, 0.9)]
    frames = [ShotFrame(id=f"{shot_id}.F{k}", position=k, time_sec=round(t, 3), file=f"frames/{shot_id}.F{k}.jpg") for k, t in enumerate(times, 1)]
    analysis = None
    if status == "analisada":
        analysis = ShotAnalysis(
            description=f"imagem de {clip_id}", tags=["paisagem"], face_to_camera=clip_id.startswith("P") and bool(has_speech),
            people_count=0, issues=issues or ["nenhuma"], usable=usable,
            best_frames=[frames[1].id, frames[2].id] if usable != "nao" else [], beauty=beauty, summary_for_editor="ok",
        )
    return Shot(
        id=shot_id, index=n, start_sec=start, end_sec=end, boundary="inicio" if n == 1 else "janela", frames=frames,
        measures=_measures(uniform), has_speech=has_speech, status=status, analysis=analysis,
    )


def _clip_entry(clip_id: str, role: str, relative: str, duration: float, has_audio: bool = True) -> CatalogClip:
    return CatalogClip(
        id=clip_id, role=role, folder=role, relative_path=relative, size_bytes=1,
        media=ClipMedia(duration_sec=duration, width=160, height=120, fps=30, codec="h264", has_audio=has_audio),
        speech=SpeechDetection(status="analisada", ratio=0.5 if role == "principal" else 0.0, speech_sec=1.0, detector="t"),
    )


def _synthetic_project(tmp_path: Path) -> tuple[ProjectVisual, ProjectCatalog]:
    """P001 (fala; T02 inutilizável, T03 com borrão), P002 (fala; T01 sem análise mas tela uniforme),
    P003 (principal sem fala), B001 (b-roll), D001 (drone sem áudio, sem análise), B002 (inutilizável)."""
    entries = [
        _clip_entry("P001", "principal", "principal/a.mp4", 10.0),
        _clip_entry("P002", "principal", "principal/b.mp4", 6.0),
        _clip_entry("P003", "principal", "principal/c.mp4", 5.0),
        _clip_entry("B001", "broll", "broll/praia.mp4", 9.0),
        _clip_entry("B002", "broll", "broll/feio.mp4", 3.0),
        _clip_entry("D001", "drone", "drone/d.mp4", 8.0, has_audio=False),
    ]
    catalog = ProjectCatalog(source_dir=tmp_path / "viagem", project_dir=tmp_path / "p", listing_hash="h", clips=entries)
    shots = {
        "P001": [
            _shot("P001", 1, 0.0, 3.0, has_speech=True),
            _shot("P001", 2, 3.0, 6.0, has_speech=True, usable="nao", issues=["obstruida"]),
            _shot("P001", 3, 6.0, 10.0, has_speech=True, usable="parcial", issues=["borrao_movimento"]),
        ],
        "P002": [_shot("P002", 1, 0.0, 3.0, has_speech=None, status="sem_analise", uniform=True), _shot("P002", 2, 3.0, 6.0, has_speech=True)],
        "P003": [_shot("P003", 1, 0.0, 5.0, has_speech=False, beauty=5)],
        "B001": [_shot("B001", 1, 0.0, 6.0, beauty=4), _shot("B001", 2, 6.0, 9.0, beauty=2)],
        "B002": [_shot("B002", 1, 0.0, 3.0, usable="nao", issues=["desfocada"])],
        "D001": [_shot("D001", 1, 0.0, 8.0, status="sem_analise")],
    }
    clips = [
        ClipVisual(clip_id=e.id, role=e.role, relative_path=e.relative_path, duration_sec=e.media.duration_sec, shots=shots[e.id])
        for e in entries
    ]
    visual = ProjectVisual(project_dir=tmp_path / "p", source_dir=tmp_path / "viagem", roles=["principal", "broll", "drone"], clips=clips)
    return visual, catalog


def _spine(tmp_path: Path) -> Timeline:
    """P001: [0, 4) e [5, 10) (pausa 4–5 cortada); P002: [0, 6). Total 15 s = 450 frames."""
    return build_spine_timeline(
        [
            ClipCut(clip_id="P001", source_path=tmp_path / "a.mp4", keep=[TimeRange(start_sec=0.0, end_sec=4.0), TimeRange(start_sec=5.0, end_sec=10.0)]),
            ClipCut(clip_id="P002", source_path=tmp_path / "b.mp4", keep=[TimeRange(start_sec=0.0, end_sec=6.0)]),
        ],
        tmp_path / "out.mp4", fps=FPS,
    )


SENTENCES = [
    Sentence(id=0, start_sec=0.2, end_sec=1.8, text="Bom dia, pessoal.", segment_ids=[0], clip_id="P001"),
    Sentence(id=1, start_sec=2.0, end_sec=3.8, text="Estamos na van.", segment_ids=[1], clip_id="P001"),
    Sentence(id=2, start_sec=4.2, end_sec=6.0, text="A estrada é longa.", segment_ids=[2], clip_id="P001"),  # 4,2–5,0 cortado
    Sentence(id=3, start_sec=6.5, end_sec=9.5, text="Frase removida pelo Diretor.", segment_ids=[3], clip_id="P001"),
    Sentence(id=4, start_sec=0.5, end_sec=2.5, text="Chegamos no lago.", segment_ids=[4], clip_id="P002"),
    Sentence(id=5, start_sec=3.0, end_sec=5.5, text="Que lugar lindo.", segment_ids=[5], clip_id="P002"),
]


def _context(tmp_path: Path, roles: Optional[list[str]] = None) -> tuple[CoverageContext, Timeline, list[VisualShot]]:
    visual, catalog = _synthetic_project(tmp_path)
    material = build_visual_material(visual, catalog, tmp_path / "visual.json", ["P001", "P002"])
    spine = _spine(tmp_path)
    kept = coverage_sentences(SENTENCES, {3}, spine)
    allowed = coverage_roles(roles or [])
    context = CoverageContext(
        project_name="viagem", fps=FPS, spine_duration_sec=spine.duration_sec, sentences=kept,
        windows=find_mandatory_windows(kept, material, ["P001", "P002"]),
        candidates=coverage_candidates(material, allowed), roles=allowed,
    )
    return context, spine, material.shots


# --- janelas obrigatórias e frases → frames ---


def test_mandatory_windows_only_where_kept_speech_meets_bad_image(tmp_path):
    context, _, _ = _context(tmp_path)
    assert [s.id for s in context.sentences] == ["S000", "S001", "S002", "S004", "S005"]  # S003 removida
    windows = {w.id: w for w in context.windows}
    # P001.T02 inutilizável e obstruída: S001 (2,0–3,8) e S002 (parte mantida 5,0–6,0).
    assert windows["O1"].clip_id == "P001" and windows["O1"].sentence_ids == ["S001", "S002"]
    assert windows["O1"].shot_ids == ["P001.T02"] and windows["O1"].reasons == ["imagem inutilizável", "obstruída/parede"]
    # P002.T01 sem análise da IA, mas a medida diz tela uniforme.
    assert windows["O2"].sentence_ids == ["S004"] and windows["O2"].reasons == ["tela uniforme"]
    # P001.T03 tem borrão, mas a única fala nela (S003) foi removida: nenhuma janela.
    assert len(context.windows) == 2


def test_sentence_trimmed_by_silence_starts_at_first_kept_frame(tmp_path):
    context, spine, _ = _context(tmp_path)
    by_id = {s.id: s for s in context.sentences}
    # S002 começa em 4,2 s no clipe, mas 4–5 s foi cortado: começa onde o trecho [5, 10) começa (frame 120).
    assert (by_id["S002"].start_frame, by_id["S002"].end_frame) == (120, 150)
    assert by_id["S002"].kept == [(5.0, 6.0)] and by_id["S002"].source_start_sec == 5.0
    assert (by_id["S000"].start_frame, by_id["S000"].end_frame) == (6, 54)
    assert (by_id["S004"].start_frame, by_id["S004"].end_frame) == (285, 345)  # P002 começa no frame 270


def test_candidates_exclude_unanalyzed_unusable_and_principal_with_speech(tmp_path):
    context, _, shots = _context(tmp_path)
    assert [c.id for c in context.candidates] == ["P003.T01", "B001.T01", "B001.T02"]
    excluded = excluded_shots(context, shots)
    assert excluded["D001.T01"] == "sem análise da IA" and "inutilizável" in excluded["B002.T01"]
    assert excluded["P001.T01"] == "principal com fala"
    only_broll, _, _ = _context(tmp_path, roles=["broll"])  # roles_to_use do briefing
    assert [c.id for c in only_broll.candidates] == ["B001.T01", "B001.T02"]


def test_visual_from_another_catalog_is_refused(tmp_path):
    visual, catalog = _synthetic_project(tmp_path)
    catalog.clips[0] = catalog.clips[0].model_copy(update={"relative_path": "principal/outro.mp4"})
    with pytest.raises(CoverageError, match="outro catálogo.*P001"):
        build_visual_material(visual, catalog, tmp_path / "visual.json", ["P001"])


def test_prompt_lists_ids_windows_and_candidates_without_times(tmp_path):
    context, _, _ = _context(tmp_path)
    context = context.model_copy(update={"instructions": ["Use bastante drone"]})
    prompt = build_coverage_prompt(context)
    assert "O1: S001–S002 (clipe P001" in prompt and "tela uniforme" in prompt
    assert "[S004] P002 2.0s Chegamos no lago." in prompt and "S003" not in prompt
    assert "B001.T01 b-roll, 6.0s, beleza 4, boa" in prompt and "D001.T01" not in prompt
    assert "Use bastante drone" in prompt and "timestamps" in prompt


# --- validação ---


def _decision(coverages=(), breaths=()) -> CoverageDecision:
    return CoverageDecision(coverages=list(coverages), breaths=list(breaths), summary="ok")


def _cov(a: str, b: str, shot: str, windows=()) -> CoverageChoice:
    return CoverageChoice(from_sentence=a, to_sentence=b, shot_id=shot, reason="combina", windows=list(windows))


GOOD = [_cov("S001", "S002", "B001.T01", ["O1"]), _cov("S004", "S005", "P003.T01", ["O2"])]


def _problems(context, shots, decision) -> list[str]:
    return [p.message for p in coverage_problems(decision, context, excluded_shots(context, shots))]


def test_valid_decision_has_no_problems(tmp_path):
    context, _, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S002", shot_id="B001.T02", size="short", reason="troca de lugar")
    assert _problems(context, shots, _decision(GOOD, [breath])) == []


@pytest.mark.parametrize(
    "coverages, expected",
    [
        ([_cov("S001", "S999", "B001.T01")], "a frase S999 não existe"),
        ([_cov("S001", "S002", "D001.T01")], "D001.T01 não pode ser cobertura (sem análise da IA)"),
        ([_cov("S001", "S002", "B002.T01")], "B002.T01 não pode ser cobertura (imagem inutilizável"),
        ([_cov("S001", "S002", "X001.T01")], "a tomada X001.T01 não existe"),
        ([_cov("S000", "S001", "B001.T01"), _cov("S001", "S002", "B001.T02")], "se sobrepõe"),
        ([_cov("S004", "S004", "B001.T01"), _cov("S005", "S005", "B001.T01")], "já foi usada"),
        ([_cov("S002", "S002", "B001.T01")], "menos que o mínimo de 1,5s"),
        ([_cov("S000", "S002", "B001.T01", ["O9"])], "a janela obrigatória O9 não existe"),
        ([_cov("S004", "S005", "P003.T01", ["O1"])], "diz atender O1"),
    ],
)
def test_validation_refuses(tmp_path, coverages, expected):
    context, _, shots = _context(tmp_path)
    problems = _problems(context, shots, _decision(coverages))
    assert any(expected in p for p in problems), problems


def test_validation_refuses_shot_of_the_clip_that_is_speaking(tmp_path):
    context, _, shots = _context(tmp_path)
    p001_silent = VisualShot(
        id="P001.T09", clip_id="P001", role="principal", source_path=tmp_path / "a.mp4", start_sec=8.0, end_sec=10.0,
        has_speech=False, status="analisada", usable="sim", beauty=3,
    )
    context = context.model_copy(update={"candidates": [*context.candidates, p001_silent]})
    problems = _problems(context, shots, _decision([_cov("S001", "S002", "P001.T09")]))
    assert any("é do clipe P001, que está falando" in p for p in problems)


def test_validation_refuses_uncovered_window_and_breath_inside_coverage(tmp_path):
    context, _, shots = _context(tmp_path)
    decision = _decision(
        [_cov("S001", "S002", "B001.T01", ["O1"])],
        [BreathChoice(after_sentence="S001", shot_id="B001.T02", size="short", reason="x")],
    )
    problems = coverage_problems(decision, context, excluded_shots(context, shots))
    assert [p.kind for p in problems] == ["invalid", "missing_window"]
    assert "cai no meio de cobertura 1" in problems[0].message
    assert "janela obrigatória O2 (S004, tela uniforme) sem cobertura: faltam S004" in problems[1].message


# --- nova tentativa e fallback ---


def test_invalid_answer_is_retried_with_the_error(tmp_path):
    context, _, shots = _context(tmp_path)
    director = MockCoverageDirector(responses=[_decision([_cov("S001", "S999", "B001.T01")]), _decision(GOOD)])
    run = run_coverage(director, context, shots)
    assert len(director.calls) == 2 and "S999" in director.calls[1].previous_error
    assert "S999" in build_coverage_prompt(director.calls[1]) and "rejeitada" in build_coverage_prompt(director.calls[1])
    assert run.origin == "ia" and [a.success for a in run.attempts] == [False, True]
    assert run.window_cover == {"O1": ["C1"], "O2": ["C2"]} and not run.warnings


def test_invalid_twice_falls_back_to_mandatory_windows(tmp_path):
    context, _, shots = _context(tmp_path)
    bad = _decision([_cov("S001", "S002", "D001.T01")])
    run = run_coverage(MockCoverageDirector(responses=[bad, bad]), context, shots)
    assert run.origin == "fallback" and "inválida em 2 tentativas" in run.origin_reason
    assert any("Cobertura da IA ignorada" in w for w in run.warnings)
    assert not run.uncovered_windows and all(c.origin == "fallback" for c in run.coverages)
    # Tomada livre de maior beleza que caiba: P003.T01 (beleza 5, 5 s) para O1 (S001–S002, 3 s).
    assert [(c.from_sentence, c.to_sentence, c.shot_id) for c in run.coverages][0] == ("S001", "S002", "P003.T01")


def test_missing_window_only_keeps_ai_choices_and_completes(tmp_path):
    context, _, shots = _context(tmp_path)
    partial = _decision([_cov("S004", "S005", "P003.T01", ["O2"])])
    run = run_coverage(MockCoverageDirector(responses=[partial, partial]), context, shots)
    assert run.origin == "ia+fallback" and not run.uncovered_windows
    origins = {(c.from_sentence, c.origin) for c in run.coverages}
    assert origins == {("S004", "ia"), ("S001", "fallback")}
    assert run.coverages[0].shot_id == "B001.T01"  # P003.T01 já usada pela IA


def test_no_coverage_ai_covers_only_mandatory_windows(tmp_path):
    context, _, shots = _context(tmp_path)
    run = run_coverage(None, context, shots)
    assert run.origin == "fallback" and "desligada" in run.origin_reason and run.provider == "none"
    assert run.window_cover == {"O1": ["C1"], "O2": ["C2"]} and not run.breaths


def test_default_mock_covers_windows_and_adds_a_breath_at_clip_change(tmp_path):
    context, _, shots = _context(tmp_path)
    director = MockCoverageDirector()
    run = run_coverage(director, context, shots)
    assert run.origin == "ia" and not run.uncovered_windows
    assert [(b.after_sentence, b.size) for b in run.breaths] == [("S002", "medium")]


def test_coverage_params_inherit_the_director():
    same = CoverageParams.inherit("gemini", "gemini-x", "gemini:gemini-y", "high", 90.0, 2)
    assert (same.provider, same.model, same.fallback, same.thinking, same.timeout_sec) == ("gemini", "gemini-x", "gemini:gemini-y", "high", 90.0)
    other = CoverageParams.inherit("gemini", "gemini-x", "gemini:gemini-y", provider="openai", thinking="off")
    assert (other.provider, other.model, other.fallback, other.thinking) == ("openai", None, None, "off")
    assert CoverageParams.inherit("none").provider == "none" and CoverageParams.inherit("mock").provider == "mock"


# --- IDs → tempos ---


def _run(context, coverages=(), breaths=()) -> CoverageRun:
    from ai_video_editor.coverage.service import window_cover

    run = CoverageRun(provider="mock", model_id="mock", windows=context.windows)
    run.coverages = [AcceptedCoverage(**c.model_dump()) for c in coverages]
    run.breaths = [AcceptedBreath(**b.model_dump()) for b in breaths]
    run.window_cover = window_cover(context, run.coverages)
    return run


def test_breath_at_clip_end_shifts_everything_after(tmp_path):
    context, spine, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S002", shot_id="B001.T02", size="short", reason="troca de lugar")
    run = _run(context, [_cov("S004", "S005", "B001.T01", ["O2"])], [breath])
    result = assemble_coverage(spine, context, run, {s.id: s for s in shots})
    final = result.timeline
    assert final.total_frames == spine.total_frames + 60  # respiro curto = 2 s
    labels = [i.label for i in final.video_main]
    assert labels == ["P001 #1", "P001 #2", "respiro R1 (B001.T02)", "P002 #1"]
    breath_item = final.video_main[2]
    assert is_breath_item(breath_item) and breath_item.use_audio and breath_item.clip_id == "B001"
    assert result.breaths[0].timeline_start == pytest.approx(9.0)  # fim do P001 na espinha
    # S004 começava no frame 285 da espinha; depois do respiro, 345 (11,5 s).
    overlay = final.video_overlay[0]
    assert overlay.timeline_start == pytest.approx(11.5) and final.item_frames(overlay) == 150  # S004–S005: 5 s
    assert result.coverages[0].timeline_start == pytest.approx(11.5)
    assert result.coverages[0].timeline_end == pytest.approx((450 + 60 - 15) / FPS)  # S005 termina em 5,5 s do P002
    # Trecho da tomada do tamanho da janela, centrado no melhor trecho (F2–F3 = 3,0–5,4 s → centro 4,2 s)
    # e limitado à tomada (0–6 s): 5 s → 1–6 s. Uma janela de 2 s fica centrada: 3,2–5,2 s.
    piece = result.coverages[0].pieces[0]
    assert (piece.source_in, piece.source_out) == (pytest.approx(1.0), pytest.approx(6.0))
    assert shot_segment({s.id: s for s in shots}["B001.T01"], 60, FPS) == (pytest.approx(3.2), 60)
    # Som ambiente baixinho da cobertura (b-roll com áudio e sem fala).
    assert final.audio_ambient and final.audio_ambient[0].volume < 0.2


def test_breath_between_sentences_without_a_cut_splits_the_item(tmp_path):
    context, spine, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S000", shot_id="B001.T02", size="short", reason="pausa")
    result = assemble_coverage(spine, context, _run(context, [], [breath]), {s.id: s for s in shots})
    final = result.timeline
    # S000 termina no frame 54 e S001 começa no 60, sem corte entre elas: divide em (54 + 60) // 2 = 57.
    first, breath_item, second = final.video_main[:3]
    assert final.item_frames(first) == 57 and is_breath_item(breath_item)
    assert second.source_in == pytest.approx(57 / FPS) and second.source_out == pytest.approx(4.0)
    assert final.total_frames == spine.total_frames + 60
    placements = clip_placements(final)["P001"]
    assert [p.concat_frame for p in placements] == [0, 57, 120] and [p.start_frame for p in placements] == [0, 117, 180]


def test_shot_shorter_than_window_is_chained_or_shortened(tmp_path):
    context, spine, shots = _context(tmp_path)
    by_id = {s.id: s for s in shots}
    short = by_id["B001.T02"].model_copy(update={"end_sec": 7.0})  # 1 s de tomada
    by_id["B001.T02"] = short
    context = context.model_copy(update={"candidates": [short if c.id == short.id else c for c in context.candidates]})
    # O1 (S001–S002, 3 s) com a tomada de 1 s: completada com a tomada livre de maior beleza.
    run = _run(context, [_cov("S001", "S002", "B001.T02", ["O1"]), _cov("S004", "S005", "B001.T01", ["O2"])])
    result = assemble_coverage(spine, context, run, by_id)
    pieces = result.coverages[0].pieces
    assert [(p.shot_id, p.chained) for p in pieces] == [("B001.T02", False), ("P003.T01", True)]
    assert pieces[0].timeline_end == pieces[1].timeline_start and pieces[1].timeline_end == pytest.approx(5.0)
    assert any("completada com P003.T01" in w for w in result.warnings)
    # Cobertura fora de janela obrigatória com tomada curta: encurtada, com aviso.
    context2 = context.model_copy(update={"windows": []})
    result2 = assemble_coverage(spine, context2, _run(context2, [_cov("S004", "S005", "B001.T02")]), by_id)
    assert len(result2.coverages[0].pieces) == 1 and any("C1 encurtada" in w for w in result2.warnings)


def test_subtitles_follow_each_range_when_a_breath_separates_them():
    words = [Word(text="Primeira.", start_sec=0.2, end_sec=0.8), Word(text="Segunda.", start_sec=2.2, end_sec=2.8)]
    transcript = Transcript(
        source_path=Path("a.mp4"), language="pt", duration_sec=3.0, model_used="mock",
        segments=[TranscriptSegment(id=0, start_sec=0.2, end_sec=2.8, text="Primeira. Segunda.", words=words)],
    )
    ranges = [TimeRange(start_sec=0.0, end_sec=1.0), TimeRange(start_sec=2.0, end_sec=3.0)]
    plain = build_clip_subtitle_blocks([ClipSubtitleSource(transcript=transcript, kept_ranges=ranges, offset_sec=0.0)], FPS)
    # Um respiro de 2 s entre os trechos: o segundo trecho começa em 3 s no vídeo final.
    shifted = build_clip_subtitle_blocks(
        [ClipSubtitleSource(transcript=transcript, kept_ranges=ranges, offset_sec=0.0, range_starts_sec=[0.0, 3.0])], FPS
    )
    assert [b.text for b in shifted] == ["Primeira.", "Segunda."]
    assert shifted[0].start_sec == pytest.approx(plain[0].start_sec)
    assert shifted[1].start_sec == pytest.approx(plain[1].start_sec + 2.0)


def test_vlog_cobertura_recipe_and_vlog_unchanged():
    vlog = get_recipe("vlog")
    assert vlog.steps == ["catalog", "silences", "transcription", "director", "timeline", "render", "subtitles", "report"]
    cobertura = get_recipe("vlog_cobertura")
    assert cobertura.multi_clip and cobertura.steps == [
        "catalog", "silences", "transcription", "director", "timeline", "visual", "coverage", "render", "subtitles", "report",
    ]
    assert cobertura.dependencies("coverage") == ["visual", "timeline", "transcription", "director"]
    assert cobertura.dependencies("render") == ["coverage"]


# --- pipeline (catálogo e silêncios com FFmpeg; render falso) ---


def _solid_video(path: Path, color: str, seconds: float, fps: int = 24) -> Path:
    """Vídeo de cor sólida com áudio mudo (b-roll fácil de reconhecer no quadro final)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(get_ffmpeg_path()), "-y", "-f", "lavfi", "-i", f"color=c={color}:size=160x120:rate={fps}:duration={seconds}",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={seconds}", "-shortest",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path.resolve()),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return path


@pytest.fixture(scope="session")
def coverage_folder(tmp_path_factory) -> Path:
    """2 principais falados (os da espinha) + 1 b-roll vermelho de 6 s (somente leitura)."""
    root = tmp_path_factory.mktemp("cobertura") / "viagem"
    for n, relative in enumerate(["principal/clip_a.mp4", "principal/clip_b.mp4"]):
        pattern, fps = PATTERNS[relative]
        path = create_tone_silence_video(root / relative, pattern, width=160, height=120, fps=fps)
        os.utime(path, (1_700_000_000 + n * 60,) * 2)
    broll = _solid_video(root / "broll" / "praia.mp4", "red", 6.0)
    os.utime(broll, (1_700_000_200,) * 2)
    return root


PARAMS = EditPipelineParams(transcribe=True, director="mock", loudness="off")


def _write_visual(project_dir: Path) -> None:
    """visual.json sintético do catálogo real do projeto: a 2ª tomada do clip_a (3–6 s) está obstruída."""
    catalog = ProjectCatalog.load(catalog_path(project_dir))
    by_id = {c.id: c for c in catalog.clips}
    assert set(by_id) == {"P001", "P002", "B001"}
    shots = {
        "P001": [_shot("P001", 1, 0.0, 3.0, has_speech=True), _shot("P001", 2, 3.0, 6.0, has_speech=True, usable="parcial", issues=["obstruida"]),
                 _shot("P001", 3, 6.0, 8.0, has_speech=True)],
        "P002": [_shot("P002", 1, 0.0, 5.0, has_speech=True)],
        "B001": [_shot("B001", 1, 0.0, 3.0, beauty=4), _shot("B001", 2, 3.0, 6.0, beauty=3)],
    }
    visual = ProjectVisual(
        project_dir=project_dir, source_dir=catalog.source_dir, roles=["principal", "broll"],
        clips=[
            ClipVisual(clip_id=i, role=by_id[i].role, relative_path=by_id[i].relative_path, duration_sec=by_id[i].media.duration_sec, shots=s)
            for i, s in shots.items()
        ],
    )
    visual.save(project_dir / "visual" / "visual.json")


def _pipeline(tmp_path: Path, executor=None, coverage_director=None) -> SpinePipeline:
    return SpinePipeline(
        projects_root=tmp_path / "projects", transcription_provider=FolderTranscriber(),
        render_executor=executor or FakeTimelineExecutor(), speech_detector=NamedSpeechDetector(),
        recipe="vlog_cobertura", coverage_director=coverage_director,
    )


# A espinha do mock (retake da 1ª fala removido): S001, S002 (P001), S003, S004 (P002).
E2E_DECISION = CoverageDecision(
    coverages=[CoverageChoice(from_sentence="S001", to_sentence="S001", shot_id="B001.T01", reason="van", windows=["O1"])],
    breaths=[BreathChoice(after_sentence="S002", shot_id="B001.T02", size="short", reason="troca de lugar")],
    summary="Cobre a janela e respira na troca de clipe.",
)


def _srt_starts(text: str) -> dict[str, float]:
    starts = {}
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        h, m, rest = lines[1].split(" --> ")[0].split(":")
        starts[" ".join(lines[2:])] = int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))
    return starts


@pytest.mark.slow
def test_edit_folder_with_coverage_shifts_subtitles_reports_and_versions(coverage_folder, tmp_path):
    # Sem visual.json: erro claro, dizendo para rodar `visual` (ou usar --receita vlog).
    with pytest.raises(CoverageError, match=r"rode `visual .*--receita vlog"):
        _pipeline(tmp_path).run(coverage_folder, PARAMS)
    project_dir, _ = catalog_location(coverage_folder.resolve(), tmp_path / "projects")
    _write_visual(project_dir)

    executor = FakeTimelineExecutor()
    director = MockCoverageDirector(responses=[E2E_DECISION])
    result = _pipeline(tmp_path, executor, director).run(coverage_folder, PARAMS)
    assert set(result.outcomes) == set(get_recipe("vlog_cobertura").steps)
    assert result.timeline_path.name == "coverage_timeline.json"

    spine = Timeline.model_validate_json((project_dir / "timeline.json").read_text(encoding="utf-8"))
    final = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    assert executor.timelines == [final]  # o render recebe a Timeline final
    run = result.coverage
    assert run.origin == "ia" and run.window_cover == {"O1": ["C1"]}
    breath = run.placed_breaths[0]
    assert final.total_frames == spine.total_frames + round(breath.duration_sec * final.fps) == spine.total_frames + 60
    assert result.final_duration_sec == pytest.approx(final.duration_sec, abs=0.001)
    assert len(final.video_overlay) == 1 and final.video_overlay[0].clip_id == "B001"
    # Som ambiente da cobertura: o b-roll tem trilha de áudio (muda) e nenhuma fala.
    assert [(a.clip_id, a.volume) for a in final.audio_ambient] == [("B001", 0.12)]

    # Legendas: a fala do P002 vem depois do respiro e fica deslocada pelos 2 s dele.
    spine_p002 = clip_placements(spine)["P002"][0].start_frame / spine.fps
    final_p002 = clip_placements(final)["P002"][0].start_frame / final.fps
    assert final_p002 == pytest.approx(spine_p002 + 2.0) and breath.timeline_start == pytest.approx(spine_p002)
    starts = _srt_starts(result.subtitles_path.read_text(encoding="utf-8"))
    assert final_p002 <= starts["Agora vamos almoçar."] < final_p002 + 0.3
    assert starts["O mar está lindo hoje."] < breath.timeline_start  # antes do respiro, não muda

    # Diff e transcrição comparativa no tempo final.
    diff = SpineDiff.model_validate_json((project_dir / "edit_diff.json").read_text(encoding="utf-8"))
    entry = diff.merged().entry_for(3)
    assert final_p002 <= entry.edited_start_sec < final_p002 + 0.3
    assert diff.final_duration_sec == pytest.approx(final.duration_sec, abs=0.001)

    # Relatório, linha do tempo e decisão.
    report = (project_dir / "edit_report.md").read_text(encoding="utf-8")
    assert "| Receita | vlog_cobertura" in report and "## Cobertura" in report
    assert "| O1 | P001 | S001 | " in report and "| C1 |" in report and "| R1 | S002 | B001.T02 | curto (2.0s)" in report
    svg = (project_dir / "edit_timeline.svg").read_text(encoding="utf-8")
    assert 'class="overlay" data-clip-id="B001"' in svg and 'class="item breath"' in svg and "respiro (sem fala)" in svg
    saved = CoverageRun.model_validate_json((project_dir / "coverage_decision.json").read_text(encoding="utf-8"))
    assert saved.placed_coverages[0].windows == ["O1"] and saved.provider == "mock" and saved.attempts[0].success

    # Versão com a Timeline renderizada e a decisão; segunda execução toda do cache.
    v1 = project_dir / "versions" / "v1"
    assert (v1 / "coverage_timeline.json").is_file() and (v1 / "coverage_decision.json").is_file() and (v1 / "timeline.json").is_file()
    assert Timeline.model_validate_json((v1 / "coverage_timeline.json").read_text(encoding="utf-8")).output_path.parent == v1
    again = _pipeline(tmp_path, executor, director).run(coverage_folder, PARAMS)
    assert set(again.outcomes.values()) == {"cached"} and len(director.calls) == 1 and again.version == 1

    # A etapa refeita com as mesmas entradas (artefato apagado) e outra decisão: o que vem depois acompanha
    # o resultado, não só as entradas da cobertura.
    (project_dir / "coverage_timeline.json").unlink()
    no_breath = E2E_DECISION.model_copy(update={"breaths": []})
    redo = _pipeline(tmp_path, executor, MockCoverageDirector(responses=[no_breath])).run(coverage_folder, PARAMS)
    assert redo.outcomes["coverage"] == "ran" and redo.outcomes["render"] == "ran" and redo.outcomes["subtitles"] == "ran"
    assert redo.final_duration_sec == pytest.approx(spine.duration_sec, abs=0.001)

    # Outra IA de cobertura muda só a etapa de cobertura em diante.
    other = _pipeline(tmp_path, executor, None).run(coverage_folder, PARAMS, coverage=CoverageParams(provider="none"))
    assert other.outcomes["timeline"] == "cached" and other.outcomes["coverage"] == "ran"
    assert other.coverage.origin == "fallback" and not other.coverage.placed_breaths
    manifest = SpineManifest.load(project_dir / "project.json")
    assert manifest.stages["coverage"].metrics["origin"] == "fallback"



@pytest.mark.slow
def test_cli_picks_vlog_cobertura_when_the_folder_has_visual(coverage_folder, tmp_path, capsys):
    projects = tmp_path / "projects"
    assert default_edit_recipe(coverage_folder, True, True, projects) == "vlog"
    project_dir, _ = catalog_location(coverage_folder.resolve(), projects)
    from ai_video_editor.pipeline.clips import load_or_build_catalog

    load_or_build_catalog(coverage_folder.resolve(), projects, lambda: _pipeline(tmp_path)._catalog_builder())
    _write_visual(project_dir)
    assert default_edit_recipe(coverage_folder, True, True, projects) == "vlog_cobertura"
    assert default_edit_recipe(coverage_folder, True, False, projects) == "vlog"  # sem transcrição não há frases
    assert default_edit_recipe(coverage_folder / "principal" / "clip_a.mp4", False, True, projects) == "fala"
    args = build_parser().parse_args(["edit", str(coverage_folder), "--coverage-provider", "mock", "--coverage-thinking", "off"])
    assert (args.coverage, args.coverage_thinking, args.coverage_model) == ("mock", "off", None)
    bad = ["edit", str(coverage_folder), "--coverage-provider", "gemini", "--coverage-fallback", "xyz:abc", "--projects-dir", str(projects)]
    assert handle_edit(build_parser().parse_args(bad)) == 1
    assert "provedor de IA desconhecido: 'xyz'" in capsys.readouterr().err


# --- render de verdade ---


def _frame_rgb(video: Path, at: float) -> tuple[int, int, int]:
    """Cor média de um quadro (1x1 px)."""
    args = [
        str(get_ffmpeg_path()), "-v", "error", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
        "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    data = subprocess.run(args, capture_output=True, check=True).stdout
    return data[0], data[1], data[2]


def _rms(video: Path, start: float, seconds: float) -> float:
    """RMS (0–1) do áudio num trecho."""
    import array
    import math

    args = [
        str(get_ffmpeg_path()), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
    ]
    samples = array.array("h", subprocess.run(args, capture_output=True, check=True).stdout)
    return math.sqrt(sum(x * x for x in samples) / max(1, len(samples))) / 32768


@pytest.mark.slow
def test_render_with_overlay_and_breath(coverage_folder, tmp_path):
    projects = tmp_path / "projects"
    project_dir, _ = catalog_location(coverage_folder.resolve(), projects)
    from ai_video_editor.pipeline.clips import load_or_build_catalog

    load_or_build_catalog(coverage_folder.resolve(), projects, lambda: _pipeline(tmp_path)._catalog_builder())
    _write_visual(project_dir)
    result = SpinePipeline(
        projects_root=projects, transcription_provider=FolderTranscriber(), speech_detector=NamedSpeechDetector(),
        recipe="vlog_cobertura", coverage_director=MockCoverageDirector(responses=[E2E_DECISION]),
    ).run(coverage_folder, PARAMS)
    spine = Timeline.model_validate_json((project_dir / "timeline.json").read_text(encoding="utf-8"))
    final = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    run = result.coverage
    breath, cover = run.placed_breaths[0], run.placed_coverages[0]

    # Duração = espinha + respiros (o render confere ±1 frame).
    assert result.final_duration_sec == pytest.approx(spine.duration_sec + breath.duration_sec, abs=1 / final.fps)

    # Quadros: b-roll vermelho na cobertura e no respiro; fala (testsrc) fora deles.
    video = result.output_path
    r, g, b = _frame_rgb(video, (cover.timeline_start + cover.timeline_end) / 2)
    assert r > 180 and g < 70 and b < 70
    r, g, b = _frame_rgb(video, breath.timeline_start + breath.duration_sec / 2)
    assert r > 180 and g < 70 and b < 70
    r, g, b = _frame_rgb(video, cover.timeline_end + 0.3)
    assert not (r > 180 and g < 70 and b < 70)

    # Áudio: a fala (tom) continua sob a cobertura; o respiro tem o som da tomada (mudo).
    assert _rms(video, cover.timeline_start + 0.2, 0.8) > 0.05
    assert _rms(video, breath.timeline_start + 0.3, breath.duration_sec - 0.6) < 0.01
