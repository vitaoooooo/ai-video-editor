"""Testes do Diretor de cobertura e do render com cobertura (`edit <pasta>`, receita `vlog_cobertura`):
janelas com imagem ruim, validação item a item, duração decidida pelo código, nova tentativa e fallback
com tomadas relacionadas, IDs → tempos, respiros deslocando legendas e relatórios, pipeline e render
ponta a ponta, e a resposta real do Gemini no vlog "Trilha La Islita" (`tests/data/coverage_real/`)."""

import json
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
    VisualMaterial,
    WindowDecision,
    annotate_screen,
    assemble_coverage,
    build_coverage_prompt,
    build_visual_material,
    coverage_candidates,
    coverage_roles,
    coverage_sentences,
    find_mandatory_windows,
    run_coverage,
)
from ai_video_editor.coverage.assemble import distribute, shot_segment
from ai_video_editor.coverage.providers.mock import MockCoverageDirector
from ai_video_editor.coverage.schema import AcceptedBreath, AcceptedCoverage, CoverageSentence, VisualShot
from ai_video_editor.coverage.service import MAX_COVERAGE_SEC, MIN_COVERAGE_SEC, excluded_shots, shot_frames
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
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


def _measures(uniform: bool = False, blurry: bool = False) -> ShotMeasures:
    return ShotMeasures(
        samples=4, blur=9.0 if blurry else 4.0, brightness=0.5, motion=0.02, luma_spread=0.05 if uniform else 0.4,
        uniform=uniform, blurry=blurry,
    )


def _shot(
    clip_id: str, n: int, start: float, end: float, has_speech: Optional[bool] = False, usable: str = "sim",
    issues: Optional[list[str]] = None, beauty: int = 3, status: str = "analisada", uniform: bool = False,
    tags: Optional[list[str]] = None, blurry: bool = False,
) -> Shot:
    shot_id = f"{clip_id}.T{n:02d}"
    times = [start + (end - start) * f for f in (0.1, 0.5, 0.9)]
    frames = [ShotFrame(id=f"{shot_id}.F{k}", position=k, time_sec=round(t, 3), file=f"frames/{shot_id}.F{k}.jpg") for k, t in enumerate(times, 1)]
    analysis = None
    if status == "analisada":
        analysis = ShotAnalysis(
            description=f"imagem de {clip_id}", tags=tags or ["paisagem"], face_to_camera=clip_id.startswith("P") and bool(has_speech),
            people_count=0, issues=issues or ["nenhuma"], usable=usable,
            best_frames=[frames[1].id, frames[2].id] if usable != "nao" else [], beauty=beauty, summary_for_editor="ok",
        )
    return Shot(
        id=shot_id, index=n, start_sec=start, end_sec=end, boundary="inicio" if n == 1 else "janela", frames=frames,
        measures=_measures(uniform, blurry), has_speech=has_speech, status=status, analysis=analysis,
    )


def _clip_entry(clip_id: str, role: str, relative: str, duration: float, has_audio: bool = True) -> CatalogClip:
    return CatalogClip(
        id=clip_id, role=role, folder=role, relative_path=relative, size_bytes=1,
        media=ClipMedia(duration_sec=duration, width=160, height=120, fps=30, codec="h264", has_audio=has_audio),
        speech=SpeechDetection(status="analisada", ratio=0.5 if role == "principal" else 0.0, speech_sec=1.0, detector="t"),
    )


def _synthetic_project(tmp_path: Path, p001_t02_usable: str = "nao") -> tuple[ProjectVisual, ProjectCatalog]:
    """P001 (fala dentro da van; T02 obstruída e inutilizável, T03 com borrão), P002 (fala; T01 sem análise
    mas tela uniforme), P003 (principal sem fala, lago, gravada logo depois de P002), B001 (b-roll da praia),
    D001 (drone sem áudio, sem análise), B002 (inutilizável)."""
    van = ["interior", "veículo"]
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
            _shot("P001", 1, 0.0, 3.0, has_speech=True, tags=van),
            _shot("P001", 2, 3.0, 6.0, has_speech=True, usable=p001_t02_usable, issues=["obstruida"], tags=van),
            _shot("P001", 3, 6.0, 10.0, has_speech=True, usable="parcial", issues=["borrao_movimento"], tags=van),
        ],
        "P002": [_shot("P002", 1, 0.0, 3.0, has_speech=None, status="sem_analise", uniform=True), _shot("P002", 2, 3.0, 6.0, has_speech=True)],
        "P003": [_shot("P003", 1, 0.0, 5.0, has_speech=False, beauty=5, tags=["lago"])],
        "B001": [_shot("B001", 1, 0.0, 6.0, beauty=4, tags=["praia", "lago"]), _shot("B001", 2, 6.0, 9.0, beauty=2, tags=["praia"])],
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


def _context(
    tmp_path: Path, roles: Optional[list[str]] = None, p001_t02_usable: str = "nao"
) -> tuple[CoverageContext, Timeline, list[VisualShot]]:
    visual, catalog = _synthetic_project(tmp_path, p001_t02_usable)
    material = build_visual_material(visual, catalog, tmp_path / "visual.json", ["P001", "P002"])
    spine = _spine(tmp_path)
    kept = coverage_sentences(SENTENCES, {3}, spine)
    screen = annotate_screen(kept, material)
    allowed = coverage_roles(roles or [])
    context = CoverageContext(
        project_name="viagem", fps=FPS, spine_duration_sec=spine.duration_sec, sentences=kept,
        windows=find_mandatory_windows(kept, material, ["P001", "P002"]),
        candidates=coverage_candidates(material, allowed), screen_shots=screen, roles=allowed,
    )
    return context, spine, material.shots


# --- janelas com imagem ruim, o que está na tela e frases → frames ---


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
    # Inutilizável e tela uniforme: as duas exigem cobertura.
    assert windows["O1"].required and windows["O2"].required


def test_window_requires_cover_only_when_every_bad_shot_is_severe(tmp_path):
    from ai_video_editor.coverage.material import is_severe

    # Obstruída mas "parcial" (ex.: os bancos do ônibus na frente): a IA pode manter.
    partial, _, _ = _context(tmp_path, p001_t02_usable="parcial")
    assert not partial.windows[0].required and partial.windows[1].required
    blur = VisualShot(
        id="P009.T01", clip_id="P009", role="principal", source_path=Path("p.mp4"), start_sec=0.0, end_sec=3.0,
        status="analisada", usable="parcial", issues=["borrao_movimento"],
    )
    assert not is_severe(blur)  # borrão só pela IA
    assert is_severe(blur.model_copy(update={"blurry": True}))  # confirmado pela medida `borrão alto`
    assert is_severe(blur.model_copy(update={"uniform": True})) and is_severe(blur.model_copy(update={"usable": "nao"}))


def test_screen_lists_the_principal_shots_under_each_sentence(tmp_path):
    context, _, _ = _context(tmp_path)
    by_id = {s.id: s for s in context.sentences}
    assert by_id["S000"].screen == ["P001.T01"]
    assert by_id["S001"].screen == ["P001.T01", "P001.T02"]  # 1,0 s na T01, 0,8 s na T02
    assert by_id["S002"].screen == ["P001.T02"] and by_id["S004"].screen == ["P002.T01"]
    assert [s.id for s in context.screen_shots] == ["P001.T01", "P001.T02", "P002.T01", "P002.T02"]


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


def test_prompt_shows_screen_windows_relations_and_recording_proximity(tmp_path):
    context, _, _ = _context(tmp_path)
    context = context.model_copy(update={"instructions": ["Use bastante drone"]})
    prompt = build_coverage_prompt(context)
    assert "O1: S001–S002 (clipe P001, ~2.8s) — imagem inutilizável, obstruída/parede — EXIGE COBERTURA" in prompt
    assert "O2: S004 (clipe P002, ~2.0s) — tela uniforme — EXIGE COBERTURA" in prompt
    # O que está na tela: na janela e na lista de frases, quando a tomada muda.
    assert "   ~ tela: P001.T01 imagem de P001 [interior, veículo]\n[S000] P001 1.6s Bom dia, pessoal." in prompt
    assert "   ~ tela: P001.T02 imagem de P001 [interior, veículo]\n[S002]" in prompt
    assert "[S004] P002 2.0s Chegamos no lago." in prompt and "S003" not in prompt
    # Candidatas relacionadas por janela e proximidade de gravação.
    assert "relacionadas: P003.T01 (gravada logo depois de P002)" in prompt
    assert "relacionadas: nenhuma pelo código" in prompt  # O1: nada de dentro da van
    assert "P003.T01 principal sem fala, 4.9s, beleza 5, boa, gravada logo depois de P002 — imagem de P003 [lago]" in prompt
    assert "B001.T01 b-roll, 6.0s, beleza 4, boa — imagem de B001 [praia, lago]" in prompt and "D001.T01" not in prompt
    assert "shot_ids" in prompt and "window_decisions" in prompt and "Contexto acima de beleza" in prompt
    assert "Use bastante drone" in prompt and "timestamps" in prompt
    partial, _, _ = _context(tmp_path, p001_t02_usable="parcial")
    assert "obstruída/parede — cobrir ou manter" in build_coverage_prompt(partial)


# --- validação item a item ---


def _decision(coverages=(), breaths=(), windows=()) -> CoverageDecision:
    return CoverageDecision(coverages=list(coverages), breaths=list(breaths), window_decisions=list(windows), summary="ok")


def _cov(a: str, b: str, *shots: str, windows=()) -> CoverageChoice:
    return CoverageChoice(from_sentence=a, to_sentence=b, shot_ids=list(shots), reason="combina", windows=list(windows))


GOOD = [_cov("S001", "S002", "B001.T01", windows=["O1"]), _cov("S004", "S004", "P003.T01", windows=["O2"])]


def _ai(context, shots, *responses) -> tuple[CoverageRun, MockCoverageDirector]:
    director = MockCoverageDirector(responses=list(responses))
    return run_coverage(director, context, shots), director


def _candidate(shot_id: str, start: float, end: float, tags: list[str], beauty: int, role: str = "broll") -> VisualShot:
    clip = shot_id.split(".")[0]
    return VisualShot(
        id=shot_id, clip_id=clip, role=role, source_path=Path(f"{clip}.mp4"), start_sec=start, end_sec=end,
        has_speech=False, status="analisada", usable="sim", tags=tags, beauty=beauty,
    )


def test_valid_decision_is_accepted_without_changes(tmp_path):
    context, _, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S002", shot_id="B001.T02", size="short", reason="troca de lugar")
    decision = _decision(GOOD, [breath])
    run, director = _ai(context, shots, decision)
    assert len(director.calls) == 1 and run.origin == "ia" and run.origin_label == "ia"
    assert [c.origin for c in run.coverages] == ["ia", "ia"] and run.breaths[0].origin == "ia"
    assert not run.discarded and not run.warnings and not run.notes
    assert run.window_cover == {"O1": ["C1"], "O2": ["C2"]}
    assert [(o.window, o.status, o.by) for o in run.window_outcomes] == [("O1", "coberta", ["C1"]), ("O2", "coberta", ["C2"])]
    assert run.responses == [decision] and run.ai_items == 3


def test_single_shot_answer_is_still_accepted():
    choice = CoverageChoice.model_validate({"from_sentence": "S001", "to_sentence": "S002", "shot_id": "B001.T01", "reason": "x"})
    assert choice.shot_ids == ["B001.T01"] and choice.shot_id == "B001.T01"
    assert "shot_id" not in CoverageChoice.model_json_schema()["properties"]  # o schema pedido à IA é só a lista


@pytest.mark.parametrize(
    "bad, kind, expected",
    [
        (_cov("S001", "S999", "B001.T02"), "ID inexistente", "a frase S999 não existe"),
        (_cov("S000", "S000", "D001.T01"), "tomada não candidata", "D001.T01 não pode ser cobertura (sem análise da IA)"),
        (_cov("S000", "S000", "B002.T01"), "tomada não candidata", "B002.T01 não pode ser cobertura (imagem inutilizável"),
        (_cov("S000", "S000", "X001.T01"), "ID inexistente", "a tomada X001.T01 não existe"),
        (_cov("S000", "S001", "B001.T02"), "sobreposição", "se sobrepõe a cobertura 1 (S001–S002)"),
        (_cov("S005", "S005", "B001.T01"), "tomada repetida", "B001.T01 já foi usada em cobertura 1 (S001–S002)"),
    ],
)
def test_invalid_item_is_discarded_and_the_rest_stays(tmp_path, bad, kind, expected):
    context, _, shots = _context(tmp_path)
    run, director = _ai(context, shots, _decision([*GOOD, bad]))
    assert len(director.calls) == 1  # 1 de 3 descartado e janelas cobertas: sem nova tentativa
    assert [d.kind for d in run.discarded] == [kind] and expected in run.discarded[0].message
    assert run.origin == "ia" and [(c.from_sentence, c.shot_id) for c in run.coverages] == [("S001", "B001.T01"), ("S004", "P003.T01")]
    assert run.origin_label == "ia (1 descartada)"
    assert run.warnings[0].startswith(f"cobertura da IA: 3 itens, 1 descartados ({kind})")
    assert expected in "\n".join(run.notes)


def test_shot_of_the_speaking_clip_is_discarded_without_dropping_the_others(tmp_path):
    context, _, shots = _context(tmp_path)
    p001_silent = _candidate("P001.T09", 8.0, 10.0, ["lago"], 3, role="principal")
    context = context.model_copy(update={"candidates": [*context.candidates, p001_silent]})
    run, director = _ai(context, shots, _decision([_cov("S000", "S000", "P001.T09"), *GOOD]))
    assert len(director.calls) == 1 and [d.kind for d in run.discarded] == ["clipe que fala"]
    assert "a tomada P001.T09 é do clipe P001, que está falando nesse trecho" in run.discarded[0].message
    assert len(run.coverages) == 2 and run.origin == "ia"


def test_invalid_shot_in_a_list_is_removed_and_the_item_adjusted(tmp_path):
    context, _, shots = _context(tmp_path)
    run, _ = _ai(context, shots, _decision([_cov("S001", "S002", "X001.T01", "B001.T01", windows=["O1"]), GOOD[1]]))
    first = run.coverages[0]
    assert first.shot_ids == ["B001.T01"] and first.origin == "ia-ajustada"
    assert first.adjustments[0].kind == "ID inexistente" and "retirada" in first.adjustments[0].message
    assert not run.discarded and run.origin_label == "ia (1 ajustada)"


def test_short_item_is_extended_to_the_shorter_free_neighbor_or_discarded(tmp_path):
    context, _, shots = _context(tmp_path)
    # S002 (1 s): a vizinha livre mais curta é S001 (S004 é de outro clipe, depois do corte).
    run, _ = _ai(context, shots, _decision([_cov("S002", "S002", "B001.T01", windows=["O1"]), GOOD[1]]))
    first = run.coverages[0]
    assert (first.from_sentence, first.to_sentence) == ("S001", "S002") and first.origin == "ia-ajustada"
    assert "1,0s é menos que o mínimo; estendida para S001–S002 (3,0s)" in first.adjustments[0].message
    # Sem vizinha livre (S001 e S004 já cobertas): descartado. O1 aqui pode ser mantida (parcial): sem nova tentativa.
    partial, _, shots = _context(tmp_path, p001_t02_usable="parcial")
    decision = _decision([_cov("S000", "S001", "B001.T01"), _cov("S004", "S004", "P003.T01"), _cov("S002", "S002", "B001.T02")])
    run, director = _ai(partial, shots, decision)
    assert len(director.calls) == 1 and [d.kind for d in run.discarded] == ["curta demais"]
    assert "dura 1,0s, menos que o mínimo de 1,5s, e não há frase vizinha livre" in run.discarded[0].message


def test_retry_only_when_a_required_window_is_left_uncovered(tmp_path):
    context, _, shots = _context(tmp_path)
    run, director = _ai(context, shots, _decision([GOOD[0]]), _decision([GOOD[1]]))
    assert len(director.calls) == 2
    retry = director.calls[1]
    assert retry.previous_error == "- janela O2 (S004, tela uniforme) exige cobertura: faltam S004"
    assert retry.accepted_items == ["cobertura S001–S002: B001.T01"]
    prompt = build_coverage_prompt(retry)
    assert "NOVA TENTATIVA" in prompt and "- cobertura S001–S002: B001.T01" in prompt
    assert "Responda SÓ com os itens que faltam" in prompt
    assert run.origin == "ia" and [a.success for a in run.attempts] == [False, True]
    assert run.window_cover == {"O1": ["C1"], "O2": ["C2"]} and not run.warnings


def test_retry_when_more_than_half_of_the_items_is_discarded(tmp_path):
    context, _, shots = _context(tmp_path)
    bad = [_cov("S000", "S000", "X001.T01"), _cov("S005", "S005", "X002.T01"), _cov("S000", "S000", "X003.T01")]
    run, director = _ai(context, shots, _decision([*GOOD, *bad]), _decision([_cov("S005", "S005", "B001.T02")]))
    assert len(director.calls) == 2
    error = director.calls[1].previous_error
    assert "X001.T01 não existe" in error and "3 de 5 itens descartados" in error
    assert [c.from_sentence for c in run.coverages] == ["S001", "S004", "S005"] and len(run.discarded) == 3
    assert run.origin == "ia"


def test_keep_is_accepted_with_a_reason_and_refused_where_cover_is_required(tmp_path):
    context, _, shots = _context(tmp_path, p001_t02_usable="parcial")  # O1 pode ser mantida; O2 (tela uniforme) não
    keep = _decision(windows=[
        WindowDecision(window="O1", action="keep", reason="a van é o assunto da fala"),
        WindowDecision(window="O2", action="keep", reason="tanto faz"),
    ])
    run, director = _ai(context, shots, keep, _decision([GOOD[1]]))
    assert len(director.calls) == 2 and "O2 exige cobertura (tela uniforme)" in director.calls[1].previous_error
    assert [d.kind for d in run.discarded] == ["janela exige cobertura"]
    outcomes = {o.window: o for o in run.window_outcomes}
    assert (outcomes["O1"].status, outcomes["O1"].reason, outcomes["O1"].origin) == ("mantida", "a van é o assunto da fala", "ia")
    assert outcomes["O2"].status == "coberta" and run.origin == "ia"
    assert run.windows_label == "1 cobertas, 1 mantidas de 2"


# --- fallback: só tomadas relacionadas ---


def test_unreadable_answer_twice_falls_back_to_related_shots_only(tmp_path):
    context, _, shots = _context(tmp_path)
    error = AIResponseValidationError("coverage", details="JSON inválido")
    run, director = _ai(context, shots, error, error)
    assert len(director.calls) == 2 and run.origin == "fallback" and run.responses == [None, None]
    assert any("Cobertura da IA ignorada: JSON inválido" in w for w in run.warnings)
    # O2: P003.T01 foi gravada logo depois de P002 → coberta. O1 (dentro da van): nada relacionado → imagem
    # original, com aviso (nunca o b-roll da praia só porque é bonito).
    assert [(c.from_sentence, c.shot_ids, c.origin) for c in run.coverages] == [("S004", ["P003.T01"], "fallback")]
    assert "P003.T01 (gravada logo depois de P002)" in run.coverages[0].reason
    outcomes = {o.window: o for o in run.window_outcomes}
    assert outcomes["O1"].status == "mantida" and outcomes["O1"].origin == "fallback"
    assert any("janela O1 (S001–S002): nenhuma tomada relacionada livre" in w for w in run.warnings)
    assert any("janelas que exigem cobertura sem cobertura completa: O1" in w for w in run.warnings)


def test_fallback_uses_tags_in_common_with_the_screen(tmp_path):
    context, _, shots = _context(tmp_path)
    van_broll = _candidate("B007.T01", 0.0, 6.0, ["veículo", "interior", "estrada"], 3)
    context = context.model_copy(update={"candidates": [*context.candidates, van_broll]})
    run = run_coverage(None, context, shots)
    assert run.origin == "fallback" and "desligada" in run.origin_reason and run.provider == "none"
    first = run.coverages[0]
    assert (first.from_sentence, first.shot_ids) == ("S001", ["B007.T01"]) and "tags veículo, interior" in first.reason
    assert run.window_cover == {"O1": ["C1"], "O2": ["C2"]} and not run.breaths


def test_fallback_does_not_repeat_the_clip_in_consecutive_pieces(tmp_path):
    context, _, shots = _context(tmp_path)
    van = ["interior", "veículo"]
    # Tomadas de 1,6 s: O1 (3 s) precisa de duas. B001.T02 é mais bonita que D009.T01, mas repetiria o B001.
    pool = [_candidate("B001.T01", 0.0, 1.6, van, 5), _candidate("B001.T02", 2.0, 3.6, van, 4), _candidate("D009.T01", 0.0, 1.6, van, 2, "drone")]
    context = context.model_copy(update={"candidates": pool, "windows": context.windows[:1]})
    run = run_coverage(None, context, shots)
    assert run.coverages[0].shot_ids == ["B001.T01", "D009.T01"]


def test_default_mock_decides_windows_and_adds_a_breath_at_clip_change(tmp_path):
    context, _, shots = _context(tmp_path)
    director = MockCoverageDirector()
    run = run_coverage(director, context, shots)
    assert run.origin == "ia" and not run.uncovered_windows and len(director.calls) == 1
    assert [c.shot_ids for c in run.coverages] == [["B001.T01"], ["P003.T01"]]
    assert [(b.after_sentence, b.size) for b in run.breaths] == [("S002", "medium")]


def test_coverage_params_inherit_the_director():
    same = CoverageParams.inherit("gemini", "gemini-x", "gemini:gemini-y", "high", 90.0, 2)
    assert (same.provider, same.model, same.fallback, same.thinking, same.timeout_sec) == ("gemini", "gemini-x", "gemini:gemini-y", "high", 90.0)
    other = CoverageParams.inherit("gemini", "gemini-x", "gemini:gemini-y", provider="openai", thinking="off")
    assert (other.provider, other.model, other.fallback, other.thinking) == ("openai", None, None, "off")
    assert CoverageParams.inherit("none").provider == "none" and CoverageParams.inherit("mock").provider == "mock"


# --- material real: vlog "Trilha La Islita" (tests/data/coverage_real) ---

REAL = Path(__file__).parent / "data" / "coverage_real"


@pytest.fixture(scope="module")
def real_material() -> tuple[VisualMaterial, dict]:
    material = VisualMaterial.model_validate_json((REAL / "material.json").read_text(encoding="utf-8"))
    return material, json.loads((REAL / "spine.json").read_text(encoding="utf-8"))


def _real_context(real_material) -> CoverageContext:
    material, spine = real_material
    kept = [CoverageSentence.model_validate(s) for s in spine["sentences"]]
    roles = coverage_roles([])
    return CoverageContext(
        project_name="trilha", fps=spine["fps"], spine_duration_sec=spine["duration_sec"], sentences=kept,
        windows=find_mandatory_windows(kept, material, spine["clips"]), candidates=coverage_candidates(material, roles),
        screen_shots=annotate_screen(kept, material), roles=roles,
    )


def _gemini() -> CoverageDecision:
    """Primeira resposta do Gemini (gemini-3.6-flash) no vlog real, na forma antiga (uma tomada por cobertura).

    Reconstruída das mensagens de erro do `coverage_decision.json` daquela execução (as respostas cruas não
    eram gravadas): mesmos 23 itens, na mesma ordem e com os mesmos trechos, e as duas tomadas do clipe que
    fala (P025.T05, P027.T03). As outras tomadas e os 4 respiros foram escolhidos aqui, entre as candidatas.
    Com a validação antiga, ela dá exatamente os 14 erros daquela execução.
    """
    return CoverageDecision.model_validate_json((REAL / "gemini_attempt1.json").read_text(encoding="utf-8"))


def test_real_material_windows_and_candidates(real_material):
    context = _real_context(real_material)
    assert len(context.sentences) == 167 and len(context.candidates) == 90
    assert [w.id for w in context.windows] == [f"O{n}" for n in range(1, 11)]
    # Nenhuma exige cobertura: todas têm tomada "parcial" (ônibus com os bancos, tour pelo quarto...).
    assert not any(w.required for w in context.windows)


def test_real_gemini_answer_is_accepted_with_adjustments(real_material):
    material, _ = real_material
    run, director = _ai(_real_context(real_material), material.shots, _gemini())
    # Antes: recusada inteira duas vezes (17 itens por duração, 2 pelo clipe que fala) e fallback geral.
    assert len(director.calls) == 1 and run.origin == "ia" and run.ai_items == 27
    assert [(d.item, d.kind) for d in run.discarded] == [
        ("cobertura 19 (S133–S134)", "clipe que fala"), ("cobertura 23 (S166–S168)", "clipe que fala"),
    ]
    assert run.origin_label == "ia (19 ajustadas, 2 descartadas)"
    assert run.warnings == [
        "cobertura da IA: 27 itens, 19 ajustados (duração), 2 descartados (clipe que fala); detalhes no edit_report.md"
    ]
    assert len(run.coverages) == 21 and len(run.breaths) == 4 and all(b.origin == "ia" for b in run.breaths)
    assert [o.status for o in run.window_outcomes] == ["coberta"] * 10
    # Nenhum pedaço passa de 8 s nem fica abaixo de 1,5 s (fora o fim de uma cobertura encurtada).
    fps = 30
    for c in run.coverages:
        caps = [shot_frames(material.by_id()[x], fps) for x in c.shot_ids]
        assert max(caps) <= MAX_COVERAGE_SEC * fps


def test_real_long_items_are_completed_in_windows_shortened_outside_and_short_ones_extended(real_material):
    material, _ = real_material
    context = _real_context(real_material)
    run, _ = _ai(context, material.shots, _gemini())
    by_start = {c.from_sentence: c for c in run.coverages}
    order = {s.id: i for i, s in enumerate(context.sentences)}

    def span(c) -> int:
        return context.sentences[order[c.to_sentence]].end_frame - context.sentences[order[c.from_sentence]].start_frame

    # O1 (S000–S001, 22,9 s) com uma tomada de 7,7 s: completada com tomadas parecidas até cobrir tudo.
    o1 = by_start["S000"]
    assert o1.shot_ids[0] == "D002.T15" and len(o1.added_shots) == 3 and o1.windows == ["O1"]
    assert sum(shot_frames(material.by_id()[x], 30) for x in o1.shot_ids) >= span(o1)
    assert all(material.by_id()[x].clip_id not in {"P001"} for x in o1.shot_ids)
    # S085–S088 (21,2 s) fora de janela: encurtada, sem tomada que a IA não escolheu.
    outside = by_start["S085"]
    assert outside.shot_ids == ["D002.T16"] and not outside.added_shots
    assert "encurtada; as tomadas cobrem 7,7s de 21,2s" in outside.adjustments[-1].message
    # S140 (1,1 s): estendida para a vizinha livre mais curta (S139), em vez de recusada.
    o9 = by_start["S139"]
    assert o9.to_sentence == "S140" and "estendida para S139–S140" in o9.adjustments[0].message


def test_real_long_item_with_enough_shots_is_distributed(real_material):
    material, _ = real_material
    context = _real_context(real_material)
    three = CoverageDecision(
        coverages=[CoverageChoice(
            from_sentence="S000", to_sentence="S001", shot_ids=["D002.T15", "D002.T07", "D002.T08"], reason="estrada e lago",
        )],
        summary="x",
    )
    run, _ = _ai(context, material.shots, three)
    first = run.coverages[0]
    assert first.origin == "ia" and first.shot_ids == ["D002.T15", "D002.T07", "D002.T08"] and not first.adjustments
    caps = [shot_frames(material.by_id()[x], 30) for x in first.shot_ids]
    pieces = distribute(687, caps)  # 22,9 s
    assert pieces == [229, 229, 229] and all(MIN_COVERAGE_SEC * 30 <= p <= MAX_COVERAGE_SEC * 30 for p in pieces)
    # Janelas sem decisão da IA: o código decide (tomada relacionada ou imagem original).
    assert run.origin == "ia+fallback" and "janelas sem decisão da IA" in run.origin_reason


def test_real_fallback_keeps_bus_and_hotel_and_covers_only_with_related_shots(real_material):
    material, _ = real_material
    run = run_coverage(None, _real_context(real_material), material.shots)
    outcomes = {o.window: o for o in run.window_outcomes}
    # Ônibus (P001) e tour pelo quarto do hotel (P007): sem tomada relacionada que cubra, a imagem fica.
    assert outcomes["O1"].status == "mantida" and outcomes["O2"].status == "mantida"
    # Cachorro perto do lago (P021): o b-roll do homem com o cachorro na beira do lago.
    o5 = run.coverages[[c.windows for c in run.coverages].index(["O5"])]
    assert {x.split(".")[0] for x in o5.shot_ids} == {"B001", "B003"}
    assert all("com tomada relacionada" in c.reason for c in run.coverages)
    for c in run.coverages:  # pedaços seguidos de clipes diferentes
        assert all(a.split(".")[0] != b.split(".")[0] for a, b in zip(c.shot_ids, c.shot_ids[1:]))


def test_real_prompt_shows_screen_and_recording_proximity(real_material):
    prompt = build_coverage_prompt(_real_context(real_material))
    assert "   ~ tela: P001.T01 Interior de uma van ou micro-ônibus de turismo" in prompt
    assert "P008.T01 principal sem fala, 3.4s, beleza 4, boa, gravada logo depois de P007" in prompt
    assert "O2: S021–S023 (clipe P007, ~58.7s)" in prompt and "cobrir ou manter" in prompt
    assert "relacionadas: B001.T01 (tags lago, animal)" in prompt


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
    run = _run(context, [_cov("S004", "S005", "B001.T01", windows=["O2"])], [breath])
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


def test_coverage_time_is_split_between_its_shots_in_order(tmp_path):
    context, spine, shots = _context(tmp_path)
    by_id = {s.id: s for s in shots}
    # O1 (S001–S002, 3 s) com B001.T02 (2,9 s) e P003.T01 (acrescentada): 1,5 s cada, em ordem, sem buraco.
    run = _run(context, [_cov("S001", "S002", "B001.T02", "P003.T01", windows=["O1"])])
    run.coverages[0].added_shots = ["P003.T01"]
    result = assemble_coverage(spine, context, run, by_id)
    pieces = result.coverages[0].pieces
    assert [(p.shot_id, p.chained) for p in pieces] == [("B001.T02", False), ("P003.T01", True)]
    assert pieces[0].timeline_start == pytest.approx(2.0) and pieces[0].timeline_end == pieces[1].timeline_start
    assert pieces[1].timeline_end == pytest.approx(5.0) and len(result.timeline.video_overlay) == 2
    assert result.coverages[0].shot_ids == ["B001.T02", "P003.T01"] and not result.warnings
    # Tomadas que não bastam (S004–S005, 5 s, com 2,9 s de tomada): o fim fica com o principal, com aviso.
    short = assemble_coverage(spine, context, _run(context, [_cov("S004", "S005", "B001.T02")]), by_id)
    assert short.coverages[0].pieces[-1].timeline_end < short.coverages[0].timeline_end
    assert any("ficam sem cobertura" in w for w in short.warnings)
    # Divisão: o mais igual possível, sem passar do limite de cada tomada.
    assert distribute(10, [100, 100]) == [5, 5] and distribute(10, [3, 100]) == [3, 7] and distribute(10, [2, 3]) == [2, 3]


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
