"""Testes do Diretor de cobertura pelo conteúdo da fala e do render com cobertura (`edit <pasta>`, receita
`vlog_cobertura`): evidências conferidas pelo código (fala literal, imagem na análise, mesma coisa,
sinônimos), mapa de assuntos, trechos com imagem ruim medidos em segundos, pessoa falando para a câmera,
limite de cobertura, tapa de tela uniforme, respiros entre blocos, IDs → tempos, pipeline com `--so-plano`,
render ponta a ponta e a resposta real do Gemini no vlog "Trilha La Islita" (`tests/data/coverage_real/`)."""

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
    TopicBlock,
    TopicMap,
    VisualMaterial,
    annotate_screen,
    assemble_coverage,
    build_coverage_prompt,
    build_topic_prompt,
    build_visual_material,
    coverage_candidates,
    coverage_roles,
    coverage_sentences,
    find_bad_spans,
    run_coverage,
    topics_by_clip,
    validate_topics,
)
from ai_video_editor.coverage import evidence as ev
from ai_video_editor.coverage import service
from ai_video_editor.coverage.assemble import distribute, shot_segment
from ai_video_editor.coverage.providers.mock import MockCoverageDirector
from ai_video_editor.coverage.schema import AcceptedBreath, AcceptedCoverage, CoverageSentence, VisualShot
from ai_video_editor.coverage.service import MAX_COVERAGE_RATIO, MAX_COVERAGE_SEC, excluded_shots, shot_frames
from ai_video_editor.director.schema import Sentence
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.pipeline import EditPipelineParams, SpinePipeline
from ai_video_editor.pipeline.clips import catalog_location, catalog_path
from ai_video_editor.pipeline.spine_schema import SpineDiff, SpineManifest
from ai_video_editor.planning import TimeRange
from ai_video_editor.planning.timeline_plan import ClipCut, build_spine_timeline, clip_placements, is_breath_item
from ai_video_editor.recipes import RecipeError, get_recipe
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
    tags: Optional[list[str]] = None, blurry: bool = False, description: Optional[str] = None,
) -> Shot:
    shot_id = f"{clip_id}.T{n:02d}"
    times = [start + (end - start) * f for f in (0.1, 0.5, 0.9)]
    frames = [ShotFrame(id=f"{shot_id}.F{k}", position=k, time_sec=round(t, 3), file=f"frames/{shot_id}.F{k}.jpg") for k, t in enumerate(times, 1)]
    analysis = None
    if status == "analisada":
        analysis = ShotAnalysis(
            description=description or f"imagem de {clip_id}", tags=tags or ["paisagem"],
            face_to_camera=clip_id.startswith("P") and bool(has_speech), people_count=0, issues=issues or ["nenhuma"],
            usable=usable, best_frames=[frames[1].id, frames[2].id] if usable != "nao" else [], beauty=beauty,
            summary_for_editor="ok",
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


def _synthetic_project(tmp_path: Path) -> tuple[ProjectVisual, ProjectCatalog]:
    """P001 (fala dentro da van, para a câmera; T02 obstruída e inutilizável, T03 "parcial" com borrão), P002
    (fala; T01 sem análise mas tela uniforme), P003 (principal sem fala, lago, vizinha de P002), B001 (b-roll
    da praia na beira do lago), D001 (drone sem áudio, sem análise), B002 (inutilizável)."""
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
            _shot("P001", 1, 0.0, 3.0, has_speech=True, tags=van, description="Mulher falando dentro da van"),
            _shot("P001", 2, 3.0, 6.0, has_speech=True, usable="nao", issues=["obstruida"], tags=van),
            _shot("P001", 3, 6.0, 10.0, has_speech=True, usable="parcial", issues=["borrao_movimento"], tags=van),
        ],
        "P002": [_shot("P002", 1, 0.0, 3.0, has_speech=None, status="sem_analise", uniform=True), _shot("P002", 2, 3.0, 6.0, has_speech=True)],
        "P003": [_shot("P003", 1, 0.0, 5.0, has_speech=False, beauty=5, tags=["lago"], description="Lago escuro entre montanhas")],
        "B001": [
            _shot("B001", 1, 0.0, 6.0, beauty=4, tags=["praia", "lago"], description="Praia de areia na beira do lago"),
            _shot("B001", 2, 6.0, 9.0, beauty=2, tags=["praia"], description="Praia vazia ao entardecer"),
        ],
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
    tmp_path: Path, roles: Optional[list[str]] = None, texts: Optional[dict[int, str]] = None
) -> tuple[CoverageContext, Timeline, list[VisualShot]]:
    visual, catalog = _synthetic_project(tmp_path)
    material = build_visual_material(visual, catalog, tmp_path / "visual.json", ["P001", "P002"])
    spine = _spine(tmp_path)
    sentences = [s.model_copy(update={"text": texts[s.id]}) if texts and s.id in texts else s for s in SENTENCES]
    kept = coverage_sentences(sentences, {3}, spine)
    screen = annotate_screen(kept, material)
    allowed = coverage_roles(roles or [])
    context = CoverageContext(
        project_name="viagem", fps=FPS, spine_duration_sec=spine.duration_sec, sentences=kept,
        bad_spans=find_bad_spans(kept, material, ["P001", "P002"], FPS),
        candidates=coverage_candidates(material, allowed), screen_shots=screen, roles=allowed,
    )
    return context, spine, material.shots


def _decision(coverages=(), breaths=()) -> CoverageDecision:
    return CoverageDecision(coverages=list(coverages), breaths=list(breaths), summary="ok")


def _cov(a: str, b: str, *shots: str, speech: str = "", image: str = "") -> CoverageChoice:
    return CoverageChoice(from_sentence=a, to_sentence=b, shot_ids=list(shots), speech_evidence=speech, image_evidence=image, reason="combina")


TOPICS = TopicMap(
    summary="viagem de van até um lago",
    blocks=[
        TopicBlock(from_sentence="S000", to_sentence="S001", topic="chegada de van", place="van"),
        TopicBlock(from_sentence="S002", to_sentence="S002", topic="a estrada", place="estrada"),
        TopicBlock(from_sentence="S004", to_sentence="S005", topic="chegada no lago", place="lago"),
    ],
)
LAKE = _cov("S004", "S004", "P003.T01", speech="no lago", image="lago")


def _ai(context, shots, *responses, topics=(TOPICS,)) -> tuple[CoverageRun, MockCoverageDirector]:
    """`topics=None`: o mapa do mock (um bloco por clipe)."""
    director = MockCoverageDirector(responses=list(responses), topics=None if topics is None else list(topics))
    return run_coverage(director, context, shots), director


def _candidate(shot_id: str, start: float, end: float, tags: list[str], beauty: int, role: str = "broll", description: str = "") -> VisualShot:
    clip = shot_id.split(".")[0]
    return VisualShot(
        id=shot_id, clip_id=clip, role=role, source_path=Path(f"{clip}.mp4"), start_sec=start, end_sec=end,
        has_speech=False, status="analisada", usable="sim", tags=tags, beauty=beauty, description=description,
    )


# --- material: frases, tela, pessoa para a câmera e trechos com imagem ruim ---


def test_bad_spans_are_measured_in_seconds_only_on_truly_bad_images(tmp_path):
    context, _, _ = _context(tmp_path)
    assert [s.id for s in context.sentences] == ["S000", "S001", "S002", "S004", "S005"]  # S003 removida
    o1, o2 = context.bad_spans
    # P001.T02 (3–6 s, inutilizável): S001 de 3,0 a 3,8 s (frames 90–114) e a parte mantida de S002 (5–6 s,
    # frames 120–150), com 0,25 s de margem sem sair da frase; pedaços seguidos viram um trecho.
    assert (o1.clip_id, o1.sentence_ids, o1.shot_ids, o1.uniform) == ("P001", ["S001", "S002"], ["P001.T02"], False)
    assert (o1.start_frame, o1.end_frame) == (82, 150) and o1.reasons == ["imagem inutilizável"]
    # P002.T01 sem análise da IA, mas a medida diz tela uniforme: S004 inteira.
    assert (o2.sentence_ids, o2.uniform, o2.start_frame, o2.end_frame) == (["S004"], True, 285, 345)
    # P001.T03 é "parcial" (borrão sem a medida): a IA disse que a imagem serve, não vira trecho ruim.
    assert len(context.bad_spans) == 2


def test_screen_and_speaking_to_camera(tmp_path):
    context, _, _ = _context(tmp_path)
    by_id = {s.id: s for s in context.sentences}
    assert by_id["S001"].screen == ["P001.T01", "P001.T02"]  # 1,0 s na T01, 0,8 s na T02
    assert by_id["S002"].screen == ["P001.T02"] and by_id["S004"].screen == ["P002.T01"]
    # Rosto e utilizável na tomada principal da frase = falando para a câmera.
    assert [s.id for s in context.sentences if s.to_camera] == ["S000", "S001", "S005"]


def test_sentence_trimmed_by_silence_starts_at_first_kept_frame(tmp_path):
    context, _, _ = _context(tmp_path)
    by_id = {s.id: s for s in context.sentences}
    assert (by_id["S002"].start_frame, by_id["S002"].end_frame) == (120, 150)
    assert by_id["S002"].kept == [(5.0, 6.0)] and by_id["S002"].source_start_sec == 5.0
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


# --- evidências (código) ---


def test_evidence_words_ignore_case_accents_punctuation_plural_and_use_synonyms():
    assert ev.words("Lá no LAGO, né?") == ["la", "no", "lago", "ne"]
    assert ev.same_word("lagos", "lago") and ev.same_word("flores", "flor") and ev.same_word("caes", "cachorro")
    assert ev.same_word("lagoa", "lago") and ev.same_word("laguna", "lago")
    assert ev.same_word("cordilheira", "montanha") and ev.same_word("morros", "montanha")
    assert ev.same_word("cao", "cachorro") and not ev.same_word("lago", "lagosta")
    assert ev.label("lagoa") == "lago"


def test_speech_evidence_is_a_literal_passage_and_its_position_is_known():
    texts = ["Bom dia.", "Agora a gente chegou na Lagoa Azul, olha só"]
    hit = ev.find_speech(texts, "chegou na lagoa azul")
    assert hit is not None and hit.sentence == 1 and 0.3 < hit.fraction < 0.7
    assert ev.find_speech(texts, "chegamos no lago") is None  # não é trecho literal
    assert ev.find_speech(texts, "dia. Agora a gente") is not None  # atravessa duas frases seguidas


def test_image_evidence_is_a_tag_or_a_description_passage_and_both_must_talk_about_the_same_thing():
    assert ev.image_matches("lago", "Vista aérea", ["paisagem", "lago"])
    assert ev.image_matches("lago escuro entre", "Lago escuro entre montanhas", [])
    assert not ev.image_matches("praia", "Lago escuro entre montanhas", ["lago"])
    assert ev.shared_concepts("na lagoa", "lago escuro") == ["lagoa"]
    assert ev.shared_concepts("bom dia pessoal", "praia de areia") == []
    # Conceitos genéricos não provam relação.
    assert ev.shared_concepts("que paisagem linda", "paisagem com árvores") == []
    assert ev.is_strong(["lagoa"], ["lago", "água"]) and not ev.is_strong(["areia"], ["praia"])


# --- mapa de assuntos ---


def test_topic_map_valid_and_refused_with_gap_overlap_or_unknown_id(tmp_path):
    context, _, _ = _context(tmp_path)
    topics, problems = validate_topics(TOPICS, context.sentences)
    assert not problems and [(t.id, t.span_label, t.start_frame, t.end_frame) for t in topics] == [
        ("A1", "S000–S001", 6, 114), ("A2", "S002", 120, 150), ("A3", "S004–S005", 285, 435),
    ]
    gap = TopicMap(blocks=[TOPICS.blocks[0], TOPICS.blocks[2]])
    assert validate_topics(gap, context.sentences) == (None, ["buraco: S002 sem bloco (antes do bloco 2 (S004–S005))"])
    overlap = TopicMap(blocks=[TOPICS.blocks[0], TopicBlock(from_sentence="S001", to_sentence="S005", topic="resto")])
    assert validate_topics(overlap, context.sentences)[1] == ["sobreposição: S001 já estão no bloco anterior (bloco 2 (S001–S005))"]
    short = TopicMap(blocks=[TOPICS.blocks[0]])
    assert validate_topics(short, context.sentences)[1] == ["buraco: S002–S005 sem bloco (no fim)"]
    unknown = TopicMap(blocks=[TopicBlock(from_sentence="S000", to_sentence="S999", topic="tudo")])
    assert "a frase S999 não existe" in validate_topics(unknown, context.sentences)[1][0]


def test_invalid_topic_map_is_retried_with_the_problems_then_replaced_by_one_block_per_clip(tmp_path):
    context, _, shots = _context(tmp_path)
    gap = TopicMap(blocks=[TOPICS.blocks[0], TOPICS.blocks[2]])
    run, director = _ai(context, shots, _decision(), topics=[gap, gap])
    assert len(director.topic_calls) == 2 and "buraco: S002 sem bloco" in director.topic_calls[1].previous_error
    assert "NOVA TENTATIVA" in build_topic_prompt(director.topic_calls[1])
    assert run.topics_origin == "fallback" and [t.topic for t in run.topics] == ["clipe P001", "clipe P002"]
    assert [a.success for a in run.topic_attempts] == [False, False] and run.topic_responses == [gap, gap]
    assert any("mapa de assuntos da IA recusado" in w for w in run.warnings)
    # A cobertura recebe o mapa validado.
    assert [t.id for t in director.calls[0].topics] == ["A1", "A2"]
    good, director = _ai(context, shots, _decision())
    assert good.topics_origin == "ia" and good.video_summary == "viagem de van até um lago" and len(good.topics) == 3
    assert [t.topic for t in topics_by_clip(context.sentences)] == ["clipe P001", "clipe P002"]


# --- coberturas com evidência ---


def test_coverage_with_valid_evidence_is_accepted(tmp_path):
    context, _, shots = _context(tmp_path)
    run, director = _ai(context, shots, _decision([LAKE]))
    assert len(director.calls) == 1 and run.origin == "ia" and not run.discarded
    (c,) = run.coverages
    assert (c.from_sentence, c.shot_ids, c.kind) == ("S004", ["P003.T01"], "conteudo")
    assert c.evidence.shared == ["lago"] and c.evidence.strong
    # S004 está toda sobre a tela uniforme O2: coberta, limitada aos segundos dela.
    assert (c.start_frame, c.end_frame, c.bad_spans) == (285, 345, ["O2"])
    assert {o.span: o.status for o in run.span_outcomes} == {"O1": "mantido", "O2": "coberto"}
    assert run.plan[2] == "S004–S005 · chegada no lago · coberto 1× (P003.T01 'no lago' ↔ 'lago')"


@pytest.mark.parametrize(
    "bad, kind, expected",
    [
        (_cov("S005", "S005", "B001.T02", image="praia"), "evidência de fala", "sem evidência de fala"),
        (_cov("S005", "S005", "B001.T02", speech="chegamos na praia", image="praia"), "evidência de fala",
         "a evidência de fala 'chegamos na praia' não está no texto de S005"),
        (_cov("S005", "S005", "B001.T02", speech="a praia", image="lago"), "evidência de imagem",
         "a evidência de imagem 'lago' não está na análise de B001.T02"),
        (_cov("S005", "S005", "B001.T02", speech="que frio", image="Praia vazia"), "evidência sem relação",
         "'que frio' e 'Praia vazia' não falam da mesma coisa"),
        (_cov("S005", "S999", "B001.T02", speech="praia", image="praia"), "ID inexistente", "a frase S999 não existe"),
        (_cov("S005", "S005", "D001.T01", speech="praia", image="praia"), "tomada não candidata", "D001.T01 não pode ser cobertura (sem análise da IA)"),
        (_cov("S005", "S005", "X001.T01", speech="praia", image="praia"), "ID inexistente", "a tomada X001.T01 não existe"),
    ],
)
def test_coverage_without_valid_evidence_or_ids_is_discarded(tmp_path, bad, kind, expected):
    context, _, shots = _context(tmp_path, texts={5: "Olha a praia vazia ali, que frio."})
    run, director = _ai(context, shots, _decision([bad, LAKE]))
    assert [d.kind for d in run.discarded] == [kind] and expected in run.discarded[0].message
    # Só o item ruim sai (1 de 2: sem nova tentativa); a cobertura do lago fica.
    assert len(director.calls) == 1 and [(c.from_sentence, c.kind) for c in run.coverages] == [("S004", "conteudo")]
    good, _ = _ai(context, shots, _decision([_cov("S005", "S005", "B001.T02", speech="a praia", image="praia")]))
    assert [c.from_sentence for c in good.coverages if c.kind == "conteudo"] == ["S005"]


def test_synonym_links_speech_and_image(tmp_path):
    context, _, shots = _context(tmp_path, texts={4: "Chegamos na lagoa."})
    run, _ = _ai(context, shots, _decision([_cov("S004", "S004", "P003.T01", speech="na lagoa", image="Lago escuro entre montanhas")]))
    (c,) = run.coverages
    assert c.evidence.shared == ["lago"] and c.evidence.strong  # "lagoa" ↔ tag "lago"


def test_shot_without_the_image_evidence_is_removed_from_the_list(tmp_path):
    context, _, shots = _context(tmp_path)
    run, _ = _ai(context, shots, _decision([_cov("S004", "S004", "B001.T02", "P003.T01", speech="lago", image="lago")]))
    (c,) = run.coverages
    assert c.shot_ids == ["P003.T01"] and c.origin == "ia-ajustada"
    assert "B001.T02 não mostra(m) 'lago'" in c.adjustments[0].message


def test_overlap_and_repeated_shot_discard_only_the_item(tmp_path):
    context, _, shots = _context(tmp_path, texts={5: "Que lago lindo, olha a praia."})
    decision = _decision([
        LAKE, _cov("S004", "S005", "B001.T01", speech="praia", image="praia"),
        _cov("S005", "S005", "P003.T01", speech="lago", image="lago"),
    ])
    run, director = _ai(context, shots, decision, _decision())
    assert [d.kind for d in run.discarded] == ["sobreposição", "tomada repetida"]
    assert "se sobrepõe a cobertura 1 (S004)" in run.discarded[0].message
    assert [c.from_sentence for c in run.coverages] == ["S004"]


def test_person_talking_to_the_camera_is_only_covered_with_strong_evidence(tmp_path):
    # S005 é a pessoa falando para a câmera. "areia" só está na descrição de B001.T01 (não é tag): fraca.
    context, _, shots = _context(tmp_path, texts={5: "Olha a praia de areia ali."})
    weak = _cov("S005", "S005", "B001.T01", speech="de areia", image="Praia de areia")
    run, _ = _ai(context, shots, _decision([weak]), _decision())
    assert [d.kind for d in run.discarded] == ["pessoa para a câmera"]
    assert "S005: a pessoa fala para a câmera; só com evidência forte" in run.discarded[0].message
    assert not [c for c in run.coverages if c.kind == "conteudo"]
    strong = _cov("S005", "S005", "B001.T01", speech="a praia", image="praia")  # "praia" é tag
    run, _ = _ai(context, shots, _decision([strong]))
    content = [c for c in run.coverages if c.kind == "conteudo"]
    assert len(content) == 1 and content[0].evidence.strong


def test_coverage_is_placed_around_where_the_evidence_is_said(tmp_path):
    context, _, shots = _context(tmp_path, texts={5: "Que lugar lindo, olha a praia ali no fim"})
    short = _candidate("B009.T01", 0.0, 1.6, ["praia"], 3, description="Praia")
    context = context.model_copy(update={"candidates": [*context.candidates, short]})
    run, _ = _ai(context, shots, _decision([_cov("S005", "S005", "B009.T01", speech="a praia", image="praia")]))
    (c,) = [c for c in run.coverages if c.kind == "conteudo"]
    # S005: frames 360–435 (2,5 s); a tomada dá 48 frames, centrados em "a praia" (perto de 60% do texto).
    assert c.end_frame - c.start_frame == 48 and 375 <= c.start_frame <= 390
    assert any("encurtada; as tomadas cobrem 1,6s de 2,5s" in n.message for n in c.adjustments)


def test_coverage_touching_bad_image_is_limited_to_the_bad_seconds(tmp_path):
    context, _, shots = _context(tmp_path, texts={1: "Estamos na van, olha o lago.", 2: "A estrada vai até o lago."})
    run, _ = _ai(context, shots, _decision([_cov("S001", "S002", "B001.T01", speech="o lago", image="lago")]))
    (c,) = [c for c in run.coverages if c.kind == "conteudo"]
    assert (c.start_frame, c.end_frame, c.bad_spans) == (82, 150, ["O1"])  # S001–S002 = 60–150; ruim = 82–150
    assert any("limitada aos segundos com imagem ruim (O1, 2,3s)" in n.message for n in c.adjustments)


def test_total_coverage_limit_drops_the_weakest_and_warns(tmp_path, monkeypatch):
    assert MAX_COVERAGE_RATIO == 0.30
    monkeypatch.setattr(service, "MAX_COVERAGE_RATIO", 0.15)  # 15 s de espinha → 2,25 s
    context, _, shots = _context(tmp_path, texts={5: "Olha a praia."})
    run, _ = _ai(context, shots, _decision([LAKE, _cov("S005", "S005", "B001.T01", speech="a praia", image="praia")]))
    # Sai primeiro a que não toca imagem ruim (S005, 2,5 s); fica a do lago sobre a tela uniforme (2 s).
    assert [c.from_sentence for c in run.coverages] == ["S004"] and run.covered_sec == pytest.approx(2.0)
    assert [d.kind for d in run.discarded] == ["limite de cobertura"]
    assert any("ficou de fora pelo limite de cobertura (15% do vídeo, 2,2s)" in w for w in run.warnings)


# --- tela uniforme e imagem ruim sem cobertura ---


def test_uniform_screen_without_speech_evidence_only_with_a_related_shot(tmp_path):
    context, _, shots = _context(tmp_path)
    # Sem evidência de fala sobre a tela uniforme: B001.T02 não tem relação com P002 → descartado.
    unrelated, _ = _ai(context, shots, _decision([_cov("S004", "S005", "B001.T02")]), _decision())
    assert unrelated.discarded[0].kind == "evidência" and "nenhuma tomada relacionada à tela uniforme (O2)" in unrelated.discarded[0].message
    # P003.T01 é vizinha de P002 no catálogo: tapa só os segundos da tela uniforme, mesmo pedido S004–S005.
    run, _ = _ai(context, shots, _decision([_cov("S004", "S005", "P003.T01")]))
    patch = next(c for c in run.coverages if c.kind == "tela_uniforme")
    assert (patch.start_frame, patch.end_frame, patch.shot_ids) == (285, 345, ["P003.T01"])
    assert "gravada logo depois de P002" in patch.evidence.relation


def test_without_ai_only_uniform_screens_are_patched_and_bad_image_is_kept_with_a_warning(tmp_path):
    context, _, shots = _context(tmp_path)
    run = run_coverage(None, context, shots)
    assert run.origin == "fallback" and "desligada" in run.origin_reason and run.topics_origin == "fallback"
    assert [(c.kind, c.shot_ids, c.origin, c.start_frame, c.end_frame) for c in run.coverages] == [
        ("tela_uniforme", ["P003.T01"], "fallback", 285, 345),
    ]
    assert {o.span: o.status for o in run.span_outcomes} == {"O1": "mantido", "O2": "coberto"}
    assert any(w.startswith("imagem ruim mantida em O1 (S001–S002, 2,3s, imagem inutilizável)") for w in run.warnings)
    assert run.plan[0] == "S000–S002 · clipe P001 · sem cobertura (nenhuma imagem que combine com a fala) · imagem ruim mantida (O1, 2,3s)"


def test_default_mock_covers_the_lake_sentence_with_the_lake_shot(tmp_path):
    context, _, shots = _context(tmp_path)
    director = MockCoverageDirector()
    run = run_coverage(director, context, shots)
    assert run.origin == "ia" and len(director.calls) == 1 and run.topics_origin == "ia"
    (c,) = run.coverages
    assert (c.from_sentence, c.shot_ids, c.evidence.speech, c.evidence.image) == ("S004", ["P003.T01"], "lago", "lago")
    # Nada da van é tapado (nenhuma imagem de van): a pessoa fica na tela, e o trecho ruim fica com aviso.
    assert run.plan[0].startswith("S000–S002 · clipe P001 · sem cobertura")


# --- respiros entre blocos ---


def test_breaths_only_between_topic_blocks_with_a_related_shot(tmp_path):
    context, _, shots = _context(tmp_path)
    breaths = [
        BreathChoice(after_sentence="S000", shot_id="B001.T02", size="short", image_evidence="praia", reason="pausa"),
        BreathChoice(after_sentence="S001", shot_id="B001.T02", size="short", image_evidence="praia", reason="troca"),
        BreathChoice(after_sentence="S002", shot_id="B001.T01", size="short", image_evidence="lago", reason="vem o lago"),
    ]
    run, _ = _ai(context, shots, _decision(breaths=breaths), _decision())
    assert [d.kind for d in run.discarded[:2]] == ["respiro fora da troca de assunto", "respiro sem relação"]
    assert "S000 não fecha um bloco de assunto" in run.discarded[0].message
    (b,) = run.breaths
    assert (b.after_sentence, b.shot_id, b.relation) == ("S002", "B001.T01", "'lago' ↔ A3 (chegada no lago)")


def test_breath_does_not_take_the_only_shot_that_patches_a_uniform_screen(tmp_path):
    context, _, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S002", shot_id="P003.T01", size="short", image_evidence="lago", reason="lago")
    run, _ = _ai(context, shots, _decision(breaths=[breath]), _decision())
    assert run.discarded[0].kind == "reservada para tela uniforme" and not run.breaths
    assert [(c.kind, c.shot_ids) for c in run.coverages] == [("tela_uniforme", ["P003.T01"])]


# --- nova tentativa ---


def test_retry_when_more_than_half_is_discarded_with_only_the_problems(tmp_path):
    context, _, shots = _context(tmp_path)
    bad = [_cov("S000", "S000", "B001.T01", image="lago"), _cov("S005", "S005", "B001.T02", speech="lindo", image="praia")]
    run, director = _ai(context, shots, _decision([LAKE, *bad]), _decision())
    assert len(director.calls) == 2
    retry = director.calls[1]
    assert "sem evidência de fala" in retry.previous_error and "2 de 3 itens descartados" in retry.previous_error
    assert retry.accepted_items == ["cobertura S004–S004: P003.T01"]
    prompt = build_coverage_prompt(retry)
    assert "NOVA TENTATIVA" in prompt and "- cobertura S004–S004: P003.T01" in prompt
    assert [a.success for a in run.attempts] == [False, True] and run.origin == "ia"


def test_unreadable_answer_twice_falls_back_to_uniform_patches_only(tmp_path):
    context, _, shots = _context(tmp_path)
    error = AIResponseValidationError("coverage", details="JSON inválido")
    run, director = _ai(context, shots, error, error)
    assert len(director.calls) == 2 and run.origin == "fallback" and run.responses == [None, None]
    assert any("Cobertura da IA ignorada: JSON inválido" in w for w in run.warnings)
    assert [(c.kind, c.shot_ids) for c in run.coverages] == [("tela_uniforme", ["P003.T01"])]


def test_old_answer_shapes_are_still_read():
    choice = CoverageChoice.model_validate({"from_sentence": "S001", "to_sentence": "S002", "shot_id": "B001.T01", "reason": "x"})
    assert choice.shot_ids == ["B001.T01"] and choice.speech_evidence == ""
    assert "shot_id" not in CoverageChoice.model_json_schema()["properties"]
    # Resposta da versão anterior (com window_decisions e windows): lida, campos novos vazios.
    old = CoverageDecision.model_validate({"coverages": [{**choice.model_dump(), "windows": ["O1"]}], "window_decisions": [], "summary": "x"})
    assert old.coverages[0].image_evidence == ""


def test_prompts_show_blocks_screen_camera_bad_image_and_evidence_rules(tmp_path):
    context, _, shots = _context(tmp_path)
    run, director = _ai(context, shots, _decision())
    topic_prompt = build_topic_prompt(director.topic_calls[0])
    assert "BLOCOS DE ASSUNTO" in topic_prompt and "TOMADAS CANDIDATAS" not in topic_prompt
    assert "   ~ tela: P001.T01 Mulher falando dentro da van [interior, veículo] (pessoa falando para a câmera)" in topic_prompt
    prompt = build_coverage_prompt(director.calls[0])
    assert "Do que o vídeo trata: viagem de van até um lago" in prompt
    assert "== A3 S004–S005 · chegada no lago (lugar/objeto: lago)" in prompt
    assert "   ! O1 imagem ruim aqui (2.3s, imagem inutilizável, S001–S002): prefira cobrir se houver imagem que combine" in prompt
    assert "O2 (S004, 2.0s): relacionadas: P003.T01 (gravada logo depois de P002)" in prompt
    assert "COPIADO LITERALMENTE" in prompt and "Na dúvida, NÃO cubra" in prompt and "no máximo 30% do vídeo" in prompt
    assert "B001.T01 b-roll, 6.0s, beleza 4, boa — Praia de areia na beira do lago [praia, lago]" in prompt
    assert "gravada logo" not in prompt.split("TOMADAS CANDIDATAS")[1]  # datas não confiáveis: sem proximidade
    assert "S003" not in prompt and "timestamps" in prompt


def test_coverage_params_inherit_the_director():
    same = CoverageParams.inherit("gemini", "gemini-x", "gemini:gemini-y", "high", 90.0, 2)
    assert (same.provider, same.model, same.fallback, same.thinking, same.timeout_sec) == ("gemini", "gemini-x", "gemini:gemini-y", "high", 90.0)
    other = CoverageParams.inherit("gemini", "gemini-x", "gemini:gemini-y", provider="openai", thinking="off")
    assert (other.provider, other.model, other.fallback, other.thinking) == ("openai", None, None, "off")
    assert CoverageParams.inherit("none").provider == "none" and CoverageParams.inherit("mock").provider == "mock"


# --- material real: vlog "Trilha La Islita" (tests/data/coverage_real) ---

REAL = Path(__file__).parent / "data" / "coverage_real"
_REAL_FPS = 30


@pytest.fixture(scope="module")
def real_material() -> tuple[VisualMaterial, dict]:
    material = VisualMaterial.model_validate_json((REAL / "material.json").read_text(encoding="utf-8"))
    return material, json.loads((REAL / "spine.json").read_text(encoding="utf-8"))


def _real_context(real_material) -> CoverageContext:
    material, spine = real_material
    kept = [CoverageSentence.model_validate(s) for s in spine["sentences"]]
    roles = coverage_roles([])
    screen = annotate_screen(kept, material)
    return CoverageContext(
        project_name="trilha", fps=spine["fps"], spine_duration_sec=spine["duration_sec"], sentences=kept,
        bad_spans=find_bad_spans(kept, material, spine["clips"], spine["fps"]), candidates=coverage_candidates(material, roles),
        screen_shots=screen, roles=roles,
    )


def _real_gemini() -> CoverageDecision:
    """Resposta crua do Gemini (gemini-3.6-flash) na execução real com o prompt da cobertura tolerante
    (`coverage_decision.json` copiado, sem caminhos): 12 coberturas e 7 respiros, sem evidências — entre
    elas, drone cobrindo todo o P001 (ônibus) e P008.T01 + 8 drones no tour do quarto (P007)."""
    run = json.loads((REAL / "gemini_run_semantica.json").read_text(encoding="utf-8"))
    return CoverageDecision.model_validate(run["responses"][0])


def _frames_of(context: CoverageContext, clip_id: str) -> tuple[int, int]:
    mine = [s for s in context.sentences if s.clip_id == clip_id]
    return mine[0].start_frame, mine[-1].end_frame


def test_real_bad_image_is_only_p007_t07_t08(real_material):
    context = _real_context(real_material)
    assert len(context.sentences) == 167 and len(context.candidates) == 90
    # Antes: 10 janelas (P001 inteiro, todo o tour do P007...). Agora só a imagem ruim de verdade.
    o1, o2 = context.bad_spans
    assert (o1.clip_id, o1.shot_ids, o1.uniform, o1.sentence_ids) == ("P007", ["P007.T07"], True, ["S022", "S023"])
    assert (o2.clip_id, o2.shot_ids, o2.uniform, o2.sentence_ids) == ("P007", ["P007.T08"], False, ["S023"])
    assert o1.duration_sec(_REAL_FPS) + o2.duration_sec(_REAL_FPS) == pytest.approx(11.4, abs=0.1)
    assert len(context.bad_spans) == 2


def test_real_gemini_answer_without_evidence_leaves_the_bus_and_the_room_tour_alone(real_material):
    material, _ = real_material
    context = _real_context(real_material)
    run, director = _ai(context, material.shots, _real_gemini(), _decision(), topics=None)
    shots = material.by_id()
    # As 11 coberturas sem evidência de fala saem; a do quarto (S021–S023, P008.T01 + 8 drones) vira só o tapa
    # da parede (P007.T07) com a tomada relacionada (P008, vizinha no catálogo): os drones saem.
    content = [c for c in run.coverages if c.kind == "conteudo"]
    assert not content and sum(1 for d in run.discarded if d.kind == "evidência de fala") == 11
    (patch,) = run.coverages
    assert patch.kind == "tela_uniforme" and patch.shot_ids == ["P008.T01"] and patch.bad_spans == ["O1"]
    o1 = context.bad_spans[0]
    assert o1.start_frame <= patch.start_frame and patch.end_frame <= o1.end_frame
    assert any("sem relação com a tela uniforme; retiradas D001.T10" in n.message for n in patch.adjustments)
    # Nada cobre o P001 (ônibus) nem o tour fora dos segundos ruins; nenhum drone no P001/P007.
    for clip in ("P001", "P007"):
        lo, hi = _frames_of(context, clip)
        for c in run.coverages:
            if c.start_frame < hi and lo < c.end_frame:
                assert c.kind == "tela_uniforme" and not any(shots[x].role == "drone" for x in c.shot_ids)
    assert {o.span: o.status for o in run.span_outcomes} == {"O1": "parcial", "O2": "mantido"}
    assert run.plan[0] == "S000–S001 · clipe P001 · sem cobertura (1 proposta(s) descartada(s): evidência de fala)"
    # Respiros só nas trocas de bloco (aqui, um bloco por clipe) e com relação com o bloco vizinho.
    ends = {t.to_sentence for t in run.topics}
    assert run.breaths and all(b.after_sentence in ends and b.relation for b in run.breaths)


def test_real_earlier_gemini_answer_also_covers_nothing_without_evidence(real_material):
    """A 1ª resposta real (cobertura obrigatória, reconstruída das mensagens de erro): 23 coberturas sem
    evidência, entre elas o drone no ônibus (S000–S001) e no tour do quarto."""
    material, _ = real_material
    earlier = CoverageDecision.model_validate_json((REAL / "gemini_attempt1.json").read_text(encoding="utf-8"))
    run, _ = _ai(_real_context(real_material), material.shots, earlier, _decision(), topics=None)
    assert not [c for c in run.coverages if c.kind == "conteudo"] and len(earlier.coverages) == 23
    assert [c.kind for c in run.coverages] == ["tela_uniforme"]


def test_real_default_mock_covers_only_with_verified_evidence(real_material):
    material, _ = real_material
    context = _real_context(real_material)
    run = run_coverage(MockCoverageDirector(), context, material.shots)
    order = {s.id: i for i, s in enumerate(context.sentences)}
    shots = material.by_id()
    content = [c for c in run.coverages if c.kind == "conteudo"]
    assert content and run.covered_sec <= MAX_COVERAGE_RATIO * context.spine_duration_sec
    for c in content:
        text = " ".join(context.sentences[k].text for k in range(order[c.from_sentence], order[c.to_sentence] + 1))
        assert ev.find_speech([text], c.evidence.speech) is not None
        assert all(ev.image_matches(c.evidence.image, shots[x].description, shots[x].tags) for x in c.shot_ids)
    # O ônibus (P001) não é coberto: nenhuma imagem do que é falado ali.
    lo, hi = _frames_of(context, "P001")
    assert not [c for c in run.coverages if c.start_frame < hi and lo < c.end_frame]
    # A parede do quarto (P007.T07) é tapada só com a tomada relacionada (P008.T01), não com um respiro.
    patch = next(c for c in run.coverages if c.kind == "tela_uniforme")
    assert patch.shot_ids == ["P008.T01"] and "P008.T01" not in {b.shot_id for b in run.breaths}


def test_real_prompts_fit_and_show_the_content(real_material):
    material, _ = real_material
    context = _real_context(real_material)
    run, director = _ai(context, material.shots, _decision(), topics=None)
    topic_prompt, prompt = build_topic_prompt(director.topic_calls[0]), build_coverage_prompt(director.calls[0])
    assert len(topic_prompt) < 30_000 and len(prompt) < 50_000
    assert "   ~ tela: P001.T01 Interior de uma van ou micro-ônibus de turismo" in topic_prompt
    assert "a montanha ali pro outro lado" in prompt  # texto quase inteiro (a evidência é copiada dele)
    assert "   ! O1 imagem ruim aqui (4.6s, tela uniforme (parede/tampa), S022–S023)" in prompt
    assert "O1 (S022–S023, 4.6s): relacionadas: P008.T01 (gravada logo depois de P007)" in prompt
    assert "P008.T01 principal sem fala, 3.4s, beleza 4, boa — Rua de cidade pequena" in prompt


# --- IDs → tempos ---


def _accepted(context, a: str, b: str, *shots: str, start: Optional[int] = None, end: Optional[int] = None, added=()) -> AcceptedCoverage:
    order = {s.id: s for s in context.sentences}
    return AcceptedCoverage(
        from_sentence=a, to_sentence=b, shot_ids=list(shots), reason="combina", added_shots=list(added),
        start_frame=order[a].start_frame if start is None else start, end_frame=order[b].end_frame if end is None else end,
    )


def _run(context, coverages=(), breaths=()) -> CoverageRun:
    run = CoverageRun(provider="mock", model_id="mock", bad_spans=context.bad_spans)
    run.coverages = list(coverages)
    run.breaths = [AcceptedBreath(**b.model_dump()) for b in breaths]
    return run


def test_breath_at_clip_end_shifts_everything_after(tmp_path):
    context, spine, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S002", shot_id="B001.T02", size="short", reason="troca de lugar")
    run = _run(context, [_accepted(context, "S004", "S005", "B001.T01")], [breath])
    result = assemble_coverage(spine, context, run, {s.id: s for s in shots})
    final = result.timeline
    assert final.total_frames == spine.total_frames + 60  # respiro curto = 2 s
    assert [i.label for i in final.video_main] == ["P001 #1", "P001 #2", "respiro R1 (B001.T02)", "P002 #1"]
    breath_item = final.video_main[2]
    assert is_breath_item(breath_item) and breath_item.use_audio and breath_item.clip_id == "B001"
    assert result.breaths[0].timeline_start == pytest.approx(9.0)  # fim do P001 na espinha
    # S004 começava no frame 285 da espinha; depois do respiro, 345 (11,5 s).
    overlay = final.video_overlay[0]
    assert overlay.timeline_start == pytest.approx(11.5) and final.item_frames(overlay) == 150  # S004–S005: 5 s
    assert result.coverages[0].timeline_end == pytest.approx((450 + 60 - 15) / FPS)
    # Trecho da tomada centrado no melhor trecho (F2–F3 = 3,0–5,4 s → centro 4,2 s) e limitado à tomada.
    piece = result.coverages[0].pieces[0]
    assert (piece.source_in, piece.source_out) == (pytest.approx(1.0), pytest.approx(6.0))
    assert shot_segment({s.id: s for s in shots}["B001.T01"], 60, FPS) == (pytest.approx(3.2), 60)
    assert final.audio_ambient and final.audio_ambient[0].volume < 0.2


def test_coverage_uses_the_frames_decided_by_the_code(tmp_path):
    context, spine, shots = _context(tmp_path)
    # Só os segundos ruins de S001–S002 (frames 82–150), não as frases inteiras (60–150).
    run = _run(context, [_accepted(context, "S001", "S002", "B001.T01", start=82, end=150)])
    result = assemble_coverage(spine, context, run, {s.id: s for s in shots})
    placed = result.coverages[0]
    assert (placed.timeline_start, placed.timeline_end) == (pytest.approx(82 / FPS, abs=1e-3), pytest.approx(5.0))
    assert result.timeline.item_frames(result.timeline.video_overlay[0]) == 68 and not result.warnings


def test_breath_between_sentences_without_a_cut_splits_the_item(tmp_path):
    context, spine, shots = _context(tmp_path)
    breath = BreathChoice(after_sentence="S000", shot_id="B001.T02", size="short", reason="pausa")
    result = assemble_coverage(spine, context, _run(context, [], [breath]), {s.id: s for s in shots})
    final = result.timeline
    first, breath_item, second = final.video_main[:3]
    assert final.item_frames(first) == 57 and is_breath_item(breath_item)
    assert second.source_in == pytest.approx(57 / FPS) and second.source_out == pytest.approx(4.0)
    placements = clip_placements(final)["P001"]
    assert [p.concat_frame for p in placements] == [0, 57, 120] and [p.start_frame for p in placements] == [0, 117, 180]


def test_coverage_time_is_split_between_its_shots_in_order(tmp_path):
    context, spine, shots = _context(tmp_path)
    by_id = {s.id: s for s in shots}
    run = _run(context, [_accepted(context, "S001", "S002", "B001.T02", "P003.T01", added=["P003.T01"])])
    result = assemble_coverage(spine, context, run, by_id)
    pieces = result.coverages[0].pieces
    assert [(p.shot_id, p.chained) for p in pieces] == [("B001.T02", False), ("P003.T01", True)]
    assert pieces[0].timeline_start == pytest.approx(2.0) and pieces[0].timeline_end == pieces[1].timeline_start
    assert pieces[1].timeline_end == pytest.approx(5.0)
    short = assemble_coverage(spine, context, _run(context, [_accepted(context, "S004", "S005", "B001.T02")]), by_id)
    assert any("ficam sem cobertura" in w for w in short.warnings)
    assert distribute(10, [100, 100]) == [5, 5] and distribute(10, [3, 100]) == [3, 7] and distribute(10, [2, 3]) == [2, 3]


def test_real_long_item_with_enough_shots_is_distributed(real_material):
    material, _ = real_material
    caps = [shot_frames(material.by_id()[x], 30) for x in ["D002.T15", "D002.T07", "D002.T08"]]
    assert distribute(687, caps) == [229, 229, 229] and max(caps) <= MAX_COVERAGE_SEC * 30


def test_subtitles_follow_each_range_when_a_breath_separates_them():
    words = [Word(text="Primeira.", start_sec=0.2, end_sec=0.8), Word(text="Segunda.", start_sec=2.2, end_sec=2.8)]
    transcript = Transcript(
        source_path=Path("a.mp4"), language="pt", duration_sec=3.0, model_used="mock",
        segments=[TranscriptSegment(id=0, start_sec=0.2, end_sec=2.8, text="Primeira. Segunda.", words=words)],
    )
    ranges = [TimeRange(start_sec=0.0, end_sec=1.0), TimeRange(start_sec=2.0, end_sec=3.0)]
    plain = build_clip_subtitle_blocks([ClipSubtitleSource(transcript=transcript, kept_ranges=ranges, offset_sec=0.0)], FPS)
    shifted = build_clip_subtitle_blocks(
        [ClipSubtitleSource(transcript=transcript, kept_ranges=ranges, offset_sec=0.0, range_starts_sec=[0.0, 3.0])], FPS
    )
    assert [b.text for b in shifted] == ["Primeira.", "Segunda."]
    assert shifted[1].start_sec == pytest.approx(plain[1].start_sec + 2.0)


def test_vlog_cobertura_recipe_and_vlog_unchanged():
    vlog = get_recipe("vlog")
    assert vlog.steps == ["catalog", "silences", "transcription", "director", "timeline", "render", "subtitles", "report"]
    cobertura = get_recipe("vlog_cobertura")
    assert cobertura.multi_clip and cobertura.steps == [
        "catalog", "silences", "transcription", "director", "timeline", "visual", "coverage", "render", "subtitles", "report",
    ]
    assert cobertura.dependencies("coverage") == ["visual", "timeline", "transcription", "director"]


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
    """visual.json sintético do catálogo real do projeto: a 2ª tomada do clip_a (3–6 s) é inutilizável; o
    b-roll é a praia com o mar."""
    catalog = ProjectCatalog.load(catalog_path(project_dir))
    by_id = {c.id: c for c in catalog.clips}
    assert set(by_id) == {"P001", "P002", "B001"}
    shots = {
        "P001": [_shot("P001", 1, 0.0, 3.0, has_speech=True), _shot("P001", 2, 3.0, 6.0, has_speech=True, usable="nao", issues=["obstruida"]),
                 _shot("P001", 3, 6.0, 8.0, has_speech=True)],
        "P002": [_shot("P002", 1, 0.0, 5.0, has_speech=True)],
        "B001": [_shot("B001", 1, 0.0, 3.0, beauty=4, tags=["praia", "mar"], description="Praia com o mar azul"),
                 _shot("B001", 2, 3.0, 6.0, beauty=3, tags=["praia", "mar"], description="Ondas do mar na praia")],
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


# A espinha do mock (retake da 1ª fala removido): S001 "Chegamos na praia.", S002 "O mar está lindo hoje."
# (P001), S003, S004 (P002).
E2E_TOPICS = TopicMap(
    summary="praia e almoço",
    blocks=[
        TopicBlock(from_sentence="S001", to_sentence="S002", topic="chegada na praia", place="praia"),
        TopicBlock(from_sentence="S003", to_sentence="S004", topic="almoço e volta ao hotel", place="hotel"),
    ],
)
E2E_DECISION = CoverageDecision(
    coverages=[CoverageChoice(from_sentence="S001", to_sentence="S001", shot_ids=["B001.T01"], speech_evidence="na praia",
                              image_evidence="praia", reason="a praia de que ela fala")],
    breaths=[BreathChoice(after_sentence="S002", shot_id="B001.T02", size="short", image_evidence="Ondas do mar na praia", reason="troca de lugar")],
    summary="Cobre a praia e respira na troca de assunto.",
)


def _director(decision=E2E_DECISION) -> MockCoverageDirector:
    return MockCoverageDirector(responses=[decision], topics=[E2E_TOPICS])


def _srt_starts(text: str) -> dict[str, float]:
    starts = {}
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        h, m, rest = lines[1].split(" --> ")[0].split(":")
        starts[" ".join(lines[2:])] = int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))
    return starts


@pytest.mark.slow
def test_edit_folder_with_coverage_shifts_subtitles_reports_and_versions(coverage_folder, tmp_path):
    with pytest.raises(CoverageError, match=r"rode `visual .*--receita vlog"):
        _pipeline(tmp_path).run(coverage_folder, PARAMS)
    project_dir, _ = catalog_location(coverage_folder.resolve(), tmp_path / "projects")
    _write_visual(project_dir)

    executor = FakeTimelineExecutor()
    director = _director()
    result = _pipeline(tmp_path, executor, director).run(coverage_folder, PARAMS)
    assert set(result.outcomes) == set(get_recipe("vlog_cobertura").steps)
    spine = Timeline.model_validate_json((project_dir / "timeline.json").read_text(encoding="utf-8"))
    final = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    assert executor.timelines == [final]
    run = result.coverage
    assert run.origin == "ia" and run.topics_origin == "ia" and [c.bad_spans for c in run.placed_coverages] == [["O1"]]
    breath = run.placed_breaths[0]
    assert final.total_frames == spine.total_frames + 60 and breath.relation.startswith("'Ondas do mar na praia' ↔ A1")
    assert len(final.video_overlay) == 1 and final.video_overlay[0].clip_id == "B001"
    assert [(a.clip_id, a.volume) for a in final.audio_ambient] == [("B001", 0.12)]

    spine_p002 = clip_placements(spine)["P002"][0].start_frame / spine.fps
    final_p002 = clip_placements(final)["P002"][0].start_frame / final.fps
    assert final_p002 == pytest.approx(spine_p002 + 2.0) and breath.timeline_start == pytest.approx(spine_p002)
    starts = _srt_starts(result.subtitles_path.read_text(encoding="utf-8"))
    assert final_p002 <= starts["Agora vamos almoçar."] < final_p002 + 0.3
    diff = SpineDiff.model_validate_json((project_dir / "edit_diff.json").read_text(encoding="utf-8"))
    assert diff.final_duration_sec == pytest.approx(final.duration_sec, abs=0.001)

    # Relatório, plano, linha do tempo e decisão.
    report = (project_dir / "edit_report.md").read_text(encoding="utf-8")
    assert "| Receita | vlog_cobertura" in report and "## Cobertura" in report and "### Plano por bloco de assunto" in report
    assert "- S001–S002 · chegada na praia · coberto 1× (B001.T01 'na praia' ↔ 'praia') · respiro depois (B001.T02)" in report
    assert "| O1 | P001 | S001 | imagem inutilizável |" in report and "'na praia' ↔ 'praia' (forte)" in report
    assert "| R1 | S002 | B001.T02 | curto (2.0s)" in report
    plan = (project_dir / "coverage_plan.md").read_text(encoding="utf-8")
    assert result.plan_path == project_dir / "coverage_plan.md" and "chegada na praia" in plan
    svg = (project_dir / "edit_timeline.svg").read_text(encoding="utf-8")
    assert 'class="overlay" data-clip-id="B001"' in svg and 'class="item breath"' in svg
    saved = CoverageRun.model_validate_json((project_dir / "coverage_decision.json").read_text(encoding="utf-8"))
    assert saved.topic_responses == [E2E_TOPICS] and saved.responses == [E2E_DECISION]

    v1 = project_dir / "versions" / "v1"
    assert (v1 / "coverage_timeline.json").is_file() and (v1 / "coverage_decision.json").is_file()
    again = _pipeline(tmp_path, executor, director).run(coverage_folder, PARAMS)
    assert set(again.outcomes.values()) == {"cached"} and len(director.calls) == 1 and again.version == 1

    (project_dir / "coverage_timeline.json").unlink()
    no_breath = E2E_DECISION.model_copy(update={"breaths": []})
    redo = _pipeline(tmp_path, executor, _director(no_breath)).run(coverage_folder, PARAMS)
    assert redo.outcomes["coverage"] == "ran" and redo.outcomes["render"] == "ran" and redo.outcomes["subtitles"] == "ran"
    assert redo.final_duration_sec == pytest.approx(spine.duration_sec, abs=0.001)

    other = _pipeline(tmp_path, executor, None).run(coverage_folder, PARAMS, coverage=CoverageParams(provider="none"))
    assert other.outcomes["timeline"] == "cached" and other.outcomes["coverage"] == "ran"
    assert other.coverage.origin == "fallback" and not other.coverage.placed_coverages and not other.coverage.placed_breaths
    manifest = SpineManifest.load(project_dir / "project.json")
    assert manifest.stages["coverage"].metrics["origin"] == "fallback"


@pytest.mark.slow
def test_plan_only_stops_after_coverage_without_rendering(coverage_folder, tmp_path, capsys, monkeypatch):
    projects = tmp_path / "projects"
    project_dir, _ = catalog_location(coverage_folder.resolve(), projects)
    from ai_video_editor.pipeline.clips import load_or_build_catalog

    load_or_build_catalog(coverage_folder.resolve(), projects, lambda: _pipeline(tmp_path)._catalog_builder())
    _write_visual(project_dir)
    executor = FakeTimelineExecutor()
    result = _pipeline(tmp_path, executor, _director()).run(coverage_folder, PARAMS, plan_only=True)
    assert executor.timelines == [] and result.plan_only and result.version is None
    assert set(result.outcomes) == {"catalog", "silences", "transcription", "director", "timeline", "visual", "coverage"}
    assert not (project_dir / "versions").exists() and not result.output_path.exists()
    assert result.final_duration_sec == pytest.approx(result.coverage.final_duration_sec)
    assert "chegada na praia" in (project_dir / "coverage_plan.md").read_text(encoding="utf-8")
    with pytest.raises(RecipeError, match="--so-plano precisa da cobertura"):
        SpinePipeline(projects_root=projects, recipe="vlog").run(coverage_folder, PARAMS, plan_only=True)
    # CLI: --so-plano com o mock imprime uma linha por bloco e não renderiza (transcrição falsa, sem Whisper).
    import ai_video_editor.cli as cli

    class FakeSpine(SpinePipeline):
        def __init__(self, **kwargs):
            kwargs.pop("render_executor")
            super().__init__(
                **kwargs, transcription_provider=FolderTranscriber(), speech_detector=NamedSpeechDetector(),
                render_executor=executor,
            )

    monkeypatch.setattr(cli, "SpinePipeline", FakeSpine)
    code = handle_edit(build_parser().parse_args(
        ["edit", str(coverage_folder), "--transcribe", "--director", "mock", "--coverage-provider", "mock",
         "--so-plano", "--projects-dir", str(projects), "--no-live"]
    ))
    out = capsys.readouterr().out
    assert code == 0 and "Plano da cobertura (sem render)" in out and "Edição concluída" not in out
    assert "  S001–S002 · clipe P001 · " in out


def test_cli_so_plano_needs_a_recipe_with_coverage(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    args = build_parser().parse_args(["edit", str(video), "--so-plano", "--projects-dir", str(tmp_path / "p")])
    assert args.so_plano and handle_edit(args) == 1
    assert "--so-plano mostra o plano da cobertura" in capsys.readouterr().err


@pytest.mark.slow
def test_cli_picks_vlog_cobertura_when_the_folder_has_visual(coverage_folder, tmp_path, capsys):
    projects = tmp_path / "projects"
    assert default_edit_recipe(coverage_folder, True, True, projects) == "vlog"
    project_dir, _ = catalog_location(coverage_folder.resolve(), projects)
    from ai_video_editor.pipeline.clips import load_or_build_catalog

    load_or_build_catalog(coverage_folder.resolve(), projects, lambda: _pipeline(tmp_path)._catalog_builder())
    _write_visual(project_dir)
    assert default_edit_recipe(coverage_folder, True, True, projects) == "vlog_cobertura"
    assert default_edit_recipe(coverage_folder, True, False, projects) == "vlog"
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
        recipe="vlog_cobertura", coverage_director=_director(),
    ).run(coverage_folder, PARAMS)
    spine = Timeline.model_validate_json((project_dir / "timeline.json").read_text(encoding="utf-8"))
    final = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    run = result.coverage
    breath, cover = run.placed_breaths[0], run.placed_coverages[0]
    assert result.final_duration_sec == pytest.approx(spine.duration_sec + breath.duration_sec, abs=1 / final.fps)

    video = result.output_path
    r, g, b = _frame_rgb(video, (cover.timeline_start + cover.timeline_end) / 2)
    assert r > 180 and g < 70 and b < 70
    r, g, b = _frame_rgb(video, breath.timeline_start + breath.duration_sec / 2)
    assert r > 180 and g < 70 and b < 70
    r, g, b = _frame_rgb(video, cover.timeline_end + 0.3)
    assert not (r > 180 and g < 70 and b < 70)
    assert _rms(video, cover.timeline_start + 0.2, 0.8) > 0.05
    assert _rms(video, breath.timeline_start + 0.3, breath.duration_sec - 0.6) < 0.01
