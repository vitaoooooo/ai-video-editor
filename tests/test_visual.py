"""Testes do módulo `visual/` e do comando `visual` (sem rede: FFmpeg sintético e IA simulada)."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from ai_video_editor import cli
from ai_video_editor.catalog import CatalogBuilder, MockSpeechDetector
from ai_video_editor.intelligence.exceptions import AIProviderError, AIResponseValidationError
from ai_video_editor.llm import ModelChain, ModelSpec
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.visual import ProjectVisual, VisualBuilder, VisualDescriber, mock_visual_responder
from ai_video_editor.visual import builder as builder_module
from ai_video_editor.visual.describer import (
    BATCH_MAX_SHOTS,
    MAX_BATCH_ATTEMPTS,
    ShotRequest,
    build_visual_prompt,
    estimate_tokens,
    make_batches,
    prompt_sections,
    validate_batch,
)
from ai_video_editor.visual.report import attention_ranges, build_visual_markdown
from ai_video_editor.visual.schema import (
    ClipVisual,
    Shot,
    ShotAnalysis,
    ShotDescription,
    ShotFrame,
    ShotMeasures,
    VisualBatchResponse,
)
from ai_video_editor.visual.shots import (
    MAX_SHOT_SEC,
    MOTION_FLOOR,
    Sample,
    ShotSpan,
    extract_frames,
    frame_times,
    mark_shaky,
    measure_shot,
    motion_threshold,
    parse_metadata_file,
    scan_clip,
    split_shots,
)

FIXED_MTIME = 1_700_000_000


# --- tomadas (puro) ---


def test_split_shots_cuts_and_long_windows():
    spans = split_shots(30.0, [5.0, 5.4, 29.6])

    # 5.4 fica a menos de 1 s do corte anterior e 29.6 a menos de 1 s do fim: ignorados
    assert [s.boundary for s in spans][:2] == ["inicio", "cena"]
    assert spans[0].end_sec == 5.0
    long_part = [s for s in spans if s.start_sec >= 5.0]
    assert len(long_part) == 4  # 25 s em janelas de até 8 s
    assert all(s.end_sec - s.start_sec <= MAX_SHOT_SEC + 1e-6 for s in spans)
    assert [s.boundary for s in long_part] == ["cena", "janela", "janela", "janela"]
    assert spans[-1].end_sec == 30.0
    assert all(a.end_sec == b.start_sec for a, b in zip(spans, spans[1:]))


def test_split_shots_short_clip_and_empty():
    assert split_shots(0.0, []) == []
    spans = split_shots(3.0, [])
    assert len(spans) == 1 and spans[0].boundary == "inicio"


def test_frame_times_at_10_50_90_percent_inside_clip():
    assert frame_times(ShotSpan(start_sec=10.0, end_sec=20.0, boundary="cena"), 60.0) == [11.0, 15.0, 19.0]
    assert frame_times(ShotSpan(start_sec=0.0, end_sec=4.0, boundary="inicio"), 3.6)[-1] <= 3.55


def test_parse_metadata_file():
    text = (
        "frame:0    pts:51200   pts_time:4\nlavfi.scene_score=0.82\n"
        "frame:1    pts:102400  pts_time:8.5\nlavfi.blur=nan\nlavfi.signalstats.YAVG=210\n"
    )
    blocks = parse_metadata_file(text)
    assert [t for t, _ in blocks] == [4.0, 8.5]
    assert blocks[1][1] == {"blur": "nan", "signalstats.YAVG": "210"}


def test_measure_shot_flags_uniform_blur_and_ignores_diff_across_cut():
    flat = [Sample(time_sec=t / 2, blur=None, yavg=210, ylow=208, yhigh=212, ydif=0) for t in range(8)]
    flat[0].ydif = 90  # diferença com o quadro de antes do corte
    uniform = measure_shot(ShotSpan(start_sec=0.0, end_sec=4.0, boundary="cena"), flat)
    assert uniform.uniform and not uniform.shaky and uniform.blur is None
    assert "tela uniforme" in uniform.hints

    blurry = [Sample(time_sec=t / 2, blur=15.0, yavg=120, ylow=40, yhigh=210, ydif=30) for t in range(8)]
    measures = measure_shot(ShotSpan(start_sec=0.0, end_sec=4.0, boundary="janela"), blurry)
    assert measures.blurry and not measures.shaky and not measures.uniform  # tremor só no projeto
    assert measures.hints == ["borrão alto"]


def test_measure_shot_wall_texture_is_uniform():
    """Parede real (P007): faixa de luma entre 0,09 e 0,12, acima do limite antigo de 0,10."""
    wall = [Sample(time_sec=t / 2, blur=5.0, yavg=120, ylow=100, yhigh=100 + 255 * s, ydif=5)
            for t, s in enumerate([0.09, 0.11, 0.12, 0.10, 0.12, 0.11, 0.25, 0.09])]
    assert measure_shot(ShotSpan(start_sec=0.0, end_sec=4.0, boundary="janela"), wall).uniform


def _motions(values: list[float]) -> list[ShotMeasures]:
    return [ShotMeasures(samples=1, brightness=0.5, motion=v, luma_spread=0.4) for v in values]


def test_shaky_is_relative_to_project_percentile():
    measures = _motions([0.09, 0.10, 0.11, 0.12, 0.10, 0.09, 0.11, 0.10, 0.12, 0.30])
    threshold = mark_shaky(measures)
    # percentil 90 interpolado: 0,12 + 0,1 × (0,30 − 0,12)
    assert threshold == pytest.approx(0.138)
    assert [m.shaky for m in measures] == [False] * 9 + [True]
    assert measures[-1].hints == ["movimento alto"]


def test_shaky_never_below_floor():
    measures = _motions([0.01, 0.02, 0.02, 0.03, 0.05])
    assert mark_shaky(measures) == MOTION_FLOOR
    assert not any(m.shaky for m in measures)
    assert motion_threshold([]) == MOTION_FLOOR


def test_shaky_all_equal_or_tiny_project_marks_nothing():
    same = _motions([0.2] * 6)
    assert mark_shaky(same) == pytest.approx(0.2)
    assert not any(m.shaky for m in same)
    single = _motions([0.5])
    assert mark_shaky(single) == 0.5 and not single[0].shaky
    pair = _motions([0.05, 0.3])
    mark_shaky(pair)
    assert [m.shaky for m in pair] == [False, True]


# --- lote e validação (puro) ---


def _shot(shot_id: str, has_speech=True, uniform=False) -> Shot:
    measures = ShotMeasures(samples=4, brightness=0.5, motion=0.01, luma_spread=0.02 if uniform else 0.6,
                            uniform=uniform)
    return Shot(
        id=shot_id, index=int(shot_id[-2:]), start_sec=0.0, end_sec=4.0, boundary="inicio",
        frames=[ShotFrame(id=f"{shot_id}.F{n}", position=n, time_sec=n, file=f"frames/{shot_id}.F{n}.jpg")
                for n in (1, 2, 3)],
        measures=measures, has_speech=has_speech,
    )


def _request(shot: Shot, tmp_path: Path) -> ShotRequest:
    paths = []
    for frame in shot.frames:
        path = tmp_path / f"{frame.id}.jpg"
        path.write_bytes(b"\xff\xd8jpeg")
        paths.append(path)
    return ShotRequest(shot=shot, clip_role="principal", image_paths=paths)


def _item(shot_id: str, best=None, usable="sim") -> dict:
    return {
        "shot_id": shot_id, "description": "Lago ao entardecer.", "tags": ["Lago", "paisagem", "lago"],
        "face_to_camera": False, "people_count": 0, "issues": ["nenhuma"], "usable": usable,
        "best_frames": best if best is not None else [f"{shot_id}.F1", f"{shot_id}.F3"],
        "beauty": 4, "summary_for_editor": "Boa para abrir.",
    }


def test_validate_batch_accepts_and_normalizes(tmp_path):
    batch = [_request(_shot("P001.T01"), tmp_path)]
    accepted, problems = validate_batch(batch, VisualBatchResponse.model_validate({"shots": [_item("P001.T01")]}))
    assert problems == []
    assert accepted["P001.T01"].tags == ["lago", "paisagem"]
    assert accepted["P001.T01"].best_frames == ["P001.T01.F1", "P001.T01.F3"]


def test_validate_batch_rejects_missing_duplicate_unknown_and_bad_frames(tmp_path):
    batch = [_request(_shot(f"P001.T0{n}"), tmp_path) for n in (1, 2, 3, 4, 5)]
    response = VisualBatchResponse.model_validate({"shots": [
        _item("P001.T01"),
        _item("P001.T02"), _item("P001.T02"),                       # duplicada
        _item("P001.T03", best=["P001.T01.F1"]),                    # miniatura de outra tomada
        _item("P001.T04", best=["P001.T04.F3", "P001.T04.F1"]),     # fora de ordem
        _item("P009.T01"),                                          # ID inexistente
    ]})  # P001.T05 faltando

    accepted, problems = validate_batch(batch, response)

    assert set(accepted) == {"P001.T01"}
    text = "\n".join(problems)
    for expected in ("P009.T01", "P001.T02", "P001.T03", "P001.T04", "P001.T05"):
        assert expected in text


def test_validate_batch_empty_best_frames_only_when_unusable(tmp_path):
    batch = [_request(_shot("P001.T01"), tmp_path), _request(_shot("P001.T02"), tmp_path)]
    response = VisualBatchResponse.model_validate({"shots": [
        _item("P001.T01", best=[], usable="nao"), _item("P001.T02", best=[], usable="sim"),
    ]})
    accepted, problems = validate_batch(batch, response)
    assert set(accepted) == {"P001.T01"} and "P001.T02" in problems[0]


def test_make_batches_and_estimate(tmp_path):
    requests = [_request(_shot(f"P001.T{n:02d}"), tmp_path) for n in range(1, 20)]
    batches = make_batches(requests)
    assert [len(b) for b in batches] == [8, 8, 3]
    assert all(len(b) <= BATCH_MAX_SHOTS for b in batches)
    calls, tokens_in, tokens_out = estimate_tokens(19, 57, "low")
    assert calls == 3
    assert tokens_in == 57 * 280 + 3 * 900 + 19 * 60
    assert tokens_out == 19 * 150
    assert estimate_tokens(0, 0, "auto") == (0, 0, 0)


def test_prompt_lists_ids_and_hints_without_clip_times(tmp_path):
    shot = _shot("P007.T03", uniform=True).model_copy(update={"start_sec": 61.0, "end_sec": 65.0})
    prompt = build_visual_prompt([_request(shot, tmp_path)], previous_error="- Tomadas sem resposta: P007.T03.")
    assert prompt_sections(prompt) == [("P007.T03", "tela uniforme")]
    assert "P007.T03.F1, P007.T03.F2, P007.T03.F3" in prompt
    assert "1:01" not in prompt and "61" not in prompt  # a IA não vê onde a tomada está no clipe
    assert "resposta anterior foi recusada" in prompt


def test_prompt_mentions_measures_only_when_flagged(tmp_path):
    clean = _shot("P001.T01")
    flagged = _shot("P001.T02", uniform=True)
    prompt = build_visual_prompt([_request(clean, tmp_path), _request(flagged, tmp_path)])
    assert prompt_sections(prompt) == [("P001.T01", ""), ("P001.T02", "tela uniforme")]
    assert prompt.count("medidas: ") == 1 and "nada a notar" not in prompt
    assert "0.4" not in prompt  # nenhum valor bruto


# --- descritor com IA simulada ---


def _describer(responses) -> tuple[VisualDescriber, MockLLMClient]:
    mock = MockLLMClient(responses=responses)
    chain = ModelChain(
        [ModelSpec(provider="mock", model_id="mock")],
        retry=RetryPolicy(max_retries=0, sleep=lambda _: None),
        clients={"mock": mock},
    )
    return VisualDescriber(chain), mock


def test_describer_retries_only_failed_shots_with_the_error(tmp_path):
    batch = [_request(_shot("P001.T01"), tmp_path), _request(_shot("P001.T02"), tmp_path)]
    describer, mock = _describer([
        {"shots": [_item("P001.T01"), _item("P009.T09")]},
        {"shots": [_item("P001.T02")]},
    ])

    outcome = describer.describe_batch(batch)

    assert set(outcome.analyses) == {"P001.T01", "P001.T02"} and not outcome.failed
    assert len(mock.calls) == 2
    retry_prompt = mock.calls[1][0]
    assert "P009.T09" in retry_prompt and "Tomadas sem resposta: P001.T02" in retry_prompt
    assert [s for s, _ in prompt_sections(retry_prompt)] == ["P001.T02"]
    assert [i.label for i in mock.image_calls[1]] == ["P001.T02.F1", "P001.T02.F2", "P001.T02.F3"]
    assert outcome.images_sent == 9 and outcome.api_calls == 2


def test_describer_marks_persistent_failures_without_analysis(tmp_path):
    batch = [_request(_shot("P001.T01"), tmp_path), _request(_shot("P001.T02"), tmp_path)]
    describer, mock = _describer([
        {"shots": [_item("P001.T01")]},
        AIResponseValidationError("mock", details="JSON quebrado"),
        {"shots": [_item("P001.T01")]},  # de novo a tomada errada
    ])

    outcome = describer.describe_batch(batch)

    assert set(outcome.analyses) == {"P001.T01"}
    assert "P001.T02" in outcome.failed and "3 tentativas" in outcome.failed["P001.T02"]
    assert len(mock.calls) == MAX_BATCH_ATTEMPTS


def test_describer_api_error_marks_batch_and_gives_up_after_repeated_failures(tmp_path):
    error = AIProviderError("Serviço fora", provider="mock")
    describer, mock = _describer([error])
    batches = [[_request(_shot(f"P001.T0{n}"), tmp_path)] for n in (1, 2, 3)]

    outcomes = [describer.describe_batch(b) for b in batches]

    assert all(o.failed and not o.analyses for o in outcomes)
    assert "erro da IA" in outcomes[0].failed["P001.T01"]
    assert "não enviada" in outcomes[2].failed["P001.T03"]
    assert len(mock.calls) == 2  # o terceiro lote não chega a ser enviado


# --- relatório ---


def test_attention_lists_speech_shots_with_obstruction(tmp_path):
    bad = ShotAnalysis(description="Parede amarela.", tags=["detalhe"], face_to_camera=False, people_count=0,
                       issues=["obstruida"], usable="nao", best_frames=[], beauty=1, summary_for_editor="Cobrir.")
    good = bad.model_copy(update={"issues": ["nenhuma"], "usable": "sim", "best_frames": ["x"]})
    shots = [_shot("P001.T01"), _shot("P001.T02"), _shot("P001.T03"), _shot("P001.T04", has_speech=False)]
    for shot, analysis in zip(shots, (good, bad, bad, bad)):
        shot.analysis, shot.status = analysis, "analisada"
    clips = [
        ClipVisual(clip_id="P001", role="principal", relative_path="p/a.mp4", duration_sec=16, shots=shots),
        ClipVisual(clip_id="B001", role="broll", relative_path="b/a.mp4", duration_sec=4,
                   shots=[_shot("B001.T01").model_copy(update={"analysis": bad, "status": "analisada"})]),
    ]
    visual = ProjectVisual(project_dir=tmp_path, source_dir=tmp_path, roles=["principal", "broll"], clips=clips)

    ranges = attention_ranges(visual)
    markdown = build_visual_markdown(visual)

    assert [(c.clip_id, [s.id for s in group]) for c, group in ranges] == [("P001", ["P001.T02", "P001.T03"])]
    assert "P001.T02–T03" in markdown and "obstruída/parede" in markdown
    assert "| principal | 1 | 4 | 4 | 25% | 0 | 3 |" in markdown


# --- CLI (sem FFmpeg) ---


def test_parse_roles():
    assert cli._parse_roles(None) == ["principal", "drone", "pov", "broll", "outro"]
    assert cli._parse_roles("principal, b-roll,principal") == ["principal", "broll"]
    with pytest.raises(ValueError):
        cli._parse_roles("paisagem")


def test_cli_visual_options():
    args = cli.build_parser().parse_args(["visual", "pasta", "--so-estimar", "--papeis", "broll"])
    assert args.so_estimar and args.visual_provider == "gemini" and args.visual_thinking == "low"
    assert args.detalhe_imagem == "auto"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["visual", "pasta", "--visual-provider", "nao-existe"])


# --- ponta a ponta com FFmpeg ---


def _make_scene_clip(path: Path) -> Path:
    """4 s de padrão nítido, 10 s de parede amarela (fatiada em 2 janelas) e 4 s do padrão desfocado, com áudio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(get_ffmpeg_path())
    args = [
        ffmpeg, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=25:duration=4",
        "-f", "lavfi", "-i", "color=c=yellow:size=480x270:rate=25:duration=10",
        "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=25:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=18",
        "-filter_complex", "[2]boxblur=8[b];[0][1][b]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(args, capture_output=True, check=True)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return path


@pytest.fixture(scope="session")
def scene_folder(tmp_path_factory) -> Path:
    """Pasta de clipes sintética (somente leitura): um principal com cortes de cena."""
    root = tmp_path_factory.mktemp("visual") / "Trilha"
    _make_scene_clip(root / "Principal" / "a.mp4")
    return root


def _visual_builder(root: Path, responses=None, **kwargs) -> tuple[VisualBuilder, MockLLMClient]:
    mock = MockLLMClient(responses=responses or [mock_visual_responder])
    chain = ModelChain([ModelSpec(provider="mock", model_id="mock")], clients={"mock": mock})
    catalog_builder = CatalogBuilder(projects_root=root, speech_detector=MockSpeechDetector(segments=[(0.0, 18.0)]))
    builder = VisualBuilder(projects_root=root, describer=VisualDescriber(chain), catalog_builder=catalog_builder,
                            **kwargs)
    return builder, mock


@pytest.mark.slow
def test_visual_end_to_end_shots_measures_frames_and_report(tmp_path, scene_folder):
    builder, mock = _visual_builder(tmp_path / "projects")

    result = builder.build(scene_folder)

    visual = result.visual
    shots = visual.all_shots()
    assert [s.id for s in shots] == ["P001.T01", "P001.T02", "P001.T03", "P001.T04"]
    assert [s.boundary for s in shots] == ["inicio", "cena", "janela", "cena"]
    assert shots[1].start_sec == pytest.approx(4.0, abs=0.1) and shots[3].start_sec == pytest.approx(14.0, abs=0.1)
    assert shots[1].measures.uniform and shots[2].measures.uniform
    assert not shots[0].measures.uniform and not shots[0].measures.blurry
    assert shots[3].measures.blurry
    assert all(s.has_speech for s in shots)
    frame = shots[1].frames[1]
    assert frame.id == "P001.T02.F2" and frame.time_sec == pytest.approx(6.5, abs=0.1)

    visual_dir = result.project_dir / "visual"
    for shot in shots:
        for f in shot.frames:
            assert (visual_dir / f.file).is_file()
    assert all(s.status == "analisada" for s in shots)
    assert shots[1].analysis.issues == ["obstruida"]
    # a IA recebe as miniaturas rotuladas com os IDs
    assert [i.label for i in mock.image_calls[0]][:3] == ["P001.T01.F1", "P001.T01.F2", "P001.T01.F3"]

    loaded = ProjectVisual.load(visual_dir / "visual.json")
    assert loaded.schema_version == 1 and loaded.ai.effective_model == "mock:mock"
    markdown = (visual_dir / "visual.md").read_text(encoding="utf-8")
    assert "## Atenção" in markdown and "P001.T02–T03" in markdown
    assert "frames/P001.T01.F1.jpg" in markdown


def _make_clip(path: Path, video: str, duration: float, audio_sec: float | None = None) -> Path:
    """Clipe sintético de uma fonte lavfi, sem cortes; com áudio de `audio_sec` s, se pedido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [str(get_ffmpeg_path()), "-v", "error", "-y", "-f", "lavfi", "-i", f"{video}:duration={duration}"]
    if audio_sec is not None:
        args += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={audio_sec}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, capture_output=True, check=True)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return path


@pytest.mark.slow
def test_visual_continuous_clip_without_scene_change(tmp_path):
    """Tomada contínua (o caso comum do vlog): o ramo de cenas não pode ficar sem quadros."""
    folder = tmp_path / "Trilha"
    _make_clip(folder / "Principal" / "continua.mp4", "testsrc2=size=480x270:rate=25", 18, audio_sec=18)
    builder, _ = _visual_builder(tmp_path / "projects")

    result = builder.build(folder)

    clip = result.visual.clips[0]
    assert clip.error is None and result.visual.warnings == []
    shots = clip.shots
    assert [s.boundary for s in shots] == ["inicio", "janela", "janela"]
    assert [(s.start_sec, s.end_sec) for s in shots] == [(0.0, 6.0), (6.0, 12.0), (12.0, 18.0)]
    for shot in shots:
        assert shot.measures.samples == 12 and shot.measures.blur is not None
        assert shot.measures.brightness > 0 and not shot.measures.uniform
        assert all((result.project_dir / "visual" / f.file).is_file() for f in shot.frames)


@pytest.mark.slow
def test_visual_very_short_clip(tmp_path):
    folder = tmp_path / "Trilha"
    _make_clip(folder / "Principal" / "curto.mp4", "testsrc2=size=480x270:rate=25", 0.4)
    builder, _ = _visual_builder(tmp_path / "projects")

    clip = builder.build(folder).visual.clips[0]

    assert clip.error is None
    assert len(clip.shots) == 1 and clip.shots[0].measures.samples == 1
    assert clip.shots[0].measures.blur is not None


@pytest.mark.slow
def test_scan_clip_without_scene_change_has_no_cuts(tmp_path):
    clip = _make_clip(tmp_path / "parede.mp4", "color=c=yellow:size=320x240:rate=25", 3)

    scan = scan_clip(clip)

    assert scan.cuts == [] and len(scan.samples) == 6


@pytest.mark.slow
def test_extract_frames_after_last_video_frame_uses_last_frame(tmp_path):
    """Áudio mais longo que o vídeo: a duração do contêiner passa do último quadro."""
    clip = _make_clip(tmp_path / "a.mp4", "testsrc2=size=320x240:rate=10", 3, audio_sec=3.6)
    outputs = [tmp_path / f"F{i}.jpg" for i in range(1, 4)]

    assert extract_frames(clip, [0.3, 1.5, 3.5], outputs)
    assert all(o.is_file() for o in outputs)


@pytest.mark.slow
def test_visual_cache_skips_ffmpeg_and_ai(tmp_path, scene_folder, monkeypatch):
    root = tmp_path / "projects"
    builder, _ = _visual_builder(root)
    builder.build(scene_folder)

    def boom(*args, **kwargs):
        raise AssertionError("não devia rodar o FFmpeg de novo")

    monkeypatch.setattr(builder_module, "scan_clip", boom)
    monkeypatch.setattr(builder_module, "extract_frames", boom)
    again, mock = _visual_builder(root)
    result = again.build(scene_folder)

    assert mock.calls == []
    assert result.visual.clips[0].cached
    assert result.visual.ai.cached_shots == 4
    assert all(s.status == "analisada" for s in result.visual.all_shots())


@pytest.mark.slow
def test_visual_force_redoes_shots_but_keeps_ai_analyses(tmp_path, scene_folder):
    root = tmp_path / "projects"
    _visual_builder(root)[0].build(scene_folder)

    again, mock = _visual_builder(root)
    result = again.build(scene_folder, force=True)

    assert not result.visual.clips[0].cached
    assert mock.calls == []
    assert all(s.status == "analisada" for s in result.visual.all_shots())


@pytest.mark.slow
def test_visual_unanswered_shot_is_retried_on_next_run(tmp_path, scene_folder):
    root = tmp_path / "projects"

    def skip_last(prompt, images, schema):
        response = mock_visual_responder(prompt, images, schema)
        return VisualBatchResponse(shots=[s for s in response.shots if s.shot_id != "P001.T04"])

    builder, _ = _visual_builder(root, responses=[skip_last])
    first = builder.build(scene_folder).visual
    status = {s.id: s.status for s in first.all_shots()}
    assert status["P001.T04"] == "sem_analise" and status["P001.T01"] == "analisada"
    assert "sem análise" in (first.project_dir / "visual" / "visual.md").read_text(encoding="utf-8")

    again, mock = _visual_builder(root)
    second = again.build(scene_folder).visual
    assert len(mock.calls) == 1
    assert [s for s, _ in prompt_sections(mock.calls[0][0])] == ["P001.T04"]
    assert all(s.status == "analisada" for s in second.all_shots())


@pytest.mark.slow
def test_cli_visual_estimate_does_not_call_ai(tmp_path, scene_folder, monkeypatch, capsys):
    from ai_video_editor.llm.providers.gemini import GeminiLLMClient

    def no_call(*args, **kwargs):
        raise AssertionError("--so-estimar não pode chamar a IA")

    monkeypatch.setattr(GeminiLLMClient, "_request", no_call)
    monkeypatch.setattr("ai_video_editor.catalog.builder.FasterWhisperVAD", lambda: MockSpeechDetector())
    root = tmp_path / "projects"

    code = cli.handle_visual(cli.build_parser().parse_args(
        ["visual", str(scene_folder), "--so-estimar", "--no-live", "--projects-dir", str(root),
         "--detalhe-imagem", "low"]
    ))

    out = capsys.readouterr().out
    assert code == 0
    assert "A analisar: 4 tomadas, 12 imagens" in out
    assert "Chamadas: 1" in out and "280 por imagem" in out
    project = next(p for p in root.iterdir() if p.name.startswith("trilha"))
    assert not (project / "visual" / "visual.json").exists()


@pytest.mark.slow
def test_cli_visual_with_mock_provider(tmp_path, scene_folder, monkeypatch, capsys):
    monkeypatch.setattr("ai_video_editor.catalog.builder.FasterWhisperVAD", lambda: MockSpeechDetector())
    root = tmp_path / "projects"

    code = cli.handle_visual(cli.build_parser().parse_args(
        ["visual", str(scene_folder), "--visual-provider", "mock", "--no-live", "--projects-dir", str(root)]
    ))

    out = capsys.readouterr().out
    assert code == 0
    assert "Análise visual concluída" in out and "Modelo efetivo: mock:mock" in out
    data = json.loads(next(root.glob("trilha-*/visual/visual.json")).read_text(encoding="utf-8"))
    assert len(data["clips"][0]["shots"]) == 4
