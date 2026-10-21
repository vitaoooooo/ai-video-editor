"""Testes do modelo Timeline, da conversão EditPlan → Timeline e do achatamento (sem FFmpeg)."""

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_editor.core.schema import ClipSegment, EditPlan, LoudnessTarget
from ai_video_editor.core.timeline import (
    TIMELINE_SAMPLE_RATE,
    Timeline,
    flatten,
    timeline_from_edit_plan,
)

A = Path("a.mp4")
B = Path("b.mp4")
C = Path("c.mp4")


def _main(path=A, source_in=0.0, source_out=1.0, **kw):
    return {"source_path": path, "source_in": source_in, "source_out": source_out, **kw}


def _over(start, duration, path=B, source_in=0.0, **kw):
    return {"source_path": path, "source_in": source_in, "source_out": source_in + duration, "timeline_start": start, **kw}


def _timeline(main, overlay=(), ambient=(), fps=30, **kw) -> Timeline:
    return Timeline(
        output_path=Path("out.mp4"),
        fps=fps,
        video_main=list(main),
        video_overlay=list(overlay),
        audio_ambient=list(ambient),
        **kw,
    )


# --- modelo e validação ---


def test_defaults_and_duration():
    tl = _timeline([_main(source_out=2.0), _main(B, 1.0, 1.5)])
    assert tl.resolution == (1920, 1080)
    assert tl.fps == 30
    assert tl.schema_version == 1
    assert tl.total_frames == 75
    assert tl.duration_sec == pytest.approx(2.5)


def test_json_roundtrip():
    tl = _timeline(
        [_main(source_out=2.0, clip_id="A001", label="fala")],
        overlay=[_over(0.5, 1.0, clip_id="D003")],
        ambient=[_over(0.0, 1.0, path=C, volume=0.5, fade_in_sec=0.2)],
        loudness=LoudnessTarget.from_preset("voz"),
    )
    again = Timeline.model_validate(json.loads(tl.model_dump_json()))
    assert again == tl


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"main": [_main(source_in=2.0, source_out=1.0)]}, "tempos invertidos"),
        ({"main": [_main(source_in=1.0, source_out=1.01)]}, "menos de 1 frame"),
        ({"main": [_main(source_out=2.0)], "overlay": [_over(1.5, 1.0)]}, "além da duração"),
        ({"main": [_main(source_out=2.0)], "overlay": [_over(0.2, 0.01)]}, "menos de 1 frame"),
        (
            {"main": [_main(source_out=3.0)], "overlay": [_over(0.5, 1.0), _over(1.2, 0.5, clip_id="D9")]},
            "overlays sobrepostos: video_overlay[1] (D9)",
        ),
        (
            {"main": [_main(source_out=1.0), _main(B, 0.0, 1.0, timeline_start=1.5)]},
            "buraco na base",
        ),
        (
            {"main": [_main(source_out=1.0), _main(B, 0.0, 1.0, timeline_start=0.5)]},
            "sobreposição na base",
        ),
        ({"main": [_main(source_out=1.0)], "ambient": [_over(0.5, 1.0)]}, "audio_ambient[0]"),
        (
            {"main": [_main(source_out=2.0)], "ambient": [_over(0.0, 1.0, fade_in_sec=0.6, fade_out_sec=0.6)]},
            "fade in + fade out",
        ),
    ],
)
def test_validation_errors(kwargs, message):
    with pytest.raises(ValidationError) as exc:
        _timeline(kwargs["main"], kwargs.get("overlay", ()), kwargs.get("ambient", ()))
    assert message in str(exc.value)


def test_validation_rejects_odd_resolution_and_unknown_fields():
    with pytest.raises(ValidationError, match="pares"):
        _timeline([_main()], resolution=(641, 360))
    with pytest.raises(ValidationError):
        Timeline.model_validate({"output_path": "o.mp4", "video_main": [_main()], "segments": []})


def test_base_timeline_start_matching_position_is_accepted():
    # 0,51s quantiza para o mesmo frame que o fim do item anterior (frame 15 a 30 fps)
    tl = _timeline([_main(source_out=0.5), _main(B, 0.0, 1.0, timeline_start=0.51)])
    assert tl.total_frames == 45


def test_adjacent_overlays_are_valid():
    tl = _timeline([_main(source_out=3.0)], overlay=[_over(0.5, 1.0), _over(1.5, 1.0, path=C)])
    assert len(tl.video_overlay) == 2


def test_volume_bounds():
    with pytest.raises(ValidationError):
        _timeline([_main(volume=2.5)])
    with pytest.raises(ValidationError):
        _timeline([_main(source_out=2.0)], ambient=[_over(0.0, 1.0, volume=-0.1)])


# --- EditPlan → Timeline ---


def test_timeline_from_edit_plan():
    plan = EditPlan(
        title="aula",
        output_path=Path("o.mp4"),
        target_resolution=(1280, 720),
        target_fps=25,
        audio_fade_sec=0.05,
        loudness=LoudnessTarget.from_preset("youtube"),
        segments=[
            ClipSegment(source_path=A, start_sec=1.0, end_sec=2.0, volume=0.8),
            ClipSegment(source_path=B, start_sec=0.5),
            ClipSegment(source_path=A, start_sec=4.0, end_sec=9.0),
        ],
    )
    tl = timeline_from_edit_plan(plan, {A: 6.0, B: 3.0})

    assert (tl.title, tl.resolution, tl.fps, tl.audio_fade_sec) == ("aula", (1280, 720), 25, 0.05)
    assert tl.loudness == plan.loudness
    assert [(i.source_path, i.source_in, i.source_out, i.volume) for i in tl.video_main] == [
        (A, 1.0, 2.0, 0.8),
        (B, 0.5, 3.0, 1.0),  # sem end_sec: vai até o fim da origem
        (A, 4.0, 6.0, 1.0),  # end_sec além da origem: limitado como no renderer do EditPlan
    ]
    assert all(i.use_audio for i in tl.video_main)
    assert tl.video_overlay == [] and tl.audio_ambient == []


def test_timeline_from_edit_plan_requires_duration_for_open_segment():
    plan = EditPlan(output_path=Path("o.mp4"), segments=[ClipSegment(source_path=A, start_sec=0.0)])
    with pytest.raises(ValueError, match="source_durations"):
        timeline_from_edit_plan(plan)


def test_edit_plan_conversion_keeps_frame_quantization():
    """Mesma regra de frames do renderer do EditPlan: round((fim - início) * fps), mínimo 1."""
    segments = [ClipSegment(source_path=A, start_sec=i * 0.37, end_sec=i * 0.37 + 0.123) for i in range(20)]
    plan = EditPlan(output_path=Path("o.mp4"), segments=segments)
    tl = timeline_from_edit_plan(plan)
    assert tl.total_frames == sum(max(1, round((s.end_sec - s.start_sec) * 30)) for s in segments)


# --- achatamento ---


def _pieces(tl: Timeline):
    return [(p.track, p.item_index, p.timeline_start_frame, p.frame_count) for p in flatten(tl).video]


def _assert_contiguous(tl: Timeline):
    flat = flatten(tl)
    position = 0
    for piece in flat.video:
        assert piece.timeline_start_frame == position
        assert piece.frame_count >= 1
        position += piece.frame_count
    assert position == flat.total_frames == tl.total_frames
    assert sum(p.frame_count for p in flat.video) == tl.total_frames


def test_flatten_without_overlay_is_the_base():
    tl = _timeline([_main(source_out=1.0), _main(B, 2.0, 2.5)])
    assert _pieces(tl) == [("main", 0, 0, 30), ("main", 1, 30, 15)]
    _assert_contiguous(tl)


def test_flatten_overlay_at_start():
    tl = _timeline([_main(source_out=2.0)], overlay=[_over(0.0, 0.5)])
    assert _pieces(tl) == [("overlay", 0, 0, 15), ("main", 0, 15, 45)]
    base = flatten(tl).video[1]
    assert base.source_in == pytest.approx(0.5)  # a base continua de onde estaria
    _assert_contiguous(tl)


def test_flatten_overlay_at_end():
    tl = _timeline([_main(source_out=2.0)], overlay=[_over(1.5, 0.5)])
    assert _pieces(tl) == [("main", 0, 0, 45), ("overlay", 0, 45, 15)]
    _assert_contiguous(tl)


def test_flatten_overlay_in_the_middle_splits_the_item():
    tl = _timeline([_main(source_in=10.0, source_out=12.0)], overlay=[_over(0.5, 1.0, source_in=3.0)])
    flat = flatten(tl)
    assert _pieces(tl) == [("main", 0, 0, 15), ("overlay", 0, 15, 30), ("main", 0, 45, 15)]
    assert flat.video[1].source_in == pytest.approx(3.0)
    assert flat.video[1].source_out == pytest.approx(4.0)
    assert flat.video[2].source_in == pytest.approx(11.5)
    assert flat.video[2].source_out == pytest.approx(12.0)


def test_flatten_overlay_on_item_boundary():
    tl = _timeline([_main(source_out=1.0), _main(B, 0.0, 1.0)], overlay=[_over(1.0, 0.5, path=C)])
    assert _pieces(tl) == [("main", 0, 0, 30), ("overlay", 0, 30, 15), ("main", 1, 45, 15)]
    assert flatten(tl).video[2].source_in == pytest.approx(0.5)


def test_flatten_overlay_ending_on_item_boundary():
    tl = _timeline([_main(source_out=1.0), _main(B, 0.0, 1.0)], overlay=[_over(0.5, 0.5, path=C)])
    assert _pieces(tl) == [("main", 0, 0, 15), ("overlay", 0, 15, 15), ("main", 1, 30, 30)]


def test_flatten_overlay_covering_whole_item():
    tl = _timeline(
        [_main(source_out=1.0), _main(B, 0.0, 1.0), _main(A, 5.0, 6.0)],
        overlay=[_over(1.0, 1.0, path=C)],
    )
    assert _pieces(tl) == [("main", 0, 0, 30), ("overlay", 0, 30, 30), ("main", 2, 60, 30)]


def test_flatten_overlay_spanning_several_items():
    tl = _timeline(
        [_main(source_out=1.0), _main(B, 0.0, 1.0), _main(A, 5.0, 6.0)],
        overlay=[_over(0.5, 2.0, path=C)],
    )
    assert _pieces(tl) == [("main", 0, 0, 15), ("overlay", 0, 15, 60), ("main", 2, 75, 15)]
    assert flatten(tl).video[2].source_in == pytest.approx(5.5)


def test_flatten_adjacent_overlays_and_unsorted_input():
    tl = _timeline(
        [_main(source_out=3.0)],
        overlay=[_over(1.5, 1.0, path=C), _over(0.5, 1.0)],  # fora de ordem na trilha
    )
    assert _pieces(tl) == [("main", 0, 0, 15), ("overlay", 1, 15, 30), ("overlay", 0, 45, 30), ("main", 0, 75, 15)]


def test_flatten_overlay_covering_whole_timeline():
    tl = _timeline([_main(source_out=1.0), _main(B, 0.0, 1.0)], overlay=[_over(0.0, 2.0, path=C)])
    assert _pieces(tl) == [("overlay", 0, 0, 60)]


def test_flatten_keeps_clip_ids():
    tl = _timeline([_main(source_out=2.0, clip_id="A001")], overlay=[_over(0.5, 0.5, clip_id="D003", label="drone")])
    assert [(p.clip_id, p.label) for p in flatten(tl).video] == [("A001", None), ("D003", "drone"), ("A001", None)]


def test_flatten_audio_plan():
    tl = _timeline(
        [_main(source_out=1.0, volume=0.5), _main(B, 0.0, 0.5, use_audio=False), _main(A, 2.0, 3.0)],
        overlay=[_over(0.5, 1.0, path=C)],
        ambient=[_over(1.0, 0.5, path=C, source_in=2.0, volume=1.5, fade_in_sec=0.1, fade_out_sec=0.2)],
    )
    flat = flatten(tl)
    # O áudio principal ignora os overlays: um pedaço por item da base, contíguo em amostras
    assert [(a.source_path, a.start_sample, a.sample_count, a.volume) for a in flat.audio_main] == [
        (A, 0, 48000, 0.5),
        (None, 48000, 24000, 1.0),  # item marcado sem áudio → silêncio
        (A, 72000, 48000, 1.0),
    ]
    assert flat.total_samples == 120000 == sum(a.sample_count for a in flat.audio_main)
    (amb,) = flat.audio_ambient
    assert (amb.source_path, amb.source_in, amb.start_sample, amb.sample_count) == (C, 2.0, 48000, 24000)
    assert (amb.volume, amb.fade_in_sec, amb.fade_out_sec) == (1.5, 0.1, 0.2)
    assert flat.sample_rate == TIMELINE_SAMPLE_RATE


def test_flatten_audio_samples_follow_frames_without_drift():
    """Com fps que não divide 48 kHz em frames inteiros, as amostras seguem as fronteiras de frame."""
    main = [_main(source_in=i, source_out=i + 0.37) for i in range(30)]
    tl = _timeline(main, fps=24)
    flat = flatten(tl)
    position = 0
    for item, audio in zip(tl.video_main, flat.audio_main):
        assert audio.start_sample == round(position * TIMELINE_SAMPLE_RATE / 24)
        position += tl.item_frames(item)
    assert flat.total_samples == round(tl.total_frames * TIMELINE_SAMPLE_RATE / 24)


def test_flatten_random_timelines_cover_duration_exactly():
    """Propriedade: para timelines aleatórias válidas, os pedaços cobrem a duração sem buraco nem sobra."""
    rng = random.Random(1234)
    for _ in range(300):
        fps = rng.choice([24, 25, 30, 60])
        main = [_main(source_in=rng.uniform(0, 5), source_out=0) for _ in range(rng.randint(1, 6))]
        for item in main:
            item["source_out"] = item["source_in"] + rng.randint(1, 90) / fps
        total = sum(round((m["source_out"] - m["source_in"]) * fps) for m in main)
        overlay, cursor = [], 0
        while cursor < total and rng.random() < 0.7:
            start = rng.randint(cursor, total - 1)
            length = rng.randint(1, total - start)
            overlay.append(_over(start / fps, length / fps))
            cursor = start + length
        rng.shuffle(overlay)
        tl = _timeline(main, overlay, fps=fps)
        _assert_contiguous(tl)
        flat = flatten(tl)
        assert sum(p.frame_count for p in flat.video if p.track == "overlay") == sum(
            tl.item_frames(o) for o in tl.video_overlay
        )
