"""Unit tests for core schemas."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from ai_video_editor.core.schema import ClipSegment, EditPlan


def test_clip_segment_valid():
    """Valid clip segment builds properly with correct duration."""
    seg = ClipSegment(
        source_path=Path("sample.mp4"),
        start_sec=1.5,
        end_sec=4.5,
        volume=1.0,
    )
    assert seg.start_sec == 1.5
    assert seg.end_sec == 4.5
    assert seg.duration == 3.0


def test_clip_segment_invalid_range():
    """End time must be strictly greater than start time."""
    with pytest.raises(ValidationError):
        ClipSegment(source_path=Path("sample.mp4"), start_sec=5.0, end_sec=3.0)

    with pytest.raises(ValidationError):
        ClipSegment(source_path=Path("sample.mp4"), start_sec=5.0, end_sec=5.0)


def test_clip_segment_negative_start():
    """Negative start timestamps are disallowed."""
    with pytest.raises(ValidationError):
        ClipSegment(source_path=Path("sample.mp4"), start_sec=-1.0)


def test_edit_plan_valid():
    """Valid EditPlan initializes with expected defaults."""
    plan = EditPlan(
        title="Sample Plan",
        output_path=Path("output.mp4"),
        segments=[
            ClipSegment(source_path=Path("a.mp4"), start_sec=0.0, end_sec=2.0),
            ClipSegment(source_path=Path("b.mp4"), start_sec=1.0, end_sec=3.0),
        ],
    )
    assert plan.target_resolution == (1280, 720)
    assert plan.target_fps == 30
    assert len(plan.segments) == 2


def test_edit_plan_empty_segments_fails():
    """Plan must have at least one segment."""
    with pytest.raises(ValidationError):
        EditPlan(output_path=Path("output.mp4"), segments=[])


def test_edit_plan_json_roundtrip():
    """EditPlan can be serialized to JSON and reconstructed."""
    original = EditPlan(
        title="Serialized Plan",
        output_path=Path("final.mp4"),
        target_resolution=(1920, 1080),
        target_fps=60,
        segments=[
            ClipSegment(source_path=Path("c1.mp4"), start_sec=0.0, end_sec=5.0),
        ],
    )
    json_data = original.model_dump_json()
    reconstructed = EditPlan.model_validate_json(json_data)
    assert reconstructed.title == original.title
    assert reconstructed.target_resolution == (1920, 1080)
    assert reconstructed.target_fps == 60
    assert len(reconstructed.segments) == 1
