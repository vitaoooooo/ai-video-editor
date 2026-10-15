"""Unit tests for the FFmpeg compiler."""

from pathlib import Path
import pytest

from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.renderer.compiler import FFmpegCompiler


def test_compiler_missing_source_file(tmp_path: Path):
    """Compiler fails fast if input file does not exist on disk."""
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        segments=[
            ClipSegment(source_path=tmp_path / "non_existent.mp4", start_sec=0.0, end_sec=2.0)
        ],
    )
    compiler = FFmpegCompiler()
    with pytest.raises(FileNotFoundError):
        compiler.compile(plan)


def test_compiler_structure_with_synthetic_pair(synthetic_video_pair, tmp_path: Path):
    """Compiler produces correct flag arguments and filter graph."""
    v1, v2 = synthetic_video_pair
    out_file = tmp_path / "render_test.mp4"

    plan = EditPlan(
        title="Test Cut",
        output_path=out_file,
        target_resolution=(1280, 720),
        target_fps=30,
        segments=[
            ClipSegment(source_path=v1, start_sec=0.5, end_sec=2.0),
            ClipSegment(source_path=v2, start_sec=1.0, end_sec=2.5),
        ],
    )

    compiler = FFmpegCompiler()
    args = compiler.compile(plan)

    # Check safe subprocess array format
    assert isinstance(args, list)
    assert "-y" in args
    assert "-filter_complex" in args
    assert "-c:v" in args
    assert "libx264" in args
    assert "-c:a" in args
    assert "aac" in args
    assert str(out_file.resolve()) in args

    # Check filtergraph composition
    fc_idx = args.index("-filter_complex")
    filter_graph = args[fc_idx + 1]
    assert "trim=start=0.5:end=2.0" in filter_graph
    assert "trim=start=1.0:end=2.5" in filter_graph
    assert "concat=n=2:v=1:a=1" in filter_graph
