"""Integration tests for the render executor with real FFmpeg processing."""

from pathlib import Path

from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.renderer.executor import RenderExecutor


def test_render_two_synthetic_videos_integration(synthetic_video_pair, tmp_path: Path):
    """Cuts two distinct synthetic clips and renders them into a single valid MP4."""
    v1, v2 = synthetic_video_pair
    out_file = tmp_path / "final_cut.mp4"

    # Segment 1: 0.5s to 2.0s (1.5s duration)
    # Segment 2: 1.0s to 2.5s (1.5s duration)
    # Total expected duration: ~3.0s
    plan = EditPlan(
        title="Integration Cut",
        output_path=out_file,
        target_resolution=(1280, 720),
        target_fps=30,
        segments=[
            ClipSegment(source_path=v1, start_sec=0.5, end_sec=2.0),
            ClipSegment(source_path=v2, start_sec=1.0, end_sec=2.5),
        ],
    )

    executor = RenderExecutor()
    result = executor.execute(plan)

    # 1. Output file exists and is non-empty
    assert result.output_path.is_file()
    assert result.size_bytes > 1000

    # 2. Resolution matches target
    assert result.resolution == (1280, 720)

    # 3. Duration is within expected tolerance (around 3.0 seconds)
    assert abs(result.duration_sec - 3.0) < 0.3
