"""Command-line interface for the AI Video Editor core engine."""

import argparse
import json
import sys
from pathlib import Path

from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.renderer.executor import RenderExecutor


def handle_plan_render(args: argparse.Namespace) -> int:
    """Executes an EditPlan defined in a JSON file."""
    plan_path = Path(args.plan_file)
    if not plan_path.is_file():
        print(f"Error: plan file not found: {plan_path}", file=sys.stderr)
        return 1

    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plan = EditPlan.model_validate(data)
    print(f"Loaded plan: '{plan.title}' with {len(plan.segments)} segments.")
    print(f"Target: {plan.target_resolution[0]}x{plan.target_resolution[1]} @ {plan.target_fps}fps")

    executor = RenderExecutor()
    result = executor.execute(plan)

    print(f"Render completed successfully!")
    print(f"Output: {result.output_path}")
    print(f"Duration: {result.duration_sec:.2f}s | Resolution: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Size: {result.size_bytes} bytes")
    return 0


def handle_quick_cut(args: argparse.Namespace) -> int:
    """Cuts slices from two videos and merges them into an output MP4."""
    v1_path = Path(args.video1)
    v2_path = Path(args.video2)
    out_path = Path(args.output)

    if not v1_path.is_file():
        print(f"Error: video 1 not found: {v1_path}", file=sys.stderr)
        return 1
    if not v2_path.is_file():
        print(f"Error: video 2 not found: {v2_path}", file=sys.stderr)
        return 1

    plan = EditPlan(
        title="Quick Cut Two Videos",
        output_path=out_path,
        target_resolution=(args.width, args.height),
        target_fps=args.fps,
        segments=[
            ClipSegment(
                source_path=v1_path,
                start_sec=args.start1,
                end_sec=args.end1,
            ),
            ClipSegment(
                source_path=v2_path,
                start_sec=args.start2,
                end_sec=args.end2,
            ),
        ],
    )

    executor = RenderExecutor()
    result = executor.execute(plan)

    print(f"Render completed successfully!")
    print(f"Output: {result.output_path}")
    print(f"Duration: {result.duration_sec:.2f}s | Resolution: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Size: {result.size_bytes} bytes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-video-editor",
        description="Core rendering engine for AI Video Editor (Phase 1 PoC)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: render
    render_parser = subparsers.add_parser("render", help="Render an EditPlan from JSON")
    render_parser.add_argument("plan_file", help="Path to EditPlan JSON file")

    # Subcommand: quick-cut
    cut_parser = subparsers.add_parser("quick-cut", help="Cut and concatenate two video files")
    cut_parser.add_argument("--video1", required=True, help="Path to first video")
    cut_parser.add_argument("--start1", type=float, default=0.0, help="Start time in seconds for video 1")
    cut_parser.add_argument("--end1", type=float, required=True, help="End time in seconds for video 1")
    cut_parser.add_argument("--video2", required=True, help="Path to second video")
    cut_parser.add_argument("--start2", type=float, default=0.0, help="Start time in seconds for video 2")
    cut_parser.add_argument("--end2", type=float, required=True, help="End time in seconds for video 2")
    cut_parser.add_argument("--output", "-o", required=True, help="Path for output MP4")
    cut_parser.add_argument("--width", type=int, default=1280, help="Output width (default: 1280)")
    cut_parser.add_argument("--height", type=int, default=720, help="Output height (default: 720)")
    cut_parser.add_argument("--fps", type=int, default=30, help="Output framerate (default: 30)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "render":
        sys.exit(handle_plan_render(args))
    elif args.command == "quick-cut":
        sys.exit(handle_quick_cut(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
