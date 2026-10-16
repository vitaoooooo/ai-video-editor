"""Orchestrates video ingestion, technical metadata probing, and frame extraction."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from ai_video_editor.analyzer.exceptions import VideoIngestError
from ai_video_editor.analyzer.extractor import FrameExtractor
from ai_video_editor.analyzer.probe import VideoProbe
from ai_video_editor.analyzer.schema import (
    BatchAnalysisResult,
    FrameExtractionConfig,
    VideoAnalysisResult,
)


def _generate_video_dir_name(video_path: Path) -> str:
    """Creates a deterministic, safe directory name based on file stem and hash."""
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in video_path.stem)
    path_hash = hashlib.sha256(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{path_hash}"


class VideoIngestService:
    """High-level service coordinating video probing and frame extraction."""

    def __init__(
        self,
        probe: Optional[VideoProbe] = None,
        extractor: Optional[FrameExtractor] = None,
        default_base_dir: Optional[Path] = None,
    ):
        self.probe = probe or VideoProbe()
        self.extractor = extractor or FrameExtractor()
        self.default_base_dir = (
            default_base_dir.resolve()
            if default_base_dir
            else Path("output/analysis").resolve()
        )

    def analyze_single(
        self,
        video_path: Path,
        output_dir: Path,
        config: Optional[FrameExtractionConfig] = None,
    ) -> VideoAnalysisResult:
        """Analyzes a single video file, saving extracted frames in output_dir."""
        cfg = config or FrameExtractionConfig()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Probe metadata
        metadata = self.probe.probe(video_path)

        # 2. Extract frames
        strategy_desc = (
            f"interval: every {cfg.interval_sec:.2f}s (max {cfg.max_frames} frames)"
            if cfg.strategy == "interval"
            else f"fixed_count: {cfg.target_count} frames (max {cfg.max_frames})"
        )

        planned_timestamps, frames = self.extractor.extract_frames(
            metadata=metadata,
            output_dir=output_dir,
            config=cfg,
        )

        return VideoAnalysisResult(
            metadata=metadata,
            sampling_strategy_used=strategy_desc,
            sampling_disclaimer=(
                "Discrete sample of frames; does not guarantee exhaustive representation of all video content."
            ),
            planned_timestamps_sec=planned_timestamps,
            frames=frames,
            output_dir=output_dir.resolve(),
            analyzed_at=datetime.now(timezone.utc),
        )

    def analyze_batch(
        self,
        video_paths: Sequence[Path],
        config: Optional[FrameExtractionConfig] = None,
        session_base_dir: Optional[Path] = None,
    ) -> BatchAnalysisResult:
        """Analyzes multiple video paths, isolating outputs into distinct session folders."""
        cfg = config or FrameExtractionConfig()
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        base = session_base_dir.resolve() if session_base_dir else self.default_base_dir
        session_dir = base / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        successful: list[VideoAnalysisResult] = []
        failed: dict[str, str] = {}

        for p in video_paths:
            path_obj = Path(p)
            try:
                video_subdir = session_dir / _generate_video_dir_name(path_obj)
                res = self.analyze_single(path_obj, output_dir=video_subdir, config=cfg)
                successful.append(res)
            except VideoIngestError as e:
                failed[str(path_obj)] = str(e)
            except Exception as e:
                failed[str(path_obj)] = f"Unexpected failure: {e}"

        batch_result = BatchAnalysisResult(
            session_id=session_id,
            session_dir=session_dir,
            total_requested=len(video_paths),
            successful=successful,
            failed=failed,
        )

        # Write analysis_report.json in session_dir
        report_file = session_dir / "analysis_report.json"
        report_file.write_text(batch_result.model_dump_json(indent=2), encoding="utf-8")

        return batch_result
