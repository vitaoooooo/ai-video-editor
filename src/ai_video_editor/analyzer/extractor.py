"""Extracts representative video frames deterministically using FFmpeg."""

import subprocess
from pathlib import Path
from typing import Optional, Tuple

from ai_video_editor.analyzer.exceptions import FrameExtractionError
from ai_video_editor.analyzer.schema import (
    ExtractedFrame,
    FrameExtractionConfig,
    VideoMetadata,
)
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path


def calculate_target_timestamps(
    duration_sec: float,
    config: FrameExtractionConfig,
) -> list[float]:
    """Calculates deterministic timestamps according to the requested sampling strategy.

    Note on precision & edge cases:
        - If interval does not divide duration evenly, the final timestamp will be the
          last multiple strictly less than duration_sec.
        - Frame count is always capped by config.max_frames (default 50).
        - For very short videos (< interval), at least a single frame at 0.0s is extracted.
    """
    if duration_sec <= 0.0:
        return [0.0]

    timestamps: list[float] = []

    if config.strategy == "interval":
        interval = max(0.01, config.interval_sec)
        t = 0.0
        # Step until reaching end of duration
        while t < duration_sec:
            timestamps.append(round(t, 3))
            t += interval
            if len(timestamps) >= config.max_frames:
                break

    elif config.strategy == "fixed_count":
        count = min(config.target_count, config.max_frames)
        if count == 1:
            timestamps.append(round(duration_sec / 2.0, 3))
        else:
            # Distribute points evenly across the available duration
            # Keeping the last sample slightly before EOF to prevent seeking beyond video
            safe_end = max(0.0, duration_sec - 0.04)
            step = safe_end / (count - 1)
            for i in range(count):
                timestamps.append(round(i * step, 3))

    else:
        # Fallback to single frame
        timestamps = [0.0]

    # Deduplicate and ensure non-empty
    unique_timestamps = sorted(list(set(timestamps)))
    return unique_timestamps[: config.max_frames] or [0.0]


class FrameExtractor:
    """Extracts image frames from video files with verifiable timestamps."""

    def __init__(self, ffmpeg_bin: Optional[Path] = None):
        self.ffmpeg_bin = ffmpeg_bin or get_ffmpeg_path()

    def extract_frames(
        self,
        metadata: VideoMetadata,
        output_dir: Path,
        config: Optional[FrameExtractionConfig] = None,
    ) -> Tuple[list[float], list[ExtractedFrame]]:
        """Extracts frames to the specified directory.

        Returns:
            Tuple of (planned_timestamps, list_of_extracted_frames)
        """
        cfg = config or FrameExtractionConfig()
        output_dir.mkdir(parents=True, exist_ok=True)

        planned_timestamps = calculate_target_timestamps(metadata.duration_sec, cfg)
        extracted_frames: list[ExtractedFrame] = []

        ext = "jpg" if cfg.image_format == "jpg" else "png"

        for idx, target_t in enumerate(planned_timestamps):
            file_name = f"frame_{idx:04d}_{int(target_t * 1000):06d}ms.{ext}"
            frame_path = output_dir / file_name

            # Subprocess execution with list arguments and shell=False
            args = [
                str(self.ffmpeg_bin),
                "-y",
                "-ss",
                f"{target_t:.3f}",
                "-i",
                str(metadata.source_path.resolve()),
                "-vframes",
                "1",
                "-q:v",
                "2",
                str(frame_path.resolve()),
            ]

            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                shell=False,
            )

            if proc.returncode != 0 or not frame_path.is_file():
                err = proc.stderr.strip() or "frame extraction non-zero exit"
                raise FrameExtractionError(
                    metadata.source_path,
                    timestamp_sec=target_t,
                    details=err,
                )

            size_bytes = frame_path.stat().st_size
            if size_bytes == 0:
                raise FrameExtractionError(
                    metadata.source_path,
                    timestamp_sec=target_t,
                    details="Extracted frame file is 0 bytes",
                )

            extracted_frames.append(
                ExtractedFrame(
                    frame_index=idx,
                    target_timestamp_sec=target_t,
                    image_path=frame_path.resolve(),
                    width=metadata.video_stream.width,
                    height=metadata.video_stream.height,
                    size_bytes=size_bytes,
                )
            )

        return planned_timestamps, extracted_frames
