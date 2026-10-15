"""Pydantic schemas for edit plans and video segments."""

from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class ClipSegment(BaseModel):
    """Represents a sliced portion of an input video."""

    source_path: Path = Field(..., description="Path to source video file")
    start_sec: float = Field(default=0.0, ge=0.0, description="Start timestamp in seconds")
    end_sec: Optional[float] = Field(default=None, description="End timestamp in seconds (None = clip end)")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Audio volume multiplier")

    @model_validator(mode="after")
    def validate_time_range(self) -> "ClipSegment":
        if self.end_sec is not None and self.end_sec <= self.start_sec:
            raise ValueError(
                f"end_sec ({self.end_sec}) must be greater than start_sec ({self.start_sec})"
            )
        return self

    @property
    def duration(self) -> Optional[float]:
        """Estimated duration if end_sec is specified."""
        if self.end_sec is not None:
            return round(self.end_sec - self.start_sec, 3)
        return None


class EditPlan(BaseModel):
    """Declarative specification of a complete video edit."""

    title: str = Field(default="Untitled Edit", description="Title or description of the edit")
    output_path: Path = Field(..., description="Destination path for rendered output MP4")
    target_resolution: Tuple[int, int] = Field(
        default=(1280, 720),
        description="Target (width, height) for standardized output canvas",
    )
    target_fps: int = Field(default=30, gt=0, description="Target frames per second")
    segments: list[ClipSegment] = Field(
        ..., min_length=1, description="Ordered list of video segments to trim and concatenate"
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "EditPlan":
        if self.target_resolution[0] <= 0 or self.target_resolution[1] <= 0:
            raise ValueError(f"Invalid target resolution: {self.target_resolution}")
        return self
