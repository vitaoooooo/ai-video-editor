"""Legendas (.srt) sincronizadas com o vídeo editado."""

from ai_video_editor.subtitles.clips import ClipSubtitleSource, build_clip_subtitle_blocks
from ai_video_editor.subtitles.remap import TimelineMapper, remap_time
from ai_video_editor.subtitles.schema import SubtitleBlock, SubtitleParams
from ai_video_editor.subtitles.srt import (
    build_subtitle_blocks,
    format_srt,
    generate_srt,
    write_srt,
)

__all__ = [
    "ClipSubtitleSource",
    "SubtitleBlock",
    "SubtitleParams",
    "TimelineMapper",
    "build_clip_subtitle_blocks",
    "build_subtitle_blocks",
    "format_srt",
    "generate_srt",
    "remap_time",
    "write_srt",
]
