"""Dados técnicos (ffprobe), miniaturas e sinais de cor de um clipe, sem IA."""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from ai_video_editor.analyzer.exceptions import InvalidMediaFormatError, NoVideoStreamError
from ai_video_editor.analyzer.probe import parse_fractional_fps
from ai_video_editor.catalog.dates import date_from_tags
from ai_video_editor.catalog.schema import ClipMedia, ColorSignals, RecordingDate
from ai_video_editor.pipeline.runner import probe_rotation
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path

THUMBNAIL_POSITIONS = (0.10, 0.50, 0.90)
THUMBNAIL_WIDTH = 320

_STAT_RE = re.compile(r"lavfi\.signalstats\.(YLOW|YHIGH|SATAVG)=([\d.]+)")


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bit_depth(stream: dict) -> Optional[int]:
    depth = _int_or_none(stream.get("bits_per_raw_sample"))
    if depth:
        return depth
    pix_fmt = stream.get("pix_fmt") or ""
    match = re.search(r"p(\d{2})(?:le|be)?$", pix_fmt)
    if match:
        return int(match.group(1))
    return 8 if pix_fmt else None


def probe_clip(path: Path) -> tuple[ClipMedia, Optional[RecordingDate]]:
    """ffprobe do clipe: dados técnicos e data das tags (se houver).

    Raises:
        InvalidMediaFormatError: arquivo ilegível.
        NoVideoStreamError: sem trilha de vídeo.
    """
    args = [str(get_ffprobe_path()), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", shell=False)
    if proc.returncode != 0:
        raise InvalidMediaFormatError(path, details=proc.stderr.strip()[:300] or "ffprobe falhou")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise InvalidMediaFormatError(path, details=f"saída do ffprobe inválida: {e}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    # Capas embutidas (attached_pic) não são a trilha de vídeo.
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if video is None or not video.get("width"):
        raise NoVideoStreamError(path)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = parse_fractional_fps(video.get("avg_frame_rate"))
    if fps <= 0:
        fps = parse_fractional_fps(video.get("r_frame_rate"))
    duration = 0.0
    for candidate in (fmt.get("duration"), video.get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue
    width, height = int(video["width"]), int(video["height"])
    rotation = probe_rotation(path)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    media = ClipMedia(
        duration_sec=round(max(0.0, duration), 3),
        width=width,
        height=height,
        fps=round(fps, 3),
        codec=video.get("codec_name", "desconhecido"),
        pix_fmt=video.get("pix_fmt"),
        bit_depth=_bit_depth(video),
        rotation=rotation,
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        bitrate=_int_or_none(fmt.get("bit_rate")),
    )
    return media, date_from_tags(fmt.get("tags", {}) or {}, video.get("tags", {}) or {})


def _thumbnail(path: Path, at_sec: float, output: Path) -> Optional[dict[str, float]]:
    """Grava a miniatura em `output` e devolve YLOW/YHIGH/SATAVG do frame (None se falhar)."""
    args = [
        str(get_ffmpeg_path()), "-hide_banner", "-nostdin", "-y",
        "-ss", f"{at_sec:.3f}", "-i", str(path),
        "-frames:v", "1", "-an",
        "-vf", f"signalstats,metadata=mode=print,scale={THUMBNAIL_WIDTH}:-2",
        "-q:v", "4", str(output),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", shell=False)
    if proc.returncode != 0 or not output.is_file():
        return None
    stats: dict[str, float] = {}
    for key, value in _STAT_RE.findall(proc.stderr):
        stats.setdefault(key, float(value))  # o primeiro frame basta
    return stats


def thumbnails_and_color(
    path: Path, media: ClipMedia, out_dir: Path
) -> tuple[list[str], Optional[ColorSignals], list[str]]:
    """Miniaturas em 10%, 50% e 90% da duração (`t10.jpg`...) e os sinais de cor desses frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    frames: list[dict[str, float]] = []
    warnings: list[str] = []
    last_safe = max(0.0, media.duration_sec - 0.1)
    for fraction in THUMBNAIL_POSITIONS:
        name = f"t{round(fraction * 100):02d}.jpg"
        stats = _thumbnail(path, min(media.duration_sec * fraction, last_safe), out_dir / name)
        if stats is None:
            warnings.append(f"miniatura em {round(fraction * 100)}% falhou")
            continue
        names.append(name)
        if {"YLOW", "YHIGH", "SATAVG"} <= stats.keys():
            frames.append(stats)
    color = None
    if frames:
        peak = float(2 ** (media.bit_depth or 8) - 1)
        spread = sum(f["YHIGH"] - f["YLOW"] for f in frames) / len(frames) / peak
        saturation = sum(f["SATAVG"] for f in frames) / len(frames) / peak
        color = ColorSignals(
            frames=len(frames),
            luma_spread=round(spread, 3),
            saturation=round(saturation, 3),
        )
    return names, color, warnings
