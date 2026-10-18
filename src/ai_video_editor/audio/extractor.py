"""Extração de áudio de vídeos via FFmpeg."""

import json
import subprocess
from pathlib import Path
from typing import Optional

from ai_video_editor.audio.exceptions import (
    AudioAnalysisError,
    AudioExtractionError,
    MediaNotFoundError,
    NoAudioStreamError,
)
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path


def probe_audio(media_path: Path, ffprobe_bin: Optional[Path] = None) -> tuple[bool, float]:
    """Retorna (tem_audio, duração_total_sec) da mídia via ffprobe."""
    path = media_path.resolve()
    if not path.is_file():
        raise MediaNotFoundError(path)

    args = [
        str(ffprobe_bin or get_ffprobe_path()),
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise AudioAnalysisError(
            f"Mídia inválida ou corrompida: {path} ({proc.stderr.strip()})", file_path=path
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AudioAnalysisError(f"Saída inválida do ffprobe: {e}", file_path=path)

    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    return has_audio, max(duration, 0.0)


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    ffmpeg_bin: Optional[Path] = None,
) -> Path:
    """Extrai o áudio do vídeo como WAV mono (PCM 16 bits) na taxa indicada."""
    has_audio, _ = probe_audio(video_path)
    if not has_audio:
        raise NoAudioStreamError(video_path.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(ffmpeg_bin or get_ffmpeg_path()),
        "-y",
        "-i",
        str(video_path.resolve()),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path.resolve()),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, shell=False)
    if proc.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise AudioExtractionError(video_path, proc.stderr.strip()[-500:] or "ffmpeg retornou erro")
    return output_path.resolve()
