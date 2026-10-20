"""Detecção determinística de silêncios com o filtro silencedetect do FFmpeg."""

import re
import subprocess
from pathlib import Path
from typing import Literal, Optional, Union

from ai_video_editor.audio.exceptions import NoAudioStreamError, SilenceDetectionError
from ai_video_editor.audio.extractor import probe_audio
from ai_video_editor.audio.levels import choose_threshold, measure_levels
from ai_video_editor.audio.schema import (
    AudioAnalysisParams,
    AudioAnalysisResult,
    SilenceInterval,
)
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path

_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+(?:e-?\d+)?)")
_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+(?:e-?\d+)?)")

# Folga abaixo da qual um trecho é considerado vazio (1 ms).
_EPS = 0.001


def parse_silencedetect(stderr: str, duration_sec: float) -> list[SilenceInterval]:
    """Converte a saída do silencedetect em intervalos.

    Casos de borda: início negativo vira 0; silêncio sem `silence_end` fecha na
    duração total; valores são limitados a [0, duração].
    """
    intervals: list[SilenceInterval] = []
    open_start: Optional[float] = None

    def close(start: float, end: float) -> None:
        start = min(max(start, 0.0), duration_sec)
        end = min(max(end, 0.0), duration_sec)
        if end - start > _EPS:
            intervals.append(SilenceInterval.from_bounds(start, end))

    for line in stderr.splitlines():
        m = _START_RE.search(line)
        if m:
            if open_start is not None:  # início duplicado: fecha o anterior
                close(open_start, float(m.group(1)))
            open_start = float(m.group(1))
            continue
        m = _END_RE.search(line)
        if m and open_start is not None:
            close(open_start, float(m.group(1)))
            open_start = None

    if open_start is not None:
        close(open_start, duration_sec)
    return intervals


def compute_speech(silences: list[SilenceInterval], duration_sec: float) -> list[SilenceInterval]:
    """Complemento dos silêncios dentro de [0, duration_sec]."""
    speech: list[SilenceInterval] = []
    cursor = 0.0
    for s in sorted(silences, key=lambda i: i.start_sec):
        if s.start_sec - cursor > _EPS:
            speech.append(SilenceInterval.from_bounds(cursor, s.start_sec))
        cursor = max(cursor, s.end_sec)
    if duration_sec - cursor > _EPS:
        speech.append(SilenceInterval.from_bounds(cursor, duration_sec))
    return speech


def detect_silences(
    media_path: Path,
    threshold_db: Union[float, Literal["auto"]] = "auto",
    min_silence_sec: float = 0.5,
    ffmpeg_bin: Optional[Path] = None,
) -> AudioAnalysisResult:
    """Detecta silêncios e calcula os intervalos de fala (complemento).

    `threshold_db="auto"` mede o áudio e escolhe o limite (ver `audio/levels.py`); um número
    usa esse limite fixo.
    """
    if min_silence_sec <= 0:
        raise SilenceDetectionError(media_path, "min_silence_sec deve ser maior que zero")

    has_audio, duration = probe_audio(media_path)
    if not has_audio:
        raise NoAudioStreamError(media_path.resolve())

    levels = None
    mode = "fixed"
    if threshold_db == "auto":
        mode = "auto"
        levels = choose_threshold(measure_levels(media_path, ffmpeg_bin))
        threshold_db = levels.threshold_db

    args = [
        str(ffmpeg_bin or get_ffmpeg_path()),
        "-hide_banner",
        "-nostats",
        "-i",
        str(media_path.resolve()),
        "-vn",
        "-af",
        f"silencedetect=noise={threshold_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise SilenceDetectionError(media_path, proc.stderr.strip()[-500:])

    silences = parse_silencedetect(proc.stderr, duration)
    return AudioAnalysisResult(
        source_path=media_path.resolve(),
        duration_sec=round(duration, 3),
        params=AudioAnalysisParams(
            threshold_db=threshold_db, min_silence_sec=min_silence_sec, threshold_mode=mode
        ),
        levels=levels,
        silences=silences,
        speech=compute_speech(silences, duration),
    )
