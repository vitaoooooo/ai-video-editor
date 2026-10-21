"""Detecção de fala por clipe: proporção da duração com voz e os trechos.

O provedor real usa o VAD (Silero) que vem com o faster-whisper, sobre o áudio decodificado pelo
FFmpeg a 16 kHz mono. O áudio é lido em blocos, então clipes longos não ocupam a memória inteira.
Sem o faster-whisper instalado, o resultado é `desconhecido` (com aviso), nunca um erro.
"""

import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ai_video_editor.catalog.schema import SpeechDetection, SpeechSegment
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path

SAMPLE_RATE = 16000
# Bloco de áudio por chamada do VAD (segundos). Trechos cortados na borda são emendados depois.
CHUNK_SEC = 60.0
# Trechos separados por menos que isso viram um só.
MERGE_GAP_SEC = 0.3
# A partir desta proporção o clipe conta como "com fala" nos resumos.
SPEECH_MIN_RATIO = 0.1

UNAVAILABLE_NOTE = "faster-whisper não instalado (pip install -e \".[transcription]\")"


class SpeechDetector(ABC):
    """Interface: detecta fala no áudio de um clipe."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador do detector (entra no cache por clipe)."""

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def detect(self, path: Path, duration_sec: float) -> SpeechDetection:
        """Fala no clipe (que tem áudio)."""


def merge_segments(segments: list[tuple[float, float]], gap_sec: float = MERGE_GAP_SEC) -> list[SpeechSegment]:
    merged: list[list[float]] = []
    for start, end in sorted(segments):
        if merged and start - merged[-1][1] <= gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [SpeechSegment(start_sec=round(s, 2), end_sec=round(e, 2)) for s, e in merged]


def speech_result(segments: list[SpeechSegment], duration_sec: float, detector: str) -> SpeechDetection:
    speech_sec = sum(s.end_sec - s.start_sec for s in segments)
    ratio = min(1.0, speech_sec / duration_sec) if duration_sec > 0 else 0.0
    return SpeechDetection(
        status="analisada", ratio=round(ratio, 3), speech_sec=round(speech_sec, 2), segments=segments,
        detector=detector,
    )


class FasterWhisperVAD(SpeechDetector):
    """VAD do faster-whisper (import tardio)."""

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250, min_silence_ms: int = 500):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._vad = None
        self._checked = False
        self._lock = threading.Lock()  # vários workers compartilham o detector

    @property
    def name(self) -> str:
        return f"faster-whisper-vad:{self.threshold}:{self.min_speech_ms}:{self.min_silence_ms}"

    def _load(self):
        with self._lock:
            if not self._checked:
                try:
                    from faster_whisper.vad import VadOptions, get_speech_timestamps
                except ImportError:
                    self._vad = None
                else:
                    options = VadOptions(
                        threshold=self.threshold,
                        min_speech_duration_ms=self.min_speech_ms,
                        min_silence_duration_ms=self.min_silence_ms,
                        speech_pad_ms=100,
                    )
                    self._vad = (get_speech_timestamps, options)
                self._checked = True
        return self._vad

    @property
    def available(self) -> bool:
        return self._load() is not None

    def detect(self, path: Path, duration_sec: float) -> SpeechDetection:
        vad = self._load()
        if vad is None:
            return SpeechDetection(status="desconhecido", detector=self.name, note=UNAVAILABLE_NOTE)
        import numpy as np

        get_speech_timestamps, options = vad
        args = [
            str(get_ffmpeg_path()), "-hide_banner", "-nostdin", "-v", "error",
            "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ]
        chunk_bytes = int(CHUNK_SEC * SAMPLE_RATE) * 4
        found: list[tuple[float, float]] = []
        offset = 0.0
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        try:
            while True:
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    break
                audio = np.frombuffer(data[: len(data) // 4 * 4], dtype=np.float32)
                for ts in get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE):
                    found.append((offset + ts["start"] / SAMPLE_RATE, offset + ts["end"] / SAMPLE_RATE))
                offset += len(audio) / SAMPLE_RATE
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
        finally:
            proc.stdout.close()
            proc.stderr.close()
            returncode = proc.wait()
        if returncode != 0 and offset == 0:
            return SpeechDetection(
                status="desconhecido", detector=self.name, note=f"áudio não decodificado: {stderr.strip()[:200]}"
            )
        return speech_result(merge_segments(found), duration_sec or offset, self.name)


class MockSpeechDetector(SpeechDetector):
    """Detector simulado: trechos fixos, ou indisponível (como sem faster-whisper)."""

    def __init__(self, segments: Optional[list[tuple[float, float]]] = None, available: bool = True):
        self.segments = segments if segments is not None else [(0.0, 1.0)]
        self._available = available
        self.calls: list[Path] = []

    @property
    def name(self) -> str:
        return "mock" if self._available else "mock-indisponivel"

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, path: Path, duration_sec: float) -> SpeechDetection:
        self.calls.append(path)
        if not self._available:
            return SpeechDetection(status="desconhecido", detector=self.name, note=UNAVAILABLE_NOTE)
        clipped = [(s, min(e, duration_sec)) for s, e in self.segments if s < duration_sec]
        return speech_result(merge_segments(clipped), duration_sec, self.name)
