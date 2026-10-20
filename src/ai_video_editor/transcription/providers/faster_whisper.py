"""Provedor de transcrição local baseado em faster-whisper (dependência opcional)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.transcription.exceptions import (
    TranscriptionDependencyError,
    TranscriptionFailedError,
    TranscriptionGpuError,
    TranscriptionMediaError,
)
from ai_video_editor.transcription.provider import TranscriptionProgress, TranscriptionProvider
from ai_video_editor.transcription.schema import (
    Transcript,
    TranscriptSegment,
    TranscriptionConfig,
    Word,
)

# Menor duração aceita para uma palavra; o Whisper às vezes devolve start == end.
_MIN_WORD_SEC = 0.001

# O Whisper espera áudio mono em 16 kHz.
_SAMPLE_RATE = 16000

# Pacotes pip da NVIDIA (extra `gpu`) cujas DLLs o ctranslate2 precisa no Windows.
_NVIDIA_PACKAGES = ("nvidia.cublas", "nvidia.cudnn")

# Tipos que só existem na GPU; na volta para a CPU viram int8.
_GPU_ONLY_COMPUTE_TYPES = {"float16", "int8_float16", "int8_bfloat16", "bfloat16"}

_registered_dll_dirs: list = []


def register_nvidia_dll_dirs() -> list[Path]:
    """Registra a pasta `bin` dos pacotes pip da NVIDIA (Windows), sem depender do PATH do sistema."""
    if not hasattr(os, "add_dll_directory"):
        return []
    added: list[Path] = []
    for package in _NVIDIA_PACKAGES:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        for location in (spec.submodule_search_locations or []) if spec else []:
            bin_dir = Path(location) / "bin"
            if bin_dir.is_dir():
                _registered_dll_dirs.append(os.add_dll_directory(str(bin_dir)))
                # O ctranslate2 carrega o cuBLAS/cuDNN com LoadLibrary simples, que ignora
                # add_dll_directory; o PATH do próprio processo (não o do sistema) cobre esse caso.
                path_entries = os.environ.get("PATH", "").split(os.pathsep)
                if str(bin_dir) not in path_entries:
                    os.environ["PATH"] = os.pathsep.join([str(bin_dir), *path_entries])
                added.append(bin_dir)
    return added


def cuda_available() -> bool:
    """True se o ctranslate2 enxerga ao menos uma GPU CUDA."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _print_warning(message: str) -> None:
    print(f"Aviso: {message}", file=sys.stderr)


class FasterWhisperProvider(TranscriptionProvider):
    """Transcreve com faster-whisper (word_timestamps=True), aceitando áudio ou vídeo."""

    @property
    def provider_name(self) -> str:
        return "faster_whisper"

    def __init__(self, on_warning: Optional[Callable[[str], None]] = None):
        self._warn = on_warning or _print_warning
        # (dispositivo, compute_type) do último modelo carregado.
        self._effective: Optional[tuple[str, str]] = None
        # Callback e duração do áudio da transcrição em curso.
        self._progress: Optional[tuple[TranscriptionProgress, float]] = None
        # Modelo carregado com `preload`: (chave da configuração, modelo, (dispositivo, compute_type)).
        self._preloaded: Optional[tuple[tuple[str, str, str], object, tuple[str, str]]] = None

    @staticmethod
    def _resolve(device: str, compute_type: str) -> tuple[str, str]:
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def effective_settings(self, config: TranscriptionConfig) -> tuple[str, str]:
        return self._resolve(config.device, config.compute_type)

    @staticmethod
    def _cpu_settings(compute_type: str) -> tuple[str, str]:
        if compute_type == "auto" or compute_type in _GPU_ONLY_COMPUTE_TYPES:
            compute_type = "int8"
        return "cpu", compute_type

    def _load_model(self, config: TranscriptionConfig):
        """Importa faster-whisper só quando necessário e instancia o modelo.

        Em `auto`, uma falha ao carregar na GPU avisa o motivo e segue na CPU; com `cuda`
        explícito, a falha vira erro com instruções de instalação.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionDependencyError(self.provider_name) from exc

        device, compute_type = self._resolve(config.device, config.compute_type)
        if device == "cuda":
            register_nvidia_dll_dirs()
            if config.device == "cuda" and not cuda_available():
                raise TranscriptionGpuError(self.provider_name, details="nenhuma GPU CUDA detectada")
            try:
                model = WhisperModel(config.model_size, device=device, compute_type=compute_type)
            except Exception as exc:
                if config.device == "cuda":
                    raise TranscriptionGpuError(self.provider_name, details=str(exc)) from exc
                self._warn(f"GPU indisponível ({exc}); transcrevendo na CPU.")
                device, compute_type = self._cpu_settings(config.compute_type)
            else:
                self._effective = (device, compute_type)
                return model
        model = WhisperModel(config.model_size, device=device, compute_type=compute_type)
        self._effective = (device, compute_type)
        return model

    @staticmethod
    def _model_key(config: TranscriptionConfig) -> tuple[str, str, str]:
        return config.model_size, config.device, config.compute_type

    def preload(self, config: TranscriptionConfig) -> None:
        """Carrega o modelo agora; o `transcribe` seguinte com a mesma configuração o reaproveita.

        Permite medir o tempo de carga separado do tempo de transcrição (benchmark).
        """
        self._effective = None
        model = self._load_model(config)
        self._preloaded = (self._model_key(config), model, self._effective or self._cpu_settings(config.compute_type))

    def _decode_audio(self, media_path: Path):
        """Decodifica o áudio com FFmpeg para um array float32 mono 16 kHz (sem usar o PyAV)."""
        import numpy as np

        args = [
            str(get_ffmpeg_path()),
            "-nostdin",
            "-i",
            str(media_path.resolve()),
            "-vn",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "-",
        ]
        try:
            proc = subprocess.run(args, capture_output=True, shell=False)
        except OSError as exc:
            raise TranscriptionMediaError(
                self.provider_name, media_path, f"Não foi possível executar o FFmpeg: {exc}"
            ) from exc
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise TranscriptionMediaError(
                self.provider_name, media_path, err or "FFmpeg retornou erro ao decodificar o áudio."
            )
        usable = len(proc.stdout) - len(proc.stdout) % 4
        if usable == 0:
            raise TranscriptionMediaError(self.provider_name, media_path, "A mídia não possui áudio.")
        return np.frombuffer(proc.stdout[:usable], dtype=np.float32)

    def transcribe(
        self,
        media_path: Path,
        config: TranscriptionConfig,
        on_progress: Optional[TranscriptionProgress] = None,
    ) -> Transcript:
        media_path = Path(media_path)
        if not media_path.is_file():
            raise TranscriptionMediaError(self.provider_name, media_path, "Arquivo inexistente.")

        if self._preloaded is not None and self._preloaded[0] == self._model_key(config):
            _, model, self._effective = self._preloaded
        else:
            self._effective = None
            model = self._load_model(config)
        device, compute_type = self._effective or self._cpu_settings(config.compute_type)
        audio = self._decode_audio(media_path)
        duration = len(audio) / _SAMPLE_RATE
        self._progress = (on_progress, duration) if on_progress is not None else None
        try:
            segments, language = self._run(model, audio, config)
        except Exception as exc:
            if device != "cuda":
                raise TranscriptionFailedError(self.provider_name, details=str(exc)) from exc
            # As DLLs do cuBLAS/cuDNN só são carregadas na primeira inferência.
            if config.device == "cuda":
                raise TranscriptionGpuError(self.provider_name, details=str(exc)) from exc
            self._warn(f"falha ao transcrever na GPU ({exc}); repetindo na CPU.")
            device, compute_type = self._cpu_settings(config.compute_type)
            try:
                from faster_whisper import WhisperModel

                cpu_model = WhisperModel(config.model_size, device=device, compute_type=compute_type)
                segments, language = self._run(cpu_model, audio, config)
            except Exception as cpu_exc:
                raise TranscriptionFailedError(self.provider_name, details=str(cpu_exc)) from cpu_exc

        return Transcript(
            source_path=media_path,
            language=language,
            duration_sec=duration,
            model_used=f"faster-whisper:{config.model_size} ({device}, {compute_type})",
            segments=segments,
        )

    def _run(self, model, audio, config: TranscriptionConfig) -> tuple[list[TranscriptSegment], str]:
        raw_segments, info = model.transcribe(
            audio,
            language=config.language,
            word_timestamps=True,
            vad_filter=config.vad_filter,
        )
        return self._convert_segments(raw_segments, self._progress), info.language

    @staticmethod
    def _convert_segments(
        raw_segments, progress: Optional[tuple[TranscriptionProgress, float]] = None
    ) -> list[TranscriptSegment]:
        """Converte os segmentos do faster-whisper (gerador) para o schema, com IDs 0..n-1.

        O gerador transcreve sob demanda; com `progress`, cada segmento consumido informa o
        fim dele em relação à duração do áudio.
        """
        segments: list[TranscriptSegment] = []
        for raw in raw_segments:
            words = [
                Word(
                    text=w.word.strip(),
                    start_sec=w.start,
                    end_sec=max(w.end, w.start + _MIN_WORD_SEC),
                    probability=w.probability,
                )
                for w in (getattr(raw, "words", None) or [])
            ]
            segments.append(
                TranscriptSegment(
                    id=len(segments),
                    start_sec=raw.start,
                    end_sec=max(raw.end, raw.start + _MIN_WORD_SEC),
                    text=raw.text.strip(),
                    words=words,
                )
            )
            if progress is not None:
                on_progress, duration = progress
                on_progress(min(raw.end, duration), duration)
        return segments
