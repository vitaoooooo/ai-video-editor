"""Codificador de vídeo do render: x264 na CPU ou NVENC na GPU NVIDIA, com detecção e volta para a CPU.

Todos os pedaços de um render usam o MESMO codificador e os mesmos parâmetros, porque são unidos com
concat `-c copy`. Se o NVENC falhar no meio (limite de sessões da GeForce, memória), o render inteiro
é refeito na CPU (ver os executores); nunca há pedaços de codificadores diferentes no mesmo vídeo.
Só a codificação vai para a GPU: decodificação e filtros continuam na CPU.
"""

import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from ai_video_editor.renderer.compiler import VIDEO_ENCODE_ARGS

EncoderChoice = Literal["auto", "cpu", "nvenc"]
ENCODER_CHOICES: tuple[str, ...] = ("auto", "cpu", "nvenc")

# Pedaços codificados ao mesmo tempo na GPU (GeForce limita as sessões NVENC simultâneas pelo driver).
DEFAULT_GPU_WORKERS = 3

# Equivalente ao x264 `-preset fast -crf 22`: qualidade constante (`-rc vbr -cq 25 -b:v 0`), GOP de 250
# e 3 B-frames como os padrões do x264, perfil high. `-cq 25` foi o primeiro valor com SSIM >= ao da CPU
# em 40 trechos 1080p reais do vlog (RTX 2060; ver docs/HISTORICO.md, 2026-10-23); `-cq 23` sai 41% maior.
NVENC_ENCODE_ARGS = [
    "-c:v", "h264_nvenc",
    "-preset", "p5",
    "-tune", "hq",
    "-rc", "vbr",
    "-cq", "25",
    "-b:v", "0",
    "-profile:v", "high",
    "-g", "250",
    "-bf", "3",
    "-pix_fmt", "yuv420p",
]

# Codificação de teste: 1 s de testsrc com os parâmetros reais, descartada (muxer null).
_TEST_SOURCE = "testsrc=size=640x360:rate=30:duration=1"
_DETECT_TIMEOUT_SEC = 30

# Mensagens do FFmpeg que identificam falha do NVENC (o prefixo `[h264_nvenc @ ...]`, a API do driver, CUDA).
_NVENC_ERROR = re.compile(r"nvenc|OpenEncodeSession|cuda|No capable devices", re.IGNORECASE)


@dataclass(frozen=True)
class NvencStatus:
    """Resultado da detecção do NVENC (`reason` explica a indisponibilidade)."""

    available: bool
    reason: str = ""


@dataclass(frozen=True)
class EncoderSelection:
    """Codificador efetivo de um render e por quê (ex.: `nvenc`, `cpu (sem GPU: ...)`)."""

    name: Literal["cpu", "nvenc"] = "cpu"
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.reason})" if self.reason else self.name

    @property
    def video_args(self) -> list[str]:
        return list(NVENC_ENCODE_ARGS if self.name == "nvenc" else VIDEO_ENCODE_ARGS)

    @property
    def fingerprint_parts(self) -> tuple:
        """O que entra no fingerprint do render. CPU: nada (o mesmo fingerprint de antes do NVENC)."""
        return () if self.name == "cpu" else (self.name, NVENC_ENCODE_ARGS)

    @property
    def fell_back(self) -> bool:
        return self.reason.startswith("NVENC falhou")


def _first_line(text: str, limit: int = 200) -> str:
    """Primeira linha útil de uma saída de erro, encurtada."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    nvenc = [line for line in lines if _NVENC_ERROR.search(line)]
    line = (nvenc or lines or ["sem detalhes"])[0]
    return line if len(line) <= limit else line[: limit - 1] + "…"


def probe_nvenc(ffmpeg_bin: Path) -> NvencStatus:
    """Confere se o FFmpeg lista o `h264_nvenc` e se uma codificação de teste funciona (driver e GPU)."""
    base = [str(ffmpeg_bin), "-hide_banner", "-nostdin"]
    try:
        listing = subprocess.run(
            [*base, "-encoders"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_DETECT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NvencStatus(False, f"não foi possível listar os codificadores do FFmpeg: {exc}")
    if not re.search(r"^\s*V\S*\s+h264_nvenc\b", listing.stdout, re.MULTILINE):
        return NvencStatus(False, "h264_nvenc não existe neste FFmpeg")
    try:
        test = subprocess.run(
            [*base, "-loglevel", "error", "-f", "lavfi", "-i", _TEST_SOURCE, *NVENC_ENCODE_ARGS, "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_DETECT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NvencStatus(False, f"codificação de teste do NVENC não terminou: {exc}")
    if test.returncode != 0:
        return NvencStatus(False, f"codificação de teste do NVENC falhou: {_first_line(test.stderr)}")
    return NvencStatus(True)


_cache: dict[str, NvencStatus] = {}
_cache_lock = threading.Lock()


def detect_nvenc(ffmpeg_bin: Path) -> NvencStatus:
    """`probe_nvenc` uma vez por FFmpeg na execução (não a cada pedaço nem a cada render)."""
    key = str(ffmpeg_bin)
    with _cache_lock:
        if key not in _cache:
            _cache[key] = probe_nvenc(ffmpeg_bin)
        return _cache[key]


def select_encoder(choice: EncoderChoice, ffmpeg_bin: Path) -> EncoderSelection:
    """Resolve `auto`/`cpu`/`nvenc` no codificador do render.

    Raises:
        RuntimeError: `nvenc` pedido explicitamente, mas indisponível nesta máquina.
    """
    if choice == "cpu":
        return EncoderSelection("cpu", "--encoder cpu")
    status = detect_nvenc(ffmpeg_bin)
    if status.available:
        return EncoderSelection("nvenc")
    if choice == "nvenc":
        raise RuntimeError(f"--encoder nvenc pedido, mas o NVENC não está disponível: {status.reason}")
    return EncoderSelection("cpu", f"sem GPU: {status.reason}")


def is_nvenc_failure(message: str) -> bool:
    """A falha de um FFmpeg veio do NVENC (sessões, memória, driver)?"""
    return bool(_NVENC_ERROR.search(message))


class NvencFailure(RuntimeError):
    """Um pedaço falhou no NVENC; o executor refaz o render inteiro na CPU."""

    def __init__(self, message: str):
        super().__init__(message)
        self.detail = _first_line(message)


def fallback_selection(failure: NvencFailure) -> EncoderSelection:
    return EncoderSelection("cpu", f"NVENC falhou: {failure.detail}")


def fallback_warning(selection: EncoderSelection) -> str:
    return f"o NVENC falhou durante o render e o vídeo inteiro foi refeito na CPU ({selection.reason})"
