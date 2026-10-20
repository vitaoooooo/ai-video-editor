"""Medição do nível de áudio em janelas curtas e escolha automática do limite de silêncio.

Regra do limite automático (`choose_threshold`):

- piso de ruído  = percentil 10 dos níveis de pico por janela de 50 ms;
- nível de fala  = percentil 90 dos mesmos níveis;
- limite         = piso + 35% da distância até a fala, mas nunca menos que piso + 3 dB
                   (margem mínima) nem mais que fala - 6 dB, e dentro de [-60, -20] dB.

Mede-se o **pico** (e não o RMS) porque o `silencedetect` compara amostras instantâneas
com o limite: uma janela só vira silêncio se todos os seus picos ficam abaixo dele.

Casos de borda (marcados como `reliable=False` e acompanhados de aviso):

- fala e piso a menos de 6 dB: sem separação confiável; o limite fica 3 dB abaixo do piso
  (conservador: não corta fala);
- fala abaixo de -50 dB: áudio quase todo em silêncio; o limite fica 3 dB acima do nível
  medido, de modo que o trecho todo conta como silêncio.
"""

import subprocess
from pathlib import Path
from typing import Optional

from ai_video_editor.audio.exceptions import SilenceDetectionError
from ai_video_editor.audio.schema import AudioLevelStats
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path

WINDOW_SEC = 0.05
_ANALYSIS_RATE = 16000
_KEY = "lavfi.astats.Overall.Peak_level="

# Piso usado no lugar de -inf (silêncio digital).
DB_FLOOR = -90.0
NOISE_PERCENTILE = 10.0
SPEECH_PERCENTILE = 90.0
THRESHOLD_FRACTION = 0.35
MIN_MARGIN_DB = 3.0
SPEECH_HEADROOM_DB = 6.0
MIN_SPREAD_DB = 6.0
THRESHOLD_MIN_DB = -60.0
THRESHOLD_MAX_DB = -20.0
NEAR_SILENCE_DB = -50.0
# Versão da regra: entra no fingerprint do cache do modo automático.
AUTO_RULE_VERSION = 1


def measure_levels(media_path: Path, ffmpeg_bin: Optional[Path] = None) -> list[float]:
    """Pico (dBFS) de cada janela de 50 ms do áudio, via astats + ametadata."""
    samples = int(_ANALYSIS_RATE * WINDOW_SEC)
    args = [
        str(ffmpeg_bin or get_ffmpeg_path()),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        str(media_path.resolve()),
        "-vn",
        "-af",
        f"aresample={_ANALYSIS_RATE},asetnsamples=n={samples}:p=0,"
        "astats=metadata=1:reset=1:measure_overall=Peak_level:measure_perchannel=none,"
        "ametadata=print:key=lavfi.astats.Overall.Peak_level:file=-",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        raise SilenceDetectionError(media_path, proc.stderr.strip()[-500:])
    return parse_levels(proc.stdout)


def parse_levels(output: str) -> list[float]:
    """Extrai os níveis da saída do ametadata; `-inf` vira `DB_FLOOR`."""
    levels: list[float] = []
    for line in output.splitlines():
        if not line.startswith(_KEY):
            continue
        try:
            value = float(line[len(_KEY):].strip())
        except ValueError:
            continue
        levels.append(max(value, DB_FLOOR))  # -inf também cai no piso
    return levels


def percentile(values: list[float], pct: float) -> float:
    """Percentil por interpolação linear (sem numpy)."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def choose_threshold(levels: list[float]) -> AudioLevelStats:
    """Aplica a regra documentada no módulo e devolve limite, medidas e avisos."""
    if not levels:
        return AudioLevelStats(
            noise_floor_db=DB_FLOOR,
            speech_level_db=DB_FLOOR,
            threshold_db=DB_FLOOR + MIN_MARGIN_DB,
            reliable=False,
            warnings=["não foi possível medir o nível do áudio; todo o áudio é tratado como silêncio"],
        )

    floor = round(percentile(levels, NOISE_PERCENTILE), 1)
    speech = round(percentile(levels, SPEECH_PERCENTILE), 1)
    spread = speech - floor
    warnings: list[str] = []

    if speech < NEAR_SILENCE_DB:
        threshold = speech + MIN_MARGIN_DB
        warnings.append(
            f"áudio quase todo em silêncio (fala medida em {speech} dB); "
            "o trecho inteiro será tratado como silêncio"
        )
    elif spread < MIN_SPREAD_DB:
        threshold = floor - MIN_MARGIN_DB
        warnings.append(
            f"ruído de fundo ({floor} dB) muito próximo do nível da fala ({speech} dB) ou áudio sem pausas: "
            "detecção de silêncio pouco confiável; o limite foi mantido abaixo do ruído e poucos cortes são esperados"
        )
    else:
        threshold = floor + max(THRESHOLD_FRACTION * spread, MIN_MARGIN_DB)
        threshold = min(threshold, speech - SPEECH_HEADROOM_DB)
        threshold = min(max(threshold, THRESHOLD_MIN_DB), THRESHOLD_MAX_DB)
    return AudioLevelStats(
        noise_floor_db=floor,
        speech_level_db=speech,
        threshold_db=round(threshold, 1),
        reliable=not warnings,
        warnings=warnings,
    )
