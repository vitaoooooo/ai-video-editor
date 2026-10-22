"""Tomadas, miniaturas e medidas de um clipe, sem IA (FFmpeg).

Uma passada do FFmpeg por clipe faz duas coisas ao mesmo tempo, gravando o resultado em
arquivos de metadados: (1) mudanças de cena (`select` com o `scene` score) e (2) medidas em
quadros amostrados (`blurdetect` e `signalstats`). As mudanças de cena dividem o clipe em
tomadas; tomadas longas viram janelas (para achar o trecho ruim dentro de um clipe longo).
Depois, três miniaturas por tomada.
"""

import math
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.visual.schema import ShotBoundary, ShotMeasures

# --- detecção de cena e tomadas ---
# `scene` score do filtro `select` (0–1) a partir do qual há mudança de cena.
SCENE_THRESHOLD = 0.30
# Mudança de cena a menos disso do limite anterior (ou do fim) é ignorada.
MIN_SHOT_SEC = 1.0
# Tomadas mais longas que isso são fatiadas em janelas iguais.
MAX_SHOT_SEC = 8.0
# Largura em que a passada de análise trabalha (detecção de cena e medidas).
ANALYSIS_WIDTH = 480

# --- medidas ---
# Quadros por segundo amostrados para as medidas.
MEASURE_FPS = 2.0
# blurdetect em ANALYSIS_WIDTH: imagem nítida fica perto de 5; desfoque forte passa de 10.
BLUR_HIGH = 9.0
# Diferença média de luma entre amostras (0–1) acima disso = movimento/tremor alto.
MOTION_HIGH = 0.08
# Faixa de luma (YHIGH − YLOW, 0–1) abaixo disso = quadro quase uniforme (parede, câmera tapada).
UNIFORM_SPREAD = 0.10
# Fração mínima de amostras quase uniformes para marcar a tomada como uniforme.
UNIFORM_MIN_FRACTION = 0.6
DARK_LUMA = 0.15
BRIGHT_LUMA = 0.88

# --- miniaturas ---
FRAME_POSITIONS = (0.10, 0.50, 0.90)
FRAME_WIDTH = 512
# Qualidade JPEG do FFmpeg (-q:v: 2 = melhor, 31 = pior); 5 é moderada.
FRAME_QUALITY = 5

_FRAME_RE = re.compile(r"^frame:\d+\s+pts:\S+\s+pts_time:([\d.]+)")
_KEY_RE = re.compile(r"^lavfi\.([\w.]+)=(\S+)")


class ShotSpan(BaseModel):
    start_sec: float
    end_sec: float
    boundary: ShotBoundary


class Sample(BaseModel):
    """Medidas de um quadro amostrado."""

    time_sec: float
    blur: Optional[float] = None
    yavg: float = 0.0
    ylow: float = 0.0
    yhigh: float = 0.0
    ydif: float = 0.0


class ClipScan(BaseModel):
    """Saída da passada do FFmpeg: tempos de mudança de cena e amostras."""

    cuts: list[float] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)


def split_shots(
    duration: float,
    cuts: list[float],
    min_shot: float = MIN_SHOT_SEC,
    max_shot: float = MAX_SHOT_SEC,
) -> list[ShotSpan]:
    """Divide o clipe nas mudanças de cena e fatia as tomadas longas em janelas iguais (≤ `max_shot`)."""
    if duration <= 0:
        return []
    bounds: list[tuple[float, ShotBoundary]] = [(0.0, "inicio")]
    for cut in sorted(cuts):
        if cut - bounds[-1][0] >= min_shot and duration - cut >= min_shot:
            bounds.append((round(cut, 3), "cena"))
    spans: list[ShotSpan] = []
    for i, (start, boundary) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else round(duration, 3)
        pieces = max(1, math.ceil((end - start) / max_shot - 1e-9))
        step = (end - start) / pieces
        for k in range(pieces):
            spans.append(ShotSpan(
                start_sec=round(start + k * step, 3),
                end_sec=round(end if k == pieces - 1 else start + (k + 1) * step, 3),
                boundary=boundary if k == 0 else "janela",
            ))
    return spans


def frame_times(span: ShotSpan, duration: float) -> list[float]:
    """Tempos das 3 miniaturas: início+10%, meio e fim−10% da tomada (dentro do clipe)."""
    last = max(0.0, duration - 0.05)
    length = span.end_sec - span.start_sec
    return [round(min(span.start_sec + length * p, last), 3) for p in FRAME_POSITIONS]


def parse_metadata_file(text: str) -> list[tuple[float, dict[str, str]]]:
    """Blocos do `metadata=print:file=...`: (pts_time, {chave: valor})."""
    blocks: list[tuple[float, dict[str, str]]] = []
    for line in text.splitlines():
        line = line.strip()
        match = _FRAME_RE.match(line)
        if match:
            blocks.append((float(match.group(1)), {}))
            continue
        match = _KEY_RE.match(line)
        if match and blocks:
            blocks[-1][1][match.group(1)] = match.group(2)
    return blocks


def _float(value: Optional[str]) -> Optional[float]:
    try:
        number = float(value) if value is not None else None
    except ValueError:
        return None
    return None if number is None or math.isnan(number) or math.isinf(number) else number


def scan_clip(path: Path, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> ClipScan:
    """Uma passada do FFmpeg: mudanças de cena (todos os quadros) e medidas (MEASURE_FPS).

    Raises:
        RuntimeError: o FFmpeg falhou.
    """
    with tempfile.TemporaryDirectory(prefix="visual-scan-") as tmp:
        # Os arquivos de metadados usam nomes relativos (cwd = tmp): caminhos do Windows com
        # `C:` quebrariam a sintaxe do filtro.
        graph = (
            f"[0:v]scale={ANALYSIS_WIDTH}:-2,format=yuv420p,split[a][b];"
            f"[a]select='gt(scene,{SCENE_THRESHOLD})',metadata=print:file=scenes.txt[o1];"
            f"[b]fps={MEASURE_FPS},blurdetect,signalstats,metadata=print:file=stats.txt[o2]"
        )
        args = [
            str(get_ffmpeg_path()), "-hide_banner", "-nostdin", "-v", "error",
            "-i", str(Path(path).resolve()), "-an", "-sn", "-dn",
            "-filter_complex", graph,
            "-map", "[o1]", "-f", "null", "-",
            "-map", "[o2]", "-f", "null", "-",
        ]
        proc = run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=tmp, shell=False)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg falhou na análise visual: {(proc.stderr or '').strip()[:300]}")
        scenes = Path(tmp, "scenes.txt")
        stats = Path(tmp, "stats.txt")
        cuts = [t for t, _ in parse_metadata_file(scenes.read_text(encoding="utf-8"))] if scenes.is_file() else []
        blocks = parse_metadata_file(stats.read_text(encoding="utf-8")) if stats.is_file() else []
    samples = [
        Sample(
            time_sec=t,
            blur=_float(keys.get("blur")),
            yavg=_float(keys.get("signalstats.YAVG")) or 0.0,
            ylow=_float(keys.get("signalstats.YLOW")) or 0.0,
            yhigh=_float(keys.get("signalstats.YHIGH")) or 0.0,
            ydif=_float(keys.get("signalstats.YDIF")) or 0.0,
        )
        for t, keys in blocks
    ]
    return ClipScan(cuts=cuts, samples=samples)


def measure_shot(span: ShotSpan, samples: list[Sample]) -> ShotMeasures:
    """Agrega as amostras dentro da tomada. A diferença de luma do 1º quadro após um corte é ignorada."""
    inside = [s for s in samples if span.start_sec <= s.time_sec < span.end_sec]
    if not inside:
        # Tomada mais curta que o intervalo de amostragem: usa a amostra mais próxima.
        middle = (span.start_sec + span.end_sec) / 2
        inside = sorted(samples, key=lambda s: abs(s.time_sec - middle))[:1]
    if not inside:
        return ShotMeasures(samples=0, brightness=0.0, motion=0.0, luma_spread=0.0)
    peak = 255.0
    blurs = [s.blur for s in inside if s.blur is not None]
    # A diferença da 1ª amostra compara com um quadro de antes do corte (ou com nada, no início).
    diffs = [s.ydif for s in (inside if span.boundary == "janela" else inside[1:])]
    spreads = [(s.yhigh - s.ylow) / peak for s in inside]
    blur = round(statistics.median(blurs), 3) if blurs else None
    brightness = round(min(1.0, statistics.fmean(s.yavg for s in inside) / peak), 3)
    motion = round(min(1.0, statistics.median(diffs) / peak), 4) if diffs else 0.0
    uniform_fraction = sum(1 for v in spreads if v < UNIFORM_SPREAD) / len(spreads)
    return ShotMeasures(
        samples=len(inside),
        blur=blur,
        brightness=brightness,
        motion=motion,
        luma_spread=round(max(0.0, min(1.0, statistics.median(spreads))), 3),
        blurry=blur is not None and blur >= BLUR_HIGH,
        shaky=motion >= MOTION_HIGH,
        uniform=uniform_fraction >= UNIFORM_MIN_FRACTION,
        dark=brightness < DARK_LUMA,
        bright=brightness > BRIGHT_LUMA,
    )


def extract_frames(path: Path, times: list[float], outputs: list[Path]) -> bool:
    """Grava as miniaturas de uma tomada num só FFmpeg (uma entrada com busca por tempo para cada).

    FRAME_WIDTH px de largura, JPEG de qualidade moderada. False se alguma falhar.
    """
    args = [str(get_ffmpeg_path()), "-hide_banner", "-nostdin", "-v", "error", "-y"]
    for at_sec in times:
        args += ["-ss", f"{at_sec:.3f}", "-i", str(path)]
    for index, output in enumerate(outputs):
        args += [
            "-map", f"{index}:v:0", "-frames:v", "1",
            "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", str(FRAME_QUALITY), str(output),
        ]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", shell=False)
    return proc.returncode == 0 and all(output.is_file() for output in outputs)


def shot_params() -> dict[str, float]:
    """Parâmetros que entram na chave do cache (mudou algum, refaz as tomadas)."""
    return {
        "scene_threshold": SCENE_THRESHOLD,
        "min_shot_sec": MIN_SHOT_SEC,
        "max_shot_sec": MAX_SHOT_SEC,
        "analysis_width": ANALYSIS_WIDTH,
        "measure_fps": MEASURE_FPS,
        "blur_high": BLUR_HIGH,
        "motion_high": MOTION_HIGH,
        "uniform_spread": UNIFORM_SPREAD,
        "uniform_min_fraction": UNIFORM_MIN_FRACTION,
        "dark_luma": DARK_LUMA,
        "bright_luma": BRIGHT_LUMA,
        "frame_width": FRAME_WIDTH,
        "frame_quality": FRAME_QUALITY,
    }
