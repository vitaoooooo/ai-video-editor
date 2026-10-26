"""Fotos de um clipe para o roteiro: quando tirar, extração com FFmpeg e descarte das quase iguais.

- **Quando:** uma a cada `interval` segundos (grade fixa: 0, 2, 4...), mais extras no início de cada
  frase que vem depois de uma pausa ≥ PAUSE_PHOTO_SEC e em cada troca de cena. Extra a menos de
  EXTRA_MERGE_SEC de outra foto não vira foto nova: só soma o motivo à que já existe.
- **Extração:** um FFmpeg por clipe (a partida custa ~1 s no Windows): a grade sai do filtro `fps`,
  os extras de um `select` com os tempos; os tempos reais vêm do `metadata=print`. Lado maior com
  PHOTO_MAX_SIDE px (sem ampliar).
- **Descarte:** cada foto é comparada com a última foto MANTIDA (miniatura cinza 16x16, diferença média
  0–255, Pillow); abaixo de DUPLICATE_MAX_DIFF é descartada, a não ser que a última mantida esteja a
  MAX_KEPT_GAP_SEC ou mais (assim nenhum trecho fica sem foto por muito tempo).
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.roteiro.schema import PhotoReason
from ai_video_editor.visual.shots import parse_metadata_file

PHOTO_INTERVAL_SEC = 2.0
# 512 px no lado maior: no Qwen2.5-VL (blocos de 28 px) uma foto 16:9 fica em ~18x10 = ~190 tokens de
# imagem, o que mantém a descrição rápida num modelo de 3B na GPU local; detalhes finos não importam aqui.
PHOTO_MAX_SIDE = 512
# Qualidade JPEG do FFmpeg (-q:v: 2 = melhor, 31 = pior); a mesma das miniaturas do `visual`.
PHOTO_QUALITY = 5
# Frase depois de pausa ≥ isso ganha uma foto no início (a cena pode ter mudado durante a pausa).
PAUSE_PHOTO_SEC = 1.0
# Extra a menos disso de outra foto só soma o motivo a ela.
EXTRA_MERGE_SEC = 0.5
# Troca de cena a menos disso do início/fim do clipe ou da troca anterior é ignorada (= `visual`).
SCENE_MIN_GAP_SEC = 1.0
# --- descarte (estimativas iniciais, a calibrar) ---
SIGNATURE_SIDE = 16
# Diferença média da miniatura cinza 16x16 (0–255) até a qual a foto é "quase igual" à anterior.
DUPLICATE_MAX_DIFF = 6.0
# Nunca mais que isso sem uma foto mantida, mesmo com a imagem parada (fala longa, câmera fixa).
MAX_KEPT_GAP_SEC = 10.0
# Termos por `select` (a expressão cresce com os extras; vários ramos no mesmo FFmpeg).
EXTRAS_PER_BRANCH = 100
# Ramos de extras por FFmpeg (mantém a linha de comando abaixo do limite do Windows).
BRANCHES_PER_PROCESS = 4
# Sobe quando muda o que é extraído (refaz as fotos do cache).
PHOTO_FORMAT_VERSION = 1

_SCALE = (
    f"scale=w='if(gte(iw,ih),min({PHOTO_MAX_SIDE},iw),-2)':h='if(gte(iw,ih),-2,min({PHOTO_MAX_SIDE},ih))'"
)
_STAMP = "metadata=add:key=roteiro:value=1"


@dataclass
class PlannedPhoto:
    time_sec: float
    reasons: list[PhotoReason]
    grid: bool  # da grade fixa (sai do `fps`) ou extra (sai do `select`)


@dataclass
class ExtractedPhoto:
    time_sec: float
    reasons: list[PhotoReason]
    file: Path


@dataclass
class PhotoSelection:
    """Resultado do descarte: (foto, diferença para a mantida anterior) e (foto, índice da mantida, diferença)."""

    kept: list[tuple[ExtractedPhoto, Optional[float]]] = field(default_factory=list)
    discarded: list[tuple[ExtractedPhoto, int, float]] = field(default_factory=list)


def scene_changes(cuts: list[float], duration: float) -> list[float]:
    """Trocas de cena válidas: longe do início, do fim e da troca anterior."""
    kept: list[float] = []
    for cut in sorted(cuts):
        if cut >= SCENE_MIN_GAP_SEC and duration - cut >= SCENE_MIN_GAP_SEC and (
            not kept or cut - kept[-1] >= SCENE_MIN_GAP_SEC
        ):
            kept.append(round(cut, 3))
    return kept


def plan_photos(
    duration: float,
    interval: float = PHOTO_INTERVAL_SEC,
    sentence_starts: Optional[list[float]] = None,
    scene_cuts: Optional[list[float]] = None,
) -> list[PlannedPhoto]:
    """Fotos planejadas em ordem de tempo (grade + extras; extras perto de outra foto viram motivo)."""
    if duration <= 0:
        return []
    count = int((duration - 1e-6) // interval) + 1
    photos = [PlannedPhoto(round(k * interval, 3), ["intervalo"], True) for k in range(count)]
    extras: list[tuple[float, PhotoReason]] = [(t, "frase") for t in sentence_starts or []]
    extras += [(t, "cena") for t in scene_cuts or []]
    for time_sec, reason in sorted(extras):
        if time_sec <= 0 or time_sec >= duration - 0.05:
            continue
        near = min(photos, key=lambda p: abs(p.time_sec - time_sec))
        if abs(near.time_sec - time_sec) < EXTRA_MERGE_SEC:
            if reason not in near.reasons:
                near.reasons.append(reason)
            continue
        photos.append(PlannedPhoto(round(time_sec, 3), [reason], False))
        photos.sort(key=lambda p: p.time_sec)
    return photos


def _rate(interval: float) -> str:
    """Taxa do `fps` como fração exata (2 s → `1/2`)."""
    rate = Fraction(1) / Fraction(interval).limit_denominator(1000)
    return f"{rate.numerator}/{rate.denominator}"


def _select(times: list[float]) -> str:
    """Primeiro quadro em ou depois de cada tempo (o anterior ainda antes dele)."""
    return "+".join(f"gte(t,{t:.3f})*lt(prev_pts*TB,{t:.3f})" for t in times)


def photo_commands(
    path: Path, interval: float, extras: list[float], ffmpeg: str
) -> list[tuple[list[str], list[str]]]:
    """Comandos do FFmpeg (args, prefixos dos ramos). O 1º faz a grade (`g`) e os primeiros extras."""
    chunks = [extras[i:i + EXTRAS_PER_BRANCH] for i in range(0, len(extras), EXTRAS_PER_BRANCH)]
    groups = [chunks[i:i + BRANCHES_PER_PROCESS] for i in range(0, len(chunks), BRANCHES_PER_PROCESS)] or [[]]
    commands = []
    for number, group in enumerate(groups):
        branches: list[tuple[str, str]] = []  # (prefixo, filtro)
        if number == 0:
            branches.append(("g", f"fps=fps={_rate(interval)}"))
        offset = sum(len(g) for g in groups[:number])
        for k, chunk in enumerate(group):
            branches.append((f"e{offset + k}", f"select='{_select(chunk)}'"))
        head = "[0:v]setpts=PTS-STARTPTS"
        if len(branches) == 1:
            graph = [f"{head},{branches[0][1]},{_SCALE},{_STAMP},metadata=print:key=roteiro:file={branches[0][0]}.txt[{branches[0][0]}]"]
        else:
            graph = [head + f",split={len(branches)}" + "".join(f"[s{name}]" for name, _ in branches)]
            graph += [
                f"[s{name}]{flt},{_SCALE},{_STAMP},metadata=print:key=roteiro:file={name}.txt[{name}]"
                for name, flt in branches
            ]
        args = [
            ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
            "-i", str(Path(path).resolve()), "-an", "-sn", "-dn", "-filter_complex", ";".join(graph),
        ]
        for name, _ in branches:
            args += ["-map", f"[{name}]", "-fps_mode", "passthrough", "-q:v", str(PHOTO_QUALITY), f"{name}_%05d.jpg"]
        commands.append((args, [name for name, _ in branches]))
    return commands


def extract_photos(
    path: Path,
    planned: list[PlannedPhoto],
    out_dir: Path,
    interval: float = PHOTO_INTERVAL_SEC,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[ExtractedPhoto]:
    """Extrai as fotos planejadas para `out_dir` (`t<ms>.jpg`), com os tempos reais dos quadros.

    Raises:
        RuntimeError: o FFmpeg falhou ou não gerou as imagens.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = [p for p in planned if not p.grid]
    grid = [p for p in planned if p.grid]
    found: list[ExtractedPhoto] = []
    with tempfile.TemporaryDirectory(prefix="roteiro-fotos-", dir=out_dir) as tmp:
        for args, names in photo_commands(path, interval, [p.time_sec for p in extras], str(get_ffmpeg_path())):
            # Nomes relativos (cwd = tmp): caminhos do Windows com `C:` quebrariam a sintaxe do filtro.
            proc = run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=tmp, shell=False)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg falhou ao extrair as fotos: {(proc.stderr or '').strip()[:300]}")
            for name in names:
                meta = Path(tmp, f"{name}.txt")
                times = [t for t, _ in parse_metadata_file(meta.read_text(encoding="utf-8"))] if meta.is_file() else []
                pool = grid if name == "g" else extras
                for index, time_sec in enumerate(times, start=1):
                    image = Path(tmp, f"{name}_{index:05d}.jpg")
                    if not image.is_file():
                        raise RuntimeError(f"FFmpeg não gerou a foto {image.name}")
                    near = min(pool, key=lambda p: abs(p.time_sec - time_sec)) if pool else None
                    reasons = list(near.reasons) if near is not None else ["intervalo"]
                    target = out_dir / f"t{round(time_sec * 1000):08d}.jpg"
                    image.replace(target)
                    found.append(ExtractedPhoto(round(time_sec, 3), reasons, target))
    if not found:
        raise RuntimeError("o FFmpeg não gerou nenhuma foto")
    found.sort(key=lambda p: p.time_sec)
    unique: list[ExtractedPhoto] = []
    for photo in found:  # dois extras no mesmo quadro: uma foto só, com os dois motivos
        if unique and unique[-1].file == photo.file:
            unique[-1].reasons = sorted(set(unique[-1].reasons) | set(photo.reasons), key=_REASON_ORDER.index)
            continue
        unique.append(photo)
    return unique


_REASON_ORDER: list[str] = ["intervalo", "frase", "cena"]


def signature(image: Path) -> bytes:
    """Miniatura cinza SIGNATURE_SIDE x SIGNATURE_SIDE (para comparar fotos)."""
    with Image.open(image) as img:
        return img.convert("L").resize((SIGNATURE_SIDE, SIGNATURE_SIDE), Image.Resampling.BILINEAR).tobytes()


def signature_diff(a: bytes, b: bytes) -> float:
    """Diferença média por pixel (0–255)."""
    return round(sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a)), 2)


def select_photos(photos: list[ExtractedPhoto], signatures: Optional[list[bytes]] = None) -> PhotoSelection:
    """Descarta as fotos quase iguais à última mantida (ver o topo do módulo). A primeira sempre fica."""
    signatures = signatures if signatures is not None else [signature(p.file) for p in photos]
    selection = PhotoSelection()
    last: Optional[int] = None  # índice (em `photos`) da última mantida
    for index, photo in enumerate(photos):
        if last is None:
            selection.kept.append((photo, None))
            last = index
            continue
        diff = signature_diff(signatures[last], signatures[index])
        if diff <= DUPLICATE_MAX_DIFF and photo.time_sec - photos[last].time_sec < MAX_KEPT_GAP_SEC:
            selection.discarded.append((photo, len(selection.kept) - 1, diff))
            continue
        selection.kept.append((photo, diff))
        last = index
    return selection


def photo_rules() -> dict[str, float]:
    return {
        "photo_max_side": PHOTO_MAX_SIDE,
        "photo_quality": PHOTO_QUALITY,
        "pause_photo_sec": PAUSE_PHOTO_SEC,
        "extra_merge_sec": EXTRA_MERGE_SEC,
        "scene_min_gap_sec": SCENE_MIN_GAP_SEC,
        "duplicate_max_diff": DUPLICATE_MAX_DIFF,
        "max_kept_gap_sec": MAX_KEPT_GAP_SEC,
    }
