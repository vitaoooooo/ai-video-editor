"""Espinha falada → Timeline: os trechos mantidos de cada clipe, em sequência, na trilha `video_main`.

Função pura (sem FFmpeg). Os trechos de cada clipe vêm do mesmo cálculo do caminho de um vídeo
(`pipeline.cutting.plan_keep_ranges`: silêncios + frases removidas ajustadas aos silêncios); aqui
eles só viram itens da base, com o `clip_id` de cada um.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ai_video_editor.core.schema import LoudnessTarget
from ai_video_editor.core.timeline import MainItem, Timeline
from ai_video_editor.planning.exceptions import PlanningError
from ai_video_editor.planning.schema import TimeRange

# Saída da espinha: Full HD; cada clipe é encaixado com letterbox/pillarbox pelo render da Timeline.
SPINE_RESOLUTION: tuple[int, int] = (1920, 1080)

# Rótulo dos itens da base que são respiros da cobertura (`respiro R1 (D001.T03)`), não fala.
BREATH_LABEL_PREFIX = "respiro "


class ClipCut(BaseModel):
    """Trechos mantidos de um clipe (tempos relativos ao clipe)."""

    clip_id: str
    source_path: Path
    keep: list[TimeRange] = Field(default_factory=list)


def spine_fps(source_fps: list[float]) -> int:
    """FPS de saída: o mais comum entre os clipes, arredondado para inteiro (a Timeline exige inteiro).

    29,97 vira 30 (cada clipe é convertido pelo filtro `fps` do render); empate fica com o primeiro clipe.
    """
    rounded = [max(1, round(f)) for f in source_fps if f > 0]
    if not rounded:
        return 30
    counts = Counter(rounded)
    best = max(counts.values())
    return next(f for f in rounded if counts[f] == best)


def usable_ranges(keep: list[TimeRange], fps: int) -> list[TimeRange]:
    """Trechos com pelo menos 1 frame no fps de saída (a Timeline recusa itens menores)."""
    return [r for r in keep if round((r.end_sec - r.start_sec) * fps) >= 1]


def build_spine_timeline(
    clips: list[ClipCut],
    output_path: Path,
    fps: int,
    resolution: tuple[int, int] = SPINE_RESOLUTION,
    title: str = "Espinha falada",
    loudness: Optional[LoudnessTarget] = None,
) -> Timeline:
    """Timeline com os trechos de cada clipe em sequência, na ordem de `clips`.

    Trechos menores que 1 frame são descartados. Cada item leva o `clip_id` e um rótulo
    `P006 #3` (clipe e número do trecho), usados nos relatórios.

    Raises:
        PlanningError: nenhum trecho sobrou em nenhum clipe.
    """
    items: list[MainItem] = []
    for clip in clips:
        for n, r in enumerate(usable_ranges(clip.keep, fps), start=1):
            items.append(
                MainItem(
                    source_path=clip.source_path,
                    source_in=r.start_sec,
                    source_out=r.end_sec,
                    clip_id=clip.clip_id,
                    label=f"{clip.clip_id} #{n}",
                )
            )
    if not items:
        raise PlanningError("nenhum trecho de fala sobrou nos clipes da espinha")
    return Timeline(
        title=title, output_path=output_path, resolution=resolution, fps=fps, loudness=loudness, video_main=items
    )


@dataclass(frozen=True)
class Placement:
    """Onde um item da base (trecho de um clipe) está na timeline.

    `concat_frame` é a posição do trecho entre os trechos do mesmo clipe, como se só eles
    estivessem em sequência (o tempo "editado" do clipe, que a legenda e o diff usam).
    """

    item_index: int
    source_in: float
    source_out: float
    start_frame: int
    frames: int
    concat_frame: int


def is_breath_item(item: MainItem) -> bool:
    """Item da base inserido como respiro (cobertura), não fala da espinha."""
    return (item.label or "").startswith(BREATH_LABEL_PREFIX)


def clip_placements(timeline: Timeline) -> dict[str, list[Placement]]:
    """Trechos de fala de cada clipe na timeline, em ordem; respiros ficam de fora.

    Funciona com os trechos de um clipe separados por respiros (a espinha com cobertura): cada
    trecho guarda onde começa na timeline e onde começaria com só o clipe em sequência.
    """
    placements: dict[str, list[Placement]] = {}
    position = 0
    for idx, item in enumerate(timeline.video_main):
        frames = timeline.item_frames(item)
        if not is_breath_item(item):
            clip = placements.setdefault(item.clip_id or "", [])
            concat = clip[-1].concat_frame + clip[-1].frames if clip else 0
            clip.append(Placement(idx, item.source_in, item.source_out, position, frames, concat))
        position += frames
    return placements


def map_source_range(
    placements: list[Placement], start_sec: float, end_sec: float, fps: int
) -> Optional[tuple[int, int, list[tuple[float, float]]]]:
    """Trecho [start_sec, end_sec) do clipe → (primeiro frame, frame final exclusivo) na timeline e os
    pedaços de origem que ficaram; None se nada dele ficou.

    Uma frase aparada pelos cortes de silêncio começa no primeiro instante mantido e termina no
    último (os cortes no meio dela não mudam os limites).
    """
    first: Optional[int] = None
    last = 0
    kept: list[tuple[float, float]] = []
    for p in placements:
        lo, hi = max(start_sec, p.source_in), min(end_sec, p.source_out)
        if hi <= lo:
            continue
        a = p.start_frame + min(p.frames, max(0, round((lo - p.source_in) * fps)))
        b = p.start_frame + min(p.frames, max(0, round((hi - p.source_in) * fps)))
        if b <= a:
            continue
        first = a if first is None else first
        last = b
        kept.append((lo, hi))
    if first is None:
        return None
    return first, last, kept


def concat_to_final(placements: list[Placement], seconds: float, fps: int, end: bool = False) -> float:
    """Tempo "editado" do clipe (trechos do clipe em sequência) → tempo na timeline.

    `end=True` trata um tempo exatamente no limite de dois trechos como fim do anterior (fim de
    frase/bloco), em vez de início do seguinte.
    """
    frame = seconds * fps
    tolerance = 1e-6 if end else -1e-6
    chosen = next((p for p in placements if frame < p.concat_frame + p.frames + tolerance), placements[-1])
    return seconds + (chosen.start_frame - chosen.concat_frame) / fps


def clip_spans(timeline: Timeline) -> dict[str, tuple[int, int]]:
    """(primeiro frame, frame final exclusivo) de cada clipe na timeline, pela ordem dos itens da base."""
    spans: dict[str, tuple[int, int]] = {}
    position = 0
    for item in timeline.video_main:
        frames = timeline.item_frames(item)
        key = item.clip_id or ""
        start = spans[key][0] if key in spans else position
        spans[key] = (start, position + frames)
        position += frames
    return spans
