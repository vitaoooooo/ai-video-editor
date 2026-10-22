"""Espinha falada → Timeline: os trechos mantidos de cada clipe, em sequência, na trilha `video_main`.

Função pura (sem FFmpeg). Os trechos de cada clipe vêm do mesmo cálculo do caminho de um vídeo
(`pipeline.cutting.plan_keep_ranges`: silêncios + frases removidas ajustadas aos silêncios); aqui
eles só viram itens da base, com o `clip_id` de cada um.
"""

from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ai_video_editor.core.schema import LoudnessTarget
from ai_video_editor.core.timeline import MainItem, Timeline
from ai_video_editor.planning.exceptions import PlanningError
from ai_video_editor.planning.schema import TimeRange

# Saída da espinha: Full HD; cada clipe é encaixado com letterbox/pillarbox pelo render da Timeline.
SPINE_RESOLUTION: tuple[int, int] = (1920, 1080)


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
