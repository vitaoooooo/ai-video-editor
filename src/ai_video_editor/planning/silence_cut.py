"""Planejador determinístico de corte de silêncio.

Dividido em duas funções puras para que outras etapas possam entrar no meio:
`plan_silence_cuts` decide os trechos a manter (lista de `TimeRange`, que pode ser
refinada depois) e `build_edit_plan` converte a lista final em `EditPlan`.
"""

from pathlib import Path

from ai_video_editor.audio.schema import AudioAnalysisResult
from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.planning.exceptions import NoSpeechDetectedError
from ai_video_editor.planning.schema import SilenceCutParams, TimeRange

# Folga abaixo da qual um trecho é considerado vazio (1 ms).
_EPS = 0.001


def plan_silence_cuts(
    audio: AudioAnalysisResult, params: SilenceCutParams | None = None
) -> list[TimeRange]:
    """Calcula os trechos a manter a partir dos intervalos de fala.

    1. Descarta falas mais curtas que `min_keep_sec` (cliques, ruído).
    2. Expande cada fala pelo padding, limitado a [0, duração].
    3. Mescla trechos sobrepostos ou separados por menos de `min_gap_to_cut_sec`;
       a mesma regra vale para as bordas (uma pausa curta no início/fim não é cortada).
    """
    params = params or SilenceCutParams()
    duration = audio.duration_sec

    speech = [s for s in audio.speech if s.end_sec - s.start_sec >= params.min_keep_sec]
    if not speech:
        details = "só ruído curto" if audio.speech else "áudio todo em silêncio"
        raise NoSpeechDetectedError(duration, details)

    expanded = sorted(
        (
            max(0.0, s.start_sec - params.padding_before_sec),
            min(duration, s.end_sec + params.padding_after_sec),
        )
        for s in speech
    )

    merged: list[list[float]] = []
    for start, end in expanded:
        if merged and start - merged[-1][1] < params.min_gap_to_cut_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    if merged[0][0] < params.min_gap_to_cut_sec:
        merged[0][0] = 0.0
    if duration - merged[-1][1] < params.min_gap_to_cut_sec:
        merged[-1][1] = duration

    return [
        TimeRange(start_sec=round(start, 3), end_sec=round(end, 3))
        for start, end in merged
        if end - start > _EPS
    ]


def removed_ranges(keep: list[TimeRange], duration_sec: float) -> list[TimeRange]:
    """Complemento dos trechos mantidos dentro de [0, duration_sec]: o que foi cortado."""
    removed: list[TimeRange] = []
    cursor = 0.0
    for r in sorted(keep, key=lambda r: r.start_sec):
        if r.start_sec - cursor > _EPS:
            removed.append(TimeRange(start_sec=round(cursor, 3), end_sec=r.start_sec))
        cursor = max(cursor, r.end_sec)
    if duration_sec - cursor > _EPS:
        removed.append(TimeRange(start_sec=round(cursor, 3), end_sec=round(duration_sec, 3)))
    return removed


def build_edit_plan(
    source_path: Path,
    keep: list[TimeRange],
    output_path: Path,
    resolution: tuple[int, int],
    fps: float,
    title: str = "Corte de silêncio",
) -> EditPlan:
    """Converte os trechos mantidos em `EditPlan` na resolução e FPS da origem.

    A resolução é arredondada para números pares (exigência do H.264 yuv420p) e o FPS
    para inteiro, já que `EditPlan.target_fps` é inteiro (29.97 vira 30).
    """
    width, height = resolution
    return EditPlan(
        title=title,
        output_path=output_path,
        target_resolution=(max(2, width - width % 2), max(2, height - height % 2)),
        target_fps=max(1, round(fps)),
        segments=[
            ClipSegment(source_path=source_path, start_sec=r.start_sec, end_sec=r.end_sec)
            for r in keep
        ],
    )
