"""Medição e formatação dos tempos de processamento das etapas do `edit` e do `revise`."""

import time
from typing import Optional

from ai_video_editor.pipeline.schema import RunTimings, StageOutcome, StageTiming

STAGE_LABELS = {
    "probe": "Metadados",
    "silences": "Silêncios",
    "transcription": "Transcrição",
    "director": "Diretor",
    "reviser": "Revisor",
    "plan": "Plano de corte",
    "render": "Renderização",
    "subtitles": "Legendas",
    "report": "Relatório",
}

_STATUS_LABELS: dict[str, str] = {"ran": "executada", "cached": "cache", "skipped": "pulada"}


def format_duration(seconds: float) -> str:
    """`8.43` -> `8,4s`; `754` -> `12min 34s`; `720` -> `12min`."""
    seconds = max(0.0, seconds)
    tenths = round(seconds, 1)
    if tenths < 60:
        return f"{tenths:.1f}".replace(".", ",") + "s"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}min" if secs == 0 else f"{minutes}min {secs:02d}s"


def format_ratio(ratio: float) -> str:
    """`0.37` -> `0,37×`."""
    return f"{ratio:.2f}".replace(".", ",") + "×"


def summarize_run(timings: RunTimings) -> Optional[str]:
    """`57min de vídeo processados em 21min (0,37× o tempo real)`; None sem duração do vídeo."""
    video = timings.video_duration_sec
    if not video or video <= 0:
        return None
    return (
        f"{format_duration(video)} de vídeo processados em {format_duration(timings.total_sec)} "
        f"({format_ratio(timings.total_sec / video)} o tempo real)"
    )


def build_timings_section(timings: RunTimings) -> str:
    """Seção "Tempo de processamento" do relatório."""
    lines = [
        "",
        "## Tempo de processamento",
        "",
        "| Etapa | Status | Tempo |",
        "| --- | --- | --- |",
    ]
    for stage in timings.stages:
        elapsed = format_duration(stage.elapsed_sec) if stage.elapsed_sec is not None else "—"
        label = STAGE_LABELS.get(stage.name, stage.name)
        lines.append(f"| {label} | {_STATUS_LABELS[stage.status]} | {elapsed} |")
    lines.append(f"| **Total** | | **{format_duration(timings.total_sec)}** |")
    summary = summarize_run(timings)
    if summary:
        lines += ["", summary + "."]
    return "\n".join(lines) + "\n"


class TimingRecorder:
    """Acumula os tempos de relógio (`time.perf_counter`) das etapas de uma execução."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._stages: list[StageTiming] = []
        self.video_duration_sec: Optional[float] = None

    def record(self, name: str, status: StageOutcome, elapsed_sec: Optional[float] = None) -> None:
        if elapsed_sec is not None:
            elapsed_sec = round(elapsed_sec, 3)
        self._stages.append(StageTiming(name=name, status=status, elapsed_sec=elapsed_sec))

    def snapshot(self, in_progress: Optional[tuple[str, float]] = None) -> RunTimings:
        """Tempos até agora. `in_progress` = (etapa, perf_counter de início) conta a etapa em curso."""
        stages = list(self._stages)
        if in_progress is not None:
            name, started = in_progress
            stages.append(
                StageTiming(name=name, status="ran", elapsed_sec=round(time.perf_counter() - started, 3))
            )
        return RunTimings(
            stages=stages,
            total_sec=round(time.perf_counter() - self._started, 3),
            video_duration_sec=self.video_duration_sec,
        )
