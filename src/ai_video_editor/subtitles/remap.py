"""Remapeamento de tempos do vídeo original para o vídeo editado."""

from ai_video_editor.planning.schema import TimeRange


class TimelineMapper:
    """Converte tempos do original em tempos do vídeo editado.

    Usa a mesma quantização do renderer (`plan_timeline`): cada trecho dura
    `max(1, round((end - start) * fps))` frames, e o início de cada trecho é a soma
    inteira de frames dos anteriores, então a legenda não escorrega com muitos cortes.
    """

    def __init__(self, ranges: list[TimeRange], fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps deve ser positivo")
        self.ranges = list(ranges)
        self.fps = fps
        self._offset_frames: list[int] = []
        self._frame_counts: list[int] = []
        total = 0
        for r in self.ranges:
            frames = max(1, round((r.end_sec - r.start_sec) * fps))
            self._offset_frames.append(total)
            self._frame_counts.append(frames)
            total += frames
        self.total_frames = total

    @property
    def duration_sec(self) -> float:
        """Duração total do vídeo editado."""
        return self.total_frames / self.fps

    def range_index(self, time_sec: float) -> int | None:
        """Índice do trecho mantido que contém `time_sec` ([start, end)), ou None."""
        for idx, r in enumerate(self.ranges):
            if r.start_sec <= time_sec < r.end_sec:
                return idx
        return None

    def range_output_bounds(self, idx: int) -> tuple[float, float]:
        """Início e fim (no vídeo editado) do trecho `idx`."""
        start = self._offset_frames[idx] / self.fps
        return start, (self._offset_frames[idx] + self._frame_counts[idx]) / self.fps

    def map_in_range(self, idx: int, time_sec: float) -> float:
        """Mapeia `time_sec` usando o trecho `idx` (o fim do trecho é aceito; fora dele, é limitado)."""
        r = self.ranges[idx]
        start, end = self.range_output_bounds(idx)
        local = min(max(time_sec - r.start_sec, 0.0), end - start)
        return start + local

    def map_time(self, time_sec: float) -> float | None:
        """Tempo no vídeo editado, ou None se `time_sec` cai em trecho removido."""
        idx = self.range_index(time_sec)
        return None if idx is None else self.map_in_range(idx, time_sec)


def remap_time(time_sec: float, ranges: list[TimeRange], fps: float) -> float | None:
    """Converte um tempo do original para o vídeo editado (None se o trecho foi removido)."""
    return TimelineMapper(ranges, fps).map_time(time_sec)
