"""Timeline com trilhas (base, overlay de vídeo, ambiente de áudio) e seu achatamento.

A Timeline é o dado que descreve uma edição montada a partir de vários clipes:

- `video_main`: itens contíguos que formam a base; a duração da timeline é a soma deles;
- `video_overlay`: itens que substituem a imagem da base numa janela (o áudio da base continua);
- áudio principal: o áudio dos próprios itens de `video_main` (ou silêncio, com `use_audio=False`);
- `audio_ambient`: itens de áudio posicionados no tempo da timeline e mixados por cima.

Todos os tempos são quantizados em frames do fps de saída (a mesma regra do renderer do
`EditPlan`), e as posições de áudio em amostras são derivadas das posições em frames, para
que áudio e vídeo nunca escorreguem um em relação ao outro.

`flatten` é uma função pura (sem FFmpeg): transforma as trilhas em uma lista de pedaços de
vídeo em sequência e em um plano de áudio, que o renderer executa reaproveitando o render
segmentado (um FFmpeg por pedaço + concat), em vez de filtros de overlay complexos.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_video_editor.core.schema import LoudnessTarget

if TYPE_CHECKING:
    from ai_video_editor.core.schema import EditPlan

TIMELINE_SCHEMA_VERSION = 1

# Taxa de amostragem do áudio produzido pelo render da Timeline (48 kHz estéreo).
TIMELINE_SAMPLE_RATE = 48000


class _SourceRange(BaseModel):
    """Trecho de um arquivo de origem: [source_in, source_out) em segundos da origem."""

    model_config = ConfigDict(extra="forbid")

    source_path: Path = Field(..., description="Arquivo de origem")
    source_in: float = Field(..., ge=0.0, description="Início do trecho na origem, em segundos")
    source_out: float = Field(..., gt=0.0, description="Fim do trecho na origem, em segundos")
    clip_id: Optional[str] = Field(default=None, description="Identificador livre do clipe (ex.: D003), para relatórios")
    label: Optional[str] = Field(default=None, description="Descrição curta do item, para relatórios")

    @model_validator(mode="after")
    def _validate_range(self):
        if self.source_out <= self.source_in:
            raise ValueError(
                f"tempos invertidos: source_out ({self.source_out}) deve ser maior que "
                f"source_in ({self.source_in})"
            )
        return self

    @property
    def duration_sec(self) -> float:
        return self.source_out - self.source_in


class MainItem(_SourceRange):
    """Item da base (`video_main`): imagem e, por padrão, áudio do próprio clipe."""

    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Volume do áudio do item")
    use_audio: bool = Field(
        default=True, description="Usar o áudio do item (False = silêncio no lugar do áudio do item)"
    )
    timeline_start: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Opcional: posição esperada na timeline; se informada, precisa bater com a soma dos anteriores",
    )


class OverlayItem(_SourceRange):
    """Item de overlay (`video_overlay`): substitui a imagem da base a partir de `timeline_start`."""

    timeline_start: float = Field(..., ge=0.0, description="Início do overlay na timeline, em segundos")


class AmbientItem(_SourceRange):
    """Item de áudio de ambiente (`audio_ambient`), mixado por cima do áudio principal."""

    timeline_start: float = Field(..., ge=0.0, description="Início do áudio na timeline, em segundos")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Volume do item na mixagem")
    fade_in_sec: float = Field(default=0.0, ge=0.0, description="Fade in do item, em segundos")
    fade_out_sec: float = Field(default=0.0, ge=0.0, description="Fade out do item, em segundos")


def _describe(track: str, idx: int, item: _SourceRange) -> str:
    """`video_overlay[1] (D003)`: identifica o item nas mensagens de erro."""
    return f"{track}[{idx}]" + (f" ({item.clip_id})" if item.clip_id else "")


class Timeline(BaseModel):
    """Edição com trilhas, validada e quantizada em frames do fps de saída."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(default=TIMELINE_SCHEMA_VERSION, description="Versão do formato da Timeline")
    title: str = Field(default="Untitled Timeline", description="Título da edição")
    output_path: Path = Field(..., description="MP4 de saída")
    resolution: Tuple[int, int] = Field(default=(1920, 1080), description="(largura, altura) da saída")
    fps: int = Field(default=30, gt=0, le=240, description="Frames por segundo da saída")
    audio_fade_sec: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Micro fade de áudio (in e out) em cada item da base, em segundos (0 desativa)",
    )
    loudness: Optional[LoudnessTarget] = Field(
        default=None, description="Normalização de loudness do áudio final em duas passadas (None mantém o volume)"
    )
    video_main: list[MainItem] = Field(..., min_length=1, description="Base contígua da timeline")
    video_overlay: list[OverlayItem] = Field(default_factory=list, description="Overlays de imagem sobre a base")
    audio_ambient: list[AmbientItem] = Field(default_factory=list, description="Áudios de ambiente posicionados")

    # --- quantização ---

    def to_frame(self, sec: float) -> int:
        """Posição/duração em segundos → frames do fps de saída (arredondamento)."""
        return round(sec * self.fps)

    def item_frames(self, item: _SourceRange) -> int:
        """Duração de um item em frames."""
        return self.to_frame(item.duration_sec)

    def to_sample(self, frame: int) -> int:
        """Posição em frames → amostra de áudio (derivada do frame, sem acumular desvio)."""
        return round(frame * TIMELINE_SAMPLE_RATE / self.fps)

    @property
    def total_frames(self) -> int:
        return sum(self.item_frames(item) for item in self.video_main)

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps

    # --- validação ---

    @model_validator(mode="after")
    def _validate_timeline(self) -> "Timeline":
        width, height = self.resolution
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f"resolução inválida {self.resolution}: largura e altura precisam ser pares e positivas")

        tracks: list[tuple[str, list[_SourceRange]]] = [
            ("video_main", self.video_main),
            ("video_overlay", self.video_overlay),
            ("audio_ambient", self.audio_ambient),
        ]
        for track, items in tracks:
            for idx, item in enumerate(items):
                if self.item_frames(item) < 1:
                    raise ValueError(
                        f"{_describe(track, idx, item)} dura {item.duration_sec:.4f}s, menos de 1 frame "
                        f"a {self.fps} fps"
                    )

        position = 0
        for idx, item in enumerate(self.video_main):
            if item.timeline_start is not None:
                expected = self.to_frame(item.timeline_start)
                if expected > position:
                    raise ValueError(
                        f"buraco na base: {_describe('video_main', idx, item)} começa em "
                        f"{item.timeline_start}s, mas o item anterior termina em {position / self.fps:.3f}s"
                    )
                if expected < position:
                    raise ValueError(
                        f"sobreposição na base: {_describe('video_main', idx, item)} começa em "
                        f"{item.timeline_start}s, antes do fim do item anterior ({position / self.fps:.3f}s)"
                    )
            position += self.item_frames(item)
        total = position

        windows = sorted(
            (self.to_frame(item.timeline_start), self.to_frame(item.timeline_start) + self.item_frames(item), idx)
            for idx, item in enumerate(self.video_overlay)
        )
        for start, end, idx in windows:
            if end > total:
                raise ValueError(
                    f"{_describe('video_overlay', idx, self.video_overlay[idx])} vai até "
                    f"{end / self.fps:.3f}s, além da duração da timeline ({total / self.fps:.3f}s)"
                )
        for (_, prev_end, prev_idx), (start, _, idx) in zip(windows, windows[1:]):
            if start < prev_end:
                raise ValueError(
                    f"overlays sobrepostos: {_describe('video_overlay', idx, self.video_overlay[idx])} começa em "
                    f"{start / self.fps:.3f}s, antes do fim de "
                    f"{_describe('video_overlay', prev_idx, self.video_overlay[prev_idx])} ({prev_end / self.fps:.3f}s)"
                )

        for idx, item in enumerate(self.audio_ambient):
            end = self.to_frame(item.timeline_start) + self.item_frames(item)
            if end > total:
                raise ValueError(
                    f"{_describe('audio_ambient', idx, item)} vai até {end / self.fps:.3f}s, além da "
                    f"duração da timeline ({total / self.fps:.3f}s)"
                )
            if item.fade_in_sec + item.fade_out_sec > item.duration_sec:
                raise ValueError(
                    f"{_describe('audio_ambient', idx, item)}: fade in + fade out "
                    f"({item.fade_in_sec + item.fade_out_sec}s) maior que a duração do item ({item.duration_sec:.3f}s)"
                )
        return self


def timeline_from_edit_plan(
    plan: "EditPlan", source_durations: Optional[Mapping[Path, float]] = None
) -> Timeline:
    """Converte os segmentos de um EditPlan em `video_main` com o áudio de cada item.

    `source_durations` (chave: caminho resolvido) dá o fim dos segmentos sem `end_sec` e limita
    os que passam do fim da origem, como o renderer do EditPlan faz.
    """
    durations = {Path(k).resolve(): v for k, v in (source_durations or {}).items()}
    items: list[MainItem] = []
    for idx, seg in enumerate(plan.segments):
        source_duration = durations.get(seg.source_path.resolve())
        end = seg.end_sec
        if source_duration is not None and source_duration > 0:
            end = source_duration if end is None else min(end, source_duration)
        if end is None:
            raise ValueError(
                f"segmento {idx} sem end_sec: informe a duração da origem ({seg.source_path}) em source_durations"
            )
        items.append(MainItem(source_path=seg.source_path, source_in=seg.start_sec, source_out=end, volume=seg.volume))
    return Timeline(
        title=plan.title,
        output_path=plan.output_path,
        resolution=plan.target_resolution,
        fps=plan.target_fps,
        audio_fade_sec=plan.audio_fade_sec,
        loudness=plan.loudness,
        video_main=items,
    )


# --- achatamento ---

VideoTrack = Literal["main", "overlay"]


@dataclass(frozen=True)
class VideoPiece:
    """Pedaço contíguo de vídeo da saída, vindo de uma única origem."""

    source_path: Path
    source_in: float  # segundos na origem
    source_out: float  # source_in + frame_count / fps
    timeline_start_frame: int
    frame_count: int
    track: VideoTrack
    item_index: int  # índice do item na sua trilha
    clip_id: Optional[str] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class MainAudioPiece:
    """Áudio de um item da base: `source_path` None = silêncio (item marcado sem áudio)."""

    source_path: Optional[Path]
    source_in: float
    volume: float
    start_sample: int
    sample_count: int
    item_index: int


@dataclass(frozen=True)
class AmbientAudioPiece:
    """Item de ambiente com a posição absoluta na timeline em amostras."""

    source_path: Path
    source_in: float
    volume: float
    start_sample: int
    sample_count: int
    fade_in_sec: float
    fade_out_sec: float
    item_index: int


@dataclass(frozen=True)
class FlatTimeline:
    """Timeline achatada: vídeo em sequência + plano de áudio (principal concatenado + ambiente)."""

    fps: int
    sample_rate: int
    total_frames: int
    total_samples: int
    video: list[VideoPiece]
    audio_main: list[MainAudioPiece]
    audio_ambient: list[AmbientAudioPiece]

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps


def flatten(timeline: Timeline) -> FlatTimeline:
    """Achata as trilhas: a base é recortada onde há overlay e cada janela vira um pedaço do overlay."""
    fps = timeline.fps

    def piece(item: _SourceRange, track: VideoTrack, idx: int, offset: int, start: int, frames: int) -> VideoPiece:
        source_in = item.source_in + offset / fps
        return VideoPiece(
            source_path=item.source_path,
            source_in=source_in,
            source_out=source_in + frames / fps,
            timeline_start_frame=start,
            frame_count=frames,
            track=track,
            item_index=idx,
            clip_id=item.clip_id,
            label=item.label,
        )

    spans: list[tuple[int, int, int, MainItem]] = []  # (início, fim, índice, item) em frames
    audio_main: list[MainAudioPiece] = []
    position = 0
    for idx, item in enumerate(timeline.video_main):
        end = position + timeline.item_frames(item)
        spans.append((position, end, idx, item))
        start_sample, end_sample = timeline.to_sample(position), timeline.to_sample(end)
        audio_main.append(
            MainAudioPiece(
                source_path=item.source_path if item.use_audio else None,
                source_in=item.source_in,
                volume=item.volume,
                start_sample=start_sample,
                sample_count=end_sample - start_sample,
                item_index=idx,
            )
        )
        position = end

    video: list[VideoPiece] = []

    def fill_with_base(lo: int, hi: int) -> None:
        """Pedaços da base que cobrem [lo, hi) da timeline."""
        for start, end, idx, item in spans:
            a, b = max(lo, start), min(hi, end)
            if a < b:
                video.append(piece(item, "main", idx, a - start, a, b - a))

    windows = sorted(
        (timeline.to_frame(o.timeline_start), idx, o) for idx, o in enumerate(timeline.video_overlay)
    )
    cursor = 0
    for start, idx, overlay in windows:
        fill_with_base(cursor, start)
        frames = timeline.item_frames(overlay)
        video.append(piece(overlay, "overlay", idx, 0, start, frames))
        cursor = start + frames
    fill_with_base(cursor, position)

    ambient = []
    for idx, item in enumerate(timeline.audio_ambient):
        start = timeline.to_frame(item.timeline_start)
        start_sample = timeline.to_sample(start)
        end_sample = timeline.to_sample(start + timeline.item_frames(item))
        ambient.append(
            AmbientAudioPiece(
                source_path=item.source_path,
                source_in=item.source_in,
                volume=item.volume,
                start_sample=start_sample,
                sample_count=end_sample - start_sample,
                fade_in_sec=item.fade_in_sec,
                fade_out_sec=item.fade_out_sec,
                item_index=idx,
            )
        )

    return FlatTimeline(
        fps=fps,
        sample_rate=TIMELINE_SAMPLE_RATE,
        total_frames=position,
        total_samples=timeline.to_sample(position),
        video=video,
        audio_main=audio_main,
        audio_ambient=ambient,
    )
