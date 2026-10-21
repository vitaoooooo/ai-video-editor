"""Compila uma Timeline achatada em comandos FFmpeg (lista de argumentos, sem shell).

Um comando por pedaço de vídeo (sem áudio, padronizado) e um por item do áudio principal,
todos independentes para rodar em paralelo; depois os concats (demuxer), a mixagem do
ambiente e o mux final. Os intermediários de áudio são PCM a 48 kHz estéreo.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ai_video_editor.core.timeline import (
    TIMELINE_SAMPLE_RATE,
    AmbientAudioPiece,
    FlatTimeline,
    MainAudioPiece,
    Timeline,
    VideoPiece,
    flatten,
)
from ai_video_editor.renderer.compiler import INPUT_READ_MARGIN_SEC, VIDEO_ENCODE_ARGS
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path

FINAL_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", str(TIMELINE_SAMPLE_RATE)]
PCM_AUDIO_ARGS = ["-c:a", "pcm_s16le", "-ar", str(TIMELINE_SAMPLE_RATE), "-ac", "2"]

# Funções de transferência de vídeo HDR (PQ e HLG).
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}

_QUIET = ["-y", "-nostdin", "-hide_banner", "-loglevel", "error"]


@dataclass(frozen=True)
class MediaInfo:
    """O que o render da Timeline precisa saber de cada origem (uma chamada ao ffprobe)."""

    duration_sec: float
    has_video: bool
    has_audio: bool
    pix_fmt: Optional[str] = None
    color_transfer: Optional[str] = None

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS


def probe_media(path: Path, ffprobe_bin: Optional[Path] = None) -> MediaInfo:
    """Duração, presença de vídeo/áudio e formato de cor da primeira trilha de vídeo."""
    args = [
        str(ffprobe_bin or get_ffprobe_path()),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,pix_fmt,color_transfer",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Falha ao inspecionar {path} com ffprobe:\n{res.stderr.strip()}")
    data = json.loads(res.stdout or "{}")
    try:
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    return MediaInfo(
        duration_sec=duration,
        has_video=video is not None,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        pix_fmt=(video or {}).get("pix_fmt"),
        color_transfer=(video or {}).get("color_transfer"),
    )


@dataclass(frozen=True)
class TimelineRenderJob:
    """Comandos de um render de Timeline e o que se espera da saída."""

    piece_commands: list[list[str]]  # vídeo e áudio principal, independentes (paralelos)
    video_list_path: Path
    video_list_content: str
    audio_list_path: Path
    audio_list_content: str
    audio_command: list[str]  # concat do áudio principal (+ mixagem do ambiente) → audio_path
    audio_path: Path
    expected_frames: int
    expected_duration_sec: float
    has_audio: bool  # algum item usa áudio de verdade (senão é só silêncio gerado)
    warnings: list[str] = field(default_factory=list)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _video_filters(width: int, height: int, fps: int, frame_count: int) -> str:
    """Padroniza o pedaço (proporção preservada com letterbox/pillarbox) e fixa a contagem de frames.

    A rotação de metadados já vem aplicada pelo autorotate do FFmpeg na decodificação.
    `format=yuv420p` converte fontes de 10 bits para 8 bits.
    """
    return (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,format=yuv420p,"
        f"tpad=stop_mode=clone:stop=-1,"
        f"trim=end_frame={frame_count},setpts=PTS-STARTPTS"
    )


def _fit_audio(sample_count: int) -> str:
    """Reamostra para 48 kHz estéreo e fixa a contagem exata de amostras (completa com silêncio)."""
    return (
        f"aresample={TIMELINE_SAMPLE_RATE}:first_pts=0,"
        f"aformat=sample_fmts=s16:channel_layouts=stereo,"
        f"apad,atrim=end_sample={sample_count},asetpts=PTS-STARTPTS"
    )


def _fades(sample_count: int, fade_in: float, fade_out: float) -> str:
    duration = sample_count / TIMELINE_SAMPLE_RATE
    chain = ""
    if fade_in > 0:
        chain += f",afade=t=in:st=0:d={_fmt(fade_in)}"
    if fade_out > 0:
        chain += f",afade=t=out:st={_fmt(duration - fade_out)}:d={_fmt(fade_out)}"
    return chain


def _silence(sample_count: int) -> str:
    return (
        f"anullsrc=channel_layout=stereo:sample_rate={TIMELINE_SAMPLE_RATE},"
        f"aformat=sample_fmts=s16:channel_layouts=stereo,"
        f"atrim=end_sample={sample_count}"
    )


class TimelineCompiler:
    """Transforma uma Timeline em comandos FFmpeg determinísticos."""

    def __init__(self, ffmpeg_bin: Optional[Path] = None, ffprobe_bin: Optional[Path] = None):
        self.ffmpeg_bin = ffmpeg_bin or get_ffmpeg_path()
        self.ffprobe_bin = ffprobe_bin or get_ffprobe_path()

    def probe_sources(self, timeline: Timeline) -> dict[Path, MediaInfo]:
        """Inspeciona cada origem distinta uma vez e confere se cada item cabe na sua origem."""
        infos: dict[Path, MediaInfo] = {}
        tolerance = 1 / timeline.fps
        tracks = [
            ("video_main", timeline.video_main, "video"),
            ("video_overlay", timeline.video_overlay, "video"),
            ("audio_ambient", timeline.audio_ambient, "audio"),
        ]
        for track, items, needs in tracks:
            for idx, item in enumerate(items):
                name = f"{track}[{idx}]" + (f" ({item.clip_id})" if item.clip_id else "")
                if not item.source_path.exists():
                    raise FileNotFoundError(f"{name}: arquivo de origem não encontrado: {item.source_path}")
                key = item.source_path.resolve()
                if key not in infos:
                    infos[key] = probe_media(key, self.ffprobe_bin)
                info = infos[key]
                if needs == "video" and not info.has_video:
                    raise ValueError(f"{name}: a origem não tem trilha de vídeo: {item.source_path}")
                if needs == "audio" and not info.has_audio:
                    raise ValueError(f"{name}: a origem não tem trilha de áudio: {item.source_path}")
                if info.duration_sec > 0 and item.source_out > info.duration_sec + tolerance:
                    raise ValueError(
                        f"{name}: termina em {item.source_out}s, depois do fim da origem "
                        f"({info.duration_sec:.3f}s): {item.source_path}"
                    )
        return infos

    @staticmethod
    def source_warnings(infos: dict[Path, MediaInfo]) -> list[str]:
        """Avisos sobre as origens (HDR convertido sem tonemapping)."""
        return [
            f"origem HDR ({info.color_transfer}) convertida para SDR 8 bits sem tonemapping; "
            f"as cores podem ficar lavadas: {path}"
            for path, info in infos.items()
            if info.is_hdr
        ]

    def compile(self, timeline: Timeline, work_dir: Path) -> TimelineRenderJob:
        infos = self.probe_sources(timeline)
        flat = flatten(timeline)

        commands: list[list[str]] = []
        video_lines = ["ffconcat version 1.0"]
        for idx, piece in enumerate(flat.video):
            path = work_dir / f"v_{idx:05d}.mov"
            commands.append(self._video_command(timeline, piece, path))
            video_lines += [f"file '{path.name}'", f"duration {_fmt(piece.frame_count / flat.fps)}"]

        audio_lines = ["ffconcat version 1.0"]
        has_audio = False
        for idx, piece in enumerate(flat.audio_main):
            path = work_dir / f"a_{idx:05d}.wav"
            source_has_audio = piece.source_path is not None and infos[piece.source_path.resolve()].has_audio
            has_audio = has_audio or source_has_audio
            commands.append(self._main_audio_command(timeline, piece, source_has_audio, path))
            audio_lines += [f"file '{path.name}'", f"duration {_fmt(piece.sample_count / flat.sample_rate)}"]

        audio_list_path = work_dir / "audio.txt"
        audio_path = work_dir / "audio.wav"
        return TimelineRenderJob(
            piece_commands=commands,
            video_list_path=work_dir / "video.txt",
            video_list_content="\n".join(video_lines) + "\n",
            audio_list_path=audio_list_path,
            audio_list_content="\n".join(audio_lines) + "\n",
            audio_command=self._audio_command(flat, audio_list_path, audio_path),
            audio_path=audio_path,
            expected_frames=flat.total_frames,
            expected_duration_sec=flat.duration_sec,
            has_audio=has_audio or bool(flat.audio_ambient),
            warnings=self.source_warnings(infos),
        )

    def _video_command(self, timeline: Timeline, piece: VideoPiece, out: Path) -> list[str]:
        width, height = timeline.resolution
        read = piece.source_out - piece.source_in + INPUT_READ_MARGIN_SEC
        return [
            str(self.ffmpeg_bin),
            *_QUIET,
            "-ss",
            _fmt(piece.source_in),
            "-t",
            _fmt(read),
            "-i",
            str(piece.source_path.resolve()),
            "-filter_complex",
            f"[0:v:0]{_video_filters(width, height, timeline.fps, piece.frame_count)}[v]",
            "-map",
            "[v]",
            "-an",
            *VIDEO_ENCODE_ARGS,
            "-r",
            str(timeline.fps),
            str(out),
        ]

    def _main_audio_command(
        self, timeline: Timeline, piece: MainAudioPiece, source_has_audio: bool, out: Path
    ) -> list[str]:
        """Áudio de um item da base com os micro fades nos cortes; silêncio se não houver áudio."""
        if not source_has_audio:
            return [
                str(self.ffmpeg_bin),
                *_QUIET,
                "-filter_complex",
                f"{_silence(piece.sample_count)}[a]",
                "-map",
                "[a]",
                *PCM_AUDIO_ARGS,
                str(out),
            ]

        fade = min(timeline.audio_fade_sec, piece.sample_count / TIMELINE_SAMPLE_RATE / 2)
        read = piece.sample_count / TIMELINE_SAMPLE_RATE + INPUT_READ_MARGIN_SEC
        chain = f"[0:a:0]volume={piece.volume},{_fit_audio(piece.sample_count)}{_fades(piece.sample_count, fade, fade)}[a]"
        return [
            str(self.ffmpeg_bin),
            *_QUIET,
            "-ss",
            _fmt(piece.source_in),
            "-t",
            _fmt(read),
            "-i",
            str(piece.source_path.resolve()),
            "-filter_complex",
            chain,
            "-map",
            "[a]",
            *PCM_AUDIO_ARGS,
            str(out),
        ]

    def _audio_command(self, flat: FlatTimeline, audio_list_path: Path, out: Path) -> list[str]:
        """Concatena o áudio principal e mixa o ambiente posicionado (amix sem normalizar volumes)."""
        args = [str(self.ffmpeg_bin), *_QUIET, "-f", "concat", "-safe", "0", "-i", str(audio_list_path)]
        if not flat.audio_ambient:
            return [*args, "-map", "0:a:0", "-c:a", "copy", str(out)]

        chains: list[str] = []
        labels = ["[0:a]"]
        for n, item in enumerate(flat.audio_ambient, start=1):
            args += [
                "-ss",
                _fmt(item.source_in),
                "-t",
                _fmt(item.sample_count / flat.sample_rate + INPUT_READ_MARGIN_SEC),
                "-i",
                str(item.source_path.resolve()),
            ]
            chains.append(self._ambient_chain(n, item))
            labels.append(f"[amb{n}]")
        chains.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0:dropout_transition=0,"
            f"aformat=sample_fmts=s16:channel_layouts=stereo,"
            f"atrim=end_sample={flat.total_samples}[a]"
        )
        return [*args, "-filter_complex", ";".join(chains), "-map", "[a]", *PCM_AUDIO_ARGS, str(out)]

    @staticmethod
    def _ambient_chain(input_index: int, item: AmbientAudioPiece) -> str:
        return (
            f"[{input_index}:a:0]volume={item.volume},{_fit_audio(item.sample_count)}"
            f"{_fades(item.sample_count, item.fade_in_sec, item.fade_out_sec)},"
            f"adelay=delays={item.start_sample}S:all=1[amb{input_index}]"
        )

    def mux_command(
        self, job: TimelineRenderJob, output_path: Path, audio_filter: Optional[str] = None
    ) -> list[str]:
        """Mux final: copia o vídeo concatenado e codifica o AAC uma única vez (com o loudnorm, se houver)."""
        return [
            str(self.ffmpeg_bin),
            *_QUIET,
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(job.video_list_path),
            "-i",
            str(job.audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *(["-af", audio_filter] if audio_filter else []),
            *FINAL_AUDIO_ARGS,
            "-movflags",
            "+faststart",
            str(output_path.resolve()),
        ]
