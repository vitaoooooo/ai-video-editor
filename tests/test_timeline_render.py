"""Testes do render da Timeline: comandos gerados, render com formatos misturados e equivalência com o EditPlan."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from ai_video_editor.core.schema import ClipSegment, EditPlan, LoudnessTarget
from ai_video_editor.core.timeline import Timeline, timeline_from_edit_plan
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path, get_ffprobe_path
from ai_video_editor.renderer.timeline_compiler import MediaInfo, TimelineCompiler
from ai_video_editor.renderer.timeline_executor import TimelineRenderExecutor


def _streams(path: Path) -> dict[str, dict]:
    args = [
        str(get_ffprobe_path()),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,duration,nb_frames,avg_frame_rate,sample_rate,channels,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    return {s["codec_type"]: s for s in data["streams"]}


def _rms(path: Path, start: float, duration: float) -> float:
    """RMS (0–1) do áudio da saída numa janela, decodificado em PCM mono."""
    args = [
        str(get_ffmpeg_path()),
        "-v",
        "error",
        "-ss",
        f"{start}",
        "-t",
        f"{duration}",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-f",
        "s16le",
        "-",
    ]
    raw = subprocess.run(args, capture_output=True, check=True).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def _gray_columns(path: Path, frame: int, width: int, height: int) -> np.ndarray:
    """Média de luminância de cada coluna de um frame da saída."""
    args = [
        str(get_ffmpeg_path()),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    raw = subprocess.run(args, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width).mean(axis=0)


def _graph(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


# --- comandos (só ffprobe) ---


def test_compile_commands(mixed_sources, tmp_path: Path):
    s = mixed_sources
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        resolution=(640, 360),
        video_main=[
            {"source_path": s["main"], "source_in": 0.0, "source_out": 1.0},
            {"source_path": s["mute"], "source_in": 0.0, "source_out": 1.0},
            {"source_path": s["main"], "source_in": 1.0, "source_out": 2.0, "use_audio": False},
        ],
        video_overlay=[{"source_path": s["vertical"], "source_in": 0.0, "source_out": 0.5, "timeline_start": 0.5}],
        audio_ambient=[
            {"source_path": s["small"], "source_in": 0.0, "source_out": 1.0, "timeline_start": 1.5, "volume": 0.5}
        ],
    )
    job = TimelineCompiler().compile(tl, tmp_path)

    video_cmds = [c for c in job.piece_commands if c[-1].endswith(".mov")]
    audio_cmds = [c for c in job.piece_commands if c[-1].endswith(".wav")]
    assert len(video_cmds) == 4  # base, overlay (termina na fronteira do item), item mudo, último item
    assert len(audio_cmds) == 3  # um por item da base: o overlay não mexe no áudio
    for cmd in video_cmds:
        assert "-an" in cmd and cmd.index("-ss") < cmd.index("-i")
        graph = _graph(cmd)
        assert "pad=640:360" in graph and "setsar=1" in graph and "format=yuv420p" in graph

    assert "[0:a:0]" in _graph(audio_cmds[0]) and "afade=t=in" in _graph(audio_cmds[0])
    assert _graph(audio_cmds[1]).startswith("anullsrc")  # origem sem áudio
    assert _graph(audio_cmds[2]).startswith("anullsrc")  # item marcado sem áudio
    assert "atrim=end_sample=48000" in _graph(audio_cmds[0])

    mix = _graph(job.audio_command)
    assert "amix=inputs=2:duration=first:normalize=0" in mix
    assert "adelay=delays=72000S" in mix and "volume=0.5" in mix
    assert job.expected_frames == 90 and job.has_audio


def test_compile_without_ambient_copies_main_audio(mixed_sources, tmp_path: Path):
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        video_main=[{"source_path": mixed_sources["mute"], "source_in": 0.0, "source_out": 1.0}],
    )
    job = TimelineCompiler().compile(tl, tmp_path)
    assert "-filter_complex" not in job.audio_command and "copy" in job.audio_command
    assert not job.has_audio


def test_compile_rejects_items_beyond_source_or_without_streams(
    mixed_sources, synthetic_audio_only_file, tmp_path: Path
):
    base = {"source_path": mixed_sources["main"], "source_in": 0.0, "source_out": 2.0}
    too_long = Timeline(
        output_path=tmp_path / "o.mp4",
        video_main=[{**base, "source_in": 2.0, "source_out": 4.0, "clip_id": "A1"}],
    )
    with pytest.raises(ValueError, match=r"video_main\[0\] \(A1\): termina em 4.0s"):
        TimelineCompiler().compile(too_long, tmp_path)

    no_video = Timeline(
        output_path=tmp_path / "o.mp4",
        video_main=[base],
        video_overlay=[{"source_path": synthetic_audio_only_file, "source_in": 0, "source_out": 1, "timeline_start": 0}],
    )
    with pytest.raises(ValueError, match="não tem trilha de vídeo"):
        TimelineCompiler().compile(no_video, tmp_path)

    no_audio = Timeline(
        output_path=tmp_path / "o.mp4",
        video_main=[base],
        audio_ambient=[{"source_path": mixed_sources["mute"], "source_in": 0, "source_out": 1, "timeline_start": 0}],
    )
    with pytest.raises(ValueError, match="não tem trilha de áudio"):
        TimelineCompiler().compile(no_audio, tmp_path)

    missing = Timeline(output_path=tmp_path / "o.mp4", video_main=[{**base, "source_path": tmp_path / "nada.mp4"}])
    with pytest.raises(FileNotFoundError):
        TimelineCompiler().compile(missing, tmp_path)


def test_hdr_sources_produce_warning():
    infos = {
        Path("hdr.mov"): MediaInfo(duration_sec=1, has_video=True, has_audio=True, pix_fmt="yuv420p10le",
                                   color_transfer="smpte2084"),
        Path("sdr.mp4"): MediaInfo(duration_sec=1, has_video=True, has_audio=True, color_transfer="bt709"),
    }
    (warning,) = TimelineCompiler.source_warnings(infos)
    assert "HDR" in warning and "hdr.mov" in warning


# --- render ---


@pytest.mark.slow
def test_render_mixed_formats_with_overlay_and_ambient(mixed_sources, tmp_path: Path):
    """Base com áudio + base muda; overlays vertical (rotação) e 4:3; ambiente no trecho mudo."""
    s = mixed_sources
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        resolution=(640, 360),
        fps=30,
        video_main=[
            {"source_path": s["main"], "source_in": 0.0, "source_out": 2.0, "clip_id": "A001"},
            {"source_path": s["mute"], "source_in": 0.5, "source_out": 2.5, "clip_id": "M001"},
        ],
        video_overlay=[
            {"source_path": s["vertical"], "source_in": 0.0, "source_out": 1.0, "timeline_start": 0.5},
            {"source_path": s["small"], "source_in": 1.0, "source_out": 1.5, "timeline_start": 2.5},
        ],
        audio_ambient=[
            {
                "source_path": s["small"],
                "source_in": 0.0,
                "source_out": 1.0,
                "timeline_start": 2.8,
                "volume": 0.8,
                "fade_in_sec": 0.1,
                "fade_out_sec": 0.1,
            }
        ],
    )
    temp_root = tmp_path / "work"
    progress: list[tuple[int, int]] = []
    started = time.perf_counter()
    result = TimelineRenderExecutor(temp_root=temp_root).execute(tl, on_progress=lambda d, t: progress.append((d, t)))
    print(f"\nRender da Timeline mista (4s, 4 formatos): {time.perf_counter() - started:.1f}s")

    assert result.resolution == (640, 360)
    assert result.frame_count == 120
    assert result.duration_sec == pytest.approx(4.0, abs=1 / 30)
    assert result.fps == pytest.approx(30)
    assert result.warnings == []
    assert progress[0] == (0, 8) and progress[-1] == (8, 8)  # 6 pedaços de vídeo + 2 de áudio

    streams = _streams(result.output_path)
    assert streams["video"]["pix_fmt"] == "yuv420p"
    assert (streams["audio"]["sample_rate"], streams["audio"]["channels"]) == ("48000", 2)
    assert abs(float(streams["audio"]["duration"]) - float(streams["video"]["duration"])) < 0.05

    # O áudio da base continua sob o overlay; o trecho mudo é silêncio, exceto onde entra o ambiente
    assert _rms(result.output_path, 0.7, 0.6) > 0.02
    assert _rms(result.output_path, 2.2, 0.4) < 0.001
    assert _rms(result.output_path, 3.0, 0.5) > 0.02

    # Overlay vertical (rotação de metadados aplicada) aparece com pillarbox: laterais pretas
    columns = _gray_columns(result.output_path, frame=30, width=640, height=360)
    assert columns[:100].max() < 20 and columns[-100:].max() < 20
    assert columns[230:410].mean() > 40  # imagem vertical de ~202 px no centro
    # Base 16:9 ocupa a tela toda (a última faixa do testsrc é clara)
    assert _gray_columns(result.output_path, frame=5, width=640, height=360)[-20:].mean() > 40

    assert list(temp_root.iterdir()) == []


@pytest.mark.slow
def test_render_timeline_with_loudness(mixed_sources, tmp_path: Path):
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        resolution=(320, 180),
        loudness=LoudnessTarget.from_preset("voz"),
        video_main=[{"source_path": mixed_sources["main"], "source_in": 0.0, "source_out": 2.0}],
    )
    result = TimelineRenderExecutor().execute(tl)
    assert result.loudness is not None and result.loudness.status == "applied"
    assert result.frame_count == 60


@pytest.mark.slow
def test_render_timeline_without_any_audio_skips_loudness(mixed_sources, tmp_path: Path):
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        resolution=(320, 180),
        loudness=LoudnessTarget.from_preset("voz"),
        video_main=[{"source_path": mixed_sources["mute"], "source_in": 0.0, "source_out": 1.0}],
    )
    result = TimelineRenderExecutor().execute(tl)
    assert result.loudness.status == "skipped"
    assert "audio" in _streams(result.output_path)  # silêncio gerado: a saída sempre tem áudio


@pytest.mark.slow
def test_edit_plan_and_timeline_render_equivalence(mixed_sources, tmp_path: Path):
    """O mesmo EditPlan pelo caminho antigo e pela Timeline: mesma duração e mesmo número de frames."""
    s = mixed_sources
    plan = EditPlan(
        output_path=tmp_path / "old.mp4",
        target_resolution=(640, 360),
        target_fps=30,
        segments=[
            ClipSegment(source_path=s["main"], start_sec=0.2, end_sec=1.1),
            ClipSegment(source_path=s["small"], start_sec=0.5, end_sec=1.37),
            ClipSegment(source_path=s["mute"], start_sec=1.0, end_sec=1.9),
            ClipSegment(source_path=s["vertical"], start_sec=0.0, end_sec=0.533),
            ClipSegment(source_path=s["main"], start_sec=2.0, end_sec=3.0),
        ],
    )
    old = RenderExecutor().execute(plan)
    tl = timeline_from_edit_plan(plan).model_copy(update={"output_path": tmp_path / "new.mp4"})
    new = TimelineRenderExecutor().execute(tl)

    old_video = _streams(old.output_path)["video"]
    new_video = _streams(new.output_path)["video"]
    assert int(old_video["nb_frames"]) == int(new_video["nb_frames"]) == tl.total_frames
    assert float(old_video["duration"]) == pytest.approx(float(new_video["duration"]), abs=1e-3)
    assert (old_video["width"], old_video["height"]) == (new_video["width"], new_video["height"])


# --- comando render ---


def test_render_command_detects_document_kind():
    from ai_video_editor.cli import is_timeline_document

    assert is_timeline_document({"output_path": "o.mp4", "video_main": []})
    assert is_timeline_document({"schema_version": 1})
    assert not is_timeline_document({"output_path": "o.mp4", "segments": []})


@pytest.mark.slow
def test_render_command_accepts_timeline_json(mixed_sources, tmp_path: Path, capsys):
    import argparse

    from ai_video_editor.cli import handle_plan_render

    doc = {
        "schema_version": 1,
        "output_path": str(tmp_path / "cli.mp4"),
        "resolution": [320, 180],
        "video_main": [{"source_path": str(mixed_sources["main"]), "source_in": 0.0, "source_out": 1.0}],
        "video_overlay": [
            {"source_path": str(mixed_sources["small"]), "source_in": 0.0, "source_out": 0.5, "timeline_start": 0.25}
        ],
    }
    plan_file = tmp_path / "timeline.json"
    plan_file.write_text(json.dumps(doc), encoding="utf-8")

    assert handle_plan_render(argparse.Namespace(plan_file=str(plan_file))) == 0
    out = capsys.readouterr().out
    assert "Timeline carregada" in out and "30 frames" in out
    assert (tmp_path / "cli.mp4").is_file()
