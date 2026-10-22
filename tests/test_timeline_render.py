"""Testes do render da Timeline: comandos gerados, render com formatos misturados e equivalência com o EditPlan."""

import json
import subprocess
import time
import wave
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

    video_cmds = job.piece_commands
    assert all(c[-1].endswith(".mov") for c in video_cmds)
    assert len(video_cmds) == 4  # base, overlay (termina na fronteira do item), item mudo, último item
    for cmd in video_cmds:
        assert "-an" in cmd and cmd.index("-ss") < cmd.index("-i")
        graph = _graph(cmd)
        assert "pad=640:360" in graph and "setsar=1" in graph and "format=yuv420p" in graph

    # Áudio inteiro num só comando: a origem com áudio (main) e o ambiente entram uma vez cada.
    assert job.audio_command.count("-i") == 2
    assert job.audio_command[job.audio_command.index("-/filter_complex") + 1] == str(job.audio_graph_path)
    chains = job.audio_graph.split(";\n")
    item0 = next(c for c in chains if c.endswith("[p0]"))
    assert item0.startswith("[s0_0]asetpts=PTS-STARTPTS") and "afade=t=in" in item0
    assert "[0:a:0]asegment=timestamps=1.000000[s0_0][g0_1]" in job.audio_graph
    assert "atrim=end_sample=48000" in item0
    assert next(c for c in chains if c.endswith("[p1]")).startswith("anullsrc")  # origem sem áudio
    assert next(c for c in chains if c.endswith("[p2]")).startswith("anullsrc")  # item marcado sem áudio
    assert "[p0][p1][p2]concat=n=3:v=0:a=1[main]" in job.audio_graph
    assert "amix=inputs=2:duration=first:normalize=0" in job.audio_graph
    assert "adelay=delays=72000S" in job.audio_graph and "volume=0.5" in job.audio_graph
    assert job.expected_frames == 90 and job.has_audio


def test_compile_without_ambient_only_generates_silence(mixed_sources, tmp_path: Path):
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        video_main=[{"source_path": mixed_sources["mute"], "source_in": 0.0, "source_out": 1.0}],
    )
    job = TimelineCompiler().compile(tl, tmp_path)
    assert "-i" not in job.audio_command and "amix" not in job.audio_graph
    assert job.audio_graph.startswith("anullsrc") and "concat=n=1:v=0:a=1[a]" in job.audio_graph
    assert not job.has_audio


def _shuffled_timeline(sources: dict[str, Path], tmp_path: Path, count: int) -> Timeline:
    """`count` itens curtos (4 a 7 frames) fora de ordem e alternando duas origens (caso patológico)."""
    items = []
    for i in range(count):
        source = sources["main"] if i % 3 else sources["small"]
        start = round((i * 0.37) % 2.5, 3)
        items.append({"source_path": source, "source_in": start, "source_out": start + (4 + i % 4) / 30})
    return Timeline(output_path=tmp_path / "out.mp4", resolution=(640, 360), video_main=items)


@pytest.fixture(scope="module")
def pulse_sources(tmp_path_factory) -> list[Path]:
    """Duas origens de 160 s cujo áudio alterna 1 s de tom e 1 s de silêncio (tom em [2k, 2k+1))."""
    root = tmp_path_factory.mktemp("pulse")
    paths = []
    for n, freq in enumerate((440, 660)):
        path = root / f"pulse_{n}.mp4"
        args = [
            str(get_ffmpeg_path()), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=160:size=160x120:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration=160",
            "-filter_complex", "[1:a]volume=eval=frame:volume='lt(mod(t,2),1)'[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "48000", str(path),
        ]
        subprocess.run(args, check=True, capture_output=True)
        paths.append(path)
    return paths


def _spine_timeline(sources: list[Path], tmp_path: Path, count: int) -> Timeline:
    """Espinha sintética: blocos de 10 itens por origem; o item j de uma origem pega 0,4 s do segundo j
    (tom quando j é par, silêncio quando é ímpar). Tempos de origem crescentes dentro de cada origem."""
    items, taken = [], [0, 0]
    for i in range(count):
        k = (i // 10) % 2
        j = taken[k]
        taken[k] += 1
        items.append({"source_path": sources[k], "source_in": j + 0.3, "source_out": j + 0.7, "label": str(j)})
    return Timeline(output_path=tmp_path / "out.mp4", resolution=(160, 120), video_main=items)


def test_audio_process_count_does_not_grow_with_items(pulse_sources, mixed_sources, tmp_path: Path):
    few = TimelineCompiler().compile(_spine_timeline(pulse_sources, tmp_path, 10), tmp_path)
    many = TimelineCompiler().compile(_spine_timeline(pulse_sources, tmp_path, 300), tmp_path)
    shuffled = TimelineCompiler().compile(_shuffled_timeline(mixed_sources, tmp_path, 300), tmp_path)
    for job, sources in ((few, 1), (many, 2), (shuffled, 2)):
        # Um processo de áudio; cada origem entra uma vez; os demais comandos são só de vídeo.
        assert job.audio_command.count("-i") == sources
        assert all("-an" in c and c[-1].endswith(".mov") for c in job.piece_commands)
    assert len(many.piece_commands) == 300
    assert many.audio_graph.count("asegment=") == 2 and "asplit" not in many.audio_graph
    assert "concat=n=300:v=0:a=1[a]" in many.audio_graph
    # Itens muito fora de ordem: a origem entra uma vez e cada item recorta sua cópia.
    assert "asplit=200" in shuffled.audio_graph and "asplit=100" in shuffled.audio_graph


def _wav_rms(path: Path, start_sample: int, count: int) -> float:
    with wave.open(str(path)) as wav:
        wav.setpos(start_sample)
        samples = np.frombuffer(wav.readframes(count), dtype=np.int16).astype(np.float64) / 32768
    return float(np.sqrt(np.mean(samples**2)))


@pytest.mark.slow
def test_audio_of_300_items_is_exact_and_in_sync(pulse_sources, mixed_sources, tmp_path: Path):
    """O áudio de 300 itens sai de um só FFmpeg, com a contagem exata de amostras e o trecho certo em cada item."""
    timeline = _spine_timeline(pulse_sources, tmp_path, 300)
    job = TimelineCompiler().compile(timeline, tmp_path)
    job.audio_graph_path.write_text(job.audio_graph, encoding="utf-8")
    RenderExecutor._run_ffmpeg(job.audio_command)
    with wave.open(str(job.audio_path)) as wav:
        assert (wav.getframerate(), wav.getnchannels()) == (48000, 2)
        assert wav.getnframes() == round(timeline.total_frames * 48000 / timeline.fps)

    position = 0
    for item in timeline.video_main:
        frames = timeline.item_frames(item)
        start, end = timeline.to_sample(position), timeline.to_sample(position + frames)
        middle = _wav_rms(job.audio_path, start + (end - start) // 4, (end - start) // 2)
        if int(item.label) % 2 == 0:
            assert middle > 0.02, f"item {item.label}: esperado tom"
        else:
            assert middle < 0.002, f"item {item.label}: esperado silêncio"
        position += frames

    shuffled = _shuffled_timeline(mixed_sources, tmp_path / "shuffled", 300)
    (tmp_path / "shuffled").mkdir()
    job = TimelineCompiler().compile(shuffled, tmp_path / "shuffled")
    job.audio_graph_path.write_text(job.audio_graph, encoding="utf-8")
    RenderExecutor._run_ffmpeg(job.audio_command)
    with wave.open(str(job.audio_path)) as wav:
        assert wav.getnframes() == round(shuffled.total_frames * 48000 / shuffled.fps)


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
    assert progress[0] == (0, 7) and progress[-1] == (7, 7)  # 6 pedaços de vídeo + 1 processo de áudio

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
