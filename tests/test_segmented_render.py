"""Tests for the segmented renderer (many cuts of the same source)."""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import pytest

from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.renderer import compiler as compiler_module
from ai_video_editor.renderer.compiler import FFmpegCompiler
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.renderer.ffmpeg_locator import get_ffprobe_path
from tests.conftest import create_synthetic_video


def _stream_durations(path: Path) -> dict[str, float]:
    args = [
        str(get_ffprobe_path()),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,duration",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    return {s["codec_type"]: float(s["duration"]) for s in data["streams"]}


@pytest.mark.slow
def test_render_200_segments_from_60s_video(tmp_path: Path):
    """Plano com 200 cortes do mesmo vídeo renderiza com duração final correta e A/V alinhados."""
    source = create_synthetic_video(tmp_path / "lecture_60s.mp4", duration_sec=60.0)
    segments = [
        ClipSegment(source_path=source, start_sec=i * 0.3, end_sec=i * 0.3 + 0.2)
        for i in range(200)
    ]
    temp_root = tmp_path / "render_tmp"
    plan = EditPlan(output_path=tmp_path / "out_200.mp4", segments=segments)

    started = time.perf_counter()
    result = RenderExecutor(temp_root=temp_root).execute(plan)
    elapsed = time.perf_counter() - started
    print(f"\nRender de 200 segmentos: {elapsed:.1f}s")

    assert result.resolution == (1280, 720)
    assert abs(result.duration_sec - 40.0) < 0.5

    streams = _stream_durations(result.output_path)
    assert abs(streams["video"] - streams["audio"]) < 0.05

    # Concat via arquivo de lista: o comando final não cresce com o número de segmentos
    assert len(result.command_executed) < 40

    # Temporários apagados ao final
    assert list(temp_root.iterdir()) == []


@pytest.mark.slow
def test_ffprobe_called_once_per_source(synthetic_video_pair, tmp_path: Path, monkeypatch):
    """Cada arquivo de origem é inspecionado com ffprobe uma única vez, mesmo com muitos cortes."""
    v1, v2 = synthetic_video_pair
    calls: Counter[str] = Counter()
    real_run = subprocess.run

    def counting_run(args, *a, **kw):
        if Path(args[0]).stem.lower() == "ffprobe":
            calls[str(Path(args[-1]).resolve())] += 1
        return real_run(args, *a, **kw)

    monkeypatch.setattr(compiler_module.subprocess, "run", counting_run)

    segments = [
        ClipSegment(source_path=(v1 if i % 2 == 0 else v2), start_sec=i * 0.1, end_sec=i * 0.1 + 0.08)
        for i in range(20)
    ]
    plan = EditPlan(output_path=tmp_path / "out.mp4", segments=segments)
    job = FFmpegCompiler().compile_segmented(plan, tmp_path)

    assert len(job.segment_commands) == 20
    assert calls == Counter({str(v1.resolve()): 1, str(v2.resolve()): 1})


def _audio_filter(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


@pytest.mark.slow
def test_audio_fade_configurable(synthetic_video_with_audio, tmp_path: Path):
    """Micro fade padrão de 20 ms em cada segmento; 0 desativa."""
    seg = ClipSegment(source_path=synthetic_video_with_audio, start_sec=0.5, end_sec=1.5)

    default_plan = EditPlan(output_path=tmp_path / "a.mp4", segments=[seg])
    graph = _audio_filter(FFmpegCompiler().compile_segmented(default_plan, tmp_path).segment_commands[0])
    assert "afade=t=in:st=0:d=0.020000" in graph
    assert "afade=t=out:st=0.980000:d=0.020000" in graph

    no_fade_plan = EditPlan(output_path=tmp_path / "b.mp4", audio_fade_sec=0.0, segments=[seg])
    graph = _audio_filter(FFmpegCompiler().compile_segmented(no_fade_plan, tmp_path).segment_commands[0])
    assert "afade" not in graph


def test_segment_commands_use_fast_seek(synthetic_video_with_audio, tmp_path: Path):
    """Seek rápido (-ss antes do -i) e contagem exata de frames/amostras por segmento."""
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        segments=[ClipSegment(source_path=synthetic_video_with_audio, start_sec=1.0, end_sec=1.5)],
    )
    cmd = FFmpegCompiler().compile_segmented(plan, tmp_path).segment_commands[0]

    assert cmd.index("-ss") < cmd.index("-i")
    graph = _audio_filter(cmd)
    assert "trim=end_frame=15" in graph
    assert "atrim=end_sample=22050" in graph


def test_open_ended_segment_uses_probed_duration(synthetic_video_with_audio, tmp_path: Path):
    """end_sec=None vai até o fim da origem, usando a duração do ffprobe."""
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        segments=[ClipSegment(source_path=synthetic_video_with_audio, start_sec=1.0)],
    )
    job = FFmpegCompiler().compile_segmented(plan, tmp_path)
    assert job.expected_duration_sec == pytest.approx(2.0, abs=0.05)


@pytest.mark.slow
def test_render_mixed_sources_without_audio(
    synthetic_video_with_audio, synthetic_video_without_audio, tmp_path: Path
):
    """Origens diferentes, uma sem áudio, renderizam com trilha de áudio contínua."""
    plan = EditPlan(
        output_path=tmp_path / "mixed.mp4",
        target_resolution=(640, 360),
        segments=[
            ClipSegment(source_path=synthetic_video_with_audio, start_sec=0.0, end_sec=1.0),
            ClipSegment(source_path=synthetic_video_without_audio, start_sec=0.5, end_sec=2.0),
            ClipSegment(source_path=synthetic_video_with_audio, start_sec=2.0),
        ],
    )
    result = RenderExecutor().execute(plan)

    assert abs(result.duration_sec - 3.5) < 0.2
    streams = _stream_durations(result.output_path)
    assert abs(streams["video"] - streams["audio"]) < 0.05


def test_segment_starting_after_source_end_fails(synthetic_video_with_audio, tmp_path: Path):
    """Segmento que começa depois do fim da origem é rejeitado antes do render."""
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        segments=[ClipSegment(source_path=synthetic_video_with_audio, start_sec=10.0, end_sec=12.0)],
    )
    with pytest.raises(ValueError):
        FFmpegCompiler().compile_segmented(plan, tmp_path)
