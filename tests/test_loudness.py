"""Testes da normalização de loudness (loudnorm em duas passadas) no render e no pipeline."""

import math
import subprocess
from pathlib import Path

import pytest

from ai_video_editor.cli import build_parser
from ai_video_editor.core.schema import ClipSegment, EditPlan, LoudnessTarget
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams, ProjectManifest
from ai_video_editor.pipeline.report import build_edit_report
from ai_video_editor.renderer.compiler import FFmpegCompiler
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.renderer.loudness import (
    LoudnessMeasurement,
    LoudnessResult,
    apply_filter,
    measure_filter,
    parse_loudnorm_output,
)
from tests.conftest import create_tone_silence_video

_LOUDNORM_STDERR = """[Parsed_loudnorm_0 @ 0000026d9f0a9f00]
{
\t"input_i" : "-35.20",
\t"input_tp" : "-31.06",
\t"input_lra" : "1.30",
\t"input_thresh" : "-45.75",
\t"output_i" : "-15.95",
\t"output_tp" : "-12.27",
\t"output_lra" : "0.00",
\t"output_thresh" : "-25.95",
\t"normalization_type" : "dynamic",
\t"target_offset" : "-0.05"
}
[out#0/null @ 0000026d9f0c4f40] video:0KiB audio:1125KiB
"""


def _tone_video(path: Path, volume_db: float, duration: float = 6.0) -> Path:
    """Vídeo com um tom de 440 Hz no volume pedido (o seno padrão do lavfi tem pico em -18 dBFS)."""
    args = [
        str(get_ffmpeg_path()), "-y", "-nostdin",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration},volume={volume_db}dB",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-shortest", str(path),
    ]
    subprocess.run(args, capture_output=True, check=True)
    return path


def _measure(path: Path) -> LoudnessMeasurement:
    """Loudness de um arquivo medido pelo próprio loudnorm do FFmpeg."""
    proc = subprocess.run(
        [
            str(get_ffmpeg_path()), "-nostdin", "-hide_banner", "-nostats", "-i", str(path),
            "-map", "0:a:0", "-af", measure_filter(LoudnessTarget.from_preset("voz")), "-f", "null", "-",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    return parse_loudnorm_output(proc.stderr)


def _plan(source: Path, output: Path, preset: str = "voz") -> EditPlan:
    return EditPlan(
        output_path=output,
        target_resolution=(320, 240),
        loudness=LoudnessTarget.from_preset(preset),
        segments=[
            ClipSegment(source_path=source, start_sec=0.0, end_sec=2.5),
            ClipSegment(source_path=source, start_sec=3.0, end_sec=6.0),
        ],
    )


@pytest.fixture(scope="module")
def quiet_video(tmp_path_factory):
    """~-35 LUFS."""
    return _tone_video(tmp_path_factory.mktemp("loudness") / "baixo.mov", volume_db=-13.0)


@pytest.fixture(scope="module")
def loud_video(tmp_path_factory):
    """Pico perto de 0 dBFS, ~-4 LUFS."""
    return _tone_video(tmp_path_factory.mktemp("loudness") / "alto.mov", volume_db=17.5)


# --- presets, parse e filtros (sem FFmpeg) ---


def test_presets_and_off():
    assert LoudnessTarget.from_preset("off") is None
    voz = LoudnessTarget.from_preset("voz")
    assert (voz.integrated_lufs, voz.true_peak_dbtp, voz.lra) == (-16.0, -1.5, 11.0)
    assert LoudnessTarget.from_preset("youtube").integrated_lufs == -14.0
    assert LoudnessTarget.from_preset("broadcast").integrated_lufs == -23.0
    assert voz.describe() == "voz (-16 LUFS, TP -1.5 dBTP, LRA 11)"


def test_edit_plan_defaults_to_no_normalization():
    plan = EditPlan(output_path=Path("o.mp4"), segments=[ClipSegment(source_path=Path("a.mp4"))])
    assert plan.loudness is None
    # Planos antigos (sem o campo) continuam válidos
    assert EditPlan.model_validate_json(plan.model_dump_json(exclude={"loudness"})).loudness is None


def test_parse_loudnorm_json_from_stderr():
    measured = parse_loudnorm_output(_LOUDNORM_STDERR)
    assert measured == LoudnessMeasurement(
        input_i=-35.2, input_tp=-31.06, input_lra=1.3, input_thresh=-45.75, target_offset=-0.05
    )
    assert not measured.is_silent


@pytest.mark.parametrize("value", ["-inf", "nan", "abc"])
def test_silent_or_invalid_measurement_is_detected(value):
    text = _LOUDNORM_STDERR.replace('"-35.20"', f'"{value}"')
    assert parse_loudnorm_output(text).is_silent


def test_parse_without_json_fails_clearly():
    with pytest.raises(ValueError, match="loudnorm"):
        parse_loudnorm_output("nada aqui")


def test_second_pass_filter_uses_measured_values_and_linear_mode():
    measured = parse_loudnorm_output(_LOUDNORM_STDERR)
    text = apply_filter(LoudnessTarget.from_preset("voz"), measured)
    assert text == (
        "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-35.20:measured_TP=-31.06:measured_LRA=1.30"
        ":measured_thresh=-45.75:offset=-0.05:linear=true:print_format=none"
    )


def test_clipping_warning_and_metrics_round_trip():
    measured = LoudnessMeasurement(input_i=-9.0, input_tp=0.3, input_lra=4.0, input_thresh=-19.0, target_offset=0.1)
    result = LoudnessResult(
        target=LoudnessTarget.from_preset("youtube"), status="applied", measured=measured, measure_sec=1.2
    )
    assert len(result.warnings) == 1 and "0.3 dBTP" in result.warnings[0]
    assert "distorção" in result.warnings[0]
    assert LoudnessResult.from_metrics({"duration_sec": 1.0, **result.to_metrics()}) == result
    assert LoudnessResult.from_metrics({"duration_sec": 1.0}) is None

    skipped = LoudnessResult(target=LoudnessTarget.from_preset("voz"), status="skipped", note="vídeo sem áudio")
    assert LoudnessResult.from_metrics(skipped.to_metrics()) == skipped
    assert skipped.warnings == ["normalização de volume pulada: vídeo sem áudio"]


def test_report_shows_measured_loudness_target_and_off():
    plan = EditPlan(output_path=Path("o.mp4"), segments=[ClipSegment(source_path=Path("a.mp4"))])
    measured = LoudnessMeasurement(input_i=-9.0, input_tp=0.3, input_lra=4.0, input_thresh=-19.0, target_offset=0.1)
    result = LoudnessResult(target=LoudnessTarget.from_preset("voz"), status="applied", measured=measured)
    args = (Path("a.mp4"), 10.0, 8.0, plan, [], EditPipelineParams(), Path("o.mp4"))

    report = build_edit_report(*args, loudness=result)
    assert "## Volume" in report
    assert "| Loudness medido (I) | -9.0 LUFS |" in report
    assert "| True peak medido (TP) | 0.3 dBTP |" in report
    assert "| Faixa de loudness medida (LRA) | 4.0 LU |" in report
    assert "| Alvo aplicado | -16 LUFS, TP -1.5 dBTP, LRA 11 LU |" in report
    assert "provável distorção" in report

    off = build_edit_report(*args[:5], EditPipelineParams(loudness="off"), args[6])
    assert "| Normalização | desligada (--loudness off) |" in off


def test_cli_loudness_option():
    parser = build_parser()
    assert parser.parse_args(["edit", "v.mp4"]).loudness == "voz"
    assert parser.parse_args(["edit", "v.mp4", "--loudness", "broadcast"]).loudness == "broadcast"
    assert parser.parse_args(["revise", "p", "pedido"]).loudness is None
    assert parser.parse_args(["revise", "p", "pedido", "--loudness", "off"]).loudness == "off"
    with pytest.raises(SystemExit):
        parser.parse_args(["edit", "v.mp4", "--loudness", "alto"])


def test_concat_without_normalization_is_unchanged(synthetic_video_with_audio, tmp_path):
    """render/quick-cut (plano sem `loudness`): o concat final não ganha filtro de áudio."""
    plan = EditPlan(
        output_path=tmp_path / "o.mp4",
        segments=[ClipSegment(source_path=synthetic_video_with_audio, end_sec=1.0)],
    )
    job = FFmpegCompiler().compile_segmented(plan, tmp_path)
    assert "-af" not in job.concat_command
    assert job.concat_command[job.concat_command.index("-c:v") + 1] == "copy"


# --- render real ---


@pytest.mark.slow
def test_quiet_tone_is_raised_to_target(quiet_video, tmp_path):
    before = _measure(quiet_video)
    assert before.input_i == pytest.approx(-35.0, abs=2.0)

    result = RenderExecutor().execute(_plan(quiet_video, tmp_path / "out.mp4"))
    assert result.loudness is not None and result.loudness.status == "applied"
    assert result.loudness.measured.input_i == pytest.approx(before.input_i, abs=1.0)
    assert result.loudness.measure_sec > 0
    assert "-c:v" in result.command_executed and "copy" in result.command_executed
    assert any("linear=true" in arg for arg in result.command_executed)

    after = _measure(result.output_path)
    assert after.input_i == pytest.approx(-16.0, abs=1.0)


@pytest.mark.slow
def test_loud_tone_is_lowered_without_exceeding_true_peak(loud_video, tmp_path):
    before = _measure(loud_video)
    assert before.input_i > -8.0

    result = RenderExecutor().execute(_plan(loud_video, tmp_path / "out.mp4"))
    after = _measure(result.output_path)
    assert after.input_i == pytest.approx(-16.0, abs=1.0)
    assert after.input_tp <= -1.0 + 0.5


@pytest.mark.slow
def test_off_keeps_audio_unchanged(quiet_video, tmp_path):
    result = RenderExecutor().execute(_plan(quiet_video, tmp_path / "out.mp4", preset="off"))
    assert result.loudness is None
    assert not any("loudnorm" in arg for arg in result.command_executed)
    assert _measure(result.output_path).input_i == pytest.approx(_measure(quiet_video).input_i, abs=0.5)


@pytest.mark.slow
def test_silent_audio_skips_with_warning(tmp_path):
    source = create_tone_silence_video(tmp_path / "mudo.mp4", [("silence", 4.0)])
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        target_resolution=(320, 240),
        loudness=LoudnessTarget.from_preset("voz"),
        segments=[ClipSegment(source_path=source)],
    )
    result = RenderExecutor().execute(plan)
    assert result.loudness.status == "skipped" and result.loudness.measured is None
    assert "sem conteúdo mensurável" in result.loudness.note
    assert result.loudness.warnings and "pulada" in result.loudness.warnings[0]
    assert not any("loudnorm" in arg for arg in result.command_executed)
    assert result.output_path.is_file()


@pytest.mark.slow
def test_video_without_audio_skips_without_measuring(synthetic_video_without_audio, tmp_path, monkeypatch):
    commands = []
    original = RenderExecutor._run_ffmpeg
    monkeypatch.setattr(
        RenderExecutor, "_run_ffmpeg", staticmethod(lambda cmd: commands.append(cmd) or original(cmd))
    )
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        target_resolution=(320, 240),
        loudness=LoudnessTarget.from_preset("voz"),
        segments=[ClipSegment(source_path=synthetic_video_without_audio)],
    )
    result = RenderExecutor().execute(plan)
    assert result.loudness.status == "skipped" and result.loudness.note == "vídeo sem áudio"
    assert not any("loudnorm" in arg for cmd in commands for arg in cmd)


@pytest.mark.slow
def test_edit_pipeline_normalizes_reports_and_caches_by_preset(quiet_video, tmp_path):
    source = tmp_path / "fala.mov"
    source.write_bytes(quiet_video.read_bytes())
    pipeline = EditPipeline(projects_root=tmp_path / "projects")

    first = pipeline.run(source, EditPipelineParams())
    metrics = ProjectManifest.load(first.manifest_path).stages["render"].metrics
    assert metrics["loudness_status"] == "applied" and metrics["loudness_preset"] == "voz"
    assert _measure(first.output_path).input_i == pytest.approx(-16.0, abs=1.0)
    report = first.report_path.read_text(encoding="utf-8")
    assert "## Volume" in report and "preset voz" in report
    assert "| Alvo aplicado | -16 LUFS, TP -1.5 dBTP, LRA 11 LU |" in report
    assert ProjectManifest.load(first.manifest_path).versions[0].loudness == "voz"

    cached = pipeline.run(source, EditPipelineParams())
    assert cached.outcomes["render"] == "cached"

    broadcast = pipeline.run(source, EditPipelineParams(loudness="broadcast"))
    assert broadcast.outcomes["silences"] == "cached"
    assert broadcast.outcomes["render"] == "ran" and broadcast.outcomes["report"] == "ran"
    assert _measure(broadcast.output_path).input_i == pytest.approx(-23.0, abs=1.0)
    assert broadcast.version == 2

    off = pipeline.run(source, EditPipelineParams(loudness="off"))
    assert "loudness_status" not in ProjectManifest.load(off.manifest_path).stages["render"].metrics
    assert "desligada" in off.report_path.read_text(encoding="utf-8")
    assert math.isclose(_measure(off.output_path).input_i, _measure(source).input_i, abs_tol=0.5)
