"""Codificador do render (x264 na CPU ou NVENC): detecção, argumentos, limite de GPU e volta para a CPU.

Nenhum teste precisa de GPU: a detecção usa um FFmpeg falso e o NVENC "roda" num `_run_ffmpeg` falso.
O render real dos testes fica na CPU (fixture `_render_on_cpu` do conftest).
"""

import dataclasses
import subprocess
import threading
import time
from pathlib import Path

import pytest

from ai_video_editor.cli import build_parser
from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams
from ai_video_editor.pipeline.project import fingerprint
from ai_video_editor.renderer import encoders
from ai_video_editor.renderer.compiler import VIDEO_ENCODE_ARGS, FFmpegCompiler
from ai_video_editor.renderer.encoders import (
    NVENC_ENCODE_ARGS,
    EncoderSelection,
    NvencFailure,
    NvencStatus,
    detect_nvenc,  # a função real (a fixture do conftest troca a do módulo)
    probe_nvenc,
    select_encoder,
)
from ai_video_editor.renderer.executor import RenderExecutor, run_parallel
from ai_video_editor.renderer.timeline_compiler import TimelineCompiler
from ai_video_editor.renderer.timeline_executor import TimelineRenderExecutor
from tests.conftest import create_tone_silence_video

FAKE_FFMPEG = Path("ffmpeg-falso")
ENCODERS_WITH_NVENC = " V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n"
ENCODERS_WITHOUT = " V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)\n"
NVENC_ERROR = "[h264_nvenc @ 000001] OpenEncodeSessionEx failed: out of memory (10): (no details)"


def _fake_ffmpeg(monkeypatch, listing: str, test_returncode: int = 0, test_stderr: str = "") -> list[list[str]]:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(list(args))
        if "-encoders" in args:
            return subprocess.CompletedProcess(args, 0, stdout=listing, stderr="")
        return subprocess.CompletedProcess(args, test_returncode, stdout="", stderr=test_stderr)

    monkeypatch.setattr(encoders.subprocess, "run", run)
    return calls


# --- detecção ---


def test_probe_nvenc_not_listed_is_unavailable(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, ENCODERS_WITHOUT)
    status = probe_nvenc(FAKE_FFMPEG)
    assert not status.available and "h264_nvenc não existe" in status.reason
    assert len(calls) == 1  # sem codificador listado, nem tenta a codificação de teste


def test_probe_nvenc_listed_but_test_fails_is_unavailable(monkeypatch):
    _fake_ffmpeg(monkeypatch, ENCODERS_WITH_NVENC, 1, "Cannot load nvcuda.dll\n" + NVENC_ERROR)
    status = probe_nvenc(FAKE_FFMPEG)
    assert not status.available
    assert status.reason.startswith("codificação de teste do NVENC falhou") and "nvcuda" in status.reason


def test_probe_nvenc_listed_and_test_works(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, ENCODERS_WITH_NVENC)
    assert probe_nvenc(FAKE_FFMPEG) == NvencStatus(True)
    test = calls[1]
    assert test[test.index("-f") + 1] == "lavfi" and "testsrc" in test[test.index("-i") + 1]
    assert test[-3:] == ["-f", "null", "-"]
    # A codificação de teste usa os parâmetros reais do render (preset/tune aceitos por este FFmpeg)
    assert test[test.index("-c:v") : test.index("-c:v") + len(NVENC_ENCODE_ARGS)] == NVENC_ENCODE_ARGS


def test_probe_nvenc_handles_missing_binary(monkeypatch):
    def run(args, **kwargs):
        raise FileNotFoundError("ffmpeg-falso")

    monkeypatch.setattr(encoders.subprocess, "run", run)
    assert not probe_nvenc(FAKE_FFMPEG).available


def test_detection_runs_once_per_ffmpeg(monkeypatch):
    calls: list[Path] = []
    monkeypatch.setattr(encoders, "_cache", {})
    monkeypatch.setattr(encoders, "probe_nvenc", lambda bin: calls.append(bin) or NvencStatus(True))
    assert detect_nvenc(FAKE_FFMPEG).available
    assert detect_nvenc(FAKE_FFMPEG).available
    assert calls == [FAKE_FFMPEG]


def test_select_encoder(monkeypatch):
    def fail(bin):
        raise AssertionError("cpu não detecta")

    monkeypatch.setattr(encoders, "detect_nvenc", fail)
    assert select_encoder("cpu", FAKE_FFMPEG).label == "cpu (--encoder cpu)"

    monkeypatch.setattr(encoders, "detect_nvenc", lambda bin: NvencStatus(False, "h264_nvenc não existe neste FFmpeg"))
    auto = select_encoder("auto", FAKE_FFMPEG)
    assert auto.name == "cpu" and auto.label == "cpu (sem GPU: h264_nvenc não existe neste FFmpeg)"
    with pytest.raises(RuntimeError, match="--encoder nvenc"):
        select_encoder("nvenc", FAKE_FFMPEG)

    monkeypatch.setattr(encoders, "detect_nvenc", lambda bin: NvencStatus(True))
    assert select_encoder("auto", FAKE_FFMPEG).label == "nvenc"
    assert select_encoder("nvenc", FAKE_FFMPEG).label == "nvenc"


def test_tests_render_on_cpu_by_default():
    assert RenderExecutor().resolve_encoder().name == "cpu"
    assert TimelineRenderExecutor().resolve_encoder().name == "cpu"


# --- argumentos ---


def test_cpu_arguments_and_fingerprint_unchanged():
    assert VIDEO_ENCODE_ARGS == ["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p"]
    cpu = EncoderSelection("cpu", "sem GPU: x")
    assert cpu.video_args == VIDEO_ENCODE_ARGS
    assert fingerprint("render", "plano", "voz", *cpu.fingerprint_parts) == fingerprint("render", "plano", "voz")
    nvenc = EncoderSelection("nvenc")
    assert fingerprint("render", "plano", "voz", *nvenc.fingerprint_parts) != fingerprint("render", "plano", "voz")


def test_nvenc_arguments():
    args = EncoderSelection("nvenc").video_args

    def value(flag: str) -> str:
        return args[args.index(flag) + 1]

    assert value("-c:v") == "h264_nvenc" and value("-preset") == "p5" and value("-tune") == "hq"
    assert (value("-rc"), value("-cq"), value("-b:v")) == ("vbr", "25", "0")
    assert value("-pix_fmt") == "yuv420p" and value("-g") == "250"


def test_same_encoder_arguments_in_every_piece(mixed_sources, tmp_path: Path):
    s = mixed_sources
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        resolution=(640, 360),
        video_main=[
            {"source_path": s["main"], "source_in": 0.0, "source_out": 1.0},
            {"source_path": s["mute"], "source_in": 0.0, "source_out": 1.0},
        ],
        video_overlay=[{"source_path": s["vertical"], "source_in": 0.0, "source_out": 0.5, "timeline_start": 0.5}],
    )
    plan = EditPlan(
        output_path=tmp_path / "plan.mp4",
        target_resolution=(640, 360),
        segments=[
            ClipSegment(source_path=s["main"], start_sec=0.0, end_sec=1.0),
            ClipSegment(source_path=s["small"], start_sec=0.5, end_sec=1.0),
        ],
    )
    nvenc = EncoderSelection("nvenc").video_args

    def encoder_args(cmd: list[str]) -> list[str]:
        start = cmd.index("-c:v")
        return cmd[start : start + len(nvenc)]

    timeline_pieces = TimelineCompiler().compile(tl, tmp_path, nvenc).piece_commands
    plan_pieces = FFmpegCompiler().compile_segmented(plan, tmp_path, nvenc).segment_commands
    assert len(timeline_pieces) > 2
    assert all(encoder_args(cmd) == nvenc for cmd in [*timeline_pieces, *plan_pieces])
    # Sem argumentos: a CPU de sempre
    default = TimelineCompiler().compile(tl, tmp_path).piece_commands
    assert all(cmd[cmd.index("-c:v") : cmd.index("-c:v") + 8] == VIDEO_ENCODE_ARGS for cmd in default)
    assert all("h264_nvenc" not in cmd for cmd in default)


# --- execução em paralelo ---


def _nvenc_cmd(i: int) -> list[str]:
    return ["ffmpeg", "-c:v", "h264_nvenc", f"pedaco_{i}.mov"]


def test_gpu_limit_caps_simultaneous_nvenc_sessions(monkeypatch):
    lock = threading.Lock()
    active = {"now": 0, "max": 0, "other": 0}

    def fake_run(cmd):
        if "h264_nvenc" not in cmd:
            active["other"] += 1
            return ""
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.02)
        with lock:
            active["now"] -= 1
        return ""

    monkeypatch.setattr(RenderExecutor, "_run_ffmpeg", staticmethod(fake_run))
    progress: list[tuple[int, int]] = []
    commands = [["ffmpeg", "audio.wav"], *(_nvenc_cmd(i) for i in range(12))]
    run_parallel(commands, 8, lambda d, t: progress.append((d, t)), EncoderSelection("nvenc"), gpu_workers=2)
    assert active["max"] == 2 and active["other"] == 1
    assert progress[0] == (0, 13) and progress[-1] == (13, 13)


def test_nvenc_error_becomes_nvenc_failure_but_other_errors_do_not(monkeypatch):
    def nvenc_breaks(cmd):
        raise RuntimeError(f"FFmpeg execution failed with code 1:\n{NVENC_ERROR}")

    monkeypatch.setattr(RenderExecutor, "_run_ffmpeg", staticmethod(nvenc_breaks))
    with pytest.raises(NvencFailure) as info:
        run_parallel([_nvenc_cmd(0)], 2, encoder=EncoderSelection("nvenc"))
    assert "OpenEncodeSessionEx" in info.value.detail
    # Na CPU, a mesma mensagem continua um erro comum (não há para onde voltar)
    with pytest.raises(RuntimeError) as cpu_info:
        run_parallel([["ffmpeg", "-c:v", "libx264"]], 2, encoder=EncoderSelection("cpu"))
    assert not isinstance(cpu_info.value, NvencFailure)

    def source_breaks(cmd):
        raise RuntimeError("FFmpeg execution failed with code 1:\nInvalid data found when processing input")

    monkeypatch.setattr(RenderExecutor, "_run_ffmpeg", staticmethod(source_breaks))
    with pytest.raises(RuntimeError) as other:
        run_parallel([_nvenc_cmd(0)], 2, encoder=EncoderSelection("nvenc"))
    assert not isinstance(other.value, NvencFailure)


def _nvenc_breaks_on_second_piece(monkeypatch) -> list[list[str]]:
    """NVENC "disponível"; o 1º pedaço na GPU passa sem rodar, o 2º falha por sessão. CPU roda de verdade."""
    monkeypatch.setattr(encoders, "detect_nvenc", lambda bin: NvencStatus(True))
    real = RenderExecutor._run_ffmpeg
    seen: list[list[str]] = []
    nvenc_calls = {"n": 0}
    lock = threading.Lock()

    def fake_run(cmd):
        seen.append(cmd)
        if "h264_nvenc" in cmd:
            with lock:
                nvenc_calls["n"] += 1
                n = nvenc_calls["n"]
            if n >= 2:
                raise RuntimeError(f"FFmpeg execution failed with code 1:\n{NVENC_ERROR}")
            return ""
        return real(cmd)

    monkeypatch.setattr(RenderExecutor, "_run_ffmpeg", staticmethod(fake_run))
    return seen


def _after_fallback(seen: list[list[str]]) -> list[list[str]]:
    last_nvenc = max(i for i, cmd in enumerate(seen) if "h264_nvenc" in cmd)
    return seen[last_nvenc + 1 :]


@pytest.mark.slow
def test_edit_plan_render_falls_back_to_cpu_entirely(monkeypatch, mixed_sources, tmp_path: Path):
    seen = _nvenc_breaks_on_second_piece(monkeypatch)
    s = mixed_sources
    plan = EditPlan(
        output_path=tmp_path / "out.mp4",
        target_resolution=(320, 180),
        segments=[ClipSegment(source_path=s["main"], start_sec=t, end_sec=t + 0.5) for t in (0.0, 1.0, 2.0)],
    )
    progress: list[tuple[int, int]] = []
    result = RenderExecutor(max_workers=1).execute(plan, on_progress=lambda d, t: progress.append((d, t)))

    assert result.encoder.name == "cpu" and result.encoder.fell_back
    assert result.encoder.label.startswith("cpu (NVENC falhou: [h264_nvenc")
    assert any("refeito na CPU" in w for w in result.warnings)
    redo = _after_fallback(seen)
    pieces = [cmd for cmd in redo if "libx264" in cmd]
    assert len(pieces) == 3  # TODOS os trechos refeitos na CPU, inclusive o que "passou" no NVENC
    assert result.duration_sec == pytest.approx(1.5, abs=0.05)
    assert progress.count((0, 3)) == 2  # o andamento recomeça na CPU


@pytest.mark.slow
def test_timeline_render_falls_back_to_cpu_entirely(monkeypatch, mixed_sources, tmp_path: Path):
    seen = _nvenc_breaks_on_second_piece(monkeypatch)
    s = mixed_sources
    tl = Timeline(
        output_path=tmp_path / "out.mp4",
        resolution=(320, 180),
        video_main=[{"source_path": s["main"], "source_in": t, "source_out": t + 0.5} for t in (0.0, 1.0, 2.0)],
    )
    result = TimelineRenderExecutor(max_workers=1).execute(tl)

    assert result.encoder.fell_back and result.frame_count == 45
    assert any("refeito na CPU" in w for w in result.warnings)
    redo = _after_fallback(seen)
    assert len([cmd for cmd in redo if "libx264" in cmd]) == 3
    assert all("h264_nvenc" not in cmd for cmd in redo)


# --- pipeline: fingerprint e relatório ---


class _EncoderExecutor(RenderExecutor):
    """Render real na CPU que se apresenta com o codificador pedido (simula a GPU)."""

    def __init__(self, selection: EncoderSelection, effective: EncoderSelection | None = None):
        super().__init__()
        self.selection = selection
        self.effective = effective or selection
        self.calls = 0

    def resolve_encoder(self) -> EncoderSelection:
        return self.selection

    def execute(self, plan, on_progress=None):
        self.calls += 1
        result = self._execute(plan, EncoderSelection("cpu"), on_progress)
        return dataclasses.replace(result, encoder=self.effective)


@pytest.fixture(scope="module")
def encoder_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("encoder") / "curto.mp4"
    return create_tone_silence_video(path, [("tone", 1.0), ("silence", 1.0), ("tone", 1.0)], width=160, height=120)


@pytest.mark.slow
def test_encoder_enters_only_the_render_fingerprint(tmp_path: Path, encoder_video: Path):
    root = tmp_path / "projects"
    cpu = EncoderSelection("cpu", "--encoder cpu")
    nvenc = EncoderSelection("nvenc")
    fallback = EncoderSelection("cpu", "NVENC falhou: sessão")

    first = EditPipeline(projects_root=root, render_executor=_EncoderExecutor(cpu)).run(encoder_video)
    report = first.report_path.read_text(encoding="utf-8")
    assert first.outcomes["render"] == "ran" and "| Codificador de vídeo | cpu (--encoder cpu) |" in report

    def fps() -> dict[str, str]:
        from ai_video_editor.pipeline.schema import ProjectManifest

        m = ProjectManifest.load(first.manifest_path)
        return {name: record.fingerprint for name, record in m.stages.items()}

    cpu_fps = fps()
    gpu = _EncoderExecutor(nvenc)
    second = EditPipeline(projects_root=root, render_executor=gpu).run(encoder_video)
    assert second.outcomes["probe"] == second.outcomes["silences"] == second.outcomes["plan"] == "cached"
    assert second.outcomes["render"] == "ran" and gpu.calls == 1
    gpu_fps = fps()
    assert {k: v for k, v in gpu_fps.items() if k not in ("render", "report")} == {
        k: v for k, v in cpu_fps.items() if k not in ("render", "report")
    }
    assert gpu_fps["render"] != cpu_fps["render"]
    assert "| Codificador de vídeo | nvenc |" in second.report_path.read_text(encoding="utf-8")

    again = _EncoderExecutor(nvenc)
    assert EditPipeline(projects_root=root, render_executor=again).run(encoder_video).outcomes["render"] == "cached"
    assert again.calls == 0

    # NVENC pedido, mas caiu para a CPU no meio: o render fica registrado como de CPU
    broke = _EncoderExecutor(nvenc, fallback)
    fell = EditPipeline(projects_root=root, render_executor=broke).run(encoder_video, force=True)
    assert fell.outcomes["render"] == "ran"
    assert fps()["render"] == cpu_fps["render"]
    assert "| Codificador de vídeo | cpu (NVENC falhou: sessão) |" in fell.report_path.read_text(encoding="utf-8")
    # ...então um pedido de CPU o reaproveita, e um novo pedido de NVENC tenta a GPU de novo
    on_cpu = _EncoderExecutor(cpu)
    assert EditPipeline(projects_root=root, render_executor=on_cpu).run(encoder_video).outcomes["render"] == "cached"
    retry = _EncoderExecutor(nvenc)
    assert EditPipeline(projects_root=root, render_executor=retry).run(encoder_video).outcomes["render"] == "ran"


# --- CLI ---


def test_cli_encoder_options():
    parser = build_parser()
    for argv in (["edit", "v.mp4"], ["render", "plano.json"]):
        args = parser.parse_args(argv)
        assert (args.encoder, args.gpu_paralelo) == ("auto", 3)
        args = parser.parse_args([*argv, "--encoder", "nvenc", "--gpu-paralelo", "2"])
        assert (args.encoder, args.gpu_paralelo) == ("nvenc", 2)
    with pytest.raises(SystemExit):
        parser.parse_args(["edit", "v.mp4", "--encoder", "hevc"])
    with pytest.raises(SystemExit):
        parser.parse_args(["render", "p.json", "--gpu-paralelo", "0"])
