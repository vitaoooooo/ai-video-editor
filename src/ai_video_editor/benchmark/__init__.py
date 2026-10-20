"""Benchmark reproduzível de transcrição (velocidade e precisão)."""

from ai_video_editor.benchmark.report import format_markdown
from ai_video_editor.benchmark.normalize import normalize_words
from ai_video_editor.benchmark.rescore import RescoreError, rescore_run, save_run
from ai_video_editor.benchmark.runner import (
    PROVIDER_FACTORIES,
    register_benchmark_provider,
    run_benchmark,
)
from ai_video_editor.benchmark.schema import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkManifest,
    BenchmarkRun,
)
from ai_video_editor.benchmark.wer import compute_wer, compute_wer_words, normalize_text

__all__ = [
    "BenchmarkConfig",
    "BenchmarkItem",
    "BenchmarkManifest",
    "BenchmarkRun",
    "PROVIDER_FACTORIES",
    "RescoreError",
    "compute_wer",
    "compute_wer_words",
    "format_markdown",
    "normalize_text",
    "normalize_words",
    "register_benchmark_provider",
    "rescore_run",
    "run_benchmark",
    "save_run",
]
