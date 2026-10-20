"""Execução do benchmark de transcrição: configurações × itens."""

import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ai_video_editor import __version__
from ai_video_editor.benchmark.schema import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkRun,
    ConfigAverage,
    ConfigRun,
    EnvironmentInfo,
    ItemResult,
    WerResult,
)
from ai_video_editor.benchmark.wer import compute_wer
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig

ProviderFactory = Callable[[BenchmarkConfig], TranscriptionProvider]


def _faster_whisper_factory(config: BenchmarkConfig) -> TranscriptionProvider:
    from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider

    return FasterWhisperProvider()


def _mock_factory(config: BenchmarkConfig) -> TranscriptionProvider:
    from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider

    return MockTranscriptionProvider()


# Provedores de transcrição que o benchmark sabe criar. Para incluir outro (ex.: Gemini),
# registre aqui uma fábrica que devolva um `TranscriptionProvider`.
PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "faster_whisper": _faster_whisper_factory,
    "mock": _mock_factory,
}


def register_benchmark_provider(name: str, factory: ProviderFactory) -> None:
    PROVIDER_FACTORIES[name] = factory


def load_reference(path: Path) -> str:
    """Lê a referência ignorando linhas de comentário (iniciadas por '#')."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def transcript_text(transcript: Transcript) -> str:
    return " ".join(segment.text for segment in transcript.segments)


def collect_environment() -> EnvironmentInfo:
    gpu = None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            gpu = proc.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from importlib.metadata import version

        faster_whisper = version("faster-whisper")
    except Exception:
        faster_whisper = None
    return EnvironmentInfo(
        project_version=__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        gpu=gpu,
        faster_whisper=faster_whisper,
    )


def _run_config(
    config: BenchmarkConfig,
    items: list[BenchmarkItem],
    factory: ProviderFactory,
    on_progress: Callable[[str], None],
) -> ConfigRun:
    transcription_config = TranscriptionConfig(
        model_size=config.model, device=config.device, compute_type=config.compute_type
    )
    run = ConfigRun(config=config, status="ok")
    try:
        provider = factory(config)
        # O modelo é carregado antes de medir; o download da primeira vez cai aqui.
        started = time.perf_counter()
        preload = getattr(provider, "preload", None)
        if preload is not None:
            preload(transcription_config)
        run.load_time_sec = time.perf_counter() - started
        run.effective_device, run.effective_compute_type = provider.effective_settings(
            transcription_config
        )
    except Exception as exc:
        run.status = "error"
        run.error = str(exc) or type(exc).__name__
        return run

    for item in items:
        on_progress(f"  {config.label} · {item.name}")
        run.items.append(_run_item(provider, transcription_config, item))
    if run.items and all(result.status == "error" for result in run.items):
        run.status = "error"
        run.error = run.items[0].error
    return run


def _run_item(
    provider: TranscriptionProvider, config: TranscriptionConfig, item: BenchmarkItem
) -> ItemResult:
    item_config = config.model_copy(update={"language": item.language})
    try:
        started = time.perf_counter()
        transcript = provider.transcribe(item.audio, item_config)
        elapsed = time.perf_counter() - started
        wer: Optional[WerResult] = None
        if item.reference is not None:
            wer = compute_wer(load_reference(item.reference), transcript_text(transcript))
    except Exception as exc:
        return ItemResult(item=item.name, status="error", error=str(exc) or type(exc).__name__)
    duration = transcript.duration_sec
    return ItemResult(
        item=item.name,
        status="ok",
        audio_duration_sec=duration,
        transcription_time_sec=elapsed,
        real_time_factor=elapsed / duration if duration > 0 else None,
        wer=wer,
    )


def compute_average(run: ConfigRun) -> ConfigAverage:
    ok = [result for result in run.items if result.status == "ok"]
    audio = sum(result.audio_duration_sec or 0.0 for result in ok)
    elapsed = sum(result.transcription_time_sec or 0.0 for result in ok)
    with_wer = [result.wer for result in ok if result.wer is not None]
    pooled: Optional[WerResult] = None
    if with_wer:
        substitutions = sum(w.substitutions for w in with_wer)
        deletions = sum(w.deletions for w in with_wer)
        insertions = sum(w.insertions for w in with_wer)
        words = sum(w.reference_words for w in with_wer)
        pooled = WerResult(
            wer=(substitutions + deletions + insertions) / words if words else 0.0,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            reference_words=words,
            hypothesis_words=sum(w.hypothesis_words for w in with_wer),
        )
    return ConfigAverage(
        config=run.config.label,
        items_ok=len(ok),
        items_total=len(run.items),
        audio_duration_sec=audio,
        transcription_time_sec=elapsed,
        real_time_factor=elapsed / audio if audio > 0 else None,
        wer=pooled,
    )


def run_benchmark(
    name: str,
    configs: list[BenchmarkConfig],
    items: list[BenchmarkItem],
    provider_factories: Optional[dict[str, ProviderFactory]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> BenchmarkRun:
    """Roda todas as configurações em todos os itens; uma falha não interrompe as demais."""
    factories = provider_factories or PROVIDER_FACTORIES
    progress = on_progress or (lambda message: None)
    runs: list[ConfigRun] = []
    for config in configs:
        factory = factories.get(config.provider)
        if factory is None:
            runs.append(
                ConfigRun(
                    config=config,
                    status="error",
                    error=f"Provedor desconhecido '{config.provider}'. Disponíveis: {', '.join(sorted(factories))}",
                )
            )
            continue
        progress(f"Configuração {config.label}")
        runs.append(_run_config(config, items, factory, progress))
    return BenchmarkRun(
        name=name,
        created_at=datetime.now().astimezone(),
        environment=collect_environment(),
        items=items,
        configs=runs,
        averages=[compute_average(run) for run in runs if run.items],
    )
