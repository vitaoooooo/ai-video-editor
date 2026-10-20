"""Execução do benchmark de transcrição: configurações × itens."""

import platform
import re
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
from ai_video_editor.benchmark.normalize import has_normalizer, normalize_words
from ai_video_editor.benchmark.wer import compute_wer_words, normalize_text
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


HYPOTHESES_DIR = "hypotheses"


def hypothesis_relpath(config: BenchmarkConfig, item_name: str, multiple_items: bool) -> str:
    """Caminho (relativo à pasta da execução) do texto gerado por uma configuração.

    `hypotheses/<config>.txt` com um item só; `hypotheses/<item>/<config>.txt` com vários.
    Os caracteres inválidos em nomes de arquivo (`:` etc.) viram `_`: `small:cuda:float16` → `small_cuda_float16`.
    """
    safe = lambda text: re.sub(r"[^\w.\-]+", "_", text)  # noqa: E731
    name = f"{safe(config.label)}.txt"
    if multiple_items:
        return f"{HYPOTHESES_DIR}/{safe(item_name)}/{name}"
    return f"{HYPOTHESES_DIR}/{name}"


def score_texts(
    reference_text: str, hypothesis_text: str, language: Optional[str]
) -> tuple[WerResult, Optional[WerResult]]:
    """(WER bruto, WER normalizado); o normalizado é None se o idioma não tem regras cadastradas."""
    raw = compute_wer_words(normalize_text(reference_text), normalize_text(hypothesis_text))
    normalized: Optional[WerResult] = None
    if has_normalizer(language):
        normalized = compute_wer_words(
            normalize_words(reference_text, language), normalize_words(hypothesis_text, language)
        )
    return raw, normalized


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
    out_dir: Optional[Path] = None,
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
        run.effective_model = provider.effective_model(transcription_config)
    except Exception as exc:
        run.status = "error"
        run.error = str(exc) or type(exc).__name__
        return run

    for item in items:
        on_progress(f"  {config.label} · {item.name}")
        hypothesis_file = (
            hypothesis_relpath(config, item.name, len(items) > 1) if out_dir is not None else None
        )
        run.items.append(_run_item(provider, transcription_config, item, out_dir, hypothesis_file))
    if run.items and all(result.status == "error" for result in run.items):
        run.status = "error"
        run.error = run.items[0].error
    return run


def _run_item(
    provider: TranscriptionProvider,
    config: TranscriptionConfig,
    item: BenchmarkItem,
    out_dir: Optional[Path] = None,
    hypothesis_file: Optional[str] = None,
) -> ItemResult:
    item_config = config.model_copy(update={"language": item.language})
    try:
        started = time.perf_counter()
        transcript = provider.transcribe(item.audio, item_config)
        elapsed = time.perf_counter() - started
        text = transcript_text(transcript)
        # Gravar já aqui preserva o texto mesmo se uma configuração seguinte falhar.
        if out_dir is not None and hypothesis_file is not None:
            target = out_dir / hypothesis_file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        wer: Optional[WerResult] = None
        wer_normalized: Optional[WerResult] = None
        if item.reference is not None:
            wer, wer_normalized = score_texts(load_reference(item.reference), text, item.language)
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
        wer_normalized=wer_normalized,
        hypothesis_file=hypothesis_file,
    )


def _pool(results: list[WerResult]) -> Optional[WerResult]:
    """WER agregado: erros somados sobre palavras de referência somadas."""
    if not results:
        return None
    substitutions = sum(w.substitutions for w in results)
    deletions = sum(w.deletions for w in results)
    insertions = sum(w.insertions for w in results)
    words = sum(w.reference_words for w in results)
    return WerResult(
        wer=(substitutions + deletions + insertions) / words if words else 0.0,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=words,
        hypothesis_words=sum(w.hypothesis_words for w in results),
        tail_words=sum(w.tail_words for w in results),
    )


def compute_average(run: ConfigRun) -> ConfigAverage:
    ok = [result for result in run.items if result.status == "ok"]
    audio = sum(result.audio_duration_sec or 0.0 for result in ok)
    elapsed = sum(result.transcription_time_sec or 0.0 for result in ok)
    pooled = _pool([result.wer for result in ok if result.wer is not None])
    pooled_normalized = _pool([result.wer_normalized for result in ok if result.wer_normalized is not None])
    return ConfigAverage(
        config=run.config.label,
        items_ok=len(ok),
        items_total=len(run.items),
        audio_duration_sec=audio,
        transcription_time_sec=elapsed,
        real_time_factor=elapsed / audio if audio > 0 else None,
        wer=pooled,
        wer_normalized=pooled_normalized,
    )


def run_benchmark(
    name: str,
    configs: list[BenchmarkConfig],
    items: list[BenchmarkItem],
    provider_factories: Optional[dict[str, ProviderFactory]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    out_dir: Optional[Path] = None,
) -> BenchmarkRun:
    """Roda todas as configurações em todos os itens; uma falha não interrompe as demais.

    Com `out_dir`, o texto gerado por cada configuração é gravado em `out_dir/hypotheses/`.
    """
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
        runs.append(_run_config(config, items, factory, progress, out_dir))
    return BenchmarkRun(
        name=name,
        created_at=datetime.now().astimezone(),
        environment=collect_environment(),
        items=items,
        configs=runs,
        averages=[compute_average(run) for run in runs if run.items],
    )
