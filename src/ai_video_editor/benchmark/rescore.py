"""Recalcula as métricas de uma execução do benchmark a partir das transcrições salvas."""

from pathlib import Path
from typing import Optional

from ai_video_editor.benchmark.report import format_markdown
from ai_video_editor.benchmark.runner import compute_average, load_reference, score_texts
from ai_video_editor.benchmark.schema import BenchmarkItem, BenchmarkRun

RESULTS_JSON = "results.json"
RESULTS_MD = "results.md"


class RescoreError(Exception):
    """A execução não pode ser recalculada (arquivo ou referência ausente)."""


def _find_reference(item: BenchmarkItem, run_dir: Path, override: Optional[Path]) -> Path:
    """Referência: a informada ou a gravada no results.json (se relativa, a partir do cwd ou das pastas acima)."""
    if override is not None:
        candidates = [Path(override)]
    elif item.reference is None:
        raise RescoreError(f"O item '{item.name}' não tem referência.")
    elif item.reference.is_absolute():
        candidates = [item.reference]
    else:
        candidates = [Path.cwd() / item.reference] + [
            parent / item.reference for parent in run_dir.resolve().parents
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RescoreError(
        f"Referência do item '{item.name}' não encontrada ({candidates[0]}). Rode a partir da raiz do "
        "projeto ou informe --reference."
    )


def rescore_run(run_dir: Path, reference: Optional[Path] = None) -> BenchmarkRun:
    """Relê `results.json` e `hypotheses/`, recalcula WER (bruto e normalizado) e as médias.

    Tempos, carga, ambiente e configurações são mantidos como foram medidos. Não transcreve nada.
    `reference` substitui a referência gravada (só vale para execuções de um item).
    """
    run_dir = Path(run_dir)
    results_path = run_dir / RESULTS_JSON
    if not results_path.is_file():
        raise RescoreError(f"{results_path} não encontrado: informe a pasta de uma execução do benchmark.")
    run = BenchmarkRun.model_validate_json(results_path.read_text(encoding="utf-8"))
    if reference is not None and len(run.items) != 1:
        raise RescoreError("--reference só vale para execuções com um item.")

    items = {item.name: item for item in run.items}
    reference_texts: dict[str, str] = {}
    scored = 0
    for config_run in run.configs:
        for result in config_run.items:
            if result.status != "ok":
                continue
            item = items[result.item]
            if item.reference is None and reference is None:
                continue
            if result.hypothesis_file is None or not (run_dir / result.hypothesis_file).is_file():
                raise RescoreError(
                    f"Transcrição salva de '{config_run.config.label}' · {result.item} não encontrada "
                    f"({result.hypothesis_file or 'results.json sem hypothesis_file'}). "
                    "Essa execução não guardou os textos; rode o benchmark de novo."
                )
            if item.name not in reference_texts:
                reference_texts[item.name] = load_reference(_find_reference(item, run_dir, reference))
            hypothesis = (run_dir / result.hypothesis_file).read_text(encoding="utf-8")
            result.wer, result.wer_normalized = score_texts(
                reference_texts[item.name], hypothesis, item.language
            )
            scored += 1
    if scored == 0:
        raise RescoreError("Nada para recalcular: nenhum item com referência e transcrição salva.")
    run.schema_version = 2
    run.averages = [compute_average(config_run) for config_run in run.configs if config_run.items]
    return run


def save_run(run: BenchmarkRun, run_dir: Path) -> str:
    """Grava results.json e results.md na pasta da execução; devolve o Markdown."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RESULTS_JSON).write_text(run.model_dump_json(indent=2), encoding="utf-8")
    markdown = format_markdown(run)
    (run_dir / RESULTS_MD).write_text(markdown + "\n", encoding="utf-8")
    return markdown
