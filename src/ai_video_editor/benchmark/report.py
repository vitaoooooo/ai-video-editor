"""Relatório Markdown do benchmark (tabelas prontas para o README)."""

from typing import Optional

from ai_video_editor.benchmark.schema import BenchmarkRun, ConfigRun, ItemResult, WerResult

_DASH = "—"


def _seconds(value: Optional[float]) -> str:
    if value is None:
        return _DASH
    if value >= 60:
        return f"{int(value // 60)}min {value % 60:04.1f}s"
    return f"{value:.1f}s"


def _rtf(value: Optional[float]) -> str:
    return _DASH if value is None else f"{value:.3f}x"


def _wer_cells(wer: Optional[WerResult]) -> list[str]:
    if wer is None:
        return [_DASH] * 5
    return [
        f"{wer.wer * 100:.1f}%",
        str(wer.substitutions),
        str(wer.deletions),
        str(wer.insertions),
        str(wer.hypothesis_words),
    ]


def _device(run: ConfigRun) -> str:
    if run.effective_device is None:
        return _DASH
    return f"{run.effective_device}, {run.effective_compute_type}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


_HEADER = ["Configuração", "Efetivo", "Carga", "Tempo", "Tempo real", "WER", "S", "D", "I", "Palavras"]


def _config_row(run: ConfigRun, result: Optional[ItemResult]) -> list[str]:
    label = f"`{run.config.label}`"
    if result is None or result.status == "error":
        return [label, _device(run), _seconds(run.load_time_sec), "falhou", _DASH] + [_DASH] * 5
    return [
        label,
        _device(run),
        _seconds(run.load_time_sec),
        _seconds(result.transcription_time_sec),
        _rtf(result.real_time_factor),
        *_wer_cells(result.wer),
    ]


def format_markdown(run: BenchmarkRun) -> str:
    env = run.environment
    lines = [f"# Benchmark de transcrição — {run.name}", ""]
    lines.append(f"- Data: {run.created_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Versão do projeto: {env.project_version} · Python {env.python} · faster-whisper {env.faster_whisper or _DASH}")
    lines.append(f"- CPU: {env.cpu} · GPU: {env.gpu or 'nenhuma detectada'}")
    lines.append(f"- Normalização antes do WER: {run.normalization}")
    lines.append(
        "- *Tempo real* = tempo de transcrição ÷ duração do áudio (menor é mais rápido). O tempo inclui a "
        "decodificação do áudio com FFmpeg e não inclui a carga do modelo, medida à parte em *Carga*."
    )
    lines.append("")

    for item in run.items:
        lines.append(f"## {item.name}")
        lines.append("")
        durations = [
            result.audio_duration_sec
            for config_run in run.configs
            for result in config_run.items
            if result.item == item.name and result.audio_duration_sec
        ]
        duration = _seconds(durations[0]) if durations else _DASH
        reference = "com referência" if item.reference else "sem referência (WER não calculado)"
        language = item.language or "automático"
        lines.append(f"Áudio: `{item.audio.name}` · duração {duration} · idioma {language} · {reference}")
        lines.append("")
        rows = []
        for config_run in run.configs:
            result = next((r for r in config_run.items if r.item == item.name), None)
            rows.append(_config_row(config_run, result))
        lines += _table(_HEADER, rows)
        lines.append("")

    if len(run.items) > 1 and run.averages:
        lines.append("## Média geral")
        lines.append("")
        lines.append(
            "Tempo real = tempo total ÷ áudio total; WER agregado = erros somados ÷ palavras de referência "
            "somadas, só dos itens com referência. Considera apenas os itens concluídos."
        )
        lines.append("")
        rows = []
        for average in run.averages:
            rows.append(
                [
                    f"`{average.config}`",
                    f"{average.items_ok}/{average.items_total}",
                    _seconds(average.audio_duration_sec),
                    _seconds(average.transcription_time_sec),
                    _rtf(average.real_time_factor),
                    *_wer_cells(average.wer),
                ]
            )
        lines += _table(
            ["Configuração", "Itens", "Áudio", "Tempo", "Tempo real", "WER", "S", "D", "I", "Palavras"], rows
        )
        lines.append("")

    substitution_blocks: list[str] = []
    for config_run in run.configs:
        for result in config_run.items:
            if result.wer is None or not result.wer.top_substitutions:
                continue
            substitution_blocks.append(f"### `{config_run.config.label}` · {result.item}")
            substitution_blocks.append("")
            substitution_blocks += _table(
                ["Referência", "Modelo escreveu", "Vezes"],
                [[s.reference, s.hypothesis, str(s.count)] for s in result.wer.top_substitutions],
            )
            substitution_blocks.append("")
    if substitution_blocks:
        lines += ["## Substituições mais frequentes", ""] + substitution_blocks

    failures = [
        f"- `{config_run.config.label}`: {config_run.error}"
        for config_run in run.configs
        if config_run.error
    ] + [
        f"- `{config_run.config.label}` · {result.item}: {result.error}"
        for config_run in run.configs
        if not config_run.error
        for result in config_run.items
        if result.error
    ]
    if failures:
        lines += ["## Falhas", ""] + failures + [""]
    return "\n".join(lines)
