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
    """WER, S, D, I."""
    if wer is None:
        return [_DASH] * 4
    return [
        f"{wer.wer * 100:.1f}%",
        str(wer.substitutions),
        str(wer.deletions),
        str(wer.insertions),
    ]


def _tail_cell(raw: Optional[WerResult]) -> str:
    return _DASH if raw is None else str(raw.tail_words)


def _words_cell(raw: Optional[WerResult]) -> str:
    return _DASH if raw is None else str(raw.hypothesis_words)


def _device(run: ConfigRun) -> str:
    if run.effective_device is None:
        return _DASH
    return f"{run.effective_device}, {run.effective_compute_type}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


_WER_HEADER = ["WER bruto", "S", "D", "I", "WER norm.", "S", "D", "I", "Após a ref.", "Palavras"]
_HEADER = ["Configuração", "Efetivo", "Carga", "Tempo", "Tempo real", *_WER_HEADER]


def _wer_block(raw: Optional[WerResult], normalized: Optional[WerResult]) -> list[str]:
    return [*_wer_cells(raw), *_wer_cells(normalized), _tail_cell(raw), _words_cell(raw)]


def _config_row(run: ConfigRun, result: Optional[ItemResult]) -> list[str]:
    label = f"`{run.config.label}`"
    if result is None or result.status == "error":
        return [label, _device(run), _seconds(run.load_time_sec), "falhou", _DASH] + [_DASH] * len(_WER_HEADER)
    return [
        label,
        _device(run),
        _seconds(run.load_time_sec),
        _seconds(result.transcription_time_sec),
        _rtf(result.real_time_factor),
        *_wer_block(result.wer, result.wer_normalized),
    ]


def format_markdown(run: BenchmarkRun) -> str:
    env = run.environment
    lines = [f"# Benchmark de transcrição — {run.name}", ""]
    lines.append(f"- Data: {run.created_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Versão do projeto: {env.project_version} · Python {env.python} · faster-whisper {env.faster_whisper or _DASH}")
    lines.append(f"- CPU: {env.cpu} · GPU: {env.gpu or 'nenhuma detectada'}")
    lines.append(f"- WER bruto: {run.normalization}")
    lines.append(f"- WER normalizado: o bruto mais, nos dois textos, {run.normalization_extra}")
    lines.append(
        "- *Após a ref.* = palavras do final da transcrição gerada sem correspondência na referência "
        "(a referência pode terminar antes da fala); contadas à parte, fora das inserções e do WER."
    )
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
                    *_wer_block(average.wer, average.wer_normalized),
                ]
            )
        lines += _table(["Configuração", "Itens", "Áudio", "Tempo", "Tempo real", *_WER_HEADER], rows)
        lines.append("")

    substitution_blocks: list[str] = []
    for config_run in run.configs:
        for result in config_run.items:
            for title, wer in (("bruto", result.wer), ("normalizado", result.wer_normalized)):
                if wer is None or not wer.top_substitutions:
                    continue
                substitution_blocks.append(f"### `{config_run.config.label}` · {result.item} · WER {title}")
                substitution_blocks.append("")
                substitution_blocks += _table(
                    ["Referência", "Modelo escreveu", "Vezes"],
                    [[s.reference, s.hypothesis, str(s.count)] for s in wer.top_substitutions],
                )
                substitution_blocks.append("")
    if substitution_blocks:
        lines += ["## Substituições mais frequentes", ""] + substitution_blocks

    tails = [
        (config_run, result)
        for config_run in run.configs
        for result in config_run.items
        if result.wer is not None and result.wer.tail_words
    ]
    if tails:
        lines += ["## Palavras após o fim da referência", ""]
        for config_run, result in tails:
            assert result.wer is not None
            lines.append(
                f"- `{config_run.config.label}` · {result.item}: {result.wer.tail_words} palavras, "
                f"começando em: “{result.wer.tail_preview}…”"
            )
        lines.append("")

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
