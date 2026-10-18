"""Relatório Markdown da edição (`edit_report.md`)."""

from pathlib import Path

from ai_video_editor.core.schema import EditPlan
from ai_video_editor.pipeline.schema import EditPipelineParams
from ai_video_editor.planning.schema import TimeRange


def format_timestamp(seconds: float) -> str:
    """`75.5` -> `01:15.500` (ou `1:01:15.500` acima de uma hora)."""
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def build_edit_report(
    video_path: Path,
    original_duration_sec: float,
    final_duration_sec: float,
    plan: EditPlan,
    removed: list[TimeRange],
    params: EditPipelineParams,
    output_path: Path,
) -> str:
    """Monta o Markdown com resumo, parâmetros e a lista de trechos removidos."""
    removed_sec = max(0.0, original_duration_sec - final_duration_sec)
    pct = removed_sec / original_duration_sec * 100 if original_duration_sec > 0 else 0.0
    width, height = plan.target_resolution
    cut = params.cut

    lines = [
        f"# Relatório de edição — {video_path.name}",
        "",
        "## Resumo",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Vídeo de origem | `{video_path}` |",
        f"| Vídeo final | `{output_path.name}` |",
        f"| Duração original | {format_timestamp(original_duration_sec)} ({original_duration_sec:.2f}s) |",
        f"| Duração final | {format_timestamp(final_duration_sec)} ({final_duration_sec:.2f}s) |",
        f"| Tempo removido | {removed_sec:.2f}s ({pct:.1f}%) |",
        f"| Número de cortes | {len(removed)} |",
        f"| Trechos mantidos | {len(plan.segments)} |",
        f"| Saída | {width}x{height} @ {plan.target_fps}fps |",
        "",
        "## Parâmetros",
        "",
        "| Parâmetro | Valor |",
        "| --- | --- |",
        f"| threshold_db | {params.threshold_db} dB |",
        f"| min_silence_sec | {params.min_silence_sec}s |",
        f"| padding_before_sec | {cut.padding_before_sec}s |",
        f"| padding_after_sec | {cut.padding_after_sec}s |",
        f"| min_gap_to_cut_sec | {cut.min_gap_to_cut_sec}s |",
        f"| min_keep_sec | {cut.min_keep_sec}s |",
        f"| audio_fade_sec | {plan.audio_fade_sec}s |",
        "",
        "## Trechos removidos",
        "",
    ]
    if removed:
        lines += ["| # | Início | Fim | Duração |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {i} | {format_timestamp(r.start_sec)} | {format_timestamp(r.end_sec)} | {r.duration:.3f}s |"
            for i, r in enumerate(removed, start=1)
        ]
    else:
        lines.append("Nenhum trecho removido.")
    return "\n".join(lines) + "\n"
