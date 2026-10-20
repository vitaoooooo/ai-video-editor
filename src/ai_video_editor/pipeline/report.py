"""Relatório Markdown da edição (`edit_report.md`)."""

from pathlib import Path
from typing import Optional

from ai_video_editor.core.schema import EditPlan
from ai_video_editor.director.schema import DirectorRun
from ai_video_editor.pipeline.schema import EditPipelineParams, RunTimings
from ai_video_editor.pipeline.timing import build_timings_section
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
    director: Optional[DirectorRun] = None,
    subtitles_path: Optional[Path] = None,
    subtitle_blocks: Optional[int] = None,
    timings: Optional[RunTimings] = None,
) -> str:
    """Monta o Markdown com resumo, parâmetros, trechos removidos, decisões do Diretor e tempos."""
    removed_sec = max(0.0, original_duration_sec - final_duration_sec)
    pct = removed_sec / original_duration_sec * 100 if original_duration_sec > 0 else 0.0
    width, height = plan.target_resolution
    cut = params.cut
    subtitle_rows = (
        [f"| Legendas | `{subtitles_path.name}` ({subtitle_blocks} blocos) |"]
        if subtitles_path is not None and subtitle_blocks is not None
        else []
    )

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
        *subtitle_rows,
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
        *([f"| subtitle_max_chars | {params.subtitle.max_chars_per_line} |"] if subtitle_rows else []),
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
    if director is not None:
        lines += ["", *_director_section(director)]
    text = "\n".join(lines) + "\n"
    if timings is not None:
        text += build_timings_section(timings)
    return text


def _cell(text: str) -> str:
    """Texto seguro para uma célula de tabela Markdown."""
    return " ".join(text.split()).replace("|", "\\|")


def _director_section(director: DirectorRun) -> list[str]:
    brief = director.brief
    target = f"{brief.target_duration_sec:.0f}s" if brief.target_duration_sec is not None else "—"
    lines = [
        "## Decisões do Diretor",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Diretor | {director.provider} ({director.model_id}) |",
        f"| Raciocínio | {director.thinking or '—'} |",
        f"| Intensidade | {brief.intensity} |",
        f"| Duração alvo | {target} |",
        f"| Remover enrolação | {'sim' if brief.remove_fillers else 'não'} |",
        f"| Instruções | {_cell(brief.instructions) if brief.instructions else '—'} |",
        f"| Frases analisadas | {len(director.sentences)} |",
    ]
    for i, attempt in enumerate(director.attempts, start=1):
        tokens = (
            f"{attempt.input_tokens} entrada / {attempt.output_tokens} saída"
            if attempt.input_tokens is not None or attempt.output_tokens is not None
            else "não informado"
        )
        status = "ok" if attempt.success else f"inválida ({attempt.error_type})"
        lines.append(f"| Tentativa {i} | {status}, {attempt.latency_sec:.2f}s, tokens: {tokens} |")
    lines.append("")

    for warning in director.warnings:
        lines += [f"> **Aviso:** {warning}", ""]
    if director.decision is None:
        return lines

    lines += [f"**Resumo:** {director.decision.summary.strip()}", ""]
    removed = director.removed_sentences
    if not removed:
        lines.append("Nenhuma frase removida.")
        return lines
    lines += [
        "| # | Início | Fim | Categoria | Texto | Motivo |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {sentence.id} | {format_timestamp(sentence.start_sec)} | {format_timestamp(sentence.end_sec)} "
        f"| {removal.category} | {_cell(sentence.text)} | {_cell(removal.reason)} |"
        for sentence, removal in removed
    ]
    return lines


def build_revision_section(
    request: str,
    base_version: int,
    new_version: int,
    summary: str,
    pacing_before: str,
    pacing_after: str,
    changes: list[tuple[int, str, str, str, str]],
) -> str:
    """Seção "Revisão" do relatório de uma versão nova.

    `changes`: (id da frase, texto, antes, depois, motivo), em ordem cronológica.
    """
    lines = [
        "",
        "## Revisão",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Versão | v{new_version} (a partir da v{base_version}) |",
        f"| Pedido | {_cell(request)} |",
        f"| Ritmo das pausas | {pacing_before} → {pacing_after} |",
        "",
        f"**Resumo:** {summary.strip() or '—'}",
        "",
    ]
    if not changes:
        lines.append("Nenhuma frase mudou de estado.")
    else:
        lines += ["| Frase | Texto | Antes | Depois | Motivo |", "| --- | --- | --- | --- | --- |"]
        lines += [
            f"| {sid} | {_cell(text)} | {before} | {after} | {_cell(reason)} |"
            for sid, text, before, after, reason in changes
        ]
    return "\n".join(lines) + "\n"
