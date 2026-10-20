"""Relatório Markdown da edição (`edit_report.md`)."""

from pathlib import Path
from typing import Optional, Union

from ai_video_editor.audio import AudioAnalysisResult
from ai_video_editor.core.schema import EditPlan
from ai_video_editor.director.schema import DirectorRun
from ai_video_editor.revision.schema import RevisionRun
from ai_video_editor.pipeline.project import EDIT_DIFF_FILE, EDIT_TRANSCRIPT_FILE
from ai_video_editor.pipeline.schema import EditDiff, EditPipelineParams, RunTimings
from ai_video_editor.pipeline.timing import build_timings_section
from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.renderer.loudness import LoudnessResult


def format_timestamp(seconds: float) -> str:
    """`75.5` -> `01:15.500` (ou `1:01:15.500` acima de uma hora)."""
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _threshold_rows(params: EditPipelineParams, audio: Optional[AudioAnalysisResult]) -> list[str]:
    """Linhas da tabela de parâmetros sobre o limite de silêncio (modo, medidas e limite efetivo)."""
    if audio is None:
        return [f"| threshold_db | {params.threshold_db} dB |"]
    mode = "auto" if audio.params.threshold_mode == "auto" else "fixo"
    rows = [f"| threshold_mode | {mode} |"]
    if audio.levels is not None:
        rows += [
            f"| noise_floor_db (piso de ruído) | {audio.levels.noise_floor_db} dB |",
            f"| speech_level_db (nível de fala) | {audio.levels.speech_level_db} dB |",
        ]
    rows.append(f"| threshold_db | {audio.params.threshold_db} dB |")
    return rows


def _audio_warnings(audio: Optional[AudioAnalysisResult]) -> list[str]:
    if audio is None or audio.levels is None or not audio.levels.warnings:
        return []
    return ["", *(f"> ⚠️ **Atenção:** {w}" for w in audio.levels.warnings)]


def _loudness_section(params: EditPipelineParams, loudness: Optional[LoudnessResult]) -> list[str]:
    """Seção "Volume": preset, loudness medido do áudio editado e alvo aplicado (ou por que não foi)."""
    lines = ["## Volume", "", "| Item | Valor |", "| --- | --- |"]
    if loudness is None:
        state = "desligada (--loudness off)" if params.loudness == "off" else "—"
        return [*lines, f"| Normalização | {state} |", ""]
    target = loudness.target
    if loudness.status == "skipped":
        lines.append(f"| Normalização | pulada: {_cell(loudness.note or '')} |")
    else:
        lines.append(f"| Normalização | loudnorm em duas passadas, preset {target.preset} |")
    measured = loudness.measured
    if measured is not None:
        lines += [
            f"| Loudness medido (I) | {measured.input_i:.1f} LUFS |",
            f"| True peak medido (TP) | {measured.input_tp:.1f} dBTP |",
            f"| Faixa de loudness medida (LRA) | {measured.input_lra:.1f} LU |",
        ]
    if loudness.status == "applied":
        lines.append(
            f"| Alvo aplicado | {target.integrated_lufs:g} LUFS, TP {target.true_peak_dbtp:g} dBTP, "
            f"LRA {target.lra:g} LU |"
        )
    if loudness.measure_sec is not None:
        lines.append(f"| Passada de medição | {loudness.measure_sec:.2f}s |")
    lines += ["", *(f"> ⚠️ **Atenção:** {w}" for w in loudness.warnings)]
    return lines if not loudness.warnings else [*lines, ""]


def build_edit_report(
    video_path: Path,
    original_duration_sec: float,
    final_duration_sec: float,
    plan: EditPlan,
    removed: list[TimeRange],
    params: EditPipelineParams,
    output_path: Path,
    director: Optional[DirectorRun] = None,
    audio: Optional[AudioAnalysisResult] = None,
    subtitles_path: Optional[Path] = None,
    subtitle_blocks: Optional[int] = None,
    transcription_model: Optional[str] = None,
    timings: Optional[RunTimings] = None,
    comparison: Optional[EditDiff] = None,
    removal_label: Optional[str] = None,
    loudness: Optional[LoudnessResult] = None,
) -> str:
    """Monta o Markdown com resumo, parâmetros, trechos removidos, decisões do Diretor e tempos.

    Com `comparison` (transcrição comparativa), mostra o status de aplicação de cada remoção;
    com `removal_label` também, abre o relatório com a contagem (`Diretor: 7 remoções pedidas...`).
    """
    removed_sec = max(0.0, original_duration_sec - final_duration_sec)
    pct = removed_sec / original_duration_sec * 100 if original_duration_sec > 0 else 0.0
    width, height = plan.target_resolution
    cut = params.cut
    subtitle_rows = (
        [f"| Legendas | `{subtitles_path.name}` ({subtitle_blocks} blocos) |"]
        if subtitles_path is not None and subtitle_blocks is not None
        else []
    )

    comparison_row = (
        f"| Transcrição comparativa | `{EDIT_TRANSCRIPT_FILE}` e `{EDIT_DIFF_FILE}` |"
        if comparison is not None
        else "| Transcrição comparativa | pulada (requer --transcribe) |"
    )
    lines = [f"# Relatório de edição — {video_path.name}", ""]
    if comparison is not None and removal_label:
        lines += [f"> **{removal_label}:** {comparison.removal_counts()}", ""]
    lines += [
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
        *([f"| Transcrição | `{transcription_model}` |"] if transcription_model else []),
        comparison_row,
        "",
        "## Parâmetros",
        "",
        "| Parâmetro | Valor |",
        "| --- | --- |",
        *_threshold_rows(params, audio),
        f"| min_silence_sec | {params.min_silence_sec}s |",
        f"| padding_before_sec | {cut.padding_before_sec}s |",
        f"| padding_after_sec | {cut.padding_after_sec}s |",
        f"| min_gap_to_cut_sec | {cut.min_gap_to_cut_sec}s |",
        f"| min_keep_sec | {cut.min_keep_sec}s |",
        f"| audio_fade_sec | {plan.audio_fade_sec}s |",
        *([f"| subtitle_max_chars | {params.subtitle.max_chars_per_line} |"] if subtitle_rows else []),
        *_audio_warnings(audio),
        "",
        *_loudness_section(params, loudness),
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
        lines += ["", *_director_section(director, comparison)]
    text = "\n".join(lines) + "\n"
    if timings is not None:
        text += build_timings_section(timings)
    return text


def _cell(text: str) -> str:
    """Texto seguro para uma célula de tabela Markdown."""
    return " ".join(text.split()).replace("|", "\\|")


def model_rows(run: Union[DirectorRun, RevisionRun]) -> list[str]:
    """Linhas da tabela com a cadeia pedida, o modelo usado e as trocas automáticas."""
    rows: list[str] = []
    if len(run.model_chain) > 1:
        rows.append(f"| Cadeia de modelos | {' → '.join(run.model_chain)} |")
    if run.effective_model_id:
        rows.append(f"| Modelo usado | {run.effective_provider}:{run.effective_model_id} |")
    for switch in run.model_switches:
        rows.append(f"| Troca de modelo | {switch.from_model} → {switch.to_model}: {_cell(switch.reason)} |")
    return rows


def application_cell(comparison: Optional[EditDiff], sentence_id: int) -> str:
    """Status de aplicação de uma remoção para uma célula de tabela (`—` sem transcrição comparativa)."""
    entry = comparison.entry_for(sentence_id) if comparison is not None else None
    if entry is None or not entry.removal_requested:
        return "—"
    return _cell(entry.application_label())


def _director_section(director: DirectorRun, comparison: Optional[EditDiff] = None) -> list[str]:
    brief = director.brief
    target = f"{brief.target_duration_sec:.0f}s" if brief.target_duration_sec is not None else "—"
    lines = [
        "## Decisões do Diretor",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Diretor | {director.provider} ({director.model_id}) |",
        *model_rows(director),
        f"| Raciocínio | {director.thinking or '—'} |",
        f"| Intensidade | {brief.intensity} |",
        f"| Duração alvo | {target} |",
        f"| Remover enrolação | {'sim' if brief.remove_fillers else 'não'} |",
        f"| Instruções | {_cell(brief.instructions) if brief.instructions else '—'} |",
        f"| Frases analisadas | {len(director.sentences)} |",
    ]
    if director.sentences:
        durations = [s.duration for s in director.sentences]
        lines.append(f"| Duração média das frases | {sum(durations) / len(durations):.1f}s |")
        lines.append(f"| Duração máxima das frases | {max(durations):.1f}s |")
    for i, attempt in enumerate(director.attempts, start=1):
        tokens = (
            f"{attempt.input_tokens} entrada / {attempt.output_tokens} saída"
            if attempt.input_tokens is not None or attempt.output_tokens is not None
            else "não informado"
        )
        status = "ok" if attempt.success else f"inválida ({attempt.error_type})"
        lines.append(f"| Tentativa {i} | {status}, {attempt.latency_sec:.2f}s, chamadas à API: {attempt.api_attempts}, tokens: {tokens} |")
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
        "| # | Início | Fim | Categoria | Texto | Motivo | Aplicação |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {sentence.id} | {format_timestamp(sentence.start_sec)} | {format_timestamp(sentence.end_sec)} "
        f"| {removal.category} | {_cell(sentence.text)} | {_cell(removal.reason)} "
        f"| {application_cell(comparison, sentence.id)} |"
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
    changes: list[tuple[int, str, str, str, str, str]],
    reviser: Optional[RevisionRun] = None,
) -> str:
    """Seção "Revisão" do relatório de uma versão nova.

    `changes`: (id da frase, texto, antes, depois, aplicação, motivo), em ordem cronológica.
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
        *(
            [f"| Revisor | {reviser.provider} ({reviser.model_id}) |", *model_rows(reviser)]
            if reviser is not None
            else []
        ),
        "",
        f"**Resumo:** {summary.strip() or '—'}",
        "",
    ]
    if not changes:
        lines.append("Nenhuma frase mudou de estado.")
    else:
        lines += [
            "| Frase | Texto | Antes | Depois | Aplicação | Motivo |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {sid} | {_cell(text)} | {before} | {after} | {application} | {_cell(reason)} |"
            for sid, text, before, after, application, reason in changes
        ]
    return "\n".join(lines) + "\n"
