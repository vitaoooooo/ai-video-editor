"""Relatórios do Editor de IA: `editor_relatorio.md` (o que a IA decidiu e o que o código fez) e
`editor_plano.md` (o plano em IDs, já convertido em tempos)."""

from pathlib import Path
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.schema import CallStats, EditorPlan, EditorRun
from ai_video_editor.pipeline.schema import RunTimings
from ai_video_editor.pipeline.timing import build_timings_section, format_duration
from ai_video_editor.roteiro.report import clock

_ORIGIN = {"ia": "IA", "cache": "IA (cache)", "fallback": "**fallback neutro**", "falha": "**falhou**"}
_TARGET_ORIGIN = {"pedido": "do pedido", "ia": "escolhida pela IA", "nenhuma": "livre"}


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _sec(seconds: Optional[float]) -> str:
    return "—" if seconds is None else f"{clock(seconds)} ({seconds:.0f}s)"


def _tokens(value: Optional[int]) -> str:
    return "—" if value is None else f"{value:,}".replace(",", ".")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _stats_row(name: str, stats: CallStats) -> str:
    return (
        f"| {name} | {stats.calls} | {stats.cached} | {stats.api_calls} | {_tokens(stats.input_tokens)} | "
        f"{_tokens(stats.output_tokens)} | {format_duration(stats.seconds)} |"
    )


def build_plan_markdown(plan: EditorPlan, material: EditorMaterial) -> str:
    lines = [
        f"# Plano do Editor de IA — {plan.project_name}",
        "",
        f"Duração: {_sec(plan.duration_sec)} · alvo: {_sec(plan.target_sec)} · {plan.fps} fps · "
        f"{plan.resolution[0]}x{plan.resolution[1]}",
        "",
    ]
    for block in plan.blocks:
        title = f" — {block.title}" if block.title else ""
        lines += [
            f"## {block.id}{title} ({clock(block.timeline_start)}–{clock(block.timeline_end)}, {_ORIGIN.get(block.origin, block.origin)})",
            "",
        ]
        for piece in block.pieces:
            ids = "–".join(piece.ids) if piece.kind == "frases" else ", ".join(piece.ids)
            lines.append(
                f"- {clock(piece.timeline_start)}–{clock(piece.timeline_end)} · {piece.kind} · {piece.clip_id} {ids} "
                f"(origem {piece.source_in:.2f}–{piece.source_out:.2f}s)"
            )
        for coverage in block.coverages:
            moments = ", ".join(coverage.moment_ids)
            lines.append(
                f"- cobertura {clock(coverage.timeline_start)}–{clock(coverage.timeline_end)} · "
                f"{coverage.from_sentence}–{coverage.to_sentence} ← {moments}"
                + (f" ({coverage.uncovered_sec:.1f}s sem cobertura no fim)" if coverage.uncovered_sec else "")
            )
        lines.append("")
    if plan.out:
        lines += ["## Fora", "", *(f"- {o.clipe}: {o.motivo or '—'}" for o in plan.out), ""]
    if plan.library_clip_ids:
        lines += ["## Acervo (cobertura e respiros)", "", ", ".join(plan.library_clip_ids), ""]
    if plan.warnings:
        lines += ["## Avisos da montagem", "", *(f"- {w}" for w in plan.warnings), ""]
    return "\n".join(lines)


def build_editor_report(
    project_name: str,
    run: EditorRun,
    plan: EditorPlan,
    material: EditorMaterial,
    recipe: str,
    output_path: Optional[Path],
    final_duration_sec: Optional[float],
    subtitles_path: Optional[Path] = None,
    subtitle_blocks: Optional[int] = None,
    encoder: Optional[str] = None,
    timings: Optional[RunTimings] = None,
    extra_warnings: Optional[list[str]] = None,
) -> str:
    structure = run.structure
    material_sec = sum(c.duration_sec for c in material.clips.values())
    model = run.effective_model or (run.model_chain[0] if run.model_chain else run.provider)
    lines = [
        f"# Editor de IA — {project_name}",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Receita | {recipe} |",
        f"| Material | {len(material.clips)} clipe(s), {clock(material_sec)} · {len(material.sentences)} frases · "
        f"{len(material.moments)} momentos |",
        f"| Pedido | {_cell(run.request) if run.request.strip() else 'sem pedido'} |",
        f"| Duração alvo | {_sec(structure.target_total_sec)} ({_TARGET_ORIGIN[structure.target_origin]}) |",
        f"| Duração do plano | {_sec(plan.duration_sec)} |",
        f"| IA | {_cell(' → '.join(run.model_chain) or run.provider)} (respondeu: {_cell(model)}) |",
        f"| Contexto | {run.budget.num_ctx} tokens ({run.budget.source}); estimativa caracteres/4 × {run.budget.token_factor:g} |",
    ]
    if run.model_switches:
        switches = "; ".join(f"{s.from_model} → {s.to_model} ({s.reason})" for s in run.model_switches)
        lines.append(f"| Trocas de modelo | {_cell(switches)} |")
    if output_path is not None and final_duration_sec is not None:
        lines.append(f"| Vídeo | `{output_path.name}` ({_sec(final_duration_sec)}, {encoder or 'cpu'}) |")
    else:
        lines.append("| Vídeo | sem render (`--so-plano`) |")
    if subtitles_path is not None:
        lines.append(f"| Legendas | `{subtitles_path.name}` ({subtitle_blocks} blocos) |")
    lines.append("")

    lines += ["## Estrutura (passada 1)", "", f"Origem: {_ORIGIN[structure.origin]}" + (
        f" · pedido {structure.prompt_mode}" if structure.prompt_mode != "completo" else ""
    )]
    if structure.notes:
        lines.append(f"Notas da IA: {_cell(structure.notes)}")
    lines += ["", "| Bloco | Título | Clipes | Alvo | No plano | Origem |", "| --- | --- | --- | --- | --- | --- |"]
    by_id = {b.id: b for b in plan.blocks}
    for block in structure.blocks:
        placed = by_id.get(block.id)
        final = f"{clock(placed.timeline_start)}–{clock(placed.timeline_end)}" if placed and placed.pieces else "—"
        lines.append(
            f"| {block.id} | {_cell(block.title or '—')} | {', '.join(block.clip_ids)} | {_sec(block.target_sec)} | "
            f"{final} | {_ORIGIN.get(placed.origin if placed else 'ia', '—')} |"
        )
    lines.append("")
    if structure.out:
        lines += ["Fora:", *(f"- {o.clipe}: {_cell(o.motivo or '—')}" for o in structure.out), ""]
    lines += [f"Acervo (cobertura e respiros): {', '.join(structure.library_clip_ids) or 'nenhum'}", ""]

    lines += ["## Blocos (passada 2)", ""]
    kept_ids = {sid for d in run.details for k in d.keep for sid in k.sentence_ids}
    for detail in run.details:
        lines += [f"### {detail.unit_id} — {', '.join(detail.clip_ids)} ({_ORIGIN[detail.origin]})", ""]
        if detail.notes:
            lines.append(f"Notas da IA: {_cell(detail.notes)}")
        lines.append(
            f"Pedido: ~{_tokens(detail.estimated_tokens)} tokens · acervo oferecido: {len(detail.library_offered)} momento(s)"
        )
        for keep in detail.keep:
            if keep.kind == "frases":
                lines.append(f"- ficou: {keep.from_sentence}–{keep.to_sentence} ({keep.clip_id}, {len(keep.sentence_ids)} frase(s))")
            else:
                extra = f", {keep.duration_sec:g}s" if keep.duration_sec else ""
                lines.append(f"- ficou: momento {keep.moment_id}{extra}")
        unit_sentences = [
            s for c in detail.clip_ids for s in material.clip_sentences.get(c, [])
            if detail.moment_ids is None or (s.moment_ids and s.moment_ids[0] in detail.moment_ids)
        ]
        cut = [s for s in unit_sentences if s.id not in kept_ids]
        for sentence in cut:
            lines.append(f"- cortou: {sentence.id} — {_cell(_short(sentence.text))}")
        for coverage in detail.coverages:
            image = "; ".join(f"{m}: {', '.join(w)}" for m, w in coverage.matched_image.items())
            lines.append(
                f"- cobertura {coverage.from_sentence}–{coverage.to_sentence} ← {', '.join(coverage.moment_ids)} · "
                f"fala: \"{_cell(coverage.speech_evidence)}\" (achou {', '.join(coverage.matched_speech)}) · "
                f"imagem: \"{_cell(coverage.image_evidence)}\" ({image})" + (f" · {_cell(coverage.reason)}" if coverage.reason else "")
            )
        for breath in detail.breaths:
            extra = f", {breath.duration_sec:g}s" if breath.duration_sec else ""
            lines.append(
                f"- respiro depois de {breath.after}: {breath.moment_id}{extra}" + (f" · {_cell(breath.reason)}" if breath.reason else "")
            )
        for attempt in detail.attempts:
            lines.append(f"- tentativa recusada: {_cell(_short(attempt, 300))}")
        lines.append("")

    lines += ["## Resumos por clipe (E9)", "", "| Clipe | Origem | Pedaços | Tom | Resumo |", "| --- | --- | --- | --- | --- |"]
    for summary in run.summaries:
        answer = summary.answer
        lines.append(
            f"| {summary.clip_id} | {_ORIGIN[summary.origin]} | {summary.parts} | {_cell(answer.tom if answer else '—')} | "
            f"{_cell(_short(answer.resumo, 160) if answer else summary.error or '—')} |"
        )
    lines.append("")

    warnings = [*run.warnings, *plan.warnings, *(extra_warnings or [])]
    lines += ["## Avisos (validação, fallbacks e montagem)", ""]
    lines += [f"- {_cell(w)}" for w in warnings] if warnings else ["Nenhum."]
    lines.append("")

    lines += [
        "## Chamadas e tokens",
        "",
        "| Passada | Pedidos | Do cache | Chamadas à API | Tokens de entrada | Tokens de saída | Tempo |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        _stats_row("Resumos (E9)", run.summary_stats),
        _stats_row("Estrutura (E10)", structure.stats),
        _stats_row("Detalhe (E11)", run.detail_stats),
        _stats_row("**Total**", run.total_stats),
        "",
    ]
    report = "\n".join(lines)
    if timings is not None:
        report += build_timings_section(timings)
    return report
