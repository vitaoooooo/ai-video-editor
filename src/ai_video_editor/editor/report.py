"""Relatórios do Editor de IA: `editor_relatorio.md` (o que a IA decidiu e o que o código fez) e
`editor_plano.md` (o plano em IDs, já convertido em tempos)."""

from pathlib import Path
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.schema import CallStats, CutAttempt, DetailRun, EditorPlan, EditorRun, StructureRun
from ai_video_editor.pipeline.schema import RunTimings
from ai_video_editor.pipeline.timing import build_timings_section, format_duration
from ai_video_editor.roteiro.report import clock

_ORIGIN = {"ia": "IA", "cache": "IA (cache)", "fallback": "**fallback neutro**", "falha": "**falhou**"}
_TARGET_ORIGIN = {"pedido": "do pedido", "texto": "lida do texto do pedido", "ia": "escolhida pela IA", "nenhuma": "livre"}
_PASSES = ("resumo", "estrutura", "cortes", "insercoes", "coberturas")


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _sec(seconds: Optional[float]) -> str:
    return "—" if seconds is None else f"{clock(seconds)} ({seconds:.0f}s)"


def _tokens(value: Optional[int]) -> str:
    return "—" if value is None else f"{value:,}".replace(",", ".")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _stats_row(name: str, stats: CallStats, normalized: int) -> str:
    return (
        f"| {name} | {stats.calls} | {stats.cached} | {stats.api_calls} | {_tokens(stats.input_tokens)} | "
        f"{_tokens(stats.output_tokens)} | {format_duration(stats.seconds)} | {normalized} |"
    )


def _normalized(ids: list[str], limit: int = 4) -> str:
    more = f" (+{len(ids) - limit})" if len(ids) > limit else ""
    return f"IDs normalizados: {len(ids)} ({'; '.join(ids[:limit])}{more})"


def _detail_lines(detail: DetailRun, material: EditorMaterial) -> list[str]:
    """Um bloco no relatório: 2a por clipe, 2b (letras → IDs), 2c (trecho → letra → momento + palavra)."""
    lines = [f"### {detail.unit_id} — {', '.join(detail.clip_ids) or '(nenhum clipe)'} ({_ORIGIN[detail.origin]})", ""]
    lines.append(
        f"Duração: fala livre {_sec(detail.available_sec)} → fala que ficou {_sec(detail.speech_sec)} → com as "
        f"inclusões {_sec(detail.result_sec)} → alvo {_sec(detail.target_sec)}"
    )
    lines += ["", "**2a cortes** (meta do clipe = alvo do bloco repartido pela fala de cada clipe, até a fala dele)", ""]
    if not detail.cut_runs:
        lines.append("- (nenhum clipe com fala)")
    for cut in detail.cut_runs:
        if cut.no_question:
            lines.append(
                f"- {cut.unit_id}: sem pergunta de cortes ({cut.no_question}): entra inteiro, {_sec(cut.result_sec)} "
                f"· meta {_sec(cut.target_sec)}"
            )
            continue
        off = " · **fora da meta**" if cut.duration_off else ""
        lines.append(
            f"- {cut.unit_id} ({_ORIGIN[cut.origin]}, ~{_tokens(cut.estimated_tokens)} tokens): {_sec(cut.available_sec)} "
            f"→ meta {_sec(cut.target_sec)} → ficou {_sec(cut.result_sec)}{off} · {len(cut.cuts)} corte(s)"
        )
        lines += [
            f"  - cortou {c.sentence_id} — {_cell(_short(material.sentences[c.sentence_id].text, 70))} · "
            f"motivo: {_cell(c.reason or '—')}"
            for c in cut.cuts if c.sentence_id in material.sentences
        ]
        if cut.normalized_ids:
            lines.append(f"  - {_cell(_normalized(cut.normalized_ids))}")
        lines += [f"  - tentativa recusada: {_cell(_short(a, 300))}" for a in cut.attempts]
        if len(cut.tries) > 1 or cut.chosen_attempt == 0:
            used = "usado o clipe inteiro (sem cortes)" if cut.chosen_attempt == 0 else f"usada a {cut.chosen_attempt}"
            lines.append("  - tentativas: " + " · ".join(_try_text(t) for t in cut.tries) + f" → {used}")
        if cut.target_limit:
            lines.append(f"  - IA: a meta não dá sem prejudicar a fala — {_cell(cut.target_limit)}")
        if cut.kept_whole:
            lines.append("  - **IA cortou tudo sem dizer fora; mantido inteiro**")
        lines += [f"  - {_cell(j)}" for j in cut.joined]
        if cut.removed:
            lines.append(f"  - **IA tirou o clipe {cut.clip_id}**" + (f" (parte {cut.unit_id})" if cut.moment_ids else ""))
    if detail.removed_clip_ids:
        lines += ["", f"IA tirou do bloco: {', '.join(detail.removed_clip_ids)}"]
    if detail.trechos:
        lines += ["", "Trechos de fala: " + "; ".join(
            f"{t.number}: {t.sentence_ids[0]}–{t.sentence_ids[-1]} ({t.clip_id}, {t.seconds:.0f}s)" for t in detail.trechos
        )]
    sub = detail.inserts
    lines += ["", f"**2b inclusões e respiros** ({_ORIGIN[sub.origin] if sub and sub.requests else 'sem pergunta'})", ""]
    if sub is not None:
        spare = "sem alvo do bloco" if sub.spare_sec is None else f"{sub.spare_sec:.1f}s".replace(".", ",")
        lines.append(f"- sobra (alvo do bloco − fala que ficou): {spare}")
        if sub.no_question:
            lines.append(f"- sem pergunta: {sub.no_question}")
        if sub.requests or sub.breath_positions or sub.skipped:
            offered = ", ".join(str(n) for n in sub.breath_positions) or "nenhuma"
            lines.append(f"- posições de respiro oferecidas: {offered}")
        if sub.skipped:
            lines.append("- posições de respiro recusadas: " + "; ".join(f"{k.number} ({k.reason})" for k in sub.skipped))
        if sub.requests:
            lines.append(f"- {len(sub.candidates)} candidato(s) com letra · ~{_tokens(sub.estimated_tokens)} tokens")
        for choice in sub.inserts:
            what = "respiro (acervo)" if choice.breath else "momento do bloco"
            extra = f", {choice.duration_sec:g}s" if choice.duration_sec else ""
            lines.append(f"- {choice.letter} → {choice.moment_id} ({what}{extra}) depois do trecho {choice.after_trecho}")
        if sub.requests and not sub.inserts:
            lines.append("- nenhuma imagem escolhida")
        lines += [f"- tentativa recusada: {_cell(_short(a, 300))}" for a in sub.attempts]
    sub = detail.coverage
    lines += ["", f"**2c coberturas** ({_ORIGIN[sub.origin] if sub and sub.requests else 'sem pergunta'})", ""]
    if sub is not None:
        if sub.skipped:
            lines.append("- trechos sem pergunta: " + "; ".join(f"{k.number} ({k.reason})" for k in sub.skipped))
        if not sub.requests:
            lines.append(f"- sem pergunta: {sub.no_question or 'nenhum candidato medido'}")
        if sub.requests:
            lines.append(
                f"- {len(sub.candidates)} candidato(s) medido(s), {sub.requests} pedido(s) · "
                f"~{_tokens(sub.estimated_tokens)} tokens"
            )
        for coverage in sub.coverages:
            evidence = "correspondência explicada, fala" if coverage.semantic else "palavra"
            line = (
                f"- trecho {coverage.trecho} → {coverage.letter} → {', '.join(coverage.moment_ids)} · {coverage.kind} · "
                f"{evidence} \"{_cell(coverage.word)}\" · cobre {coverage.from_sentence}–{coverage.to_sentence}"
            )
            if coverage.image_evidence and coverage.image_evidence != coverage.word:
                line += f" · mostra: {_cell(_short(coverage.image_evidence, 80))}"
            if coverage.reason:
                line += f" · porque: {_cell(_short(coverage.reason, 80))}"
            lines.append(line)
        if sub.nones:
            lines.append(f"- \"nenhuma\" nos trechos {', '.join(str(n) for n in sub.nones)}")
        lines += [f"- tentativa recusada: {_cell(_short(a, 300))}" for a in sub.attempts]
    if detail.warnings:
        lines += ["", "Avisos:", *(f"- {_cell(_short(w, 300))}" for w in detail.warnings)]
    lines.append("")
    return lines


def _try_text(attempt: CutAttempt) -> str:
    what = " (fora: true)" if attempt.out else " (cortou tudo)" if attempt.all_cut else ""
    return f"{attempt.attempt}: {attempt.result_sec:.1f}s{what}".replace(".", ",", 1)


def _target_cell(structure: StructureRun) -> str:
    text = f"{_sec(structure.target_total_sec)} ({_TARGET_ORIGIN[structure.target_origin]}"
    if structure.target_note:
        text += f": {structure.target_note}"
    return _cell(text + ")")


def _share(speech: float, target: Optional[float]) -> str:
    return f"{_sec(speech)}" + (f" ({100 * speech / target:.0f}% do alvo)" if target else "")


def _structure_speech_lines(structure: StructureRun) -> list[str]:
    """Fala dos clipes escolhidos × alvo (antes e depois da pergunta de acréscimo por falta de duração)."""
    if structure.speech_sec is None:
        return []
    target = structure.target_total_sec
    if structure.speech_sec_first is None:
        return [f"Fala dos clipes escolhidos: {_share(structure.speech_sec, target)}"]
    added = ", ".join(
        f"{c} → {structure.added_blocks.get(c, '?')}" for c in structure.added_clip_ids
    ) or "nenhum (a IA não acrescentou clipes)"
    return [
        f"Fala dos clipes escolhidos: {_share(structure.speech_sec_first, target)} → acréscimo "
        f"({_cell(structure.shortfall)}) → {_share(structure.speech_sec, target)}",
        f"Clipes acrescentados (→ bloco em que entraram): {added}",
    ]


def breath_total_line(plan: EditorPlan) -> str:
    """Total de respiro no plano, em segundos e % da duração."""
    total = sum(p.timeline_end - p.timeline_start for b in plan.blocks for p in b.pieces if p.kind == "respiro")
    share = 100 * total / plan.duration_sec if plan.duration_sec > 0 else 0.0
    return f"{total:.1f}s ({share:.0f}% do vídeo)".replace(".", ",", 1)


def _ratio(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _calibration_lines(run: EditorRun) -> list[str]:
    """Tokens estimados × reais por pedido (os do cache não têm real) e a razão por passada."""
    measured = [r for r in run.requests if not r.cached and r.input_tokens and r.estimated_tokens]
    lines = [
        "## Tokens estimados × reais por pedido",
        "",
        f"Estimativa: caracteres/4 × {run.budget.token_factor:g} (`editor/budget.py`). Razão real/estimado acima de 1 "
        "= a estimativa subestima (o pedido pode estourar o contexto).",
        "",
    ]
    if not measured:
        return lines + ["Nenhum pedido com tokens reais nesta execução (tudo do cache ou provedor sem contagem).", ""]
    lines += ["| Passada | Pedidos medidos | Real/estimado (mín–máx) |", "| --- | --- | --- |"]
    for kind in _PASSES:
        ratios = [r.input_tokens / r.estimated_tokens for r in measured if r.kind == kind]
        if ratios:
            lines.append(f"| {kind} | {len(ratios)} | {_ratio(min(ratios))}–{_ratio(max(ratios))} |")
    lines += ["", "| Passada | Unidade | Tentativa | Estimados | Reais | Saída |", "| --- | --- | --- | --- | --- | --- |"]
    for r in measured:
        lines.append(
            f"| {r.kind} | {r.unit} | {r.attempt} | {_tokens(r.estimated_tokens)} | {_tokens(r.input_tokens)} | "
            f"{_tokens(r.output_tokens)} |"
        )
    lines.append("")
    return lines


def build_plan_markdown(plan: EditorPlan, material: EditorMaterial) -> str:
    lines = [
        f"# Plano do Editor de IA — {plan.project_name}",
        "",
        f"Duração: {_sec(plan.duration_sec)} · alvo: {_sec(plan.target_sec)} · {plan.fps} fps · "
        f"{plan.resolution[0]}x{plan.resolution[1]} · respiros: {breath_total_line(plan)}",
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
        f"| Duração alvo | {_target_cell(structure)} |",
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
    lines.append(f"| Respiros | {breath_total_line(plan)} |")
    lines.append("")

    lines += [
        "## Estrutura (passada 1)", "",
        f"Origem: {_ORIGIN[structure.origin]} · pedido ~{_tokens(structure.estimated_tokens)} tokens (uma linha por clipe)",
    ]
    if structure.notes:
        lines.append(f"Notas da IA: {_cell(structure.notes)}")
    if structure.normalized_ids:
        lines.append(_cell(_normalized(structure.normalized_ids)))
    lines += _structure_speech_lines(structure)
    if structure.order:
        lines.append(f"Ordem dos blocos: {structure.order}")
    lines += ["", "| Bloco | Título | Clipes | Alvo | No plano | Origem |", "| --- | --- | --- | --- | --- | --- |"]
    by_id = {b.id: b for b in plan.blocks}
    for block in structure.blocks:
        placed = by_id.get(block.id)
        final = f"{clock(placed.timeline_start)}–{clock(placed.timeline_end)}" if placed and placed.pieces else "—"
        target = _sec(block.target_sec)
        if block.ai_target_sec is not None and block.ai_target_sec != block.target_sec:
            target += f" (IA: {block.ai_target_sec:.0f}s)"
        lines.append(
            f"| {block.id} | {_cell(block.title or '—')} | {', '.join(block.clip_ids)} | {target} | "
            f"{final} | {_ORIGIN.get(placed.origin if placed else 'ia', '—')} |"
        )
    lines.append("")
    if structure.out:
        lines += ["Fora:", *(f"- {o.clipe}: {_cell(o.motivo or '—')}" for o in structure.out), ""]
    lines += [f"Acervo (cobertura e respiros): {', '.join(structure.library_clip_ids) or 'nenhum'}", ""]
    lines += [f"- tentativa recusada: {_cell(_short(a, 300))}" for a in structure.attempts]
    if structure.attempts:
        lines.append("")

    lines += ["## Blocos (passada 2: 2a cortes, 2b inclusões e respiros, 2c coberturas)", ""]
    for detail in run.details:
        lines += _detail_lines(detail, material)

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

    normalized = {
        "summary": sum(len(s.normalized_ids) for s in run.summaries),
        "structure": len(structure.normalized_ids),
        "cuts": sum(len(c.normalized_ids) for d in run.details for c in d.cut_runs),
    }
    lines += [
        "## Chamadas e tokens",
        "",
        "IDs normalizados: escritos pela IA fora do formato do roteiro (ex.: frase com o clipe na frente, `###` em "
        "volta) e trocados pelo código antes da validação. Muitos = o pedido ainda induz o formato errado.",
        "",
        "| Passada | Pedidos | Do cache | Chamadas à API | Tokens de entrada | Tokens de saída | Tempo | IDs normalizados |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        _stats_row("Resumos (E9)", run.summary_stats, normalized["summary"]),
        _stats_row("Estrutura (E10)", structure.stats, normalized["structure"]),
        _stats_row("2a Cortes (E11)", run.cuts_stats, normalized["cuts"]),
        _stats_row("2b Inclusões (E11)", run.inserts_stats, 0),
        _stats_row("2c Coberturas (E11)", run.coverage_stats, 0),
        _stats_row("**Total**", run.total_stats, sum(normalized.values())),
        "",
    ]
    lines += _calibration_lines(run)
    report = "\n".join(lines)
    if timings is not None:
        report += build_timings_section(timings)
    return report
