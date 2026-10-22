"""Relatório legível da análise visual (`visual/visual.md`)."""

from ai_video_editor.catalog.schema import ROLES
from ai_video_editor.visual.schema import ISSUE_LABELS, ClipVisual, ProjectVisual, Shot

# Problemas que, numa tomada do principal com fala, pedem cobertura obrigatória.
ATTENTION_ISSUES = {"obstruida", "borrao_movimento"}


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "—"


def has_issue(shot: Shot) -> bool:
    return shot.analysis is not None and any(i != "nenhuma" for i in shot.analysis.issues)


def needs_cover(clip: ClipVisual, shot: Shot) -> bool:
    """Tomada do principal com fala (ou fala desconhecida) e imagem ruim.

    Com análise: problema obstruída ou borrão de movimento. Sem análise: as medidas (tela
    uniforme ou borrão alto).
    """
    if clip.role != "principal" or shot.has_speech is False:
        return False
    if shot.analysis is not None:
        return bool(ATTENTION_ISSUES & set(shot.analysis.issues))
    return shot.measures is not None and (shot.measures.uniform or shot.measures.blurry)


def _reason(shot: Shot) -> str:
    if shot.analysis is not None:
        return ", ".join(ISSUE_LABELS[i] for i in shot.analysis.issues if i in ATTENTION_ISSUES)
    return "medidas: " + ", ".join(shot.measures.hints) if shot.measures else ""


def attention_ranges(visual: ProjectVisual) -> list[tuple[ClipVisual, list[Shot]]]:
    """Tomadas que pedem cobertura, juntando as consecutivas do mesmo clipe."""
    ranges: list[tuple[ClipVisual, list[Shot]]] = []
    for clip in visual.clips:
        current: list[Shot] = []
        for shot in clip.shots:
            if needs_cover(clip, shot):
                current.append(shot)
            elif current:
                ranges.append((clip, current))
                current = []
        if current:
            ranges.append((clip, current))
    return ranges


def _frame_links(shot: Shot) -> str:
    return " ".join(f"[F{f.position}]({f.file})" for f in shot.frames)


def _problems(shot: Shot) -> str:
    parts = []
    if shot.analysis is not None:
        parts += [ISSUE_LABELS[i] for i in shot.analysis.issues if i != "nenhuma"]
    hints = shot.measures.hints if shot.measures else []
    if hints:
        parts.append(f"(medidas: {', '.join(hints)})")
    return ", ".join(parts) or "—"


def _summary(visual: ProjectVisual) -> list[str]:
    lines = [
        "| Papel | Clipes | Tomadas | Analisadas | Utilizáveis | Com rosto | Com problema |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    totals = [0] * 6
    for role in ROLES:
        clips = [c for c in visual.clips if c.role == role]
        if not clips:
            continue
        shots = [s for c in clips for s in c.shots]
        analysed = [s for s in shots if s.analysis is not None]
        usable = sum(1 for s in analysed if s.analysis.usable != "nao")
        faces = sum(1 for s in analysed if s.analysis.face_to_camera)
        issues = sum(1 for s in analysed if has_issue(s))
        row = [len(clips), len(shots), len(analysed), usable, faces, issues]
        totals = [a + b for a, b in zip(totals, row)]
        lines.append(
            f"| {role} | {row[0]} | {row[1]} | {row[2]} | {_pct(usable, len(analysed))} | {faces} | {issues} |"
        )
    lines.append(
        f"| **total** | {totals[0]} | {totals[1]} | {totals[2]} | {_pct(totals[3], totals[2])} | "
        f"{totals[4]} | {totals[5]} |"
    )
    return lines


def build_visual_markdown(visual: ProjectVisual) -> str:
    shots = visual.all_shots()
    lines = [
        "# Análise visual",
        "",
        f"- Projeto: `{visual.project_dir}`",
        f"- Clipes: {len(visual.clips)} · tomadas: {len(shots)} · papéis: {', '.join(visual.roles)}",
    ]
    ai = visual.ai
    if ai is None:
        lines.append("- IA: não chamada (só tomadas e medidas)")
    else:
        model = ai.effective_model or (ai.model_chain[0] if ai.model_chain else ai.provider)
        tokens = (
            f"{ai.input_tokens or 0:,} de entrada / {ai.output_tokens or 0:,} de saída".replace(",", ".")
            if ai.input_tokens is not None or ai.output_tokens is not None else "não informados"
        )
        lines.append(
            f"- IA: `{model}` · raciocínio {ai.thinking or '—'} · detalhe da imagem {ai.image_detail} · "
            f"{ai.batches} lote(s), {ai.api_calls} chamada(s), {ai.images_sent} imagens · tokens: {tokens}"
        )
        if ai.cached_shots:
            lines.append(f"- Análises reaproveitadas do cache: {ai.cached_shots}")
        for switch in ai.model_switches:
            lines.append(f"- Troca de modelo: {switch.from_model} → {switch.to_model} ({switch.reason})")
    missing = sum(1 for s in shots if s.status == "sem_analise")
    if missing:
        lines.append(f"- **{missing} tomada(s) sem análise** (serão pedidas de novo na próxima execução)")
    lines += ["", "## Resumo", "", *_summary(visual), ""]

    lines += ["## Atenção: fala com imagem ruim", ""]
    ranges = attention_ranges(visual)
    if not ranges:
        lines.append("Nenhum trecho de fala do principal com imagem obstruída ou borrão de movimento.")
    else:
        lines.append(
            "Trechos do principal com fala e imagem obstruída ou com borrão de movimento: candidatos a "
            "cobertura obrigatória (B-roll, drone, POV)."
        )
        lines.append("")
        for clip, group in ranges:
            first, last = group[0], group[-1]
            ids = first.id if first is last else f"{first.id}–{last.id.rsplit('.', 1)[-1]}"
            reasons = sorted({r for s in group for r in _reason(s).split(", ") if r})
            speech = "" if all(s.has_speech for s in group) else " (fala não confirmada)"
            lines.append(
                f"- **{clip.clip_id}** {_clock(first.start_sec)}–{_clock(last.end_sec)} ({ids}): "
                f"{', '.join(reasons)}{speech}"
            )
    lines.append("")

    lines += ["## Por clipe", ""]
    for clip in visual.clips:
        speech = {"analisada": "fala analisada", "desconhecido": "fala desconhecida", "sem_audio": "sem áudio"}.get(
            clip.speech_status or "", "fala desconhecida"
        )
        lines += [f"### {clip.clip_id} — {clip.relative_path}", ""]
        lines.append(f"{clip.role} · {_clock(clip.duration_sec)} · {len(clip.shots)} tomada(s) · {speech}")
        lines.append("")
        if clip.error:
            lines += [f"Erro: {clip.error}", ""]
            continue
        lines += [
            "| Tomada | Tempo | Miniaturas | Descrição | Tags | Problemas | Utilizável | Beleza |",
            "|---|---|---|---|---|---|---|---:|",
        ]
        for shot in clip.shots:
            time_range = f"{_clock(shot.start_sec)}–{_clock(shot.end_sec)}"
            if shot.analysis is not None:
                a = shot.analysis
                face = " 🗣" if a.face_to_camera else ""
                best = f" (melhor: {', '.join(f.rsplit('.', 1)[-1] for f in a.best_frames)})" if a.best_frames else ""
                lines.append(
                    f"| {shot.id}{face} | {time_range} | {_frame_links(shot)} | {_cell(a.description)} | "
                    f"{_cell(', '.join(a.tags))} | {_cell(_problems(shot))} | {a.usable}{best} | {a.beauty} |"
                )
            else:
                status = "sem análise" if shot.status == "sem_analise" else "não pedida"
                lines.append(
                    f"| {shot.id} | {time_range} | {_frame_links(shot)} | _{status}_ | — | "
                    f"{_cell(_problems(shot))} | — | — |"
                )
        notes = [(s.id, s.analysis.summary_for_editor) for s in clip.shots if s.analysis and s.analysis.summary_for_editor]
        if notes:
            lines.append("")
            lines += [f"- {sid}: {_cell(note)}" for sid, note in notes]
        failed = [s for s in clip.shots if s.status == "sem_analise" and s.note]
        if failed:
            lines.append("")
            lines += [f"- {s.id} sem análise: {_cell(s.note)}" for s in failed]
        lines.append("")

    if visual.warnings:
        lines += ["## Avisos", ""] + [f"- {w}" for w in visual.warnings] + [""]
    return "\n".join(lines)
