"""Relatórios da espinha falada (`edit <pasta>`): transcrição comparativa por clipe (`edit_transcript.md` +
`edit_diff.json`), linha do tempo colorida por clipe (`edit_timeline.svg`) e `edit_report.md`.

Cada clipe usa a mesma regra do caminho de um vídeo (`build_edit_diff`: status de aplicação de cada
remoção pela interseção com o que saiu); os tempos "editados" são deslocados para onde o clipe
começa no vídeo final, então batem com o .srt e o player.
"""

import math
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import escape

from ai_video_editor.core.timeline import Timeline
from ai_video_editor.coverage.schema import BREATH_SIZE_LABELS, CoverageRun
from ai_video_editor.director import DirectorRun, build_clip_sentences
from ai_video_editor.pipeline.edit_diff import _entry_line, build_edit_diff
from ai_video_editor.pipeline.project import (
    COVERAGE_DECISION_FILE,
    COVERAGE_TIMELINE_FILE,
    EDIT_DIFF_FILE,
    EDIT_TIMELINE_FILE,
    EDIT_TRANSCRIPT_FILE,
    TIMELINE_FILE,
)
from ai_video_editor.pipeline.report import (
    _cell,
    _loudness_section,
    application_cell,
    format_timestamp,
    model_rows,
)
from ai_video_editor.pipeline.schema import DiffEntry, EditPipelineParams, RunTimings, _plural, format_seconds
from ai_video_editor.pipeline.spine_schema import ClipDiff, SpineDiff, SpinePlan, SpineSelection, UnusedClip
from ai_video_editor.pipeline.timing import build_timings_section
from ai_video_editor.planning import removed_ranges
from ai_video_editor.planning.timeline_plan import clip_placements, clip_spans, concat_to_final, is_breath_item
from ai_video_editor.renderer.loudness import LoudnessResult
from ai_video_editor.transcription.schema import Transcript

# --- transcrição comparativa por clipe ---


def _shift(entry: DiffEntry, offset: float, to_final: Optional[Callable[[float, bool], float]] = None) -> DiffEntry:
    """Leva os tempos "editados" do clipe para o vídeo final (deslocamento fixo, ou `to_final` com respiros)."""
    move = to_final or (lambda t, _end: t + offset)
    update = {}
    if entry.edited_start_sec is not None:
        update["edited_start_sec"] = round(move(entry.edited_start_sec, False), 3)
    if entry.edited_end_sec is not None:
        update["edited_end_sec"] = round(move(entry.edited_end_sec, True), 3)
    return entry.model_copy(update=update) if update else entry


def build_spine_diff(
    project_name: str,
    selection: SpineSelection,
    plan: SpinePlan,
    timeline: Timeline,
    director: Optional[DirectorRun],
    transcripts: Optional[dict[str, Transcript]],
    params: EditPipelineParams,
    final: Optional[Timeline] = None,
) -> SpineDiff:
    """Diff de cada clipe da espinha (inclusive os que ficaram fora) contra os trechos da Timeline.

    Com `final` (a Timeline com cobertura), os tempos editados são os do vídeo final, com os respiros
    deslocando o que vem depois deles.
    """
    fps = timeline.fps
    spans = clip_spans(timeline)
    placements = clip_placements(final) if final is not None else {}
    if director is not None:
        sentences = director.sentences
    elif transcripts is not None:
        sentences = build_clip_sentences([(c.id, transcripts[c.id]) for c in selection.clips], params.sentences)
    else:
        sentences = []
    removals = {}
    if director is not None and director.decision is not None:
        removals = {r.sentence_id: r for r in director.decision.removals}

    clips: list[ClipDiff] = []
    position = 0
    for clip in selection.clips:
        clip_plan = plan.plan_for(clip.id)
        start, end = spans.get(clip.id, (position, position))
        position = end
        clip_sentences = [x for x in sentences if x.clip_id == clip.id]
        diff = build_edit_diff(
            clip.name,
            clip_sentences,
            {x.id: removals[x.id] for x in clip_sentences if x.id in removals},
            clip_plan.keep,
            clip_plan.cut_results,
            fps,
            clip.duration_sec,
            (end - start) / fps,
            min_pause_sec=params.transcript_min_pause_sec,
        )
        offset = start / fps
        to_final = None
        if clip.id in placements:
            clip_placed = placements[clip.id]
            offset = clip_placed[0].start_frame / fps
            to_final = lambda t, end, placed=clip_placed: concat_to_final(placed, t, fps, end)  # noqa: E731
        diff = diff.model_copy(update={"entries": [_shift(e, offset, to_final) for e in diff.entries]})
        clips.append(
            ClipDiff(clip_id=clip.id, file=clip.relative_path, offset_sec=round(offset, 3), dropped=clip_plan.dropped, diff=diff)
        )
    return SpineDiff(
        project_name=project_name,
        original_duration_sec=round(sum(c.duration_sec for c in selection.clips), 3),
        final_duration_sec=round((final or timeline).duration_sec, 3),
        clips=clips,
        unused=selection.unused,
    )


def _clock(seconds: float) -> str:
    """`75.9` -> `1:15`; `3725` -> `1:02:05`."""
    total = int(seconds + 1e-6)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _clip_title(clip: ClipDiff) -> str:
    """`Clipe P006 — principal/P006.mp4 (0:34 → 0:21)`."""
    durations = f"{_clock(clip.diff.original_duration_sec)} → {_clock(clip.diff.final_duration_sec)}"
    status = f" — fora da espinha: {clip.dropped}" if clip.dropped else ""
    return f"Clipe {clip.clip_id} — {clip.file} ({durations}){status}"


def _unused_lines(unused: list[UnusedClip]) -> list[str]:
    if not unused:
        return ["Todos os clipes do catálogo entraram na espinha."]
    lines = ["| Clipe | Arquivo | Papel | Duração | Motivo |", "| --- | --- | --- | --- | --- |"]
    for clip in unused:
        duration = _clock(clip.duration_sec) if clip.duration_sec is not None else "—"
        reason = clip.reason + (" — disponível para cobertura (rodada 3)" if clip.coverage else "")
        lines.append(f"| {clip.id or '—'} | `{clip.relative_path}` | {clip.role} | {duration} | {_cell(reason)} |")
    return lines


def render_spine_transcript(diff: SpineDiff) -> str:
    """Markdown da transcrição comparativa, com uma seção por clipe, na ordem do vídeo."""
    merged = diff.merged()
    kept = merged.count("kept")
    removed = len(merged.sentences) - kept
    removed_sec = max(0.0, diff.original_duration_sec - diff.final_duration_sec)
    lines = [
        f"# Transcrição comparativa — {diff.project_name}",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Clipes na espinha | {sum(1 for c in diff.clips if not c.dropped)} de {len(diff.clips)} |",
        f"| Duração original (clipes da espinha) | {format_timestamp(diff.original_duration_sec)} ({diff.original_duration_sec:.2f}s) |",
        f"| Duração final | {format_timestamp(diff.final_duration_sec)} ({diff.final_duration_sec:.2f}s) |",
        f"| Tempo removido | {format_seconds(removed_sec)} |",
        f"| Frases mantidas | {kept} |",
        f"| Frases marcadas para remoção | {removed} ({merged.application_counts()}) |",
        f"| Pausas removidas | {format_seconds(merged.removed_pause_sec, 1)} "
        f"(no texto, só as de {format_seconds(merged.min_pause_marker_sec, 1)} ou mais) |",
        "",
        "Legenda: `[original no clipe → vídeo final]` · ~~riscada~~ = removida · ◐ remoção parcial · "
        "⚠️ remoção não aplicada (continua no vídeo) · ⏸ pausa removida",
        "",
    ]
    for clip in diff.clips:
        lines += [f"## {_clip_title(clip)}", ""]
        if not clip.diff.entries:
            lines += ["*(sem frases transcritas)*", ""]
        for entry in clip.diff.entries:
            lines += [_entry_line(entry, None), ""]
    lines += ["## Clipes não usados", "", *_unused_lines(diff.unused), ""]
    return "\n".join(lines)


# --- linha do tempo colorida por clipe ---

WIDTH = 1000
_MARGIN = 24
_BAR_WIDTH = WIDTH - 2 * _MARGIN
_BAR_HEIGHT = 34
_ROW_GAP = 30
_HEADER_HEIGHT = 74
_LEGEND_ROW = 20
_ROW_SEC = 600.0
_TICK_STEPS = (5, 10, 15, 30, 60, 120, 300, 600)
_BACKGROUND = "#ffffff"
_BORDER = "#d0d7de"
_TEXT = "#1f2328"
_MUTED = "#57606a"
# Paleta categórica (alterna tons para clipes vizinhos ficarem distinguíveis).
CLIP_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#9d755d", "#eeca3b")
# Com cobertura: faixa de overlay acima da trilha da fala e cor dos respiros.
_OVERLAY_LANE = 18
_OVERLAY_HEIGHT = 12
OVERLAY_COLOR = "#2f3b4c"
BREATH_COLOR = "#a0a7b1"


def render_spine_timeline(
    diff: SpineDiff, timeline: Timeline, final: Optional[Timeline] = None, coverage: Optional[CoverageRun] = None
) -> str:
    """SVG do vídeo final: uma trilha com os trechos de cada clipe na cor do clipe (corte = fresta branca).

    Com `final` (Timeline com cobertura), a trilha é a do vídeo final, com os respiros em cinza, e uma
    faixa acima mostra as coberturas (overlay).
    """
    shown = final or timeline
    fps = shown.fps
    duration = max(shown.duration_sec, 0.001)
    rows = math.ceil(duration / _ROW_SEC) if duration > _ROW_SEC else 1
    row_sec = _ROW_SEC if rows > 1 else duration
    scale = _BAR_WIDTH / row_sec
    lane = _OVERLAY_LANE if final is not None else 0
    row_pitch = lane + _BAR_HEIGHT + _ROW_GAP
    used = [c for c in diff.clips if not c.dropped]
    legend_items = len(used) + (2 if final is not None else 0)
    legend_rows = math.ceil(legend_items / 5) or 1
    height = _HEADER_HEIGHT + rows * row_pitch + legend_rows * _LEGEND_ROW + 16
    color = {c.clip_id: CLIP_COLORS[i % len(CLIP_COLORS)] for i, c in enumerate(used)}

    pct = max(0.0, diff.original_duration_sec - diff.final_duration_sec) / max(diff.original_duration_sec, 0.001) * 100
    title = (
        f"{diff.project_name}: {len(used)} clipes, {_clock(diff.original_duration_sec)} → "
        f"{_clock(diff.final_duration_sec)} ({f'{pct:.1f}'.replace('.', ',')}% removido)"
    )
    subtitle = (
        f"{_plural(len(timeline.video_main), 'trecho', 'trechos')} no vídeo final · "
        f"{_plural(len(diff.unused), 'clipe não usado', 'clipes não usados')}"
    )
    if final is not None:
        breaths = sum(1 for item in final.video_main if is_breath_item(item))
        covers = len(coverage.placed_coverages) if coverage is not None else len(final.video_overlay)
        subtitle += (
            f" · {_plural(covers, 'cobertura', 'coberturas')} · {_plural(breaths, 'respiro', 'respiros')}"
            f" (espinha {_clock(timeline.duration_sec)} → final {_clock(final.duration_sec)})"
        )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        f'role="img" aria-label="{escape(title, {chr(34): "&quot;"})}">',
        f"<title>{escape(title)}</title>",
        f'<rect class="background" x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="10" '
        f'fill="{_BACKGROUND}" stroke="{_BORDER}"/>',
        f'<text x="{_MARGIN}" y="32" font-size="18" font-weight="600" fill="{_TEXT}">{escape(title)}</text>',
        f'<text x="{_MARGIN}" y="54" font-size="12.5" fill="{_MUTED}">{escape(subtitle)}</text>',
    ]

    def row_y(row: int) -> float:
        """Topo da trilha da fala na linha `row` (a faixa de overlay fica acima)."""
        return _HEADER_HEIGHT + row * row_pitch + lane

    step = next((s for s in _TICK_STEPS if row_sec / s <= 12), _TICK_STEPS[-1])
    for row in range(rows):
        start = row * row_sec
        length = min(row_sec, duration - start)
        tick = 0
        while tick <= length + 0.001:
            x = _MARGIN + tick * scale
            y = row_y(row) + _BAR_HEIGHT
            anchor = "start" if tick == 0 else ("end" if x > WIDTH - _MARGIN - 20 else "middle")
            out.append(f'<line x1="{x:.2f}" y1="{y}" x2="{x:.2f}" y2="{y + 5}" stroke="{_MUTED}" stroke-width="1"/>')
            out.append(
                f'<text x="{x:.2f}" y="{y + 17}" font-size="11" fill="{_MUTED}" text-anchor="{anchor}">'
                f"{_clock(start + tick)}</text>"
            )
            tick += step

    def bars(a: float, b: float, y_of: Callable[[int], float], height: float, fill: str, gap: float = 0.6) -> list[str]:
        rects = []
        for row in range(min(rows - 1, int(a // row_sec)), min(rows - 1, int(max(a, b - 0.001) // row_sec)) + 1):
            x0 = max(a, row * row_sec) - row * row_sec
            x1 = min(b, (row + 1) * row_sec) - row * row_sec
            width = max(0.6, (x1 - x0) * scale - gap)  # fresta de 0,6 px marca o corte
            rects.append(
                f'<rect x="{_MARGIN + x0 * scale:.2f}" y="{y_of(row)}" width="{width:.2f}" '
                f'height="{height}" fill="{fill}"/>'
            )
        return rects

    position = 0
    for item in shown.video_main:
        frames = shown.item_frames(item)
        a, b = position / fps, (position + frames) / fps
        position += frames
        breath = final is not None and is_breath_item(item)
        fill = BREATH_COLOR if breath else color.get(item.clip_id or "", CLIP_COLORS[0])
        rects = bars(a, b, row_y, _BAR_HEIGHT, fill)
        tip = (
            f"{item.label or item.clip_id} · final {_clock(a)}–{_clock(b)} · "
            f"no clipe {format_timestamp(item.source_in)}–{format_timestamp(item.source_out)}"
        )
        css = "item breath" if breath else "item"
        out.append(
            f'<g class="{css}" data-clip-id="{escape(item.clip_id or "")}"><title>{escape(tip)}</title>{"".join(rects)}</g>'
        )

    if final is not None:
        for item in final.video_overlay:
            a = final.to_frame(item.timeline_start) / fps
            b = a + final.item_frames(item) / fps
            rects = bars(a, b, lambda row: row_y(row) - lane + (lane - _OVERLAY_HEIGHT) / 2, _OVERLAY_HEIGHT, OVERLAY_COLOR, 1.0)
            tip = (
                f"{item.label or item.clip_id} · final {_clock(a)}–{_clock(b)} · "
                f"na tomada {format_timestamp(item.source_in)}–{format_timestamp(item.source_out)}"
            )
            out.append(
                f'<g class="overlay" data-clip-id="{escape(item.clip_id or "")}"><title>{escape(tip)}</title>{"".join(rects)}</g>'
            )

    legend_y = row_y(rows) - lane + 4
    out.append('<g class="legend">')
    entries = [
        (color[c.clip_id], f"{c.clip_id} {_clock(c.diff.original_duration_sec)} → {_clock(c.diff.final_duration_sec)}")
        for c in used
    ]
    if final is not None:
        entries += [(OVERLAY_COLOR, "cobertura (por cima da fala)"), (BREATH_COLOR, "respiro (sem fala)")]
    for i, (fill, label) in enumerate(entries):
        x = _MARGIN + (i % 5) * (_BAR_WIDTH / 5)
        y = legend_y + (i // 5) * _LEGEND_ROW
        out.append(f'<rect x="{x:.0f}" y="{y}" width="14" height="14" rx="2" fill="{fill}"/>')
        out.append(f'<text x="{x + 20:.0f}" y="{y + 11.5}" font-size="12.5" fill="{_TEXT}">{escape(label)}</text>')
    out += ["</g>", "</svg>"]
    return "\n".join(out) + "\n"


def write_spine_diff(
    diff: SpineDiff,
    timeline: Timeline,
    directory: Path,
    with_transcript: bool,
    final: Optional[Timeline] = None,
    coverage: Optional[CoverageRun] = None,
) -> None:
    """Grava `edit_diff.json`, `edit_timeline.svg` e, com transcrição, `edit_transcript.md`."""
    (directory / EDIT_DIFF_FILE).write_text(diff.model_dump_json(indent=2), encoding="utf-8")
    (directory / EDIT_TIMELINE_FILE).write_text(render_spine_timeline(diff, timeline, final, coverage), encoding="utf-8")
    transcript = directory / EDIT_TRANSCRIPT_FILE
    if with_transcript:
        transcript.write_text(render_spine_transcript(diff), encoding="utf-8")
    else:
        transcript.unlink(missing_ok=True)


def removed_range_count(plan: SpinePlan, diff: SpineDiff) -> int:
    """Trechos removidos dos clipes da espinha (pausas e frases; um clipe que ficou fora conta como um)."""
    return sum(len(removed_ranges(plan.plan_for(c.clip_id).keep, c.diff.original_duration_sec)) for c in diff.clips)


# --- relatório ---

_ORIGIN_LABELS = {
    "ia": "IA",
    "ia+fallback": "IA, completada pelo código",
    "fallback": "código (fallback)",
}


def _cover_cell(seconds: float) -> str:
    return f"{format_timestamp(seconds)}"


def coverage_lines(coverage: CoverageRun) -> list[str]:
    """Seção "Cobertura": modelo e origem, janelas obrigatórias e quem atendeu, coberturas e respiros."""
    covered = len(coverage.windows) - len(coverage.uncovered_windows)
    breath_sec = sum(b.duration_sec for b in coverage.placed_breaths)
    model = coverage.effective_model_id or coverage.model_id
    lines = [
        "## Cobertura",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Diretor de cobertura | {coverage.provider} ({model}) |",
        *(
            [f"| Cadeia de modelos | {' → '.join(coverage.model_chain)} |"]
            if coverage.model_chain else []
        ),
        *(
            [f"| Troca de modelo | {_cell(s.from_model + ' → ' + s.to_model + ': ' + s.reason)} |" for s in coverage.model_switches]
        ),
        f"| Raciocínio | {coverage.thinking or '—'} |",
        f"| Origem | {_ORIGIN_LABELS[coverage.origin]}"
        + (f": {_cell(coverage.origin_reason)}" if coverage.origin_reason else "") + " |",
        f"| Tomadas candidatas | {coverage.candidates} (papéis: {', '.join(coverage.roles) or '—'}) |",
        f"| Janelas obrigatórias | {covered} de {len(coverage.windows)} cobertas |",
        f"| Coberturas | {len(coverage.placed_coverages)} |",
        f"| Respiros | {len(coverage.placed_breaths)} (+{breath_sec:.1f}s) |",
        f"| Duração | espinha {coverage.spine_duration_sec:.2f}s → final {coverage.final_duration_sec:.2f}s |",
        f"| Instruções | {_cell(' | '.join(coverage.instructions)) if coverage.instructions else '—'} |",
        f"| Decisão | `{COVERAGE_DECISION_FILE}`; Timeline renderizada: `{COVERAGE_TIMELINE_FILE}` |",
    ]
    for i, attempt in enumerate(coverage.attempts, start=1):
        tokens = (
            f"{attempt.input_tokens} entrada / {attempt.output_tokens} saída"
            if attempt.input_tokens is not None or attempt.output_tokens is not None
            else "não informado"
        )
        status = "ok" if attempt.success else f"inválida ({attempt.error_type})"
        lines.append(
            f"| Tentativa {i} | {status}, {attempt.latency_sec:.2f}s, chamadas à API: {attempt.api_attempts}, tokens: {tokens} |"
        )
    lines.append("")
    for warning in coverage.warnings:
        lines += [f"> **Aviso:** {_cell(warning)}", ""]
    if coverage.summary:
        lines += [f"**Resumo:** {coverage.summary}", ""]

    lines += ["### Janelas obrigatórias", ""]
    if coverage.windows:
        lines += ["| Janela | Clipe | Frases | Motivo | Tomadas ruins | Atendida por |", "| --- | --- | --- | --- | --- | --- |"]
        for w in coverage.windows:
            by = ", ".join(coverage.window_cover.get(w.id, [])) or "**sem cobertura**"
            lines.append(
                f"| {w.id} | {w.clip_id} | {w.span_label} | {_cell(', '.join(w.reasons))} | {', '.join(w.shot_ids)} | {by} |"
            )
    else:
        lines.append("Nenhuma: a imagem do principal está boa em toda a fala mantida.")
    lines += ["", "### Coberturas (por cima da fala)", ""]
    if coverage.placed_coverages:
        lines += [
            "| # | Frases | Tomada | No vídeo | Motivo | Obrigatória | Origem |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in coverage.placed_coverages:
            span = c.from_sentence if c.from_sentence == c.to_sentence else f"{c.from_sentence}–{c.to_sentence}"
            shots = ", ".join(dict.fromkeys(p.shot_id for p in c.pieces)) or c.shot_id
            lines.append(
                f"| {c.id} | {span} | {shots} | {_cover_cell(c.timeline_start)}–{_cover_cell(c.timeline_end)} | "
                f"{_cell(c.reason)} | {', '.join(c.windows) or 'não'} | {c.origin} |"
            )
    else:
        lines.append("Nenhuma cobertura.")
    lines += ["", "### Respiros (inseridos entre as falas)", ""]
    if coverage.placed_breaths:
        lines += [
            "| # | Depois de | Tomada | Tamanho | No vídeo | Som | Motivo | Origem |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for b in coverage.placed_breaths:
            lines.append(
                f"| {b.id} | {b.after_sentence} | {b.shot_id} | {BREATH_SIZE_LABELS[b.size]} ({b.duration_sec:.1f}s) | "
                f"{_cover_cell(b.timeline_start)} | {'da tomada' if b.audio else 'mudo'} | {_cell(b.reason)} | {b.origin} |"
            )
    else:
        lines.append("Nenhum respiro.")
    lines.append("")
    return lines


def _director_lines(director: DirectorRun) -> list[str]:
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
    for i, attempt in enumerate(director.attempts, start=1):
        tokens = (
            f"{attempt.input_tokens} entrada / {attempt.output_tokens} saída"
            if attempt.input_tokens is not None or attempt.output_tokens is not None
            else "não informado"
        )
        status = "ok" if attempt.success else f"inválida ({attempt.error_type})"
        lines.append(
            f"| Tentativa {i} | {status}, {attempt.latency_sec:.2f}s, chamadas à API: {attempt.api_attempts}, tokens: {tokens} |"
        )
    lines.append("")
    for warning in director.warnings:
        lines += [f"> **Aviso:** {warning}", ""]
    if director.decision is not None:
        lines += [f"**Resumo:** {director.decision.summary.strip()}", ""]
    return lines


def build_spine_report(
    source_dir: Path,
    selection: SpineSelection,
    plan: SpinePlan,
    timeline: Timeline,
    diff: SpineDiff,
    params: EditPipelineParams,
    output_path: Path,
    final_duration_sec: float,
    director: Optional[DirectorRun] = None,
    subtitles_path: Optional[Path] = None,
    subtitle_blocks: Optional[int] = None,
    transcription_model: Optional[str] = None,
    timings: Optional[RunTimings] = None,
    loudness: Optional[LoudnessResult] = None,
    encoder: Optional[str] = None,
    recipe_name: str = "vlog",
    coverage: Optional[CoverageRun] = None,
) -> str:
    """Markdown do `edit_report.md`: resumo, resumo por clipe, clipes não usados, parâmetros, volume,
    Diretor, cobertura (se houve), o que saiu de cada clipe (categoria, motivo, aplicação) e tempos."""
    original = diff.original_duration_sec
    removed_sec = max(0.0, original - final_duration_sec)
    pct = removed_sec / original * 100 if original > 0 else 0.0
    merged = diff.merged()
    has_transcript = bool(merged.sentences)
    has_decision = director is not None and director.decision is not None
    width, height = timeline.resolution
    fps_note = ""
    rounded = {round(f, 3) for f in plan.source_fps if abs(f - round(f)) > 0.001}
    if rounded:
        fps_note = f" (clipes a {', '.join(f'{f:g}'.replace('.', ',') for f in sorted(rounded))} fps convertidos)"
    cut = params.cut
    cuts = removed_range_count(plan, diff)

    lines = [f"# Relatório de edição — {source_dir.name}", ""]
    if has_decision and has_transcript:
        lines += [f"> **Diretor:** {merged.removal_counts()}", ""]
    used = [c for c in diff.clips if not c.dropped]
    lines += [
        "## Resumo",
        "",
        "| Item | Valor |",
        "| --- | --- |",
        f"| Pasta de origem | `{source_dir}` |",
        (
            "| Receita | vlog (espinha falada: clipes principais com fala, em ordem; sem cobertura) |"
            if recipe_name == "vlog"
            else f"| Receita | {recipe_name} (espinha falada + cobertura de drone/b-roll/POV e respiros) |"
        ),
        f"| Vídeo final | `{output_path.name}` |",
        f"| Clipes na espinha | {len(used)} de {len(diff.clips)} principais com fala ({len(diff.unused)} não usados) |",
        f"| Duração dos clipes | {format_timestamp(original)} ({original:.2f}s) |",
        f"| Duração final | {format_timestamp(final_duration_sec)} ({final_duration_sec:.2f}s) |",
        f"| Tempo removido | {removed_sec:.2f}s ({pct:.1f}%) |",
        f"| Número de cortes | {cuts} |",
        f"| Trechos mantidos | {len(timeline.video_main)} |",
        *(
            [
                f"| Cobertura | {len(coverage.placed_coverages)} coberturas, {len(coverage.placed_breaths)} respiros "
                f"(+{sum(b.duration_sec for b in coverage.placed_breaths):.1f}s), janelas obrigatórias "
                f"{len(coverage.windows) - len(coverage.uncovered_windows)}/{len(coverage.windows)} "
                f"({_ORIGIN_LABELS[coverage.origin]}) |"
            ]
            if coverage is not None
            else []
        ),
        f"| Saída | {width}x{height} @ {timeline.fps}fps{fps_note} |",
        *([f"| Codificador de vídeo | {_cell(encoder)} |"] if encoder else []),
        (
            f"| Timeline | `{TIMELINE_FILE}` |"
            if coverage is None
            else f"| Timeline | espinha `{TIMELINE_FILE}`; renderizada (com cobertura) `{COVERAGE_TIMELINE_FILE}` |"
        ),
        *(
            [f"| Legendas | `{subtitles_path.name}` ({subtitle_blocks} blocos) |"]
            if subtitles_path is not None and subtitle_blocks is not None
            else []
        ),
        *([f"| Transcrição | `{transcription_model}` |"] if transcription_model else []),
        (
            f"| Transcrição comparativa | `{EDIT_TRANSCRIPT_FILE}` e `{EDIT_DIFF_FILE}` |"
            if has_transcript
            else "| Transcrição comparativa | pulada (requer --transcribe) |"
        ),
        (
            f"| Linha do tempo | `{EDIT_TIMELINE_FILE}` (cor por clipe) |"
            if coverage is None
            else f"| Linha do tempo | `{EDIT_TIMELINE_FILE}` (cor por clipe, faixa de cobertura e respiros) |"
        ),
        "",
        "## Resumo por clipe",
        "",
        "| Clipe | Arquivo | Original | Final | Removido | Trechos | Frases removidas | Limite de silêncio | Situação |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for clip in diff.clips:
        clip_plan = plan.plan_for(clip.clip_id)
        removed_here = sum(1 for e in clip.diff.sentences if e.removal_requested)
        threshold = f"{clip_plan.threshold_db} dB" if clip_plan.threshold_db is not None else "—"
        situation = f"fora: {clip.dropped}" if clip.dropped else "na espinha"
        cut_sec = max(0.0, clip.diff.original_duration_sec - clip.diff.final_duration_sec)
        lines.append(
            f"| {clip.clip_id} | `{clip.file}` | {_clock(clip.diff.original_duration_sec)} | "
            f"{_clock(clip.diff.final_duration_sec)} | {cut_sec:.1f}s | {len(clip_plan.keep)} | {removed_here} | "
            f"{threshold} | {_cell(situation)} |"
        )
    lines += ["", "## Clipes não usados", "", *_unused_lines(diff.unused), ""]
    lines += [
        "## Parâmetros",
        "",
        "| Parâmetro | Valor |",
        "| --- | --- |",
        f"| threshold_db | {params.threshold_db}{'' if params.threshold_db == 'auto' else ' dB'} (por clipe) |",
        f"| min_silence_sec | {params.min_silence_sec}s |",
        f"| padding_before_sec | {cut.padding_before_sec}s |",
        f"| padding_after_sec | {cut.padding_after_sec}s |",
        f"| min_gap_to_cut_sec | {cut.min_gap_to_cut_sec}s |",
        f"| min_keep_sec | {cut.min_keep_sec}s |",
        f"| audio_fade_sec | {timeline.audio_fade_sec}s |",
        "",
        *_loudness_section(params, loudness),
    ]
    if director is not None:
        lines += _director_lines(director)
    if coverage is not None:
        lines += coverage_lines(coverage)

    lines += ["## O que saiu de cada clipe", ""]
    for clip in diff.clips:
        lines += [f"### {_clip_title(clip)}", ""]
        removed = [e for e in clip.diff.sentences if e.removal_requested]
        pause_sec = clip.diff.removed_pause_sec
        if not removed:
            lines += [f"Nenhuma frase removida; {format_seconds(pause_sec, 1)} de pausas removidas.", ""]
            continue
        lines += [
            f"{_plural(len(removed), 'frase marcada', 'frases marcadas')} para remoção "
            f"({clip.diff.application_counts()}); {format_seconds(pause_sec, 1)} de pausas removidas.",
            "",
            "| # | Início | Fim | Categoria | Texto | Motivo | Aplicação |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {e.sentence_id} | {format_timestamp(e.original_start_sec)} | {format_timestamp(e.original_end_sec)} "
            f"| {e.category} | {_cell(e.text or '')} | {_cell(e.reason or '')} "
            f"| {application_cell(merged, e.sentence_id)} |"
            for e in removed
        ]
        lines.append("")
    text = "\n".join(lines) + "\n"
    if timings is not None:
        text += build_timings_section(timings)
    return text
