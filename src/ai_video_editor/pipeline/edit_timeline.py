"""Linha do tempo dos cortes (`edit_timeline.svg`), desenhada a partir do `EditDiff`.

Uma barra horizontal representa o vídeo original; cada item do diff (frase ou pausa removida)
vira um grupo `<g class="item">` com a cor do seu destino. Vídeos longos quebram em várias
linhas, e um item que atravessa a quebra ganha um retângulo em cada linha (o grupo continua
único). O SVG tem fundo próprio, então fica legível em páginas claras e escuras.
"""

import math
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

from ai_video_editor.pipeline.project import EDIT_TIMELINE_FILE
from ai_video_editor.pipeline.schema import DiffEntry, EditDiff, _plural, format_seconds

WIDTH = 1000
_MARGIN = 24
_BAR_WIDTH = WIDTH - 2 * _MARGIN
_BAR_HEIGHT = 34
_ROW_GAP = 30
_HEADER_HEIGHT = 74
_LEGEND_HEIGHT = 44
# Até 10 minutos cabem numa linha; acima disso, uma linha por 10 minutos.
_ROW_SEC = 600.0
# Itens removidos menores que isso na escala ainda aparecem (pausas de 1s num vídeo longo).
_MIN_MARK_PX = 1.2
_TICK_STEPS = (5, 10, 15, 30, 60, 120, 300, 600)

_BACKGROUND = "#ffffff"
_BORDER = "#d0d7de"
_TEXT = "#1f2328"
_MUTED = "#57606a"

# (chave, cor, rótulo da legenda), na ordem da legenda.
CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("kept", "#4c78a8", "Fala mantida"),
    ("pause", "#f2b134", "Pausa removida"),
    ("removed", "#d1495b", "Frase removida (Diretor/revisão)"),
    ("partial", "#8e6cc0", "Remoção parcial ou não aplicada"),
    ("gap", "#e6e9ed", "Sem fala / pausa curta"),
)
_COLORS = {key: color for key, color, _ in CATEGORIES}


def category(entry: DiffEntry) -> str:
    """Chave da cor de um item: `kept`, `pause`, `removed` ou `partial`."""
    if entry.type == "pause":
        return "pause"
    if entry.status == "applied":
        return "removed"
    if entry.status in ("partial", "not_applied"):
        return "partial"
    return "kept"


def _clock(seconds: float) -> str:
    """`75.9` -> `1:15`; `3725` -> `1:02:05` (trunca, como o player mostra)."""
    total = int(seconds + 1e-6)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _tick_step(row_sec: float) -> int:
    """Passo das marcas de tempo: o menor da lista que dá no máximo ~12 marcas por linha."""
    return next((s for s in _TICK_STEPS if row_sec / s <= 12), _TICK_STEPS[-1])


def _item_title(entry: DiffEntry) -> str:
    span = f"{_clock(entry.original_start_sec)}–{_clock(entry.original_end_sec)}"
    if entry.type == "pause":
        return f"{span} · pausa de {format_seconds(entry.duration_sec, 1)} removida"
    text = " ".join((entry.text or "").split())
    if len(text) > 120:
        text = text[:117] + "..."
    if entry.status == "kept":
        return f"{span} · {text}"
    detail = entry.application_label()
    if entry.reason:
        detail += f" · {entry.category}: {entry.reason}"
    return f"{span} · {text} — {detail}"


def render_edit_timeline(diff: EditDiff) -> str:
    """SVG da linha do tempo dos cortes, com título, marcas de tempo e legenda."""
    duration = max(diff.original_duration_sec, 0.001)
    rows = math.ceil(duration / _ROW_SEC) if duration > _ROW_SEC else 1
    row_sec = _ROW_SEC if rows > 1 else duration
    scale = _BAR_WIDTH / row_sec
    row_pitch = _BAR_HEIGHT + _ROW_GAP
    height = _HEADER_HEIGHT + rows * row_pitch + _LEGEND_HEIGHT

    def row_y(row: int) -> float:
        return _HEADER_HEIGHT + row * row_pitch

    removed_sec = max(0.0, diff.original_duration_sec - diff.final_duration_sec)
    pct = removed_sec / duration * 100
    removed_sentences = sum(1 for e in diff.sentences if e.status == "applied")
    pauses = [e for e in diff.entries if e.type == "pause"]
    version = f" · v{diff.version}" if diff.version is not None else ""
    pct_text = f"{pct:.1f}".replace(".", ",")
    title = (
        f"{diff.video_name}{version}: {_clock(diff.original_duration_sec)} → "
        f"{_clock(diff.final_duration_sec)} ({pct_text}% removido)"
    )
    partial = diff.count("partial") + diff.count("not_applied")
    subtitle = (
        f"{_plural(removed_sentences, 'frase removida', 'frases removidas')} · "
        f"{_plural(partial, 'parcial/não aplicada', 'parciais/não aplicadas')} · "
        f"{_plural(len(pauses), 'pausa', 'pausas')} de {format_seconds(diff.min_pause_marker_sec, 1)} ou mais · "
        f"{format_seconds(diff.removed_pause_sec, 1)} de pausas no total"
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

    # Trilhos (o vão entre os itens é fala não transcrita, silêncio ou pausa curta) e marcas de tempo.
    step = _tick_step(row_sec)
    for row in range(rows):
        start = row * row_sec
        length = min(row_sec, duration - start)
        y = row_y(row)
        out.append(
            f'<rect class="track" x="{_MARGIN}" y="{y}" width="{length * scale:.2f}" height="{_BAR_HEIGHT}" '
            f'fill="{_COLORS["gap"]}"/>'
        )
        tick = 0
        while tick <= length + 0.001:
            x = _MARGIN + tick * scale
            out.append(
                f'<line x1="{x:.2f}" y1="{y + _BAR_HEIGHT}" x2="{x:.2f}" y2="{y + _BAR_HEIGHT + 5}" '
                f'stroke="{_MUTED}" stroke-width="1"/>'
            )
            anchor = "start" if tick == 0 else ("end" if x > WIDTH - _MARGIN - 20 else "middle")
            out.append(
                f'<text x="{x:.2f}" y="{y + _BAR_HEIGHT + 17}" font-size="11" fill="{_MUTED}" '
                f'text-anchor="{anchor}">{_clock(start + tick)}</text>'
            )
            tick += step

    # Itens: mantidos primeiro, removidos por cima (uma pausa de 1s não some atrás de uma frase).
    ordered = sorted(diff.entries, key=lambda e: category(e) != "kept")
    for entry in ordered:
        key = category(entry)
        rects = []
        first_row = min(rows - 1, int(entry.original_start_sec // row_sec))
        last_row = min(rows - 1, int(max(entry.original_start_sec, entry.original_end_sec - 0.001) // row_sec))
        for row in range(first_row, last_row + 1):
            a = max(entry.original_start_sec, row * row_sec) - row * row_sec
            b = min(entry.original_end_sec, (row + 1) * row_sec) - row * row_sec
            x, w = _MARGIN + a * scale, max(0.0, b - a) * scale
            if key != "kept" and w < _MIN_MARK_PX:
                x, w = x - (_MIN_MARK_PX - w) / 2, _MIN_MARK_PX
            rects.append(
                f'<rect x="{x:.2f}" y="{row_y(row)}" width="{w:.2f}" height="{_BAR_HEIGHT}" fill="{_COLORS[key]}"/>'
            )
        attrs = f' data-sentence-id="{entry.sentence_id}"' if entry.sentence_id is not None else ""
        out.append(
            f'<g class="item {key}"{attrs}><title>{escape(_item_title(entry))}</title>{"".join(rects)}</g>'
        )

    # Legenda
    legend_y = row_y(rows) + 4
    x = float(_MARGIN)
    out.append('<g class="legend">')
    for key, color, label in CATEGORIES:
        out.append(f'<rect x="{x:.0f}" y="{legend_y}" width="14" height="14" rx="2" fill="{color}"/>')
        out.append(
            f'<text x="{x + 20:.0f}" y="{legend_y + 11.5}" font-size="12.5" fill="{_TEXT}">{escape(label)}</text>'
        )
        x += 20 + len(label) * 6.9 + 22
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_edit_timeline(diff: EditDiff, directory: Path) -> Path:
    """Grava `edit_timeline.svg` em `directory`."""
    path = directory / EDIT_TIMELINE_FILE
    path.write_text(render_edit_timeline(diff), encoding="utf-8")
    return path


def timeline_from_diff_file(diff_path: Path, output: Optional[Path] = None) -> Path:
    """Desenha a linha do tempo a partir de um `edit_diff.json` existente (ao lado dele, por padrão)."""
    diff = EditDiff.model_validate_json(diff_path.read_text(encoding="utf-8"))
    target = output or diff_path.parent / EDIT_TIMELINE_FILE
    target.write_text(render_edit_timeline(diff), encoding="utf-8")
    return target
