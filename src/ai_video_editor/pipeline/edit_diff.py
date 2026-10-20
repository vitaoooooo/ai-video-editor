"""Transcrição comparativa (`edit_transcript.md`) e diff estruturado (`edit_diff.json`).

Parte do plano final: cada frase marcada para remoção recebe o status de aplicação medido
pela interseção dela com os trechos que saíram do vídeo (aplicada se saiu >= 90%). Os tempos
no vídeo editado usam o mesmo `TimelineMapper` das legendas, então batem com o .srt e o player.
"""

from pathlib import Path
from typing import Optional

from ai_video_editor.director.schema import Sentence, SentenceRemoval
from ai_video_editor.pipeline.project import EDIT_DIFF_FILE, EDIT_TRANSCRIPT_FILE
from ai_video_editor.pipeline.report import format_timestamp
from ai_video_editor.pipeline.schema import (
    DiffEntry,
    EditDiff,
    RemovalOrigin,
    SentenceDiffStatus,
    format_seconds,
)
from ai_video_editor.planning import SentenceCutResult, TimeRange, kept_overlap, removed_ranges, subtract_ranges
from ai_video_editor.subtitles import TimelineMapper

# Fração da frase que precisa sair do vídeo para a remoção contar como aplicada.
APPLIED_MIN_FRACTION = 0.9
# Abaixo disso (segundos) consideramos que nada da frase saiu.
_NOTHING_REMOVED_SEC = 0.01
_EPS = 0.001

_NOT_APPLIED_NOTES = {
    "protected": "protegida pela folga das frases vizinhas",
    "undone": "corte desfeito para não deixar meia palavra (o limite da frase invade uma frase mantida)",
}
_UNREACHED_NOTE = "o corte ajustado aos silêncios não alcançou esta frase"
_PARTIAL_NOTE = "corte limitado pelo silêncio mais próximo e pela folga das frases vizinhas"


def _cut_for(sentence: Sentence, results: list[SentenceCutResult]) -> Optional[SentenceCutResult]:
    """Corte pedido que contém a frase (frases removidas consecutivas dividem o mesmo corte)."""
    for result in results:
        if result.requested.start_sec - _EPS <= sentence.start_sec and sentence.end_sec <= result.requested.end_sec + _EPS:
            return result
    return None


def removal_status(
    sentence: Sentence, keep: list[TimeRange], cut: Optional[SentenceCutResult]
) -> tuple[SentenceDiffStatus, float, float, Optional[str]]:
    """(status, segundos removidos, % removido, nota) de uma frase marcada para remoção."""
    duration = sentence.end_sec - sentence.start_sec
    removed = max(0.0, duration - kept_overlap(sentence.start_sec, sentence.end_sec, keep))
    pct = min(100.0, removed / duration * 100)
    removed, pct = round(removed, 3), round(pct, 1)
    if removed / duration >= APPLIED_MIN_FRACTION:
        return "applied", removed, pct, None
    if cut is not None and cut.applied is None:
        return "not_applied", removed, pct, _NOT_APPLIED_NOTES[cut.outcome]
    if removed < _NOTHING_REMOVED_SEC or cut is None:
        return "not_applied", removed, pct, _UNREACHED_NOTE
    return "partial", removed, pct, _PARTIAL_NOTE


def _edited_span(start: float, end: float, mapper: TimelineMapper) -> tuple[Optional[float], Optional[float]]:
    """Início e fim, no vídeo editado, da parte de [start, end] que ficou no vídeo."""
    pieces = [
        (idx, max(start, r.start_sec), min(end, r.end_sec))
        for idx, r in enumerate(mapper.ranges)
        if min(end, r.end_sec) - max(start, r.start_sec) > _EPS
    ]
    if not pieces:
        return None, None
    first, last = pieces[0], pieces[-1]
    return round(mapper.map_in_range(first[0], first[1]), 3), round(mapper.map_in_range(last[0], last[2]), 3)


def _edited_point(time_sec: float, mapper: TimelineMapper) -> float:
    """Ponto do vídeo editado onde fica um trecho removido (início do próximo trecho mantido)."""
    for idx, r in enumerate(mapper.ranges):
        if r.start_sec >= time_sec - _EPS:
            return round(mapper.range_output_bounds(idx)[0], 3)
    return round(mapper.duration_sec, 3)


def build_edit_diff(
    video_name: str,
    sentences: list[Sentence],
    removals: dict[int, SentenceRemoval],
    keep: list[TimeRange],
    cut_results: list[SentenceCutResult],
    fps: float,
    original_duration_sec: float,
    final_duration_sec: float,
    min_pause_sec: float = 1.0,
    origins: Optional[dict[int, RemovalOrigin]] = None,
    version: Optional[int] = None,
    base_version: Optional[int] = None,
    request: Optional[str] = None,
    restored: Optional[dict[int, str]] = None,
    removed_now: Optional[set[int]] = None,
) -> EditDiff:
    """Monta o diff da fala original contra o plano final.

    Args:
        removals: Frases marcadas para remoção nesta versão (por ID).
        keep: Trechos mantidos do plano final (os mesmos usados nas legendas).
        cut_results: Destino de cada corte de frase (explica remoções não aplicadas/parciais).
        min_pause_sec: Pausas removidas a partir desta duração viram item; as menores só
            entram no total.
        origins: Quem removeu cada frase (padrão: Diretor).
        restored: Em revisões, frases que voltaram ao vídeo e o motivo.
        removed_now: Em revisões, frases removidas por esta revisão.
    """
    mapper = TimelineMapper(keep, fps)
    origins = origins or {}
    restored = restored or {}
    removed_now = removed_now or set()
    entries: list[DiffEntry] = []

    for sentence in sentences:
        edited_start, edited_end = _edited_span(sentence.start_sec, sentence.end_sec, mapper)
        entry = DiffEntry(
            type="sentence",
            original_start_sec=sentence.start_sec,
            original_end_sec=sentence.end_sec,
            edited_start_sec=edited_start,
            edited_end_sec=edited_end,
            sentence_id=sentence.id,
            text=sentence.text,
            status="kept",
        )
        removal = removals.get(sentence.id)
        if removal is not None:
            status, removed_sec, pct, note = removal_status(sentence, keep, _cut_for(sentence, cut_results))
            entry = entry.model_copy(
                update={
                    "status": status,
                    "removed_sec": removed_sec,
                    "removed_pct": pct,
                    "origin": origins.get(sentence.id, "director"),
                    "category": removal.category,
                    "reason": removal.reason,
                    "application_note": note,
                    "revision_change": "removed" if sentence.id in removed_now else None,
                }
            )
        elif sentence.id in restored:
            entry = entry.model_copy(
                update={"revision_change": "restored", "restoration_reason": restored[sentence.id]}
            )
        entries.append(entry)

    # Pausas: o que saiu do vídeo fora das frases marcadas para remoção.
    removed_sentence_spans = [
        TimeRange(start_sec=s.start_sec, end_sec=s.end_sec) for s in sentences if s.id in removals
    ]
    pauses = subtract_ranges(removed_ranges(keep, original_duration_sec), removed_sentence_spans)
    for pause in pauses:
        if pause.duration < min_pause_sec:
            continue
        point = _edited_point(pause.end_sec, mapper)
        entries.append(
            DiffEntry(
                type="pause",
                original_start_sec=pause.start_sec,
                original_end_sec=pause.end_sec,
                edited_start_sec=point,
                edited_end_sec=point,
                removed_sec=pause.duration,
                removed_pct=100.0,
            )
        )

    entries.sort(key=lambda e: (e.original_start_sec, e.type == "sentence"))
    return EditDiff(
        video_name=video_name,
        version=version,
        base_version=base_version,
        request=request,
        original_duration_sec=round(original_duration_sec, 3),
        final_duration_sec=round(final_duration_sec, 3),
        removed_pause_sec=round(sum(p.duration for p in pauses), 3),
        min_pause_marker_sec=min_pause_sec,
        entries=entries,
    )


def _span(entry: DiffEntry) -> str:
    """`[00:04.600 → 00:01.200]` (original → editado); `—` quando nada ficou no editado."""
    edited = format_timestamp(entry.edited_start_sec) if entry.edited_start_sec is not None else "—"
    return f"`[{format_timestamp(entry.original_start_sec)} → {edited}]`"


def _removal_detail(entry: DiffEntry, request: Optional[str]) -> str:
    """`removida pelo Diretor (repetition): motivo` ou `removida nesta revisão (...)`."""
    if entry.revision_change == "removed" and request:
        who = f"removida nesta revisão (pedido: “{request}”)"
    elif entry.origin == "revision":
        who = "removida em revisão anterior"
    else:
        who = "removida pelo Diretor"
    return f"{who} · {entry.category}: {entry.reason}"


def _entry_line(entry: DiffEntry, request: Optional[str]) -> str:
    if entry.type == "pause":
        return f"⏸ pausa de {format_seconds(entry.duration_sec, 1)} removida `[{format_timestamp(entry.original_start_sec)}]`"
    text = " ".join((entry.text or "").split())
    new_mark = "🆕 " if entry.revision_change == "removed" else ""
    if entry.status == "kept":
        if entry.revision_change == "restored":
            return f"✅ {_span(entry)} {text} — *restaurada nesta revisão: {entry.restoration_reason}*"
        return f"{_span(entry)} {text}"
    if entry.status == "applied":
        return (
            f"{new_mark}~~{text}~~ `[{format_timestamp(entry.original_start_sec)}–"
            f"{format_timestamp(entry.original_end_sec)}]` — {_removal_detail(entry, request)}"
        )
    if entry.status == "partial":
        return (
            f"◐ {new_mark}{_span(entry)} {text} — **remoção parcial**: saíram "
            f"{format_seconds(entry.removed_sec)} de {format_seconds(entry.duration_sec)} "
            f"({entry.removed_pct:.0f}%); {entry.application_note}. Pedido: {_removal_detail(entry, request)}"
        )
    return (
        f"⚠️ {new_mark}{_span(entry)} {text} — **remoção não aplicada**, continua no vídeo: "
        f"{entry.application_note}. Pedido: {_removal_detail(entry, request)}"
    )


def render_edit_transcript(diff: EditDiff) -> str:
    """Markdown da transcrição comparativa, em ordem cronológica."""
    kept = diff.count("kept")
    removed = len(diff.sentences) - kept
    removed_sec = max(0.0, diff.original_duration_sec - diff.final_duration_sec)
    lines = [
        f"# Transcrição comparativa — {diff.video_name}",
        "",
        "| Item | Valor |",
        "| --- | --- |",
    ]
    if diff.version is not None:
        lines.append(
            f"| Versão | v{diff.version}"
            + (f" (revisão da v{diff.base_version})" if diff.base_version is not None else "")
            + " |"
        )
    if diff.request:
        request = " ".join(diff.request.split()).replace("|", "\\|")
        lines.append(f"| Pedido | {request} |")
    lines += [
        f"| Duração original | {format_timestamp(diff.original_duration_sec)} ({diff.original_duration_sec:.2f}s) |",
        f"| Duração final | {format_timestamp(diff.final_duration_sec)} ({diff.final_duration_sec:.2f}s) |",
        f"| Tempo removido | {format_seconds(removed_sec)} |",
        f"| Frases mantidas | {kept} |",
        f"| Frases marcadas para remoção | {removed}: {diff.count('applied')} aplicadas, "
        f"{diff.count('partial')} parciais, {diff.count('not_applied')} não aplicadas |",
        f"| Pausas removidas | {format_seconds(diff.removed_pause_sec, 1)} "
        f"(no texto, só as de {format_seconds(diff.min_pause_marker_sec, 1)} ou mais) |",
        "",
        "Legenda: `[original → editado]` · ~~riscada~~ = removida · ◐ remoção parcial · "
        "⚠️ remoção não aplicada (continua no vídeo) · ✅ restaurada · 🆕 removida nesta revisão · "
        "⏸ pausa removida",
        "",
    ]
    for entry in diff.entries:
        lines += [_entry_line(entry, diff.request), ""]
    return "\n".join(lines)


def write_edit_diff(diff: EditDiff, directory: Path) -> tuple[Path, Path]:
    """Grava `edit_transcript.md` e `edit_diff.json` em `directory`."""
    transcript_path = directory / EDIT_TRANSCRIPT_FILE
    diff_path = directory / EDIT_DIFF_FILE
    transcript_path.write_text(render_edit_transcript(diff), encoding="utf-8")
    diff_path.write_text(diff.model_dump_json(indent=2), encoding="utf-8")
    return transcript_path, diff_path


def remove_edit_diff(directory: Path) -> None:
    """Apaga a transcrição comparativa de uma execução anterior (ficou obsoleta)."""
    for name in (EDIT_TRANSCRIPT_FILE, EDIT_DIFF_FILE):
        (directory / name).unlink(missing_ok=True)
