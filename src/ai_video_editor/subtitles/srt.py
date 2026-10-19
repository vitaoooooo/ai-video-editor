"""Geração de legendas a partir das palavras da transcrição e escrita em SRT."""

from pathlib import Path

from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.subtitles.remap import TimelineMapper
from ai_video_editor.subtitles.schema import SubtitleBlock, SubtitleParams
from ai_video_editor.transcription.schema import Transcript, Word

_STRONG_PUNCT = (".", "!", "?", "…")
_WEAK_PUNCT = (",", ";", ":", "—", "–")
# Fração da capacidade do bloco a partir da qual uma vírgula já justifica quebrar.
_WEAK_BREAK_FILL = 0.6


class _TimedWord:
    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end


def _wrap(words: list[str], max_chars: int) -> list[str]:
    """Quebra palavras em linhas de até `max_chars` (palavra maior que o limite fica sozinha)."""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _fits(texts: list[str], params: SubtitleParams) -> bool:
    return len(_wrap(texts, params.max_chars_per_line)) <= params.max_lines


def _group_words(words: list[_TimedWord], params: SubtitleParams) -> list[list[_TimedWord]]:
    """Agrupa palavras (já no tempo editado, de um único trecho) em blocos."""
    capacity = params.max_chars_per_line * params.max_lines
    groups: list[list[_TimedWord]] = []
    current: list[_TimedWord] = []

    def close() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in words:
        if current:
            texts = [w.text for w in current] + [word.text]
            too_long = not _fits(texts, params)
            too_slow = word.end - current[0].start > params.max_duration_sec
            gap = word.start - current[-1].end > params.max_gap_sec
            if too_long or too_slow or gap:
                close()
        current.append(word)

        duration = current[-1].end - current[0].start
        chars = len(" ".join(w.text for w in current))
        if word.text.endswith(_STRONG_PUNCT) and duration >= params.min_duration_sec:
            close()
        elif word.text.endswith(_WEAK_PUNCT) and chars >= capacity * _WEAK_BREAK_FILL:
            close()
    close()
    return groups


def _best_range(mapper: TimelineMapper, word: Word, params: SubtitleParams) -> int | None:
    """Trecho mantido onde está a maior parte da palavra (o primeiro, se empatar), ou None se descartada."""
    duration = word.end_sec - word.start_sec
    if duration <= 0:
        return mapper.range_index(word.start_sec)
    best_idx: int | None = None
    best_overlap = 0.0
    total = 0.0
    for idx, r in enumerate(mapper.ranges):
        overlap = min(word.end_sec, r.end_sec) - max(word.start_sec, r.start_sec)
        if overlap <= 0:
            continue
        total += overlap
        if overlap > best_overlap + 1e-9:
            best_idx, best_overlap = idx, overlap
    if best_idx is None:
        return None
    if total < params.min_kept_fraction * duration and total < params.min_kept_sec:
        return None
    return best_idx


def build_subtitle_blocks(
    transcript: Transcript,
    kept_ranges: list[TimeRange],
    fps: float,
    params: SubtitleParams | None = None,
) -> list[SubtitleBlock]:
    """Gera blocos de legenda sincronizados com o vídeo editado.

    Uma palavra só é descartada se a parte dela em trechos mantidos for pequena demais
    (menor que `min_kept_fraction` da duração E menor que `min_kept_sec`). A palavra mantida
    vai para o trecho onde está a maior parte dela, com o tempo limitado às bordas dele.
    Um bloco nunca atravessa um corte; os blocos não se sobrepõem.
    """
    params = params or SubtitleParams()
    mapper = TimelineMapper(kept_ranges, fps)

    words: list[Word] = sorted(
        (w for seg in transcript.segments for w in seg.words), key=lambda w: w.start_sec
    )

    per_range: dict[int, list[_TimedWord]] = {}
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        idx = _best_range(mapper, word, params)
        if idx is None:
            continue
        r = mapper.ranges[idx]
        start = mapper.map_in_range(idx, max(word.start_sec, r.start_sec))
        end = mapper.map_in_range(idx, min(word.end_sec, r.end_sec))
        per_range.setdefault(idx, []).append(_TimedWord(text, start, end))

    blocks: list[SubtitleBlock] = []
    previous_end = 0.0
    for idx in sorted(per_range, key=lambda i: mapper.range_output_bounds(i)[0]):
        range_end = mapper.range_output_bounds(idx)[1]
        groups = _group_words(per_range[idx], params)
        for pos, group in enumerate(groups):
            start = max(group[0].start, previous_end)
            end = group[-1].end
            limit = groups[pos + 1][0].start if pos + 1 < len(groups) else range_end
            if end - start < params.min_duration_sec:
                end = min(start + params.min_duration_sec, max(limit, end))
            end = min(end, range_end)
            if end <= start:
                continue
            lines = _wrap([w.text for w in group], params.max_chars_per_line)
            blocks.append(
                SubtitleBlock(index=len(blocks) + 1, start_sec=start, end_sec=end, lines=lines)
            )
            previous_end = end
    return blocks


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_srt(blocks: list[SubtitleBlock]) -> str:
    """Serializa os blocos no formato SRT."""
    parts = [
        f"{n}\n{_format_timestamp(b.start_sec)} --> {_format_timestamp(b.end_sec)}\n{b.text}\n"
        for n, b in enumerate(blocks, start=1)
    ]
    return "\n".join(parts)


def write_srt(blocks: list[SubtitleBlock], path: Path) -> Path:
    """Grava o SRT em UTF-8 e retorna o caminho."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_srt(blocks), encoding="utf-8", newline="\n")
    return path


def generate_srt(
    transcript: Transcript,
    kept_ranges: list[TimeRange],
    fps: float,
    output_path: Path,
    params: SubtitleParams | None = None,
) -> Path:
    """Atalho: gera os blocos e grava o .srt do vídeo editado."""
    return write_srt(build_subtitle_blocks(transcript, kept_ranges, fps, params), output_path)
