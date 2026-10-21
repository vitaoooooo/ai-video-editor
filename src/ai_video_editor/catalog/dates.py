"""Data de gravação de um clipe: tags do arquivo → padrão no nome → data de modificação.

Tudo é convertido para UTC. Tags sem fuso são tratadas como UTC (é o que a especificação do
`creation_time` diz); datas no nome do arquivo não têm fuso e são lidas como horário local desta
máquina. Câmeras com relógio errado se corrigem com `--ajuste-relogio`.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ai_video_editor.catalog.schema import DateSource, RecordingDate

# Antes disso é relógio zerado (1904/1970) ou bateria descarregada; não é data de gravação.
MIN_PLAUSIBLE_YEAR = 2000

# DJI_20240115123045_0001, VID_20240115_123045, PXL_20240115_123045123, 2024-01-15 12-30-45 (OBS)...
_FILENAME_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])"
    r"[ _T-]?([01]\d|2[0-3])[-_.h:]?([0-5]\d)[-_.m:]?([0-5]\d)"
)


def _parse_tag(value: str) -> Optional[datetime]:
    text = value.strip()
    if not text:
        return None
    # "-0300" → "-03:00" (fromisoformat do 3.11 aceita os dois, mas não "+0000Z" etc.)
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text.replace(" ", "T", 1))
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _plausible(value: datetime) -> bool:
    return value.year >= MIN_PLAUSIBLE_YEAR


def date_from_tags(format_tags: dict, stream_tags: dict) -> Optional[RecordingDate]:
    """Primeira data plausível entre as tags do container e da trilha de vídeo."""
    candidates: list[tuple[DateSource, Optional[str]]] = [
        ("quicktime", format_tags.get("com.apple.quicktime.creationdate")),
        ("container", format_tags.get("creation_time")),
        ("stream", stream_tags.get("creation_time")),
    ]
    for source, raw in candidates:
        if not raw:
            continue
        parsed = _parse_tag(str(raw))
        if parsed is not None and _plausible(parsed):
            return RecordingDate(value=parsed, source=source, raw=str(raw))
    return None


def date_from_filename(name: str) -> Optional[RecordingDate]:
    match = _FILENAME_RE.search(name)
    if match is None:
        return None
    try:
        local = datetime(*(int(g) for g in match.groups()))
    except ValueError:
        return None
    if not _plausible(local):
        return None
    return RecordingDate(value=local.astimezone().astimezone(timezone.utc), source="filename", raw=match.group(0))


def date_from_mtime(mtime_ns: int) -> RecordingDate:
    value = datetime.fromtimestamp(mtime_ns / 1e9, tz=timezone.utc)
    return RecordingDate(value=value, source="mtime", raw=value.isoformat(), reliable=False)


def recording_date(path: Path, tag_date: Optional[RecordingDate], mtime_ns: int) -> RecordingDate:
    return tag_date or date_from_filename(path.name) or date_from_mtime(mtime_ns)


def is_future(value: datetime, now: Optional[datetime] = None) -> bool:
    return value > (now or datetime.now(timezone.utc)) + timedelta(days=1)
