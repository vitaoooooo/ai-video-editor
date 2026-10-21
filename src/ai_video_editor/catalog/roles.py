"""Papel de cada clipe pela subpasta, e parsing de `--papel` e `--ajuste-relogio`."""

import re
import unicodedata

from ai_video_editor.catalog.schema import ROLES, Role

# Nome normalizado (sem acentos, minúsculo, só letras e números) → papel.
_SYNONYMS: dict[str, Role] = {
    "principal": "principal",
    "main": "principal",
    "aroll": "principal",
    "acam": "principal",
    "drone": "drone",
    "pov": "pov",
    "gopro": "pov",
    "broll": "broll",
    "bcam": "broll",
    "outro": "outro",
    "outros": "outro",
}

_OFFSET_RE = re.compile(r"^([+-])?(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$")


def normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_name.lower())


def role_for_folder(folder: str) -> Role:
    """`Drone`, `B-Roll`, `GoPros`, `Câmera`... → papel; o que não for reconhecido vira `outro`."""
    key = normalize_name(folder)
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    if key.endswith("s") and key[:-1] in _SYNONYMS:  # plural: drones, povs
        return _SYNONYMS[key[:-1]]
    return "outro"


def parse_role(value: str) -> Role:
    """Papel escrito pelo usuário (aceita os mesmos sinônimos das pastas)."""
    key = normalize_name(value)
    role = _SYNONYMS.get(key) or (_SYNONYMS.get(key[:-1]) if key.endswith("s") else None)
    if role is None:
        raise ValueError(f"papel desconhecido '{value}' (use: {', '.join(ROLES)})")
    return role


def parse_role_override(text: str) -> tuple[str, Role]:
    """`--papel "Minha Pasta=broll"` → ("Minha Pasta", "broll")."""
    folder, sep, role = text.rpartition("=")
    if not sep or not folder.strip() or not role.strip():
        raise ValueError(f"esperado pasta=papel, recebido '{text}'")
    return folder.strip(), parse_role(role.strip())


def parse_offset(text: str) -> float:
    """`+3h`, `-1h30m`, `+90s`, `2m` → segundos."""
    compact = text.strip().lower().replace(" ", "")
    match = _OFFSET_RE.match(compact)
    if not compact or match is None or not any(match.group(i) for i in (2, 3, 4)):
        raise ValueError(f"ajuste inválido '{text}' (exemplos: +3h, -1h30m, +45s)")
    sign, hours, minutes, seconds = match.groups()
    total = float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
    return -total if sign == "-" else total


def parse_clock_adjustment(text: str) -> tuple[Role, float]:
    """`--ajuste-relogio drone=+3h` → ("drone", 10800.0)."""
    role, sep, offset = text.partition("=")
    if not sep:
        raise ValueError(f"esperado papel=ajuste (ex: drone=+3h), recebido '{text}'")
    return parse_role(role.strip()), parse_offset(offset)


def format_offset(seconds: float) -> str:
    """10800 → `+3h`; -5430 → `-1h30m30s`."""
    sign = "-" if seconds < 0 else "+"
    rest = round(abs(seconds))
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours}h" if hours else "", f"{minutes}m" if minutes else "", f"{secs}s" if secs else ""]
    return sign + ("".join(parts) or "0s")
