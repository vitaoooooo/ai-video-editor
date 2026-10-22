"""Papel de cada clipe pela subpasta, e parsing de `--papel` e `--ajuste-relogio`."""

import re
import unicodedata
from typing import Optional

from ai_video_editor.catalog.schema import ROLES, Role, RoleSource

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

# Valem só como nome exato da pasta: "outro" dentro de um nome maior não diz nada.
_EXACT_ONLY: frozenset[Role] = frozenset({"outro"})

_OFFSET_RE = re.compile(r"^([+-])?(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$")


def normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_name.lower())


def _lookup(key: str) -> Optional[Role]:
    """Sinônimo exato, ou o plural dele (drones, povs)."""
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    if key.endswith("s") and key[:-1] in _SYNONYMS:
        return _SYNONYMS[key[:-1]]
    return None


def _tokens(name: str) -> list[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return [t for t in re.split(r"[^a-z0-9]+", ascii_name.lower()) if t]


def match_folder_role(folder: str) -> tuple[Role, RoleSource, list[Role]]:
    """Papel da pasta, de onde ele veio e todos os papéis encontrados no nome.

    1. Nome inteiro igual a um sinônimo (`Drone`, `B-Roll`, `GoPros`) → `padrão`.
    2. Senão, um sinônimo como palavra do nome (`Osmo Pocket - PRINCIPAL`, `Imagens de Drone`),
       inclusive duas palavras vizinhas juntas (`Fx30 - B Roll`) → `contém`.
    Nenhum papel, ou papéis diferentes no mesmo nome (`Drone e POV`) → `outro`.
    """
    exact = _lookup(normalize_name(folder))
    if exact is not None:
        return exact, "padrão", [exact]
    tokens = _tokens(folder)
    found: list[Role] = []
    for candidate in tokens + [a + b for a, b in zip(tokens, tokens[1:])]:
        role = _lookup(candidate)
        if role is not None and role not in _EXACT_ONLY and role not in found:
            found.append(role)
    if len(found) == 1:
        return found[0], "contém", found
    return "outro", "padrão", found


def role_for_folder(folder: str) -> Role:
    """`Drone`, `B-Roll`, `Osmo Pocket - PRINCIPAL`... → papel; o que não for reconhecido vira `outro`."""
    return match_folder_role(folder)[0]


def parse_role(value: str) -> Role:
    """Papel escrito pelo usuário (aceita os mesmos sinônimos das pastas)."""
    role = _lookup(normalize_name(value))
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
