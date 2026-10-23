"""Relação entre uma tomada candidata e o que está na tela (sem IA): proximidade na gravação e tags.

Usada no prompt (o que mostrar ao lado de cada janela e candidata) e pelo código quando ele escolhe
tomadas sozinho (fallback e completar trechos longos): contexto vem antes de beleza.

- **Vizinha na gravação:** do mesmo papel e até `NEIGHBOR_CLIPS` posições do clipe que fala na ordem
  do catálogo (P008 é "gravada logo depois de P007"); de outro papel, só quando as duas datas vêm da
  gravação (não do nome do arquivo) e ficam a até `NEIGHBOR_MINUTES`.
- **Tags em comum:** pelo menos `MIN_COMMON_TAGS` tags específicas (fora de `GENERIC_TAGS`) em comum com
  as tomadas da tela; se a tela tem só uma tag específica, basta ela.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ai_video_editor.coverage.schema import VisualShot

NEIGHBOR_CLIPS = 1
NEIGHBOR_MINUTES = 15.0
# Tags que aparecem em quase tudo e não dizem "o mesmo lugar" (no material real, "árvore" está em 40 das
# 90 candidatas: ligava uma cerca com parquinho a uma ilha no lago).
GENERIC_TAGS = ("pessoa", "rosto", "detalhe", "paisagem", "árvore", "céu")
MIN_COMMON_TAGS = 2

_CLIP_ID = re.compile(r"^([A-Z])(\d+)$")


@dataclass(frozen=True)
class Relation:
    """Por que uma tomada tem a ver com a tela. `score` maior = mais relacionada."""

    neighbor: Optional[str]
    tags: tuple[str, ...]

    @property
    def score(self) -> tuple[int, int]:
        return (1 if self.neighbor else 0, len(self.tags))

    @property
    def label(self) -> str:
        parts = [self.neighbor] if self.neighbor else []
        if self.tags:
            parts.append("tags " + ", ".join(self.tags))
        return "; ".join(parts)


def _clip_position(clip_id: str) -> Optional[tuple[str, int]]:
    match = _CLIP_ID.match(clip_id)
    return (match.group(1), int(match.group(2))) if match else None


def neighbor_note(shot: VisualShot, screen: VisualShot) -> Optional[str]:
    """"gravada logo depois de P007" se a tomada é vizinha do clipe da tela na gravação; senão None."""
    if shot.clip_id == screen.clip_id:
        return None
    a, b = _clip_position(shot.clip_id), _clip_position(screen.clip_id)
    if a and b and a[0] == b[0] and abs(a[1] - b[1]) <= NEIGHBOR_CLIPS:
        return f"gravada logo {'depois' if a[1] > b[1] else 'antes'} de {screen.clip_id}"
    if shot.date_reliable and screen.date_reliable and shot.recorded_at and screen.recorded_at:
        minutes = (shot.recorded_at - screen.recorded_at).total_seconds() / 60
        if abs(minutes) <= NEIGHBOR_MINUTES:
            return f"gravada {abs(minutes):.0f} min {'depois' if minutes >= 0 else 'antes'} de {screen.clip_id}"
    return None


def specific_tags(tags: Iterable[str]) -> list[str]:
    return [t for t in dict.fromkeys(tags) if t not in GENERIC_TAGS]


def relation(shot: VisualShot, screens: list[VisualShot]) -> Optional[Relation]:
    """Relação da tomada com as tomadas da tela (None = sem relação)."""
    neighbor = next((n for s in screens if (n := neighbor_note(shot, s))), None)
    screen_tags = specific_tags(t for s in screens for t in s.tags)
    common = tuple(t for t in specific_tags(shot.tags) if t in screen_tags)
    needed = min(MIN_COMMON_TAGS, len(screen_tags))
    tags = common if needed and len(common) >= needed else ()
    if neighbor is None and not tags:
        return None
    return Relation(neighbor=neighbor, tags=tags)


def related_shots(candidates: Iterable[VisualShot], screens: list[VisualShot]) -> list[tuple[VisualShot, Relation]]:
    """Candidatas relacionadas com a tela, da mais relacionada para a menos (desempate: beleza)."""
    found = [(c, r) for c in candidates if (r := relation(c, screens)) is not None]
    return sorted(found, key=lambda x: (x[1].score, x[0].beauty or 0), reverse=True)


def recording_note(shot: VisualShot, spine_clips: list[VisualShot]) -> Optional[str]:
    """Proximidade de gravação com algum clipe que fala (para a lista de candidatas do prompt)."""
    return next((n for s in spine_clips if (n := neighbor_note(shot, s))), None)
