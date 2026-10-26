"""Momentos: fotos seguidas do mesmo clipe que mostram a mesma coisa viram um trecho com ID.

Uma foto continua o momento da anterior quando (1) não há troca de cena entre elas, (2) o lugar é o
mesmo (ou um dos dois não foi dito), (3) as tags em comum chegam a MOMENT_MIN_COMMON_TAGS (ou a todas
as tags da que tem menos) e (4) o momento não passa de MAX_MOMENT_SEC. Foto sem descrição não quebra
o momento (não há com o que comparar), só a troca de cena e a duração máxima.

Os momentos cobrem o clipe inteiro sem buracos: cada um vai da sua 1ª foto até a 1ª do seguinte (o 1º
começa em 0 e o último termina no fim do clipe). As medidas de borrão/movimento do intervalo vêm das
amostras do `scan_clip` (as mesmas do `visual`).
"""

import unicodedata
from collections import Counter
from typing import Optional

from ai_video_editor.roteiro.schema import Moment, PhotoDescription, RoteiroPhoto, RoteiroSentence
from ai_video_editor.visual.schema import ShotMeasures
from ai_video_editor.visual.shots import Sample, ShotSpan, measure_shot

# Tags em comum para duas fotos seguidas serem o mesmo momento (estimativa inicial, a calibrar).
MOMENT_MIN_COMMON_TAGS = 2
# Duração máxima de um momento: mais que isso perde a granularidade para o editor.
MAX_MOMENT_SEC = 20.0
# Tags somadas que vão para o momento.
MOMENT_MAX_TAGS = 8


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", " ".join(text.lower().split()))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def similar(a: PhotoDescription, b: PhotoDescription) -> bool:
    """Mesmo lugar (quando os dois dizem) e tags suficientes em comum."""
    if a.place and b.place and _plain(a.place) != _plain(b.place):
        return False
    tags_a, tags_b = {_plain(t) for t in a.tags}, {_plain(t) for t in b.tags}
    needed = min(MOMENT_MIN_COMMON_TAGS, len(tags_a), len(tags_b))
    return len(tags_a & tags_b) >= needed


def _issues(descriptions: list[PhotoDescription], measures: Optional[ShotMeasures]) -> list[str]:
    issues: list[str] = []
    for description in descriptions:
        if description.issue != "nenhum" and description.issue not in issues:
            issues.append(description.issue)
    if measures is not None:
        for flag, label in (
            (measures.blurry, "borrão alto"), (measures.shaky, "movimento alto"), (measures.uniform, "tela uniforme"),
            (measures.dark, "escura"), (measures.bright, "clara"),
        ):
            if flag and label not in issues:
                issues.append(label)
    return issues


def _representative(descriptions: list[PhotoDescription], tags: list[str]) -> Optional[PhotoDescription]:
    """A descrição com mais tags do momento (empate: a mais antiga)."""
    if not descriptions:
        return None
    wanted = {_plain(t) for t in tags}
    return max(descriptions, key=lambda d: (len({_plain(t) for t in d.tags} & wanted), -descriptions.index(d)))


def _summed_tags(descriptions: list[PhotoDescription]) -> list[str]:
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    for description in descriptions:
        for tag in description.tags:
            counts[tag] += 1
            first.setdefault(tag, len(first))
    return sorted(counts, key=lambda t: (-counts[t], first[t]))[:MOMENT_MAX_TAGS]


def group_photos(
    photos: list[RoteiroPhoto], duration: float, scene_cuts: list[float]
) -> list[list[RoteiroPhoto]]:
    """Grupos de fotos de UM clipe (em ordem de tempo), pelas regras do topo do módulo."""
    groups: list[list[RoteiroPhoto]] = []
    for index, photo in enumerate(photos):
        if groups:
            current = groups[-1]
            previous = current[-1]
            cut_between = any(previous.time_sec < cut <= photo.time_sec for cut in scene_cuts)
            # Com a foto, o momento vai até a foto seguinte (ou o fim do clipe).
            start = 0.0 if len(groups) == 1 else current[0].time_sec
            end = photos[index + 1].time_sec if index + 1 < len(photos) else duration
            too_long = end - start > MAX_MOMENT_SEC
            described = [p for p in current if p.description is not None]
            unlike = (
                photo.description is not None and described
                and not similar(described[-1].description, photo.description)
            )
            if not (cut_between or too_long or unlike):
                current.append(photo)
                continue
        groups.append([photo])
    return groups


def build_moments(
    clip_id: str,
    photos: list[RoteiroPhoto],
    duration: float,
    scene_cuts: list[float],
    samples: Optional[list[Sample]] = None,
) -> list[Moment]:
    """Momentos de UM clipe, cobrindo [0, duração]. Sem fotos, um momento só (o clipe inteiro)."""
    groups = group_photos(photos, duration, scene_cuts) if photos else [[]]
    moments: list[Moment] = []
    for number, group in enumerate(groups, start=1):
        start = 0.0 if number == 1 else group[0].time_sec
        end = groups[number][0].time_sec if number < len(groups) else duration
        descriptions = [p.description for p in group if p.description is not None]
        tags = _summed_tags(descriptions)
        best = _representative(descriptions, tags)
        span = ShotSpan(start_sec=start, end_sec=max(end, start + 1e-3), boundary="cena" if number > 1 else "inicio")
        measures = measure_shot(span, samples) if samples else None
        moments.append(Moment(
            id=f"{clip_id}.M{number:02d}",
            clip_id=clip_id,
            start_sec=round(start, 3),
            end_sec=round(end, 3),
            photo_ids=[p.id for p in group],
            description=best.what if best is not None else "",
            place=best.place if best is not None else "",
            people=max((d.people for d in descriptions), default=0),
            talking_to_camera=sum(d.talking_to_camera for d in descriptions) * 2 > len(descriptions) if descriptions else False,
            tags=tags,
            measures=measures,
            scene_change=any(abs(start - cut) < 1.0 for cut in scene_cuts) and number > 1,
        ))
    return moments


def finalize_issues(moments: list[Moment], descriptions_of: dict[str, PhotoDescription]) -> None:
    """Problemas de cada momento: os da IA (fotos) e os das medidas (depois de `mark_shaky`)."""
    for moment in moments:
        found = [descriptions_of[p] for p in moment.photo_ids if p in descriptions_of]
        moment.issues = _issues(found, moment.measures)


def link_sentences(sentences: list[RoteiroSentence], moments: list[Moment]) -> None:
    """Liga cada frase aos momentos que ela atravessa (mesmo clipe, intervalos que se sobrepõem)."""
    by_clip: dict[str, list[Moment]] = {}
    for moment in moments:
        by_clip.setdefault(moment.clip_id, []).append(moment)
    for sentence in sentences:
        sentence.moment_ids = []
        for moment in by_clip.get(sentence.clip_id, []):
            overlaps = sentence.start_sec < moment.end_sec and sentence.end_sec > moment.start_sec
            if overlaps:
                sentence.moment_ids.append(moment.id)
                moment.sentence_ids.append(sentence.id)
        if not sentence.moment_ids:  # frase além do fim do vídeo (áudio mais longo): o último momento
            clip_moments = by_clip.get(sentence.clip_id, [])
            if clip_moments:
                sentence.moment_ids.append(clip_moments[-1].id)
                clip_moments[-1].sentence_ids.append(sentence.id)


def moment_rules() -> dict[str, float]:
    return {"moment_min_common_tags": MOMENT_MIN_COMMON_TAGS, "max_moment_sec": MAX_MOMENT_SEC}
