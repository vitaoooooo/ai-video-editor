"""Material da cobertura, sem IA: tomadas do `visual.json`, frases mantidas na espinha e o que está na
tela por baixo de cada uma (e se a pessoa fala para a câmera), trechos com imagem ruim medidos em frames e
tomadas candidatas."""

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ai_video_editor.core.timeline import Timeline
from ai_video_editor.coverage.schema import (
    BadSpan,
    CoverageSentence,
    VisualMaterial,
    VisualShot,
    sentence_label,
)
from ai_video_editor.director.schema import Sentence
from ai_video_editor.planning.timeline_plan import clip_placements, map_source_range

if TYPE_CHECKING:  # `visual` e `catalog` importam o pipeline (ciclo): só para tipos
    from ai_video_editor.catalog.schema import ProjectCatalog
    from ai_video_editor.visual.schema import ProjectVisual

# Papéis que podem ser cobertura. Do principal, só tomadas sem fala (`has_speech=False`).
COVERAGE_ROLES = ("drone", "broll", "pov", "principal")
# Problemas da IA mostrados ao lado do que está na tela (contexto; só os graves viram trecho ruim).
BAD_ISSUES = ("obstruida", "borrao_movimento")
# Como no `visual.md` (`visual/schema.py: ISSUE_LABELS`).
BAD_ISSUE_LABELS = {"obstruida": "obstruída/parede", "borrao_movimento": "borrão de movimento"}
# Margem em volta da imagem ruim (o tapa não começa nem termina em cima do defeito) e o mínimo de imagem
# ruim (parte mantida da fala) para virar um trecho.
BAD_MARGIN_SEC = 0.25
MIN_BAD_OVERLAP_SEC = 0.3


class CoverageError(ValueError):
    """Material da cobertura ausente ou de outro catálogo."""


def missing_visual_message(source_dir: Path, visual_path: Path) -> str:
    return (
        f"análise visual não encontrada ({visual_path}): rode `visual \"{source_dir}\"` antes "
        "(ou use --receita vlog para a espinha sem cobertura)"
    )


def build_visual_material(
    visual: "ProjectVisual", catalog: "ProjectCatalog", visual_path: Path, spine_clip_ids: Iterable[str]
) -> VisualMaterial:
    """Tomadas do `visual.json` com caminho e áudio do catálogo atual.

    Raises:
        CoverageError: o `visual.json` foi feito com outro catálogo (clipe sumiu ou mudou de ID/arquivo).
    """
    by_id = {c.id: c for c in catalog.clips if c.id is not None}
    source_dir = Path(catalog.source_dir)
    stale = [
        clip.clip_id for clip in visual.clips
        if clip.clip_id not in by_id or by_id[clip.clip_id].relative_path != clip.relative_path
    ]
    if stale:
        raise CoverageError(
            f"a análise visual é de outro catálogo (clipes {', '.join(stale[:5])} mudaram ou sumiram): "
            f"rode `visual \"{source_dir}\"` de novo (ou use --receita vlog)"
        )
    shots: list[VisualShot] = []
    for clip in visual.clips:
        entry = by_id[clip.clip_id]
        has_audio = bool(entry.media and entry.media.has_audio)
        recorded = entry.adjusted_at or (entry.recorded_at.value if entry.recorded_at else None)
        reliable = entry.order_source == "data" and entry.recorded_at is not None and entry.recorded_at.reliable
        for n, shot in enumerate(clip.shots):
            analysis = shot.analysis if shot.status == "analisada" else None
            shots.append(
                VisualShot(
                    id=shot.id, clip_id=clip.clip_id, role=clip.role,
                    source_path=(source_dir / entry.relative_path).resolve(),
                    start_sec=shot.start_sec, end_sec=shot.end_sec, last_in_clip=n == len(clip.shots) - 1,
                    frame_times={f.id: f.time_sec for f in shot.frames},
                    has_audio=has_audio, has_speech=shot.has_speech, status=shot.status,
                    uniform=bool(shot.measures and shot.measures.uniform),
                    blurry=bool(shot.measures and shot.measures.blurry),
                    usable=analysis.usable if analysis else None,
                    issues=list(analysis.issues) if analysis else [],
                    description=analysis.description if analysis else "",
                    tags=list(analysis.tags) if analysis else [],
                    beauty=analysis.beauty if analysis else None,
                    face_to_camera=analysis.face_to_camera if analysis else None,
                    best_frames=list(analysis.best_frames) if analysis else [],
                    recorded_at=recorded, date_reliable=reliable,
                )
            )
    warnings = []
    analyzed = {c.clip_id for c in visual.clips}
    missing = [c for c in spine_clip_ids if c not in analyzed]
    if missing:
        warnings.append(
            f"clipes da espinha sem análise visual ({', '.join(missing)}): imagem ruim neles não é detectada"
        )
    return VisualMaterial(visual_path=visual_path, roles=list(visual.roles), shots=shots, warnings=warnings)


def coverage_sentences(sentences: list[Sentence], removed: set[int], spine: Timeline) -> list[CoverageSentence]:
    """Frases mantidas que aparecem na espinha, na ordem do vídeo, com a posição em frames."""
    placements = clip_placements(spine)
    result: list[CoverageSentence] = []
    for sentence in sentences:
        if sentence.id in removed or sentence.clip_id not in placements:
            continue
        mapped = map_source_range(placements[sentence.clip_id], sentence.start_sec, sentence.end_sec, spine.fps)
        if mapped is None:
            continue
        start, end, kept = mapped
        result.append(
            CoverageSentence(
                id=sentence_label(sentence.id), sentence_id=sentence.id, clip_id=sentence.clip_id,
                text=sentence.text, source_start_sec=kept[0][0], source_end_sec=kept[-1][1],
                start_frame=start, end_frame=end, kept=kept,
            )
        )
    return sorted(result, key=lambda s: s.start_frame)


def bad_shot_reasons(shot: VisualShot) -> list[str]:
    """Problemas da imagem desta tomada do principal (vazio = nenhum); contexto para o prompt."""
    reasons = []
    if shot.usable == "nao":
        reasons.append("imagem inutilizável")
    reasons += [BAD_ISSUE_LABELS[i] for i in BAD_ISSUES if i in shot.issues]
    if shot.uniform:
        reasons.append("tela uniforme")
    return reasons


def is_severe(shot: VisualShot) -> bool:
    """Imagem ruim de verdade: tela uniforme (parede, tampa), inutilizável pela IA ou borrão de movimento
    confirmado pela medida `borrão alto`. "Parcial" é a IA dizendo que a imagem serve (a pessoa no ônibus com
    os bancos na frente): não vira trecho ruim."""
    return shot.uniform or shot.usable == "nao" or ("borrao_movimento" in shot.issues and shot.blurry)


def severe_reasons(shot: VisualShot) -> list[str]:
    reasons = []
    if shot.uniform:
        reasons.append("tela uniforme")
    if shot.usable == "nao":
        reasons.append("imagem inutilizável")
    if "borrao_movimento" in shot.issues and shot.blurry:
        reasons.append("borrão de movimento (medido)")
    return reasons


def _overlap(kept: list[tuple[float, float]], start: float, end: float) -> float:
    return sum(max(0.0, min(hi, end) - max(lo, start)) for lo, hi in kept)


def annotate_screen(sentences: list[CoverageSentence], material: VisualMaterial) -> list[VisualShot]:
    """Preenche `screen` de cada frase (tomadas do clipe que fala por baixo da parte mantida, mais tempo
    primeiro) e `to_camera` (a tomada principal tem rosto e é utilizável); devolve essas tomadas, na ordem em
    que aparecem."""
    by_clip: dict[str, list[VisualShot]] = {}
    for shot in material.shots:
        by_clip.setdefault(shot.clip_id, []).append(shot)
    used: dict[str, VisualShot] = {}
    for sentence in sentences:
        hits = [(_overlap(sentence.kept, s.start_sec, s.end_sec), s) for s in by_clip.get(sentence.clip_id, [])]
        hits = sorted((h for h in hits if h[0] > 0), key=lambda h: (-h[0], h[1].start_sec))
        sentence.screen = [s.id for _, s in hits]
        sentence.to_camera = bool(hits) and hits[0][1].face_to_camera is True and hits[0][1].usable == "sim"
        for _, shot in hits:
            used.setdefault(shot.id, shot)
    return list(used.values())


def _spine_frame(sentence: CoverageSentence, t: float, fps: int) -> int:
    """Tempo do clipe dentro da frase → frame da espinha (os cortes no meio da frase não ocupam frames)."""
    offset = sum(max(0.0, min(t, hi) - lo) for lo, hi in sentence.kept)
    return min(sentence.end_frame, sentence.start_frame + round(offset * fps))


def find_bad_spans(
    sentences: list[CoverageSentence], material: VisualMaterial, spine_clip_ids: Iterable[str], fps: int
) -> list[BadSpan]:
    """Trechos com imagem ruim de verdade, em frames da espinha: parte mantida das frases ∩ tomadas graves
    (`is_severe`) do clipe que fala, com `BAD_MARGIN_SEC` de cada lado (sem sair da frase). Pedaços seguidos
    do mesmo clipe e do mesmo tipo (tela uniforme ou não) viram um trecho; trecho com menos de
    `MIN_BAD_OVERLAP_SEC` de imagem ruim é ignorado. Tomadas sem análise da IA contam pela medida `tela uniforme`.
    """
    margin = round(BAD_MARGIN_SEC * fps)
    spine_ids = set(spine_clip_ids)
    severe: dict[str, list[VisualShot]] = {}
    for shot in material.shots:
        if shot.clip_id in spine_ids and is_severe(shot):
            severe.setdefault(shot.clip_id, []).append(shot)

    # (início, fim, frames ruins, clipe, tela uniforme, tomada, frase)
    pieces: list[tuple[int, int, int, str, bool, VisualShot, CoverageSentence]] = []
    for sentence in sentences:
        for shot in severe.get(sentence.clip_id, []):
            for lo, hi in sentence.kept:
                a, b = max(lo, shot.start_sec), min(hi, shot.end_sec)
                if b <= a:
                    continue
                start, end = _spine_frame(sentence, a, fps), _spine_frame(sentence, b, fps)
                if end <= start:
                    continue
                pieces.append((
                    max(sentence.start_frame, start - margin), min(sentence.end_frame, end + margin), end - start,
                    sentence.clip_id, shot.uniform, shot, sentence,
                ))
    pieces.sort(key=lambda p: (p[0], p[1]))

    groups: list[list] = []
    for piece in pieces:
        last = groups[-1] if groups else None
        if last and last[3] == piece[3] and last[4] == piece[4] and piece[0] <= last[1] + 2 * margin:
            last[1] = max(last[1], piece[1])
            last[2] += piece[2]
            last[5].append(piece[5])
            last[6].append(piece[6])
        else:
            groups.append([piece[0], piece[1], piece[2], piece[3], piece[4], [piece[5]], [piece[6]]])

    spans: list[BadSpan] = []
    for start, end, bad_frames, clip_id, uniform, shots, _ in groups:
        if bad_frames < MIN_BAD_OVERLAP_SEC * fps - 1e-9:
            continue
        touched = [s.id for s in sentences if s.start_frame < end and start < s.end_frame]
        shot_ids = list(dict.fromkeys(s.id for s in shots))
        reasons = list(dict.fromkeys(r for s in shots for r in severe_reasons(s)))
        if not uniform:
            reasons = [r for r in reasons if r != "tela uniforme"]
        spans.append(
            BadSpan(
                id=f"O{len(spans) + 1}", clip_id=clip_id, sentence_ids=touched, shot_ids=shot_ids, reasons=reasons,
                uniform=uniform, start_frame=start, end_frame=end,
            )
        )
    return spans


def is_candidate(shot: VisualShot, roles: Iterable[str]) -> bool:
    """Pode ser cobertura: papel permitido, analisada, utilizável e, no principal, sem fala."""
    if shot.role not in roles or shot.status != "analisada" or shot.usable in (None, "nao"):
        return False
    return shot.role != "principal" or shot.has_speech is False


def coverage_roles(roles_to_use: Iterable[str]) -> list[str]:
    """Papéis de cobertura, filtrados pelos `roles_to_use` do briefing (vazio = todos)."""
    wanted = set(roles_to_use)
    return [r for r in COVERAGE_ROLES if not wanted or r in wanted]


def coverage_candidates(material: VisualMaterial, roles: Iterable[str]) -> list[VisualShot]:
    roles = list(roles)
    return [s for s in material.shots if is_candidate(s, roles)]
