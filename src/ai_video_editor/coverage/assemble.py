"""IDs → tempos: a decisão de cobertura aceita vira a Timeline final (função pura, sem FFmpeg).

- Respiros viram itens de `video_main` inseridos na espinha (com o som da própria tomada), o que
  desloca tudo o que vem depois.
- Coberturas viram itens de `video_overlay` do início da primeira frase ao fim da última, na timeline
  final (o áudio da fala continua); o trecho da tomada é centrado no melhor trecho (`best_frames`).
- O som ambiente de cada cobertura entra baixinho em `audio_ambient` (tomadas com áudio e sem fala).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.core.timeline import AmbientItem, MainItem, OverlayItem, Timeline
from ai_video_editor.coverage.schema import (
    CoverageContext,
    CoverageRun,
    CoverageSentence,
    OverlayPiece,
    PlacedBreath,
    PlacedCoverage,
    VisualShot,
)
from ai_video_editor.coverage.service import BREATH_SECONDS, best_free_shot, shot_span
from ai_video_editor.planning.timeline_plan import BREATH_LABEL_PREFIX, clip_placements

# Volume do som da tomada num respiro (é o único som naquele momento).
BREATH_VOLUME = 1.0
# Som ambiente da cobertura, por baixo da fala, com fades curtos.
COVERAGE_AMBIENT_VOLUME = 0.12
AMBIENT_FADE_SEC = 0.3
# Sobra mínima para encadear outra tomada quando a escolhida é curta (menos que isso pisca na tela).
MIN_CHAIN_SEC = 0.75


@dataclass
class AssembledCoverage:
    timeline: Timeline
    coverages: list[PlacedCoverage] = field(default_factory=list)
    breaths: list[PlacedBreath] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


def shot_segment(shot: VisualShot, frames: int, fps: int) -> tuple[float, int]:
    """(início na origem, frames) de um trecho da tomada com até `frames`, centrado no melhor trecho."""
    lo, hi = shot_span(shot)
    available = int(math.floor((hi - lo) * fps + 1e-6))
    frames = max(0, min(frames, available))
    length = frames / fps
    times = [shot.frame_times[f] for f in shot.best_frames if f in shot.frame_times]
    center = (min(times) + max(times)) / 2 if times else (lo + hi) / 2
    start = min(max(center - length / 2, lo), hi - length)
    return round(start, 6), frames


def _insertion_frame(spine: Timeline, sentences: list[CoverageSentence], k: int) -> int:
    """Onde entra um respiro depois da frase `k` (frame da espinha).

    No primeiro corte entre o fim da frase e o início da próxima do mesmo clipe (o corte de silêncio
    já deixou a folga); sem corte ali, no meio do caminho; na última frase do clipe, no fim do clipe.
    """
    sentence = sentences[k]
    ends = [p.start_frame + p.frames for p in clip_placements(spine)[sentence.clip_id]]
    following = sentences[k + 1] if k + 1 < len(sentences) and sentences[k + 1].clip_id == sentence.clip_id else None
    if following is None:
        return max(ends)
    cuts = [e for e in ends if sentence.end_frame <= e <= following.start_frame]
    return min(cuts) if cuts else (sentence.end_frame + following.start_frame) // 2


def _split(item: MainItem, frames: int, fps: int) -> tuple[MainItem, MainItem]:
    cut = item.source_in + frames / fps
    data = item.model_dump()
    return MainItem(**{**data, "source_out": cut}), MainItem(**{**data, "source_in": cut})


def assemble_coverage(
    spine: Timeline,
    context: CoverageContext,
    run: CoverageRun,
    shots: dict[str, VisualShot],
    title: Optional[str] = None,
) -> AssembledCoverage:
    """Timeline final (espinha + respiros + coberturas + ambiente) e onde cada item foi parar."""
    fps = spine.fps
    sentences = context.sentences
    order = {s.id: i for i, s in enumerate(sentences)}
    used = {c.shot_id for c in run.coverages} | {b.shot_id for b in run.breaths}
    warnings: list[str] = []

    # 1. Respiros: posição na espinha e o item da base de cada um.
    insertions: list[tuple[int, int, MainItem, PlacedBreath]] = []
    for n, breath in enumerate(sorted(run.breaths, key=lambda b: order[b.after_sentence]), start=1):
        k = order[breath.after_sentence]
        shot = shots[breath.shot_id]
        start, frames = shot_segment(shot, round(BREATH_SECONDS[breath.size] * fps), fps)
        audio = shot.has_audio and shot.has_speech is not True
        item = MainItem(
            source_path=shot.source_path, source_in=start, source_out=start + frames / fps, clip_id=shot.clip_id,
            label=f"{BREATH_LABEL_PREFIX}R{n} ({shot.id})", use_audio=audio, volume=BREATH_VOLUME,
        )
        placed = PlacedBreath(
            id=f"R{n}", after_sentence=breath.after_sentence, shot_id=shot.id, clip_id=shot.clip_id, size=breath.size,
            reason=breath.reason, origin=breath.origin, source_in=item.source_in, source_out=round(item.source_out, 6),
            timeline_start=0.0, duration_sec=round(frames / fps, 3), audio=audio,
        )
        insertions.append((_insertion_frame(spine, sentences, k), k, item, placed))
    insertions.sort(key=lambda x: (x[0], x[1]))

    # 2. Base nova: itens da espinha divididos onde um respiro entra no meio deles.
    base: list[MainItem] = []
    position, q = 0, 0
    for item in spine.video_main:
        end = position + spine.item_frames(item)
        current, cursor = item, position
        while q < len(insertions) and insertions[q][0] < end:
            at = insertions[q][0]
            if at > cursor:
                head, current = _split(current, at - cursor, fps)
                base.append(head)
                cursor = at
            base.append(insertions[q][2])
            q += 1
        base.append(current)
        position = end
    base += [x[2] for x in insertions[q:]]

    breaths: list[PlacedBreath] = []
    shifted = 0
    for at, _, item, placed in insertions:
        breaths.append(placed.model_copy(update={"timeline_start": round((at + shifted) / fps, 3)}))
        shifted += spine.item_frames(item)

    def to_final(frame: int, end: bool) -> int:
        return frame + sum(spine.item_frames(x[2]) for x in insertions if (x[0] < frame if end else x[0] <= frame))

    # 3. Coberturas: janela final de cada uma e os trechos de tomada (encadeando se a escolhida for curta).
    in_window = {cid for cids in run.window_cover.values() for cid in cids}
    overlays: list[OverlayItem] = []
    ambient: list[AmbientItem] = []
    placed_coverages: list[PlacedCoverage] = []
    for n, coverage in enumerate(run.coverages, start=1):
        cid = f"C{n}"
        i, j = order[coverage.from_sentence], order[coverage.to_sentence]
        a, b = to_final(sentences[i].start_frame, False), to_final(sentences[j].end_frame, True)
        speaking = {sentences[x].clip_id for x in range(i, j + 1)}
        windows = [w for w, cids in run.window_cover.items() if cid in cids]
        pieces: list[OverlayPiece] = []
        shot, cursor, chained = shots[coverage.shot_id], a, False
        while shot is not None and cursor < b:
            start, frames = shot_segment(shot, b - cursor, fps)
            if frames < 1:
                break
            pieces.append(
                OverlayPiece(
                    shot_id=shot.id, clip_id=shot.clip_id, source_in=start, source_out=round(start + frames / fps, 6),
                    timeline_start=round(cursor / fps, 6), timeline_end=round((cursor + frames) / fps, 6), chained=chained,
                )
            )
            cursor += frames
            left = (b - cursor) / fps
            if cursor >= b:
                break
            if left < MIN_CHAIN_SEC:
                warnings.append(f"{cid}: a tomada {shot.id} é curta; os últimos {_fmt(left)} ficam sem cobertura")
                break
            if cid not in in_window:
                warnings.append(f"{cid} encurtada: a tomada {shot.id} cobre {_fmt((cursor - a) / fps)} de {_fmt((b - a) / fps)}")
                break
            nxt = best_free_shot(context.candidates, left, speaking, used)
            if nxt is None:
                warnings.append(f"{cid}: sem tomada livre para completar {_fmt(left)} da janela obrigatória")
                break
            warnings.append(f"{cid}: a tomada {shot.id} é curta; completada com {nxt.id} ({_fmt(left)})")
            used.add(nxt.id)
            shot, chained = nxt, True
        for piece in pieces:
            source = shots[piece.shot_id]
            overlays.append(
                OverlayItem(
                    source_path=source.source_path, source_in=piece.source_in, source_out=piece.source_out,
                    timeline_start=piece.timeline_start, clip_id=source.clip_id, label=f"cobertura {cid} ({piece.shot_id})",
                )
            )
            if source.has_audio and source.has_speech is not True:
                duration = piece.source_out - piece.source_in
                fade = min(AMBIENT_FADE_SEC, duration / 4)
                ambient.append(
                    AmbientItem(
                        source_path=source.source_path, source_in=piece.source_in, source_out=piece.source_out,
                        timeline_start=piece.timeline_start, volume=COVERAGE_AMBIENT_VOLUME,
                        fade_in_sec=fade, fade_out_sec=fade, clip_id=source.clip_id, label=f"ambiente {cid}",
                    )
                )
        placed_coverages.append(
            PlacedCoverage(
                id=cid, from_sentence=coverage.from_sentence, to_sentence=coverage.to_sentence, shot_id=coverage.shot_id,
                reason=coverage.reason, origin=coverage.origin, windows=windows,
                timeline_start=round(a / fps, 3), timeline_end=round(b / fps, 3), pieces=pieces,
            )
        )

    data = spine.model_dump(exclude={"video_main", "video_overlay", "audio_ambient"})
    timeline = Timeline.model_validate(
        {
            **data,
            "title": title or spine.title,
            "video_main": [x.model_dump() for x in base],
            "video_overlay": [x.model_dump() for x in overlays],
            "audio_ambient": [x.model_dump() for x in ambient],
        }
    )
    return AssembledCoverage(timeline=timeline, coverages=placed_coverages, breaths=breaths, warnings=warnings)
