"""E12: o plano em IDs vira a Timeline (função pura, sem FFmpeg e sem decisão editorial).

- Faixa de frases (`de`/`ate`): o mesmo corte do `edit` (`pipeline.cutting.plan_keep_ranges`: silêncios do
  clipe + todas as outras frases do clipe removidas, cortes levados para dentro dos silêncios, sem pedaço de
  palavra), limitado à janela da faixa (no máximo `EDGE_SEC` antes da 1ª e depois da última frase, nunca
  dentro das frases vizinhas). Sem picote: dentro da faixa, só as pausas de pelo menos
  `INTERNAL_PAUSE_MIN_SEC` são cortadas (as menores ficam) e um pedaço com menos de `MIN_PIECE_SEC` é unido
  ao vizinho (`merge_pieces`; o `plan_keep_ranges` compartilhado com o `edit` não muda). Cada pedaço vira um
  item de `video_main` com o som do clipe, rotulado só com as frases que caem nele; os micro fades são os da
  Timeline (`audio_fade_sec`).
- Momento incluído (2b, só momento sem fala, candidato do próprio bloco): um item da base com o som da
  própria imagem; a duração pedida, limitada ao momento (sem pedido, o momento inteiro).
- Respiro (2b, candidato do acervo): o mesmo, entre os trechos, com o som mais baixo (rótulo `respiro ...`),
  a partir de `KeepRange.start_sec` (o trecho centrado no momento, já limitado pela validação).
- Cobertura (2c): `video_overlay` do início da 1ª frase coberta ao fim da última, na Timeline final (a
  fala continua); vários momentos dividem o tempo na ordem dada, o mais igual possível sem passar da duração de
  cada um (`coverage.assemble.distribute`); se não bastam, o fim fica sem cobertura (aviso) — nunca estica
  nem repete. O som da imagem entra baixinho em `audio_ambient` (momento sem fala, clipe com áudio).
  Início ou fim a menos de `COVERAGE_SNAP_SEC` da borda do item da base vai para a borda: a sobra seria um
  pedaço de poucos frames, que o render encadeia mal (regra técnica, não editorial).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ai_video_editor.audio import AudioAnalysisResult
from ai_video_editor.core.schema import LoudnessTarget
from ai_video_editor.core.timeline import AmbientItem, MainItem, OverlayItem, Timeline
from ai_video_editor.coverage.assemble import AMBIENT_FADE_SEC, COVERAGE_AMBIENT_VOLUME, BREATH_VOLUME, distribute
from ai_video_editor.director.schema import Sentence
from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.schema import (
    EditorPlan,
    EditorRun,
    KeepRange,
    PlanBlock,
    PlanCoverage,
    PlanPiece,
)
from ai_video_editor.pipeline.cutting import plan_keep_ranges
from ai_video_editor.planning import NoSpeechDetectedError, PlanningError, SilenceCutParams, TimeRange
from ai_video_editor.planning.timeline_plan import (
    BREATH_LABEL_PREFIX,
    SPINE_RESOLUTION,
    Placement,
    map_source_range,
    spine_fps,
)
from ai_video_editor.subtitles import ClipSubtitleSource
from ai_video_editor.transcription.schema import Transcript

# Folga máxima da faixa antes da 1ª e depois da última frase (a do corte de silêncio já vem dentro disso).
EDGE_SEC = 0.5
# Sobra máxima entre a cobertura e a borda do item da base que vira cobertura (evita pedaço de poucos frames).
COVERAGE_SNAP_SEC = 0.5
# Dentro de uma faixa de frases mantidas, só pausas (o que o corte tiraria entre dois pedaços) de pelo menos
# isto são cortadas; as menores ficam (medido no vlog real: pedaços de 0,1 a 0,6 s, frase picotada em 22).
INTERNAL_PAUSE_MIN_SEC = 1.0
# Pedaço de faixa mais curto que isto é unido ao vizinho (o da pausa menor).
MIN_PIECE_SEC = 0.6
# Uma frase "cai" num pedaço quando a sobreposição passa disto (o rótulo do pedaço lista só essas frases).
SENTENCE_OVERLAP_SEC = 0.05


@dataclass
class ClipSource:
    """O arquivo de um clipe e o que a montagem precisa dele."""

    clip_id: str
    path: Path
    fps: float
    duration_sec: float
    has_audio: bool = True
    audio: Optional[AudioAnalysisResult] = None  # silêncios (clipes com fala)
    transcript: Optional[Transcript] = None


@dataclass
class _Entry:
    """Item da base antes de virar `MainItem` (com a origem para o plano)."""

    kind: str  # frases, momento, respiro
    clip_id: str
    ids: list[str]
    source_in: float
    source_out: float
    unit_id: str
    block_id: str
    label: str
    use_audio: bool = True
    volume: float = 1.0
    start_frame: int = 0
    frames: int = 0


@dataclass
class Assembled:
    timeline: Timeline
    plan: EditorPlan
    warnings: list[str] = field(default_factory=list)


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


def faixa_ranges(
    keep: KeepRange, material: EditorMaterial, source: ClipSource, cut: SilenceCutParams
) -> list[TimeRange]:
    """Trechos da faixa no clipe: o corte do `edit` com as outras frases do clipe removidas, na janela da faixa."""
    clip_sentences = material.clip_sentences[keep.clip_id]
    wanted = set(keep.sentence_ids)
    first = material.sentences[keep.sentence_ids[0]]
    last = material.sentences[keep.sentence_ids[-1]]
    before = [s for s in clip_sentences if s.index < first.index]
    after = [s for s in clip_sentences if s.index > last.index]
    lo = first.start_sec - EDGE_SEC
    if before:
        lo = max(lo, min(before[-1].end_sec, first.start_sec))
    hi = last.end_sec + EDGE_SEC
    if after:
        hi = min(hi, max(after[0].start_sec, last.end_sec))
    lo, hi = max(0.0, lo), min(source.duration_sec, hi)
    if source.audio is None:
        start = max(lo, first.start_sec - cut.padding_before_sec)
        return [TimeRange(start_sec=round(start, 3), end_sec=round(min(hi, last.end_sec + cut.padding_after_sec), 3))]
    sentences = [
        Sentence(id=s.index, start_sec=s.start_sec, end_sec=s.end_sec, text=s.text, segment_ids=[0], clip_id=s.clip_id)
        for s in clip_sentences
    ]
    removed = {s.index for s in clip_sentences if s.id not in wanted}
    try:
        ranges, _ = plan_keep_ranges(source.audio, sentences, removed, cut)
    except NoSpeechDetectedError:
        return []
    result = []
    for r in ranges:
        a, b = max(r.start_sec, lo), min(r.end_sec, hi)
        if b - a > 0.001:
            result.append(TimeRange(start_sec=round(a, 3), end_sec=round(b, 3)))
    return merge_pieces(result, cut.padding_before_sec + cut.padding_after_sec)


def merge_pieces(ranges: list[TimeRange], padding_sec: float = 0.0) -> list[TimeRange]:
    """Pedaços de UMA faixa (frases seguidas do mesmo clipe, sem frase removida no meio): pausa entre dois
    pedaços menor que `INTERNAL_PAUSE_MIN_SEC` fica (os pedaços se juntam); depois, cada pedaço com menos de
    `MIN_PIECE_SEC` é unido ao vizinho mais perto (a pausa entre eles volta).

    A pausa é o silêncio entre as falas: o intervalo entre os pedaços mais as folgas que o corte manteve em
    volta da fala (`padding_sec` = folga antes + depois)."""
    merged: list[TimeRange] = []
    for r in sorted(ranges, key=lambda r: r.start_sec):
        if merged and r.start_sec - merged[-1].end_sec + padding_sec < INTERNAL_PAUSE_MIN_SEC - 1e-6:
            merged[-1] = TimeRange(start_sec=merged[-1].start_sec, end_sec=max(merged[-1].end_sec, r.end_sec))
        else:
            merged.append(r)
    while len(merged) > 1:
        short = next((k for k, r in enumerate(merged) if r.end_sec - r.start_sec < MIN_PIECE_SEC), None)
        if short is None:
            break
        gap_before = merged[short].start_sec - merged[short - 1].end_sec if short > 0 else None
        gap_after = merged[short + 1].start_sec - merged[short].end_sec if short + 1 < len(merged) else None
        other = short - 1 if gap_after is None or (gap_before is not None and gap_before <= gap_after) else short + 1
        a, b = sorted((short, other))
        merged[a:b + 1] = [TimeRange(start_sec=merged[a].start_sec, end_sec=merged[b].end_sec)]
    return merged


def assemble_plan(
    run: EditorRun,
    material: EditorMaterial,
    sources: dict[str, ClipSource],
    cut: SilenceCutParams,
    output_path: Path,
    title: str,
    loudness: Optional[LoudnessTarget] = None,
    project_name: str = "",
) -> Assembled:
    """Timeline final e o plano (IDs + tempos).

    Raises:
        PlanningError: nada sobrou para a base, ou a Timeline montada é inválida.
    """
    structure = run.structure
    warnings: list[str] = []
    used_clips = [c for d in run.details for c in d.clip_ids]
    used_clips += [m.split(".")[0] for d in run.details for c in d.coverages for m in c.moment_ids]
    fps = spine_fps([sources[c].fps for c in dict.fromkeys(used_clips) if c in sources])

    entries: list[_Entry] = []
    for detail in run.details:
        for keep in detail.keep:
            if keep.kind == "momento":
                moment = material.moments[keep.moment_id]
                start = moment.start_sec if keep.start_sec is None else min(max(keep.start_sec, moment.start_sec), moment.end_sec)
                length = moment.end_sec - start if keep.duration_sec is None else min(keep.duration_sec, moment.end_sec - start)
                has_audio = sources[moment.clip_id].has_audio
                entries.append(_Entry(
                    kind="respiro" if keep.breath else "momento", clip_id=moment.clip_id, ids=[moment.id],
                    source_in=start, source_out=round(start + length, 3), unit_id=detail.unit_id,
                    block_id=detail.block_id,
                    label=f"{BREATH_LABEL_PREFIX}{moment.id}" if keep.breath else f"momento {moment.id}",
                    use_audio=has_audio, volume=BREATH_VOLUME if keep.breath else 1.0,
                ))
                continue
            ranges = faixa_ranges(keep, material, sources[keep.clip_id], cut)
            if not ranges:
                warnings.append(f"{detail.unit_id}: {keep.from_sentence}–{keep.to_sentence} sem trecho de fala depois do corte")
            entries += [_faixa_entry(keep, r, detail.unit_id, detail.block_id, material) for r in ranges]

    position = 0
    base: list[_Entry] = []
    for entry in entries:
        frames = round((entry.source_out - entry.source_in) * fps)
        if frames < 1:
            warnings.append(f"{entry.label}: menos de 1 frame, fora da montagem")
            continue
        entry.start_frame, entry.frames = position, frames
        position += frames
        base.append(entry)
    if not base:
        raise PlanningError("nada sobrou para a faixa principal do vídeo (todas as faixas e momentos vazios)")

    overlays: list[OverlayItem] = []
    ambient: list[AmbientItem] = []
    coverages: dict[str, list[PlanCoverage]] = {}
    for detail in run.details:
        for n, coverage in enumerate(detail.coverages, start=1):
            placed = _place_coverage(coverage, detail.unit_id, n, base, material, sources, fps, overlays, ambient, warnings)
            if placed is not None:
                coverages.setdefault(detail.block_id, []).append(placed)

    main = [
        MainItem(
            source_path=sources[e.clip_id].path, source_in=e.source_in, source_out=e.source_out, clip_id=e.clip_id,
            label=e.label, use_audio=e.use_audio, volume=e.volume,
        )
        for e in base
    ]
    try:
        timeline = Timeline(
            title=title, output_path=output_path, resolution=SPINE_RESOLUTION, fps=fps, loudness=loudness,
            video_main=main, video_overlay=overlays, audio_ambient=ambient,
        )
    except ValueError as exc:
        raise PlanningError(f"a Timeline montada do plano é inválida: {exc}") from exc

    blocks: list[PlanBlock] = []
    origins = {"ia": 0, "cache": 0, "fallback": 2}
    for block in structure.blocks:
        details = [d for d in run.details if d.block_id == block.id]
        pieces = [_piece(e, fps) for e in base if e.block_id == block.id]
        worst = max((d.origin for d in details), key=lambda o: origins.get(o, 0), default="ia")
        blocks.append(PlanBlock(
            id=block.id, title=block.title, clip_ids=block.clip_ids, origin=worst,
            timeline_start=pieces[0].timeline_start if pieces else 0.0,
            timeline_end=pieces[-1].timeline_end if pieces else 0.0,
            pieces=pieces, coverages=coverages.get(block.id, []),
            notes=[f"{d.unit_id}: {d.notes}" for d in details if d.notes],
        ))
    plan = EditorPlan(
        project_name=project_name, fps=fps, resolution=SPINE_RESOLUTION, duration_sec=round(timeline.duration_sec, 3),
        target_sec=structure.target_total_sec, blocks=blocks, out=structure.out,
        library_clip_ids=structure.library_clip_ids, warnings=warnings,
    )
    return Assembled(timeline=timeline, plan=plan, warnings=warnings)


def piece_sentences(keep: KeepRange, r: TimeRange, material: EditorMaterial) -> list[str]:
    """Frases da faixa que caem no pedaço (sobreposição > `SENTENCE_OVERLAP_SEC`); nenhuma = a mais perto."""
    def overlap(sid: str) -> float:
        sentence = material.sentences[sid]
        return min(r.end_sec, sentence.end_sec) - max(r.start_sec, sentence.start_sec)

    inside = [sid for sid in keep.sentence_ids if overlap(sid) > SENTENCE_OVERLAP_SEC]
    return inside or [max(keep.sentence_ids, key=overlap)]


def _faixa_entry(keep: KeepRange, r: TimeRange, unit_id: str, block_id: str, material: EditorMaterial) -> _Entry:
    """Pedaço de uma faixa; `ids` e o rótulo = a 1ª e a última frase que caem NESTE pedaço."""
    inside = piece_sentences(keep, r, material)
    return _Entry(
        kind="frases", clip_id=keep.clip_id, ids=[inside[0], inside[-1]], source_in=r.start_sec,
        source_out=r.end_sec, unit_id=unit_id, block_id=block_id, label=f"{keep.clip_id} {inside[0]}–{inside[-1]}",
    )


def _piece(entry: _Entry, fps: int) -> PlanPiece:
    return PlanPiece(
        kind=entry.kind, clip_id=entry.clip_id, ids=list(dict.fromkeys(entry.ids)), source_in=entry.source_in,
        source_out=entry.source_out, timeline_start=round(entry.start_frame / fps, 3),
        timeline_end=round((entry.start_frame + entry.frames) / fps, 3),
    )


def _snap_to_items(a: int, b: int, base: list[_Entry], snap: int) -> tuple[int, int]:
    """Leva o início/fim da cobertura (frames) para a borda do item da base quando a sobra é menor que `snap`."""
    for e in base:
        start, end = e.start_frame, e.start_frame + e.frames
        if start < a < end and a - start <= snap:
            a = start
        if start < b < end and end - b <= snap:
            b = end
    return a, b


def _place_coverage(coverage, unit_id, n, base, material, sources, fps, overlays, ambient, warnings) -> Optional[PlanCoverage]:
    label = f"cobertura {unit_id}.C{n} ({coverage.from_sentence}–{coverage.to_sentence})"
    placements = [
        Placement(k, e.source_in, e.source_out, e.start_frame, e.frames, 0)
        for k, e in enumerate(base)
        if e.kind == "frases" and e.clip_id == coverage.clip_id and e.unit_id == unit_id
    ]
    first, last = material.sentences[coverage.from_sentence], material.sentences[coverage.to_sentence]
    mapped = map_source_range(placements, first.start_sec, last.end_sec, fps)
    if mapped is None:
        warnings.append(f"{label}: as frases cobertas não ficaram no vídeo; cobertura ignorada")
        return None
    a, b = _snap_to_items(*mapped[:2], base, round(COVERAGE_SNAP_SEC * fps))
    moments = [material.moments[m] for m in coverage.moment_ids]
    caps = [max(0, int(m.duration * fps + 1e-6)) for m in moments]
    alloc = distribute(b - a, caps)
    cursor = a
    pieces: list[PlanPiece] = []
    for moment, frames in zip(moments, alloc):
        if frames < 1:
            continue
        source = sources[moment.clip_id]
        source_out = round(moment.start_sec + frames / fps, 6)
        overlays.append(OverlayItem(
            source_path=source.path, source_in=moment.start_sec, source_out=source_out, timeline_start=round(cursor / fps, 6),
            clip_id=moment.clip_id, label=f"{label} {moment.id}",
        ))
        if source.has_audio and not moment.sentence_ids:
            fade = min(AMBIENT_FADE_SEC, frames / fps / 4)
            ambient.append(AmbientItem(
                source_path=source.path, source_in=moment.start_sec, source_out=source_out,
                timeline_start=round(cursor / fps, 6), volume=COVERAGE_AMBIENT_VOLUME, fade_in_sec=fade, fade_out_sec=fade,
                clip_id=moment.clip_id, label=f"ambiente {moment.id}",
            ))
        pieces.append(PlanPiece(
            kind="cobertura", clip_id=moment.clip_id, ids=[moment.id], source_in=moment.start_sec,
            source_out=round(source_out, 3), timeline_start=round(cursor / fps, 3), timeline_end=round((cursor + frames) / fps, 3),
        ))
        cursor += frames
    uncovered = (b - cursor) / fps
    if cursor < b:
        warnings.append(f"{label}: os últimos {_fmt(uncovered)} ficam sem cobertura (os momentos são mais curtos)")
    return PlanCoverage(
        from_sentence=coverage.from_sentence, to_sentence=coverage.to_sentence, moment_ids=coverage.moment_ids,
        speech_evidence=coverage.speech_evidence, image_evidence=coverage.image_evidence, reason=coverage.reason,
        timeline_start=round(a / fps, 3), timeline_end=round(b / fps, 3), pieces=pieces,
        uncovered_sec=round(max(0.0, uncovered), 3),
    )


def subtitle_sources(timeline: Timeline, sources: dict[str, ClipSource]) -> list[ClipSubtitleSource]:
    """Uma fonte de legenda por item da base com transcrição (um bloco nunca atravessa um corte)."""
    result = []
    position = 0
    for item in timeline.video_main:
        frames = timeline.item_frames(item)
        source = sources.get(item.clip_id or "")
        if source is not None and source.transcript is not None and item.use_audio:
            result.append(ClipSubtitleSource(
                transcript=source.transcript,
                kept_ranges=[TimeRange(start_sec=item.source_in, end_sec=item.source_out)],
                offset_sec=position / timeline.fps,
            ))
        position += frames
    return result
