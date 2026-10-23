"""Legendas de vários clipes em sequência (espinha falada montada numa Timeline)."""

from typing import Optional

from pydantic import BaseModel

from ai_video_editor.planning.schema import TimeRange
from ai_video_editor.subtitles.schema import SubtitleBlock, SubtitleParams
from ai_video_editor.subtitles.srt import build_subtitle_blocks
from ai_video_editor.transcription.schema import Transcript


class ClipSubtitleSource(BaseModel):
    """Fala de um clipe e onde os trechos dele começam no vídeo final."""

    transcript: Transcript
    kept_ranges: list[TimeRange]  # trechos do clipe na ordem da timeline (tempos do clipe)
    offset_sec: float  # início do primeiro trecho do clipe no vídeo final
    # Início de cada trecho no vídeo final, quando os trechos do clipe não estão colados (respiros da
    # cobertura entre eles); None = trechos em sequência a partir de `offset_sec`.
    range_starts_sec: Optional[list[float]] = None


def build_clip_subtitle_blocks(
    clips: list[ClipSubtitleSource], fps: float, params: SubtitleParams | None = None
) -> list[SubtitleBlock]:
    """Blocos de legenda do vídeo final, clipe a clipe.

    Cada clipe usa a regra de um vídeo só (`build_subtitle_blocks`: palavra vai para o trecho
    onde está a maior parte dela, bloco nunca atravessa um corte), com os mesmos frames da
    Timeline, e é deslocado para onde o clipe começa no vídeo final. Como os trechos de um
    clipe terminam onde o próximo clipe começa, os blocos não se sobrepõem. Com `range_starts_sec`,
    cada bloco vai para onde o seu trecho está (um bloco nunca atravessa um corte, então nunca
    atravessa um respiro inserido entre dois trechos).
    """
    blocks: list[SubtitleBlock] = []
    for clip in clips:
        if not clip.kept_ranges:
            continue
        concat = []  # início de cada trecho com os trechos do clipe em sequência (frames inteiros, como o render)
        position = 0
        for r in clip.kept_ranges:
            concat.append(position / fps)
            position += max(1, round((r.end_sec - r.start_sec) * fps))
        for block in build_subtitle_blocks(clip.transcript, clip.kept_ranges, fps, params):
            offset = clip.offset_sec
            if clip.range_starts_sec is not None:
                k = max(i for i, start in enumerate(concat) if start <= block.start_sec + 1e-6)
                offset = clip.range_starts_sec[k] - concat[k]
            blocks.append(
                block.model_copy(
                    update={
                        "index": len(blocks) + 1,
                        "start_sec": block.start_sec + offset,
                        "end_sec": block.end_sec + offset,
                    }
                )
            )
    return blocks
