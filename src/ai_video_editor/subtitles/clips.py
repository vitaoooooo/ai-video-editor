"""Legendas de vários clipes em sequência (espinha falada montada numa Timeline)."""

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


def build_clip_subtitle_blocks(
    clips: list[ClipSubtitleSource], fps: float, params: SubtitleParams | None = None
) -> list[SubtitleBlock]:
    """Blocos de legenda do vídeo final, clipe a clipe.

    Cada clipe usa a regra de um vídeo só (`build_subtitle_blocks`: palavra vai para o trecho
    onde está a maior parte dela, bloco nunca atravessa um corte), com os mesmos frames da
    Timeline, e é deslocado para onde o clipe começa no vídeo final. Como os trechos de um
    clipe terminam onde o próximo clipe começa, os blocos não se sobrepõem.
    """
    blocks: list[SubtitleBlock] = []
    for clip in clips:
        if not clip.kept_ranges:
            continue
        for block in build_subtitle_blocks(clip.transcript, clip.kept_ranges, fps, params):
            blocks.append(
                block.model_copy(
                    update={
                        "index": len(blocks) + 1,
                        "start_sec": block.start_sec + clip.offset_sec,
                        "end_sec": block.end_sec + clip.offset_sec,
                    }
                )
            )
    return blocks
