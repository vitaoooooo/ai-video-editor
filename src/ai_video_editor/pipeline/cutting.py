"""Trechos mantidos a partir dos silêncios e das frases removidas (usado por `edit` e `revise`)."""

from typing import Optional

from ai_video_editor.audio import AudioAnalysisResult
from ai_video_editor.director.schema import Sentence
from ai_video_editor.director.sentences import split_by_removal
from ai_video_editor.planning import (
    NoSpeechDetectedError,
    SentenceCutResult,
    SilenceCutParams,
    TimeRange,
    apply_sentence_cuts,
    plan_silence_cuts,
)


def plan_keep_ranges(
    audio: AudioAnalysisResult,
    sentences: list[Sentence],
    removed_ids: Optional[set[int]],
    cut: SilenceCutParams,
) -> tuple[list[TimeRange], list[SentenceCutResult]]:
    """(trechos mantidos, destino de cada corte de frase). `removed_ids=None` = sem Diretor.

    Raises:
        NoSpeechDetectedError: nada sobra (sem fala, ou todas as frases foram removidas).
    """
    keep = plan_silence_cuts(audio, cut)
    results: list[SentenceCutResult] = []
    if removed_ids is not None:
        cuts, kept_sentences = split_by_removal(sentences, removed_ids)
        keep, results = apply_sentence_cuts(keep, cuts, kept_sentences, audio.silences, cut)
        if not keep:
            raise NoSpeechDetectedError(audio.duration_sec, "o Diretor removeu toda a fala")
    return keep, results
