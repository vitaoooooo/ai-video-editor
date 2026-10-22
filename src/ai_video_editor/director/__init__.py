"""Diretor de IA: decide quais frases da transcrição remover, referenciando só IDs."""

from ai_video_editor.director.provider import EditDirector
from ai_video_editor.director.schema import (
    DirectorClip,
    DirectorContext,
    DirectorDecision,
    DirectorRun,
    Sentence,
    SentenceParams,
    SentenceRemoval,
    UserBrief,
)
from ai_video_editor.director.sentences import build_clip_sentences, build_sentences
from ai_video_editor.director.service import run_director, validate_decision

__all__ = [
    "DirectorClip",
    "DirectorContext",
    "DirectorDecision",
    "DirectorRun",
    "EditDirector",
    "Sentence",
    "SentenceParams",
    "SentenceRemoval",
    "UserBrief",
    "build_clip_sentences",
    "build_sentences",
    "run_director",
    "validate_decision",
]
