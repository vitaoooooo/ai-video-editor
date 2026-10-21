"""Catálogo de um projeto com clipes: papéis, IDs, dados técnicos, datas, fala e miniaturas."""

from ai_video_editor.catalog.builder import CATALOG_STAGE_LABELS, CATALOG_STAGES, CatalogBuilder
from ai_video_editor.catalog.schema import CATALOG_SCHEMA_VERSION, CatalogClip, ProjectCatalog
from ai_video_editor.catalog.speech import FasterWhisperVAD, MockSpeechDetector, SpeechDetector

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CATALOG_STAGES",
    "CATALOG_STAGE_LABELS",
    "CatalogBuilder",
    "CatalogClip",
    "FasterWhisperVAD",
    "MockSpeechDetector",
    "ProjectCatalog",
    "SpeechDetector",
]
