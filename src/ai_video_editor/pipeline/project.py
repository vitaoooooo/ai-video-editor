"""Pasta de projeto: identificação do vídeo, nomes de artefatos e fingerprints de cache."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel

DEFAULT_PROJECTS_ROOT = Path("output/projects")

MANIFEST_FILE = "project.json"
PROBE_FILE = "probe.json"
AUDIO_ANALYSIS_FILE = "audio_analysis.json"
TRANSCRIPT_FILE = "transcript.json"
DIRECTOR_DECISION_FILE = "director_decision.json"
EDIT_PLAN_FILE = "edit_plan.json"
EDIT_REPORT_FILE = "edit_report.md"
EDIT_TRANSCRIPT_FILE = "edit_transcript.md"
EDIT_DIFF_FILE = "edit_diff.json"
EDIT_TIMELINE_FILE = "edit_timeline.svg"
# Projetos com vários clipes (`edit <pasta>`).
SPINE_CLIPS_FILE = "spine_clips.json"
SILENCES_FILE = "silences.json"
TRANSCRIPTION_FILE = "transcription.json"
SPINE_PLAN_FILE = "spine_plan.json"
TIMELINE_FILE = "timeline.json"
# Cache por clipe (silêncios e transcrição), fora do projeto: trocar um clipe muda o projeto, mas não
# refaz os outros clipes. `<projects>/_cache/edit_clips/<hash do arquivo>/<etapa>-<fingerprint>.json`.
CLIP_STAGE_CACHE_DIR = Path("_cache") / "edit_clips"

# Hash por amostragem: lê só alguns blocos, então vídeos de vários GB não pagam leitura completa.
_HASH_CHUNK = 4 * 1024 * 1024


def file_hash(path: Path, length: int = 10) -> str:
    """Hash curto do arquivo: tamanho + blocos do início, meio e fim (SHA-256)."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        if size <= 3 * _HASH_CHUNK:
            digest.update(f.read())
        else:
            for offset in (0, size // 2, size - _HASH_CHUNK):
                f.seek(offset)
                digest.update(f.read(_HASH_CHUNK))
    return digest.hexdigest()[:length]


def slugify(name: str, max_len: int = 40) -> str:
    """Nome seguro para pasta: ASCII, sem espaços nem símbolos."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", ascii_name).strip("-_")
    return (slug[:max_len] or "video").lower()


def project_dir_for(video_path: Path, video_hash: str, root: Path = DEFAULT_PROJECTS_ROOT) -> Path:
    """`<root>/<nome-do-video>-<hash curto>`."""
    return root / f"{slugify(video_path.stem)}-{video_hash}"


def output_video_name(video_path: Path) -> str:
    return f"{slugify(video_path.stem)}_edited.mp4"


def subtitles_file_name(video_path: Path) -> str:
    """`<video>_edited.srt`: mesmo nome base do vídeo final, para os players carregarem sozinhos."""
    return Path(output_video_name(video_path)).with_suffix(".srt").name


def fingerprint(*parts: object) -> str:
    """Hash estável das entradas de uma etapa (modelos Pydantic, dicts, strings, números)."""

    def normalize(value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Path):
            return str(value)
        return value

    payload = json.dumps([normalize(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
