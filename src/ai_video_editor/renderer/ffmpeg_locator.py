"""Utility to locate ffmpeg and ffprobe binaries across environments."""

import os
import shutil
from pathlib import Path
from typing import Optional


def _candidate_paths(binary_name: str) -> list[Path]:
    """Generates candidate file paths for the given binary."""
    ext = ".exe" if os.name == "nt" else ""
    full_name = f"{binary_name}{ext}"

    paths: list[Path] = []

    # 1. Environment variable override
    env_var = f"{binary_name.upper()}_PATH"
    if env_val := os.environ.get(env_var):
        paths.append(Path(env_val))

    # 2. System PATH
    if which_path := shutil.which(binary_name):
        paths.append(Path(which_path))

    # 3. Known fallback locations on Windows
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            paths.append(Path(local_app_data) / "Programs" / "Stremio" / full_name)
            paths.append(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / full_name)

        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        paths.append(Path(program_files) / "ffmpeg" / "bin" / full_name)
        paths.append(Path(r"C:\ProgramData\chocolatey\bin") / full_name)

    return paths


def find_binary(binary_name: str) -> Path:
    """Finds binary or raises FileNotFoundError with diagnostic instructions."""
    candidates = _candidate_paths(binary_name)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    searched_str = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not locate '{binary_name}' executable.\n"
        f"Searched candidates:\n  - {searched_str}\n"
        f"Please install ffmpeg or set {binary_name.upper()}_PATH environment variable."
    )


def get_ffmpeg_path() -> Path:
    """Returns the resolved path to ffmpeg executable."""
    return find_binary("ffmpeg")


def get_ffprobe_path() -> Path:
    """Returns the resolved path to ffprobe executable."""
    return find_binary("ffprobe")
