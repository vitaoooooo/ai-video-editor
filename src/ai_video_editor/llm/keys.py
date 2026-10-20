"""Chaves de API: variável de ambiente ou `.env` local. Nunca são logadas nem impressas."""

import os
from pathlib import Path
from typing import Optional

from ai_video_editor.intelligence.exceptions import AIAuthenticationError


def find_api_key(env_var: str) -> Optional[str]:
    """Valor de `env_var` no ambiente ou no `.env` da pasta atual; None se não houver."""
    key = os.environ.get(env_var)
    if key and key.strip():
        return key.strip()

    env_file = Path(".env")
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass
    return None


def resolve_api_key(env_var: str, provider: str) -> str:
    """Como `find_api_key`, mas lança `AIAuthenticationError` se a chave não existir."""
    key = find_api_key(env_var)
    if key is None:
        raise AIAuthenticationError(provider, details=f"Variável de ambiente {env_var} não encontrada.")
    return key
