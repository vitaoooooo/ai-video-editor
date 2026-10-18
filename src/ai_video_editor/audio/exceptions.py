"""Exceções de domínio para extração e análise de áudio."""

from pathlib import Path
from typing import Optional


class AudioAnalysisError(Exception):
    """Exceção base para erros de extração e análise de áudio."""

    def __init__(self, message: str, file_path: Optional[Path] = None):
        super().__init__(message)
        self.file_path = file_path


class MediaNotFoundError(AudioAnalysisError):
    """Lançada quando o arquivo de mídia não existe."""

    def __init__(self, file_path: Path):
        super().__init__(f"Arquivo de mídia não encontrado: {file_path}", file_path=file_path)


class NoAudioStreamError(AudioAnalysisError):
    """Lançada quando a mídia não possui nenhuma faixa de áudio."""

    def __init__(self, file_path: Path):
        super().__init__(f"O arquivo não possui faixa de áudio: {file_path}", file_path=file_path)


class AudioExtractionError(AudioAnalysisError):
    """Lançada quando o FFmpeg falha ao extrair o áudio."""

    def __init__(self, file_path: Path, details: str = ""):
        msg = f"Falha ao extrair áudio de '{file_path}'"
        if details:
            msg += f": {details}"
        super().__init__(msg, file_path=file_path)


class SilenceDetectionError(AudioAnalysisError):
    """Lançada quando a detecção de silêncio falha ou tem parâmetros inválidos."""

    def __init__(self, file_path: Optional[Path], details: str = ""):
        msg = "Falha na detecção de silêncio"
        if file_path is not None:
            msg += f" em '{file_path}'"
        if details:
            msg += f": {details}"
        super().__init__(msg, file_path=file_path)
