"""Exceções de domínio para a camada de transcrição."""


class TranscriptionError(Exception):
    """Exceção base para falhas de transcrição."""

    def __init__(self, message: str, provider: str = "", details: str = ""):
        super().__init__(f"[{provider.upper()}] {message}" if provider else message)
        self.provider = provider
        self.details = details

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} Detalhes: {self.details}" if self.details else base


class TranscriptionDependencyError(TranscriptionError):
    """Lançada quando a dependência opcional do provedor não está instalada."""

    def __init__(self, provider: str, extra: str = "transcription"):
        msg = (
            f"Dependência do provedor '{provider}' não instalada. "
            f'Instale com: pip install -e ".[{extra}]"'
        )
        super().__init__(msg, provider=provider)


class TranscriptionMediaError(TranscriptionError):
    """Lançada quando o arquivo de mídia não existe ou não pode ser lido."""

    def __init__(self, provider: str, media_path: object, details: str = ""):
        super().__init__(
            f"Não foi possível ler a mídia '{media_path}'.", provider=provider, details=details
        )


class TranscriptionFailedError(TranscriptionError):
    """Lançada quando o modelo falha durante a transcrição."""

    def __init__(self, provider: str, details: str = ""):
        super().__init__(
            f"Falha durante a transcrição com o provedor '{provider}'.",
            provider=provider,
            details=details,
        )
