"""Exceções de domínio do planejamento de cortes."""


class PlanningError(Exception):
    """Exceção base para falhas ao montar um plano de edição."""


class NoSpeechDetectedError(PlanningError):
    """Lançada quando não sobra nenhum trecho de fala para manter."""

    def __init__(self, duration_sec: float, details: str = ""):
        msg = (
            f"Nenhum trecho de fala detectado no vídeo ({duration_sec:.2f}s): "
            "tudo seria cortado. Tente um threshold mais baixo (ex: -45 dB) "
            "ou um min_silence_sec maior."
        )
        if details:
            msg += f" ({details})"
        super().__init__(msg)
        self.duration_sec = duration_sec
