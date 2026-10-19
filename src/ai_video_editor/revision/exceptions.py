"""Exceções do ciclo de revisão."""


class RevisionError(Exception):
    """Base das falhas de revisão que não criam versão."""


class RevisionRejectedError(RevisionError):
    """Pedido impossível, ambíguo, sem efeito ou respondido de forma inválida pelo revisor."""


class ProjectStateError(RevisionError):
    """A pasta não é um projeto revisável (sem edição concluída, transcrição ou versão pedida)."""
