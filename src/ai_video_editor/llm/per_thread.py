"""Atributo com um valor por thread, para chamadas à IA em paralelo não misturarem métricas."""

import threading
from typing import Any


class PerThread:
    """Descritor: cada thread lê o que ela mesma gravou (ou o padrão, se ainda não gravou).

    Uso: `last_metrics = PerThread(None)` no corpo da classe. As métricas da "última chamada"
    (cliente, cadeia, etapa) são lidas logo depois da chamada, na mesma thread.
    """

    def __init__(self, default: Any = None):
        self.default = default
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @staticmethod
    def _local(obj: Any) -> threading.local:
        # dict.setdefault é atômico no CPython: duas threads recebem o mesmo `local`.
        return obj.__dict__.setdefault("_per_thread", threading.local())

    def __get__(self, obj: Any, owner: type = None) -> Any:
        if obj is None:
            return self
        return getattr(self._local(obj), self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        setattr(self._local(obj), self.name, value)
