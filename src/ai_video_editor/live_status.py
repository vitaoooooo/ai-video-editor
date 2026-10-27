"""Linha de status ao vivo no terminal: tempo decorrido, progresso e tempo restante da etapa em curso.

Só exibe; quem informa o progresso é o pipeline (por callbacks). Usa `\\r` e completa com
espaços para apagar o resto da linha (funciona no cmd e no PowerShell sem sequências ANSI).
Fora de um terminal (ou com `live=False`) não reescreve nada: imprime só as linhas finais.
"""

import shutil
import sys
import threading
import time
from typing import Callable, Optional, TextIO

from ai_video_editor.pipeline.schema import StageProgress
from ai_video_editor.pipeline.timing import format_duration

# Estimativa de tempo restante só a partir desta fração concluída.
MIN_FRACTION_FOR_ETA = 0.10

REFRESH_INTERVAL_SEC = 1.0


def format_media_time(seconds: float) -> str:
    """Posição na mídia: `23min` a partir de 1 minuto, senão `42,0s`."""
    if seconds < 60:
        return format_duration(seconds)
    return f"{int(seconds // 60)}min"


def format_remaining(seconds: float) -> str:
    """`faltam ~3min` ou `faltam ~40s`."""
    if seconds < 60:
        return f"faltam ~{max(1, round(seconds))}s"
    return f"faltam ~{round(seconds / 60)}min"


def describe_progress(progress: StageProgress) -> str:
    """`41% (23min de 57min)`, `118/258 trechos (46%)` ou `12/40 clipes (30%)`; vazio sem progresso mensurável."""
    fraction = progress.fraction
    if fraction is None:
        return ""
    pct = int(fraction * 100)
    if progress.unit in ("items", "clips", "photos", "calls"):
        noun = {"items": "trechos", "clips": "clipes", "photos": "fotos", "calls": "pedidos"}[progress.unit]
        return f"{int(progress.done)}/{int(progress.total)} {noun} ({pct}%)"
    return f"{pct}% ({format_media_time(progress.done)} de {format_media_time(progress.total)})"


def estimate_remaining(elapsed_sec: float, fraction: Optional[float]) -> Optional[float]:
    """Segundos restantes pela taxa média até agora; None antes de 10% concluídos."""
    if fraction is None or fraction < MIN_FRACTION_FOR_ETA or fraction >= 1.0:
        return None
    return elapsed_sec * (1.0 - fraction) / fraction


class LiveStatus:
    """Linha ao vivo da etapa em curso, com avisos impressos acima dela.

    Uso: `begin(rótulo)` → `update(...)`/`message(...)` → `end(linha final)`. A thread de
    atualização nasce no primeiro `begin` e vive até `close()` (ou o fim do `with`), que também
    não deixa a linha pendurada em erro ou Ctrl+C.
    `clock`, `width` e `auto_refresh=False` existem para os testes.
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        live: Optional[bool] = None,
        clock: Callable[[], float] = time.monotonic,
        width: Optional[int] = None,
        auto_refresh: bool = True,
        interval_sec: float = REFRESH_INTERVAL_SEC,
    ):
        self.stream = stream or sys.stdout
        if live is None:
            isatty = getattr(self.stream, "isatty", None)
            live = bool(isatty and isatty())
        self.live = live
        self.clock = clock
        self._width = width
        self.auto_refresh = auto_refresh
        self.interval_sec = interval_sec
        self._lock = threading.RLock()
        self._label: Optional[str] = None
        self._started = 0.0
        self._progress = StageProgress()
        self._shown = 0  # caracteres da linha ao vivo na tela
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "LiveStatus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(interrupted=exc_type is not None and issubclass(exc_type, KeyboardInterrupt))

    @property
    def active(self) -> bool:
        return self._label is not None

    def begin(self, label: str, announce: bool = True) -> None:
        """Inicia a etapa. Fora do modo ao vivo, `announce` imprime `rótulo...` (como sempre foi)."""
        with self._lock:
            if self.active:
                self.end(f"{self._label}: interrompida")
            self._label = label
            self._started = self.clock()
            self._progress = StageProgress()
            if not self.live:
                if announce:
                    self._write_line(f"{label}...")
                return
            self._draw()
        self._start_thread()

    def update(
        self,
        done: Optional[float] = None,
        total: Optional[float] = None,
        unit: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        """Atualiza o progresso; campos None mantêm o valor anterior (a nota sobrevive ao progresso)."""
        with self._lock:
            changes = {
                k: v for k, v in {"done": done, "total": total, "unit": unit, "note": note}.items() if v is not None
            }
            self._progress = self._progress.model_copy(update=changes)

    def apply(self, progress: StageProgress) -> None:
        """Aplica um `StageProgress` vindo do pipeline (nota vazia não apaga a anterior)."""
        self.update(progress.done, progress.total, progress.unit if progress.done is not None else None,
                    progress.note or None)

    def message(self, text: str) -> None:
        """Imprime uma linha permanente; a linha ao vivo volta logo abaixo."""
        with self._lock:
            if self.live and self.active:
                self._clear()
                self._write_line(text)
                self._draw()
            else:
                self._write_line(text)

    def end(self, final_line: str) -> None:
        """Troca a linha ao vivo pela linha final da etapa."""
        with self._lock:
            if self.live and self.active:
                self._clear()
            self._label = None
            self._write_line(final_line)

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self._started) if self.active else 0.0

    def render(self) -> str:
        """Texto da linha ao vivo: `[3/8] Transcrição... 2min 13s — 41% (23min de 57min), faltam ~3min`."""
        with self._lock:
            if not self.active:
                return ""
            elapsed = self.elapsed()
            text = f"{self._label}... {format_duration(elapsed)}"
            details = [describe_progress(self._progress)]
            remaining = estimate_remaining(elapsed, self._progress.fraction)
            if remaining is not None:
                details.append(format_remaining(remaining))
            details.append(self._progress.note)
            details = [d for d in details if d]
            return f"{text} — {', '.join(details)}" if details else text

    def refresh(self) -> None:
        """Redesenha a linha ao vivo (a thread chama a cada `interval_sec`)."""
        with self._lock:
            if self.live and self.active:
                self._draw()

    def close(self, interrupted: bool = False) -> None:
        """Encerra a thread; etapa ainda aberta vira `interrompida` (Ctrl+C) ou `FALHOU`."""
        with self._lock:
            if self.active and self.live:
                status = "interrompida" if interrupted else "FALHOU"
                self.end(f"{self._label}: {status} ({format_duration(self.elapsed())})")
            self._label = None
        self._stop_thread()

    # --- escrita ---

    def _columns(self) -> int:
        if self._width is not None:
            return self._width
        return shutil.get_terminal_size(fallback=(100, 24)).columns

    def _draw(self) -> None:
        # Uma coluna a menos: escrever na última coluna faz alguns consoles pularem de linha.
        text = self.render()[: max(10, self._columns() - 1)]
        padding = " " * max(0, self._shown - len(text))
        self.stream.write("\r" + text + padding)
        self.stream.flush()
        self._shown = len(text)

    def _clear(self) -> None:
        if self._shown:
            self.stream.write("\r" + " " * self._shown + "\r")
            self._shown = 0

    def _write_line(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()

    # --- thread de atualização ---

    def _start_thread(self) -> None:
        if not (self.live and self.auto_refresh) or self._thread is not None:
            return
        stop = threading.Event()
        self._stop = stop

        def loop() -> None:
            while not stop.wait(self.interval_sec):
                try:
                    self.refresh()
                except Exception:  # noqa: BLE001 — terminal fechado etc.; a exibição não derruba o pipeline
                    return

        self._thread = threading.Thread(target=loop, name="live-status", daemon=True)
        self._thread.start()

    def _stop_thread(self) -> None:
        thread, stop = self._thread, self._stop
        self._thread = self._stop = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
