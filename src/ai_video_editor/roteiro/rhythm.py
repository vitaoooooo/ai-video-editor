"""Ritmo de cada frase, medido pelo programa: volume, velocidade e pausas.

- **Volume:** nível da frase = percentil RHYTHM_LEVEL_PERCENTILE dos picos das janelas de 50 ms
  (`audio/levels.py`) dentro dela (o percentil alto ignora as pausas entre palavras); referência =
  mediana dos níveis das frases do mesmo clipe. A classe sai da diferença (dB) para a referência:
  relativo, porque cada câmera/microfone grava num nível.
- **Velocidade:** palavras por segundo da frase (palavras do Whisper ÷ duração da frase).
- **Pausas:** antes = da frase anterior do clipe (ou do início do clipe) até esta; depois = desta até
  a próxima (ou o fim do clipe). O tempo das palavras do Whisper costuma engolir parte da pausa, então
  um silêncio detectado (`silencedetect`) que encosta no início/fim da frase vale se for maior.

Todos os limites são **estimativas iniciais, a calibrar** no material real; os valores medidos ficam no
`roteiro.json` (`SentenceRhythm`) para isso.
"""

import statistics
from typing import Optional

from ai_video_editor.audio.levels import WINDOW_SEC, percentile
from ai_video_editor.audio.schema import SilenceInterval
from ai_video_editor.director.schema import Sentence
from ai_video_editor.roteiro.schema import SentenceRhythm, Speed, Volume
from ai_video_editor.transcription.schema import Transcript

# --- volume (dB em relação à mediana das frases do clipe; estimativa inicial, a calibrar) ---
RHYTHM_LEVEL_PERCENTILE = 75.0
WHISPER_MAX_DB = -12.0  # até aqui: sussurro
LOW_MAX_DB = -5.0  # até aqui: baixo
LOUD_MIN_DB = 5.0  # a partir daqui: alto
# --- velocidade (palavras por segundo; estimativa inicial para português falado, a calibrar) ---
SLOW_MAX_WPS = 1.8  # até aqui: lenta
FAST_MIN_WPS = 3.6  # a partir daqui: rápida
# Frases com menos palavras que isso não têm velocidade (uma palavra solta não diz nada).
MIN_WORDS_FOR_SPEED = 3
# --- pausas ---
# Silêncio detectado que termina (ou começa) a até isso do início (ou fim) da frase conta como a pausa dela.
PAUSE_SNAP_SEC = 0.5


def classify_volume(relative_db: float) -> Volume:
    if relative_db <= WHISPER_MAX_DB:
        return "sussurro"
    if relative_db <= LOW_MAX_DB:
        return "baixo"
    if relative_db >= LOUD_MIN_DB:
        return "alto"
    return "normal"


def classify_speed(words_per_sec: float) -> Speed:
    if words_per_sec <= SLOW_MAX_WPS:
        return "lenta"
    if words_per_sec >= FAST_MIN_WPS:
        return "rápida"
    return "normal"


def sentence_words(sentence: Sentence, transcript: Transcript) -> int:
    """Palavras da frase: as do Whisper nos segmentos dela (ou as do texto, sem tempos por palavra)."""
    by_id = {seg.id: seg for seg in transcript.segments}
    count = sum(len(by_id[i].words) for i in sentence.segment_ids if i in by_id)
    return count or len(sentence.text.split())


def window_level(levels: list[float], start_sec: float, end_sec: float) -> Optional[float]:
    """Percentil RHYTHM_LEVEL_PERCENTILE dos picos das janelas dentro de [início, fim)."""
    first = max(0, int(start_sec / WINDOW_SEC))
    last = min(len(levels), max(first + 1, int(round(end_sec / WINDOW_SEC))))
    inside = levels[first:last]
    return round(percentile(inside, RHYTHM_LEVEL_PERCENTILE), 1) if inside else None


def _silence_before(start: float, floor: float, silences: list[SilenceInterval]) -> float:
    """Maior silêncio que termina perto do início da frase (sem voltar antes de `floor`)."""
    best = 0.0
    for silence in silences:
        if abs(silence.end_sec - start) <= PAUSE_SNAP_SEC and silence.end_sec > floor:
            best = max(best, silence.end_sec - max(silence.start_sec, floor))
    return best


def _silence_after(end: float, ceiling: float, silences: list[SilenceInterval]) -> float:
    best = 0.0
    for silence in silences:
        if abs(silence.start_sec - end) <= PAUSE_SNAP_SEC and silence.start_sec < ceiling:
            best = max(best, min(silence.end_sec, ceiling) - silence.start_sec)
    return best


def measure_rhythm(
    sentences: list[Sentence],
    transcript: Transcript,
    duration_sec: float,
    levels: Optional[list[float]] = None,
    silences: Optional[list[SilenceInterval]] = None,
) -> list[SentenceRhythm]:
    """Ritmo das frases de UM clipe (em ordem), na mesma ordem.

    `levels`: picos por janela de 50 ms do clipe (sem eles, volume fica sem medida).
    `silences`: silêncios detectados do clipe (sem eles, as pausas são só o intervalo entre frases).
    """
    silences = silences or []
    measured = [window_level(levels, s.start_sec, s.end_sec) if levels else None for s in sentences]
    known = [value for value in measured if value is not None]
    reference = statistics.median(known) if known else None
    rhythms: list[SentenceRhythm] = []
    for k, sentence in enumerate(sentences):
        previous_end = sentences[k - 1].end_sec if k > 0 else 0.0
        next_start = sentences[k + 1].start_sec if k + 1 < len(sentences) else max(duration_sec, sentence.end_sec)
        gap_before = max(0.0, sentence.start_sec - previous_end)
        gap_after = max(0.0, next_start - sentence.end_sec)
        pause_before = max(gap_before, _silence_before(sentence.start_sec, previous_end, silences))
        pause_after = max(gap_after, _silence_after(sentence.end_sec, next_start, silences))
        words = sentence_words(sentence, transcript)
        duration = sentence.end_sec - sentence.start_sec
        wps = round(words / duration, 2) if duration > 0 else None
        level = measured[k]
        relative = round(level - reference, 1) if level is not None and reference is not None else None
        rhythms.append(SentenceRhythm(
            volume=classify_volume(relative) if relative is not None else None,
            level_db=level,
            relative_db=relative,
            speed=classify_speed(wps) if wps is not None and words >= MIN_WORDS_FOR_SPEED else None,
            words=words,
            words_per_sec=wps,
            pause_before_sec=round(pause_before, 2),
            pause_after_sec=round(pause_after, 2),
        ))
    return rhythms


def rhythm_rules() -> dict[str, float]:
    """Constantes do ritmo, gravadas no roteiro.json (para calibrar com os valores medidos)."""
    return {
        "rhythm_level_percentile": RHYTHM_LEVEL_PERCENTILE,
        "whisper_max_db": WHISPER_MAX_DB,
        "low_max_db": LOW_MAX_DB,
        "loud_min_db": LOUD_MIN_DB,
        "slow_max_wps": SLOW_MAX_WPS,
        "fast_min_wps": FAST_MIN_WPS,
        "min_words_for_speed": MIN_WORDS_FOR_SPEED,
        "pause_snap_sec": PAUSE_SNAP_SEC,
    }
