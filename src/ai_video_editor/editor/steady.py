"""Pontas de câmera mexida e a janela estável de um trecho de imagem (medida, sem decisão editorial).

As medidas são as amostras de movimento do roteiro (`RoteiroClip.motion`: diferença de luma 0–1 entre amostras
seguidas do scan, `MEASURE_FPS` por segundo). A amostra do tempo t mede o movimento em (t − passo, t].

Uma amostra é **instável** quando o movimento passa do limite de "movimento alto" do projeto
(`ProjectRoteiro.motion_threshold`, o percentil do `visual`; sem ele, `MOTION_FLOOR`) ou quando é um pico
no clipe: pelo menos `EDGE_MOTION_FLOOR` e `EDGE_MOTION_RATIO` vezes a mediana do clipe inteiro (o jeito normal
daquela câmera; a do trecho não serve num trecho curto, em que os picos podem ser metade das amostras). O
pico relativo pega a câmera parada que é pega/posta no lugar (no vlog real, P012: 1–2/255 parada e 13–16/255
quando o casal volta para pegá-la, abaixo do limite do projeto, que é de câmera na mão).

- `steady_range`: o trecho sem as amostras instáveis seguidas das pontas (câmera sendo posta, pega ou
  carregada). Nada é tirado se o trecho é todo instável ou se sobraria menos de `EDGE_MIN_KEEP_SEC`.
- `steady_window`: dentro do `steady_range`, a janela de duração pedida com o menor movimento médio (empate:
  a mais perto do centro). Sem amostras, a janela centrada (o comportamento de antes).

Quem usa: respiro e inclusão (validação da 2b), cobertura (montagem) e as pontas fora das frases de um clipe
com fala (montagem; o corte nunca entra numa frase).
"""

import statistics
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.visual.shots import MEASURE_FPS, MOTION_FLOOR

# Pico relativo: amostra com pelo menos isto de movimento (0–1; 0,03 ≈ 8/255)...
EDGE_MOTION_FLOOR = 0.03
# ...e pelo menos tantas vezes a mediana do clipe é instável (além de tudo acima do limite do projeto).
EDGE_MOTION_RATIO = 4.0
# Tirar as pontas nunca deixa o trecho com menos que isto (senão, fica como está).
EDGE_MIN_KEEP_SEC = 1.0
# Passo entre amostras (o intervalo que cada amostra mede).
SAMPLE_STEP_SEC = 1.0 / MEASURE_FPS


def _samples(material: EditorMaterial, clip_id: str, start: float, end: float) -> list[tuple[float, float]]:
    """Amostras que medem um intervalo dentro de [start, end]."""
    clip = material.clips.get(clip_id)
    if clip is None:
        return []
    return [(t, m) for t, m in clip.motion if start + SAMPLE_STEP_SEC - 1e-6 <= t <= end + 1e-6]


def _threshold(material: EditorMaterial) -> float:
    value = material.roteiro.motion_threshold
    return value if value is not None else MOTION_FLOOR


def unstable_flags(material: EditorMaterial, clip_id: str, samples: list[tuple[float, float]]) -> list[bool]:
    """Para cada amostra: instável (acima do limite do projeto ou pico relativo à mediana do clipe)."""
    if not samples:
        return []
    median = statistics.median(m for _, m in material.clips[clip_id].motion)
    limit = _threshold(material)
    return [m > limit or (m >= EDGE_MOTION_FLOOR and m >= EDGE_MOTION_RATIO * median) for _, m in samples]


def steady_range(material: EditorMaterial, clip_id: str, start: float, end: float) -> tuple[float, float]:
    """[start, end] sem as pontas instáveis (ver o topo do módulo)."""
    samples = _samples(material, clip_id, start, end)
    flags = unstable_flags(material, clip_id, samples)
    if len(samples) < 2 or all(flags):
        return start, end
    a, b = start, end
    k = 0
    while flags[k]:
        a = samples[k][0]
        k += 1
    j = len(samples) - 1
    while flags[j]:
        b = samples[j][0] - SAMPLE_STEP_SEC
        j -= 1
    if b - a < EDGE_MIN_KEEP_SEC:
        return start, end
    return round(a, 3), round(b, 3)


def steady_window(
    material: EditorMaterial, clip_id: str, start: float, end: float, length: Optional[float]
) -> tuple[float, float]:
    """Janela de `length` s (None = o trecho estável inteiro) com o menor movimento médio dentro do trecho estável."""
    a, b = steady_range(material, clip_id, start, end)
    if length is None or length >= b - a:
        return a, b
    samples = _samples(material, clip_id, a, b)
    middle = a + (b - a - length) / 2
    if not samples:
        return round(middle, 3), round(middle + length, 3)
    starts = sorted({a, b - length, *(t - SAMPLE_STEP_SEC for t, _ in samples if a <= t - SAMPLE_STEP_SEC <= b - length)})

    def score(s: float) -> tuple[float, float]:
        inside = [m for t, m in samples if s + SAMPLE_STEP_SEC - 1e-6 <= t <= s + length + 1e-6]
        return (statistics.fmean(inside) if inside else 0.0, abs(s - middle))

    best = min(starts, key=score)
    return round(best, 3), round(best + length, 3)
