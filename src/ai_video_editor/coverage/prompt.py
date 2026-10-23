"""Prompt do Diretor de cobertura (independente do provedor). Compacto: só IDs, durações e textos curtos."""

from ai_video_editor.coverage.schema import CoverageContext, VisualShot
from ai_video_editor.coverage.service import (
    BREATH_EVERY_SEC,
    BREATH_SECONDS,
    MAX_COVERAGE_SEC,
    MIN_BREATH_SHOT_SEC,
    MIN_COVERAGE_SEC,
    max_breaths,
    shot_length,
)

# Sobe quando o prompt muda (entra no fingerprint da etapa).
COVERAGE_PROMPT_VERSION = 1
# Texto das frases e descrições das tomadas no prompt.
SENTENCE_CHARS = 90
DESCRIPTION_CHARS = 110

_ROLE_LABELS = {"drone": "drone", "broll": "b-roll", "pov": "POV", "principal": "principal sem fala"}
_USABLE_LABELS = {"sim": "boa", "parcial": "parcial"}


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sec(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _shot_line(shot: VisualShot) -> str:
    parts = [
        f"{shot.id} {_ROLE_LABELS.get(shot.role, shot.role)}",
        _sec(shot_length(shot)),
        f"beleza {shot.beauty or '?'}",
        _USABLE_LABELS.get(shot.usable or "", shot.usable or "?"),
    ]
    if shot.face_to_camera:
        parts.append("rosto")
    line = ", ".join(parts) + f" — {_short(shot.description, DESCRIPTION_CHARS)}"
    if shot.tags:
        line += f" [{', '.join(shot.tags[:6])}]"
    return line


def build_coverage_prompt(context: CoverageContext) -> str:
    """Regras, janelas obrigatórias, frases mantidas (com o clipe e a duração no vídeo) e tomadas candidatas."""
    sizes = ", ".join(f"{name} ≈ {sec:g}s" for name, sec in BREATH_SECONDS.items())
    lines = [
        "Você é um editor de vídeo montando a cobertura de um vlog: imagens de drone, b-roll e POV que "
        "aparecem por cima da fala (o som da fala continua) e respiros curtos entre as falas.",
        f"Projeto: '{context.project_name}', {_sec(context.spine_duration_sec)} de fala montada, "
        f"{len(context.sentences)} frases, {len(context.candidates)} tomadas candidatas.",
        "",
        "TAREFA: escolha coberturas (de_frase..ate_frase + tomada) e, se fizer sentido, respiros "
        "(uma tomada sem fala inserida depois de uma frase; o vídeo fica mais longo).",
        "",
        "REGRAS OBRIGATÓRIAS:",
        "1. TODAS as janelas obrigatórias (O1, O2...) precisam ficar cobertas: a imagem do principal ali "
        "é ruim. Uma janela longa pode ser coberta por várias coberturas seguidas.",
        f"2. Cada cobertura dura entre {_sec(MIN_COVERAGE_SEC)} e {_sec(MAX_COVERAGE_SEC)} no vídeo "
        "(some as durações das frases); uma frase sozinha mais longa que o máximo pode ser coberta inteira.",
        "3. Coberturas não podem se sobrepor (uma frase em no máximo uma cobertura).",
        "4. Cada tomada pode ser usada uma única vez (em uma cobertura ou em um respiro).",
        "5. Nunca cubra uma frase com uma tomada do mesmo clipe que está falando nela.",
        "6. Prefira a tomada mais longa que o trecho; se for mais curta, o código completa com outra.",
        "7. Escolha imagens que combinem com o que está sendo dito (lugar, ação, clima); beleza alta ajuda.",
        "8. Não cubra tudo: a pessoa falando para a câmera também conta a história. Cubra as janelas "
        "obrigatórias e alguns trechos onde a imagem ajuda (paisagem descrita, deslocamento, chegada).",
        f"9. Respiros: no máximo {max_breaths(context)} (um a cada {BREATH_EVERY_SEC:.0f}s de fala), entre "
        f"ideias ou na troca de lugar/clipe, nunca no meio de uma cobertura; tamanhos: {sizes} "
        f"(tomada com pelo menos {_sec(MIN_BREATH_SHOT_SEC)}).",
        "10. Use SOMENTE os IDs listados (frases S..., tomadas como D001.T05, janelas O...). Não "
        "escreva tempos ou timestamps.",
        "11. Escreva 'reason' e 'summary' em português, curtos e específicos.",
        "",
    ]
    if context.instructions:
        lines += ["INSTRUÇÕES DO PRODUTOR E DO USUÁRIO:", *(f"- {_short(i, 400)}" for i in context.instructions), ""]
    if context.previous_error:
        lines += [
            "ATENÇÃO: sua resposta anterior foi rejeitada por este motivo:",
            context.previous_error,
            "Corrija o problema nesta nova resposta.",
            "",
        ]
    lines.append("JANELAS OBRIGATÓRIAS (imagem ruim por baixo da fala):")
    if context.windows:
        by_id = {s.id: s for s in context.sentences}
        for w in context.windows:
            seconds = sum(by_id[x].duration_sec(context.fps) for x in w.sentence_ids)
            lines.append(f"{w.id}: {w.span_label} (clipe {w.clip_id}, ~{_sec(seconds)}) — {', '.join(w.reasons)}")
    else:
        lines.append("(nenhuma)")
    lines += ["", "FRASES MANTIDAS, na ordem do vídeo (ID, clipe, duração no vídeo, texto):"]
    for s in context.sentences:
        lines.append(f"[{s.id}] {s.clip_id} {_sec(s.duration_sec(context.fps))} {_short(s.text, SENTENCE_CHARS)}")
    lines += ["", f"TOMADAS CANDIDATAS (papéis: {', '.join(context.roles)}; ID, papel, duração, beleza 1–5, qualidade):"]
    lines += [_shot_line(shot) for shot in context.candidates] or ["(nenhuma)"]
    lines += [
        "",
        "Responda em JSON com 'coverages' (from_sentence, to_sentence, shot_id, reason, windows), "
        "'breaths' (after_sentence, shot_id, size: short|medium|long, reason; pode ser vazia) e 'summary'.",
    ]
    return "\n".join(lines)
