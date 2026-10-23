"""Prompt do Diretor de cobertura (independente do provedor). Compacto: só IDs, durações e textos curtos.

Mostra o que está na tela por baixo da fala (descrição e tags da tomada do principal, na troca de
tomada), as janelas com imagem ruim com as candidatas relacionadas e, em cada candidata, a
proximidade de gravação com os clipes que falam: contexto antes de beleza.
"""

from typing import Optional

from ai_video_editor.coverage.related import recording_note, related_shots
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
COVERAGE_PROMPT_VERSION = 2
# Texto das frases e descrições das tomadas no prompt.
SENTENCE_CHARS = 90
DESCRIPTION_CHARS = 110
# Candidatas relacionadas listadas por janela.
RELATED_PER_WINDOW = 6

_ROLE_LABELS = {"drone": "drone", "broll": "b-roll", "pov": "POV", "principal": "principal sem fala"}
_USABLE_LABELS = {"sim": "boa", "parcial": "parcial"}


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sec(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _tags(shot: VisualShot) -> str:
    return f" [{', '.join(shot.tags[:6])}]" if shot.tags else ""


def _shot_line(shot: VisualShot, note: Optional[str] = None) -> str:
    parts = [
        f"{shot.id} {_ROLE_LABELS.get(shot.role, shot.role)}",
        _sec(shot_length(shot)),
        f"beleza {shot.beauty or '?'}",
        _USABLE_LABELS.get(shot.usable or "", shot.usable or "?"),
    ]
    if shot.face_to_camera:
        parts.append("rosto")
    if note:
        parts.append(note)
    return ", ".join(parts) + f" — {_short(shot.description, DESCRIPTION_CHARS)}{_tags(shot)}"


def _screen_line(shot: VisualShot) -> str:
    text = _short(shot.description, DESCRIPTION_CHARS) if shot.description else "(sem descrição)"
    return f"   ~ tela: {shot.id} {text}{_tags(shot)}"


def build_coverage_prompt(context: CoverageContext) -> str:
    """Regras, janelas com imagem ruim, frases mantidas (com o que está na tela) e tomadas candidatas."""
    sizes = ", ".join(f"{name} ≈ {sec:g}s" for name, sec in BREATH_SECONDS.items())
    screens = {s.id: s for s in context.screen_shots}
    lines = [
        "Você é um editor de vídeo montando a cobertura de um vlog: imagens de drone, b-roll e POV que "
        "aparecem por cima da fala (o som da fala continua) e respiros curtos entre as falas.",
        f"Projeto: '{context.project_name}', {_sec(context.spine_duration_sec)} de fala montada, "
        f"{len(context.sentences)} frases, {len(context.candidates)} tomadas candidatas.",
        "",
        "TAREFA: escolha coberturas (de_frase..ate_frase + tomadas em ordem), decida cada janela com imagem "
        "ruim (cobrir ou manter) e, se fizer sentido, respiros (uma tomada sem fala inserida depois de uma "
        "frase; o vídeo fica mais longo).",
        "",
        "REGRAS:",
        "1. Janelas com imagem ruim (O1, O2...): para CADA uma, dê uma decisão em 'window_decisions': "
        "'cover' (e liste a cobertura em 'coverages', com tomada relacionada) ou 'keep' (a imagem original "
        "fica), com o motivo. Manter é certo quando a imagem é o conteúdo (ex.: o tour pelo quarto) ou o "
        "defeito é leve (ex.: a pessoa falando no ônibus, com os bancos na frente). Janelas marcadas "
        "EXIGE COBERTURA não podem ficar com 'keep'.",
        "2. Contexto acima de beleza: a imagem tem de ter relação com o que está na tela (linhas '~ tela') "
        "ou com o que é dito — mesmo lugar, mesma ação, gravada logo antes/depois. Nunca cubra com uma "
        "paisagem sem relação só porque é bonita.",
        f"3. Cada tomada mostra no máximo ~{MAX_COVERAGE_SEC:g}s. Para um trecho longo, liste em 'shot_ids' "
        "tomadas suficientes, em ordem (ex.: 20s → 3 tomadas); o código divide o tempo entre elas. Uma "
        f"cobertura tem pelo menos {_sec(MIN_COVERAGE_SEC)} (se a frase for curta, inclua a vizinha).",
        "4. Coberturas não podem se sobrepor (uma frase em no máximo uma cobertura).",
        "5. Cada tomada pode ser usada uma única vez (em uma cobertura ou em um respiro).",
        "6. Nunca use uma tomada do mesmo clipe que está falando no trecho.",
        "7. Não cubra tudo: a pessoa falando para a câmera também conta a história. Decida as janelas e cubra "
        "alguns trechos onde a imagem ajuda (paisagem descrita, deslocamento, chegada).",
        f"8. Respiros: no máximo {max_breaths(context)} (um a cada {BREATH_EVERY_SEC:.0f}s de fala), entre "
        f"ideias ou na troca de lugar/clipe, nunca no meio de uma cobertura; tamanhos: {sizes} "
        f"(tomada com pelo menos {_sec(MIN_BREATH_SHOT_SEC)}).",
        "9. Use SOMENTE os IDs listados (frases S..., tomadas como D001.T05, janelas O...). Não "
        "escreva tempos ou timestamps.",
        "10. Escreva 'reason' e 'summary' em português, curtos e específicos.",
        "",
    ]
    if context.instructions:
        lines += ["INSTRUÇÕES DO PRODUTOR E DO USUÁRIO:", *(f"- {_short(i, 400)}" for i in context.instructions), ""]
    if context.previous_error:
        lines += ["NOVA TENTATIVA: sua resposta anterior foi aceita em parte. Itens já aceitos (não repita; as "
                  "tomadas e frases deles estão ocupadas):"]
        lines += [f"- {item}" for item in context.accepted_items] or ["- (nenhum)"]
        lines += [
            "Problemas da resposta anterior (itens rejeitados):",
            context.previous_error,
            "Responda SÓ com os itens que faltam: coberturas para as janelas indicadas e substitutos dos itens "
            "rejeitados.",
            "",
        ]

    lines.append("JANELAS COM IMAGEM RUIM (por baixo da fala):")
    if context.windows:
        by_id = {s.id: s for s in context.sentences}
        for w in context.windows:
            seconds = sum(by_id[x].duration_sec(context.fps) for x in w.sentence_ids)
            rule = "EXIGE COBERTURA" if w.required else "cobrir ou manter"
            lines.append(f"{w.id}: {w.span_label} (clipe {w.clip_id}, ~{_sec(seconds)}) — {', '.join(w.reasons)} — {rule}")
            shown = dict.fromkeys(x for s in w.sentence_ids for x in by_id[s].screen)
            on_screen = [screens[x] for x in shown if x in screens]
            if on_screen:
                lines.append(
                    "   na tela: "
                    + " | ".join(f"{s.id} {_short(s.description, DESCRIPTION_CHARS) or '(sem descrição)'}{_tags(s)}" for s in on_screen[:3])
                )
            speaking = {by_id[x].clip_id for x in w.sentence_ids}
            free = [c for c in context.candidates if c.clip_id not in speaking]
            bad = [screens[x] for x in w.shot_ids if x in screens]
            related = related_shots(free, bad)[:RELATED_PER_WINDOW]
            lines.append(
                "   relacionadas: " + ("; ".join(f"{c.id} ({r.label})" for c, r in related) if related else "nenhuma pelo código (veja a fala)")
            )
    else:
        lines.append("(nenhuma)")

    lines += ["", "FRASES MANTIDAS, na ordem do vídeo (ID, clipe, duração no vídeo, texto); '~ tela' = o que aparece na imagem a partir dali:"]
    previous = None
    for s in context.sentences:
        current = s.screen[0] if s.screen else None
        if current and current != previous and current in screens:
            lines.append(_screen_line(screens[current]))
        previous = current
        lines.append(f"[{s.id}] {s.clip_id} {_sec(s.duration_sec(context.fps))} {_short(s.text, SENTENCE_CHARS)}")

    spine_shots = list({s.clip_id: s for s in context.screen_shots}.values())
    lines += [
        "",
        f"TOMADAS CANDIDATAS (papéis: {', '.join(context.roles)}; ID, papel, duração, beleza 1–5, qualidade, "
        "proximidade de gravação com os clipes que falam):",
    ]
    lines += [_shot_line(shot, recording_note(shot, spine_shots)) for shot in context.candidates] or ["(nenhuma)"]
    lines += [
        "",
        "Responda em JSON com 'coverages' (from_sentence, to_sentence, shot_ids: lista em ordem, reason, windows), "
        "'window_decisions' (window, action: cover|keep, reason; uma por janela), 'breaths' (after_sentence, "
        "shot_id, size: short|medium|long, reason; pode ser vazia) e 'summary'.",
    ]
    return "\n".join(lines)
