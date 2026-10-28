"""Prompts do Diretor de cobertura (independentes do provedor), em duas chamadas:

1. **Mapa de assuntos:** só as frases mantidas e o que está na tela (sem as candidatas): a IA entende do
   que o vídeo trata e divide as frases em blocos de assunto contíguos.
2. **Cobertura:** as frases dentro dos blocos (com o que está na tela, se a pessoa fala para a câmera e
   onde a imagem é ruim) e as candidatas (descrição, tags, papel, duração, beleza e, com datas confiáveis,
   a proximidade de gravação). Cada cobertura cita a evidência de fala e a de imagem; o código confere.

**Por que duas chamadas:** no vlog real (167 frases, 90 candidatas) o prompt de uma chamada só tinha ~41
mil caracteres, ~20 mil deles de frases e tela. O mapa sai numa chamada pequena (~22 mil, sem as
candidatas) e focada em entender o vídeo; a 2ª mostra cada frase dentro do seu bloco; e cada resposta é
validada e refeita sozinha (um mapa recusado não derruba as coberturas, e vice-versa). Custo: ~50% a mais
de tokens de entrada que uma chamada só.
"""

from typing import Optional

from ai_video_editor.coverage.related import recording_note, related_shots
from ai_video_editor.coverage.schema import CoverageContext, VisualShot
from ai_video_editor.coverage.service import (
    BREATH_EVERY_SEC,
    BREATH_SECONDS,
    MAX_COVERAGE_RATIO,
    MAX_COVERAGE_SEC,
    MIN_BREATH_SHOT_SEC,
    MIN_COVERAGE_SEC,
    max_breaths,
    shot_length,
)

# Sobe quando um dos prompts muda (entra no fingerprint da etapa).
COVERAGE_PROMPT_VERSION = 3
# Texto das frases (quase inteiro: a evidência de fala é copiada dele) e descrições das tomadas no prompt.
SENTENCE_CHARS = 600
DESCRIPTION_CHARS = 110
# Candidatas relacionadas listadas por tela uniforme.
RELATED_PER_SPAN = 6

_ROLE_LABELS = {"drone": "drone", "broll": "b-roll", "pov": "POV", "principal": "principal sem fala"}
_USABLE_LABELS = {"sim": "boa", "parcial": "parcial"}
_ISSUE_LABELS = {"obstruida": "obstruída", "borrao_movimento": "borrão"}


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sec(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _tags(shot: VisualShot) -> str:
    return f" [{', '.join(shot.tags[:8])}]" if shot.tags else ""


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


def _screen_line(shot: VisualShot, to_camera: bool) -> str:
    text = _short(shot.description, DESCRIPTION_CHARS) if shot.description else "(sem descrição)"
    flags = []
    if to_camera:
        flags.append("pessoa falando para a câmera")
    issues = [_ISSUE_LABELS[i] for i in shot.issues if i in _ISSUE_LABELS]
    if shot.usable == "parcial" and issues:
        flags.append(f"parcial: {', '.join(issues)}")
    extra = f" ({'; '.join(flags)})" if flags else ""
    return f"   ~ tela: {shot.id} {text}{_tags(shot)}{extra}"


def _sentence_lines(context: CoverageContext, with_bad: bool) -> list[str]:
    """Frases mantidas com o que está na tela (na troca de tomada) e, se pedido, onde a imagem é ruim."""
    screens = {s.id: s for s in context.screen_shots}
    starts = {}
    if with_bad:
        for span in context.bad_spans:
            starts.setdefault(span.sentence_ids[0], []).append(span)
    topics = {t.from_sentence: t for t in context.topics} if with_bad else {}
    lines: list[str] = []
    previous = None
    for s in context.sentences:
        if s.id in topics:
            t = topics[s.id]
            place = f" (lugar/objeto: {t.place})" if t.place else ""
            lines.append(f"== {t.id} {t.span_label} · {t.topic}{place}")
            previous = None
        current = s.screen[0] if s.screen else None
        if current and current != previous and current in screens:
            lines.append(_screen_line(screens[current], s.to_camera))
        previous = current
        for span in starts.get(s.id, []):
            what = "tela uniforme (parede/tampa)" if span.uniform else ", ".join(span.reasons)
            lines.append(
                f"   ! {span.id} imagem ruim aqui ({_sec(span.duration_sec(context.fps))}, {what}, {span.span_label}): "
                "prefira cobrir se houver imagem que combine"
            )
        lines.append(f"[{s.id}] {s.clip_id} {_sec(s.duration_sec(context.fps))} {_short(s.text, SENTENCE_CHARS)}")
    return lines


def build_topic_prompt(context: CoverageContext) -> str:
    """1ª chamada: do que o vídeo trata e os blocos de assunto."""
    lines = [
        "Você é um editor de vídeo. Antes de escolher imagens de cobertura, entenda do que este vlog trata e o "
        "que é falado em cada trecho.",
        f"Projeto: '{context.project_name}', {_sec(context.spine_duration_sec)} de fala montada, {len(context.sentences)} frases.",
        "",
        "TAREFA: divida as frases em BLOCOS DE ASSUNTO: frases seguidas que falam da mesma coisa, no mesmo "
        "lugar/momento (ex.: chegada de ônibus, quarto do hotel, trilha até o lago).",
        "",
        "REGRAS:",
        "1. Os blocos seguem a ordem das frases, um depois do outro: o primeiro começa na primeira frase, cada "
        "um começa na frase seguinte à última do anterior e o último termina na última frase. Sem buracos e "
        "sem sobreposição.",
        "2. Troque de bloco quando muda o lugar, o momento ou o assunto (a troca de clipe costuma ser uma "
        "troca de lugar, mas um clipe pode ter mais de um assunto).",
        "3. 'topic': o assunto em poucas palavras, em português. 'place': o lugar ou objeto principal falado "
        "ou mostrado (ex.: lago, ônibus, quarto do hotel); vazio se não houver.",
        "4. 'summary': do que o vídeo trata, em uma frase.",
        "5. Use SOMENTE os IDs de frases listados (S...). Não escreva tempos.",
        "",
    ]
    if context.previous_error:
        lines += ["NOVA TENTATIVA: o mapa anterior foi recusado. Problemas:", context.previous_error, "Refaça o mapa inteiro.", ""]
    lines += ["FRASES MANTIDAS, na ordem do vídeo (ID, clipe, duração, texto); '~ tela' = o que aparece na imagem a partir dali:"]
    lines += _sentence_lines(context, with_bad=False)
    lines += ["", "Responda em JSON com 'summary' e 'blocks' (from_sentence, to_sentence, topic, place)."]
    return "\n".join(lines)


def _uniform_lines(context: CoverageContext) -> list[str]:
    """Telas uniformes e as candidatas relacionadas (as únicas que podem tapar sem evidência de fala)."""
    screens = {s.id: s for s in context.screen_shots}
    lines = []
    for span in context.bad_spans:
        if not span.uniform:
            continue
        free = [c for c in context.candidates if c.clip_id != span.clip_id]
        bad = [screens[x] for x in span.shot_ids if x in screens]
        related = related_shots(free, bad)[:RELATED_PER_SPAN]
        found = "; ".join(f"{c.id} ({r.label})" for c, r in related) if related else "nenhuma (a imagem original fica)"
        lines.append(f"{span.id} ({span.span_label}, {_sec(span.duration_sec(context.fps))}): relacionadas: {found}")
    return lines


def build_coverage_prompt(context: CoverageContext) -> str:
    """2ª chamada: regras com evidências, frases por bloco (tela, câmera, imagem ruim) e candidatas."""
    sizes = ", ".join(f"{name} ≈ {sec:g}s" for name, sec in BREATH_SECONDS.items())
    lines = [
        "Você é um editor de vídeo montando a cobertura de um vlog: imagens de drone, b-roll e POV que "
        "aparecem por cima da fala (o som da fala continua) e respiros curtos entre os assuntos.",
        f"Projeto: '{context.project_name}', {_sec(context.spine_duration_sec)} de fala montada, "
        f"{len(context.sentences)} frases, {len(context.topics)} blocos de assunto, {len(context.candidates)} tomadas candidatas.",
    ]
    if context.video_summary:
        lines.append(f"Do que o vídeo trata: {_short(context.video_summary, 300)}")
    lines += [
        "",
        "TAREFA: cubra só onde a imagem MOSTRA o que é falado ou o lugar onde a cena acontece (fala do lago → "
        "imagem do lago). Na dúvida, NÃO cubra: a pessoa continua na tela. Respiros só entre blocos de assunto.",
        "",
        "REGRAS:",
        "1. Evidências obrigatórias em cada cobertura: 'speech_evidence' = trecho COPIADO LITERALMENTE do texto "
        "das frases cobertas que fala da coisa; 'image_evidence' = uma tag ou trecho COPIADO da descrição de CADA "
        "tomada listada que mostra essa coisa. As duas precisam falar da mesma coisa (ex.: 'chegamos no lago' ↔ "
        "'lago'). O código confere: evidência que não está no texto ou na tomada, ou que não fala da mesma "
        "coisa, descarta a cobertura. Palavras genéricas (pessoa, paisagem, árvore, céu, vista) não contam.",
        "2. Pessoa falando para a câmera (marcado na linha '~ tela'): só cubra se a coisa falada for uma TAG da "
        "tomada (o assunto principal da imagem).",
        "3. '! imagem ruim aqui': prefira cobrir esse trecho se houver imagem que combine com a fala; a "
        "cobertura fica limitada aos segundos ruins. Sem imagem que combine, deixe: a imagem original fica. "
        "Tela uniforme (parede/tampa) pode ser tapada sem evidência de fala, só com as tomadas relacionadas "
        "listadas abaixo.",
        f"4. Não cubra tudo: no máximo {MAX_COVERAGE_RATIO:.0%} do vídeo com cobertura; a pessoa falando também "
        "conta a história.",
        f"5. Cada tomada mostra no máximo ~{MAX_COVERAGE_SEC:g}s; o código divide o tempo entre as tomadas listadas "
        "e põe a imagem perto de quando a coisa é dita (um trecho longo pode ficar coberto só em parte). Uma "
        f"cobertura tem pelo menos {_sec(MIN_COVERAGE_SEC)}.",
        "6. Coberturas não se sobrepõem; cada tomada é usada uma única vez; nunca use tomada do clipe que está "
        "falando no trecho.",
        f"7. Respiros: no máximo {max_breaths(context)} (um a cada {BREATH_EVERY_SEC:.0f}s de fala), só depois da "
        "ÚLTIMA frase de um bloco (troca de lugar/momento), com uma tomada do lugar do bloco anterior ou do "
        "seguinte: 'image_evidence' = tag ou trecho da descrição dela que mostra esse lugar/assunto. Tamanhos: "
        f"{sizes} (tomada com pelo menos {_sec(MIN_BREATH_SHOT_SEC)}).",
        "8. Use SOMENTE os IDs listados (frases S..., tomadas como D001.T05). Não escreva tempos ou timestamps.",
        "9. Escreva 'reason' e 'summary' em português, curtos e específicos.",
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
            "Responda SÓ com substitutos válidos dos itens rejeitados (ou nenhum, se não houver imagem que combine).",
            "",
        ]
    uniform = _uniform_lines(context)
    if uniform:
        lines += ["TELAS UNIFORMES (podem ser tapadas sem evidência de fala, só com as relacionadas):", *uniform, ""]
    lines += [
        "FRASES MANTIDAS POR BLOCO DE ASSUNTO, na ordem do vídeo (ID, clipe, duração no vídeo, texto); '~ tela' = "
        "o que aparece na imagem a partir dali:",
    ]
    lines += _sentence_lines(context, with_bad=True)
    spine_shots = list({s.clip_id: s for s in context.screen_shots}.values())
    lines += [
        "",
        f"TOMADAS CANDIDATAS (papéis: {', '.join(context.roles)}; ID, papel, duração, beleza 1–5, qualidade, "
        "proximidade de gravação quando as datas são confiáveis — descrição [tags]):",
    ]
    lines += [_shot_line(shot, recording_note(shot, spine_shots)) for shot in context.candidates] or ["(nenhuma)"]
    lines += [
        "",
        "Responda em JSON com 'coverages' (from_sentence, to_sentence, shot_ids: lista em ordem, speech_evidence, "
        "image_evidence, reason), 'breaths' (after_sentence, shot_id, size: short|medium|long, image_evidence, "
        "reason; pode ser vazia) e 'summary'.",
    ]
    return "\n".join(lines)

