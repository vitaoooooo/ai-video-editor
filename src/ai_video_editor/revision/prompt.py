"""Prompt do revisor de IA (independente do provedor)."""

from ai_video_editor.director.schema import UserBrief
from ai_video_editor.director.service import allowed_removal_ratio, speech_duration
from ai_video_editor.revision.schema import RevisionContext, SentenceState


def build_reviser_prompt(states: list[SentenceState], brief: UserBrief, context: RevisionContext) -> str:
    """Monta o prompt do revisor: regras, briefing original, histórico, pedido e estado das frases."""
    sentences = [s.sentence for s in states]
    total = speech_duration(sentences)
    limit = allowed_removal_ratio(sentences, brief)
    lines = [
        "Você é um editor de vídeo experiente ajustando uma edição já feita de um vídeo falado "
        "(aula, tutorial ou pessoa falando para a câmera), a partir de um pedido do usuário.",
        f"Vídeo: '{context.video_name}', duração {context.duration_sec:.1f}s, "
        f"idioma {context.language or 'desconhecido'}. Tempo de fala nas frases: {total:.1f}s.",
        f"Você está revisando a versão v{context.base_version}.",
        "",
        "TAREFA: devolva o NOVO conjunto COMPLETO de frases removidas, aplicando o pedido sobre o "
        "estado atual. Toda frase que não estiver em 'removals' fica no vídeo. Você pode restaurar "
        "frases removidas antes e remover frases mantidas. As pausas silenciosas são cortadas "
        "por outra etapa.",
        "",
        "REGRAS OBRIGATÓRIAS:",
        "1. Mude só o que o pedido exige; mantenha as demais decisões do estado atual.",
        "2. Referencie as frases SOMENTE pelo ID entre colchetes. Não invente IDs e não "
        "escreva tempos ou timestamps.",
        "3. Cada ID pode aparecer no máximo uma vez em 'removals'.",
        "4. Toda frase que estava REMOVIDA e agora volta ao vídeo deve aparecer em 'restorations' "
        "com o motivo, e 'restorations' não pode listar mais nada.",
        "5. Toda frase removida (nova ou mantida) leva 'category' e 'reason' em português.",
        f"6. Não remova mais que {limit:.0%} do tempo de fala.",
        "7. 'pacing' ajusta o ritmo das pausas entre cortes: 'natural' (mais folga, fala mais "
        "respirada), 'normal' (padrão) ou 'tight' (menos folga, mais seco). Use só se o pedido "
        "falar de ritmo, pausas ou de cortar mais/menos secamente; senão deixe vazio.",
        "8. Se o pedido for impossível com essas frases (ex.: o assunto não aparece) ou tão ambíguo "
        "que você teria que chutar, devolva feasible=false e explique em 'problem'. Não invente.",
        "9. 'summary' em português, curto, dizendo o que mudou.",
        "",
        "CATEGORIAS: repetition, false_start, filler, off_topic, other (explique em 'reason').",
        "",
        "BRIEFING ORIGINAL DA EDIÇÃO:",
        f"- Intensidade: {brief.intensity}",
    ]
    if brief.target_duration_sec is not None:
        lines.append(f"- Duração final desejada: cerca de {brief.target_duration_sec:.0f}s.")
    if not brief.remove_fillers:
        lines.append("- NÃO remova frases por serem enrolação: a categoria 'filler' está proibida.")
    if brief.instructions:
        lines.append(f"- Pedido original: {brief.instructions.strip()}")
    lines.append(f"- Ritmo atual das pausas: {context.base_pacing}")
    if context.history:
        lines += ["", "PEDIDOS ANTERIORES (do mais antigo ao mais recente):"]
        lines += [f"- v{item.version}: {item.request.strip()}" for item in context.history]
    lines += ["", "NOVO PEDIDO DO USUÁRIO:", context.request.strip()]
    if context.previous_error:
        lines += [
            "",
            "ATENÇÃO: sua resposta anterior foi rejeitada por este motivo:",
            f"{context.previous_error}",
            "Corrija o problema nesta nova resposta.",
        ]
    lines += ["", "FRASES (ID, duração, estado atual e texto):"]
    for s in states:
        if s.removed:
            state = f"REMOVIDA [{s.category or 'other'}: {(s.reason or '').strip()}]"
        else:
            state = "mantida"
        lines.append(f"[{s.sentence.id}] ({s.sentence.duration:.1f}s) {state} — {s.sentence.text}")
    lines += [
        "",
        "Responda em JSON com 'feasible', 'problem', 'removals' (o conjunto completo), "
        "'restorations', 'pacing' e 'summary'.",
    ]
    return "\n".join(lines)
