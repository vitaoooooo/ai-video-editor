"""Prompt do Diretor de IA (independente do provedor)."""

from ai_video_editor.director.schema import DirectorContext, Sentence, UserBrief
from ai_video_editor.director.service import allowed_removal_ratio, speech_duration

_INTENSITY_GUIDE = {
    "light": (
        "LEVE: remova apenas problemas evidentes — frases recomeçadas, repetições quase "
        "literais e enrolação sem conteúdo. Na maioria dos vídeos isso é pouco ou nada."
    ),
    "medium": (
        "MÉDIA: além dos problemas evidentes, remova repetições de ideias já ditas e "
        "digressões curtas que não acrescentam nada ao assunto."
    ),
    "aggressive": (
        "AGRESSIVA: deixe o vídeo enxuto; remova também desvios do assunto e explicações "
        "redundantes, mas mantenha toda informação necessária para entender o conteúdo."
    ),
}


def build_director_prompt(sentences: list[Sentence], brief: UserBrief, context: DirectorContext) -> str:
    """Monta o prompt do Diretor: regras, briefing do usuário e frases numeradas."""
    total = speech_duration(sentences)
    limit = allowed_removal_ratio(sentences, brief)
    lines = [
        "Você é um editor de vídeo experiente revisando a transcrição de um vídeo falado "
        "(aula, tutorial ou pessoa falando para a câmera).",
        f"Vídeo: '{context.video_name}', duração {context.duration_sec:.1f}s, "
        f"idioma {context.language or 'desconhecido'}. Tempo de fala nas frases: {total:.1f}s.",
        "",
        "TAREFA: escolha quais frases devem ser REMOVIDAS do vídeo. As pausas silenciosas já "
        "são cortadas automaticamente por outra etapa; não se preocupe com elas.",
        "",
        "REGRAS OBRIGATÓRIAS:",
        "1. Seja conservador. Na dúvida, MANTENHA a frase.",
        "2. Nunca remova uma frase que carrega informação nova (um passo, um conceito, um dado, "
        "um exemplo, uma conclusão), mesmo que esteja mal falada.",
        "3. Em repetições e frases recomeçadas, remova a versão pior ou incompleta e mantenha a "
        "melhor (normalmente a última, completa).",
        "4. Referencie as frases SOMENTE pelo ID entre colchetes. Não invente IDs e não "
        "escreva tempos ou timestamps.",
        "5. Cada ID pode aparecer no máximo uma vez.",
        f"6. Não remova mais que {limit:.0%} do tempo de fala.",
        "7. Escreva 'reason' e 'summary' em português, de forma curta e específica.",
        "",
        "CATEGORIAS: repetition (repete algo já dito), false_start (frase começada e "
        "abandonada ou refeita logo depois), filler (enrolação sem conteúdo), off_topic "
        "(fora do assunto do vídeo), other (outro motivo; explique).",
        "",
        "BRIEFING DO USUÁRIO:",
        f"- Intensidade: {_INTENSITY_GUIDE[brief.intensity]}",
    ]
    if brief.target_duration_sec is not None:
        lines.append(
            f"- Duração final desejada: cerca de {brief.target_duration_sec:.0f}s. Corte o necessário "
            "para se aproximar disso, priorizando o que tem menos informação."
        )
    if not brief.remove_fillers:
        lines.append("- NÃO remova frases por serem enrolação: a categoria 'filler' está proibida.")
    if brief.instructions:
        lines.append(f"- Pedido do usuário: {brief.instructions.strip()}")
    if context.previous_error:
        lines += [
            "",
            "ATENÇÃO: sua resposta anterior foi rejeitada por este motivo:",
            f"{context.previous_error}",
            "Corrija o problema nesta nova resposta.",
        ]
    lines += ["", "FRASES (ID, duração e texto):"]
    lines += [f"[{s.id}] ({s.duration:.1f}s) {s.text}" for s in sentences]
    lines += ["", "Responda em JSON com 'removals' (pode ser vazia) e 'summary'."]
    return "\n".join(lines)
