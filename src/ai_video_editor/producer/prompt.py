"""Prompt do Produtor (independente do provedor)."""

from ai_video_editor.producer.schema import MaterialSummary, ProducerContext, RecipeInfo


def _clock(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}min{secs:02d}s"


def describe_material(material: MaterialSummary) -> list[str]:
    """Linhas do resumo do material, uma por papel."""
    kind = "um arquivo de vídeo" if material.kind == "arquivo" else "uma pasta de clipes"
    lines = [f"Material: {kind}, {material.clips} clipe(s), {_clock(material.duration_sec)} no total."]
    for r in material.roles:
        if r.speech_share is not None:
            speech = f"fala em {r.speech_share:.0%} do tempo, {r.clips_with_speech}/{r.clips} clipes com fala"
        elif r.speech_unknown:
            speech = "fala não analisada"
        else:
            speech = "sem áudio"
        if r.speech_share is not None and r.speech_unknown:
            speech += f", {r.speech_unknown} sem análise"
        log = f"; {r.log_suspects} com suspeita de perfil log" if r.log_suspects else ""
        lines.append(
            f"- papel {r.role}: {r.clips} clipe(s), {_clock(r.duration_sec)}, {speech}; "
            f"formatos: {', '.join(r.formats)}{log}"
        )
    return lines


def build_producer_prompt(
    request: str, material: MaterialSummary, recipes: list[RecipeInfo], context: ProducerContext
) -> str:
    """Monta o prompt do Produtor: papel, regras, material resumido, receitas e pedido."""
    lines = [
        "Você é o produtor de um editor de vídeo automático. Lê o pedido do usuário e um resumo do "
        "material e escreve o briefing que guia as próximas etapas. Você não corta nada: escolhe uma "
        "receita existente e escreve metas e instruções.",
        "",
        "REGRAS OBRIGATÓRIAS:",
        "1. 'recipe' é exatamente o nome de uma das receitas abaixo. Nunca invente receitas ou etapas.",
        "2. A receita precisa servir para o material: receita de 1 vídeo não serve para vários clipes com "
        "fala; receita que precisa de fala não serve para material sem fala.",
        "3. 'roles_to_use' só pode citar papéis que existem no material.",
        "4. 'target_duration_sec' só se o usuário pedir uma duração, e menor que a duração do material.",
        "5. 'understood' é UMA frase em português começando com 'Entendi: ', dizendo o que será feito.",
        "6. 'intensity' (quanto cortar da fala): light = só erros evidentes; medium = também repetições "
        "e digressões; aggressive = vídeo enxuto. Sem pedido de corte forte, use light.",
        "7. 'pace' (pausas entre falas): natural = mais folga; normal; tight = ritmo rápido.",
        "8. 'director_instructions': o que o Diretor de cortes da fala deve priorizar, tirado do pedido. "
        "'coverage_instructions': como usar B-roll, drone, POV e som ambiente (ex.: 'respiros de "
        "paisagem com som ambiente', 'abrir com drone'); vazio se não houver material de cobertura.",
        "9. 'warnings': dúvidas ou partes do pedido que não dá para atender; em português.",
        "",
        *describe_material(material),
        "",
        "RECEITAS DISPONÍVEIS:",
    ]
    for recipe in recipes:
        clips = "vários clipes" if recipe.multi_clip else "1 vídeo"
        speech = "precisa de fala" if recipe.needs_speech else "não precisa de fala"
        lines.append(f"- {recipe.name}: {recipe.description} ({clips}; {speech})")
    if context.forced_recipe:
        lines += [
            "",
            f"O usuário escolheu a receita '{context.forced_recipe}'. Use exatamente ela e preencha o resto.",
        ]
    if context.previous_error:
        lines += [
            "",
            "ATENÇÃO: sua resposta anterior foi rejeitada por este motivo:",
            context.previous_error,
            "Corrija o problema nesta nova resposta.",
        ]
    lines += [
        "",
        "PEDIDO DO USUÁRIO:",
        request.strip() or "(vazio: o usuário não pediu nada específico; proponha a edição mais natural "
        "para o material)",
        "",
        "Responda em JSON no schema pedido.",
    ]
    return "\n".join(lines)
