"""Pedidos das três passadas do Editor de IA. As regras editoriais ficam AQUI, no texto para a IA, e não no
código: o código só confere segurança (IDs, sobreposição, evidência) e converte IDs em tempos.

Cada passada tem a sua versão: mudar o pedido de uma passada não invalida o cache das outras.
"""

from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.schema import ClipSummary, SummaryNote
from ai_video_editor.roteiro.report import clock

# Sobem quando o pedido ou o schema da resposta da passada mudam (invalida só o cache daquela passada).
SUMMARY_PROMPT_VERSION = "resumo-v1"
STRUCTURE_PROMPT_VERSION = "estrutura-v1"
DETAIL_PROMPT_VERSION = "detalhe-v1"

# Resumo no modo compacto da estrutura (quando a lista completa não cabe).
COMPACT_SUMMARY_CHARS = 180


def _sec(seconds: Optional[float]) -> str:
    if seconds is None:
        return "livre"
    return f"{clock(seconds)} ({round(seconds)} s)"


@dataclass
class EditorContext:
    """O pedido do usuário como chega ao editor (do briefing, se houver; senão da CLI)."""

    request: str = ""
    target_sec: Optional[float] = None
    director_instructions: str = ""
    coverage_instructions: str = ""


def _retry_lines(errors: list[str]) -> list[str]:
    if not errors:
        return []
    return [
        "",
        "Sua resposta anterior foi recusada pelo programa por estes problemas:",
        *(f"- {e}" for e in errors[:20]),
        "Corrija e responda de novo com o JSON completo.",
    ]


# --- E9: resumo de um clipe ---


def build_summary_prompt(section: str, part: int = 1, parts: int = 1, errors: Optional[list[str]] = None) -> str:
    lines = [
        "Você ajuda um editor de vídeo. Resuma UM clipe do material bruto a partir do roteiro dele: momentos "
        "(trechos com a mesma imagem, descritos por uma IA de visão, com problemas entre colchetes) e frases "
        "(fala transcrita, com ID e marcas de ritmo). Responda só com o JSON pedido, em português.",
    ]
    if parts > 1:
        lines.append(f"Este é o pedaço {part} de {parts} do clipe (o clipe é grande): resuma só o que está aqui.")
    lines += [
        "",
        "Campos:",
        "- resumo: 2 a 4 frases: do que o clipe trata, onde, quem aparece.",
        '- destaques: até 3 trechos fortes, cada um {"ids": ["S012"] ou ["P007.M04"], "motivo": "por quê"}.',
        "- problemas: até 3 (imagem ruim, fala repetida ou recomeçada, áudio), no mesmo formato.",
        "- tom: uma ou duas palavras (ex.: divertido, informativo, contemplativo).",
        "- tem_fala_para_camera: true se alguém fala olhando para a câmera.",
        "Use só IDs que aparecem no roteiro abaixo.",
        "",
        "Roteiro do clipe:",
        "",
        section.strip(),
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)


# --- E10: passada 1 (estrutura) ---


@dataclass
class StructureClip:
    """Um clipe na lista da passada 1: a linha (ID, papel, duração...) e o resumo, se houver."""

    clip_id: str
    line: str
    summary: Optional[ClipSummary] = None


def _notes(notes: list[SummaryNote]) -> str:
    return "; ".join(f"{', '.join(n.ids) or '—'} ({n.motivo})" if n.motivo else ", ".join(n.ids) for n in notes)


def clip_entry(clip: StructureClip, compact: bool = False) -> str:
    lines = [f"- {clip.line}"]
    answer = clip.summary.answer if clip.summary is not None else None
    if answer is None:
        lines.append("  resumo: (sem resumo: a IA falhou neste clipe)")
        return "\n".join(lines)
    resumo = answer.resumo
    if compact and len(resumo) > COMPACT_SUMMARY_CHARS:
        resumo = resumo[:COMPACT_SUMMARY_CHARS].rstrip() + "…"
    lines.append(f"  resumo: {resumo}")
    if not compact:
        if answer.destaques:
            lines.append(f"  destaques: {_notes(answer.destaques)}")
        if answer.problemas:
            lines.append(f"  problemas: {_notes(answer.problemas)}")
    extra = [f"tom: {answer.tom}"] if answer.tom else []
    extra.append(f"fala para a câmera: {'sim' if answer.tem_fala_para_camera else 'não'}")
    lines.append(f"  {'; '.join(extra)}")
    return "\n".join(lines)


def _request_lines(context: EditorContext) -> list[str]:
    lines = []
    if context.request.strip():
        lines.append(f'Pedido do usuário: "{context.request.strip()}"')
    else:
        lines.append("Sem pedido do usuário: escolha você o que fazer com o material.")
    if context.director_instructions.strip():
        lines.append(f"Instruções para os cortes da fala: {context.director_instructions.strip()}")
    if context.coverage_instructions.strip():
        lines.append(f"Instruções para a cobertura: {context.coverage_instructions.strip()}")
    return lines


def build_structure_prompt(
    context: EditorContext, clips: list[StructureClip], compact: bool = False, errors: Optional[list[str]] = None
) -> str:
    if context.target_sec is not None:
        target = f"Duração alvo do vídeo final: {_sec(context.target_sec)}. Use esse valor em duracao_total_alvo_sec."
    else:
        target = "Sem duração alvo: escolha você a duração do vídeo final e diga qual em duracao_total_alvo_sec."
    lines = [
        "Você é o editor de um vídeo. Monte a ESTRUTURA do vídeo final a partir dos clipes do material bruto: "
        "blocos na ordem em que vão aparecer, cada um com os clipes que contam aquela parte. "
        "Responda só com o JSON pedido, em português.",
        "",
        *_request_lines(context),
        target,
        "",
        "Regras de estilo:",
        "- Um bloco = um assunto ou uma parte da história; dentro dele, os clipes na ordem em que devem aparecer.",
        "- Qualquer clipe pode estar num bloco, de qualquer papel (um drone pode abrir o vídeo; um vídeo sem fala "
        "pode ser só de imagens).",
        "- Clipes que você não puser em nenhum bloco nem em \"fora\" ficam no acervo: os momentos deles podem "
        "aparecer por cima da fala (cobertura) ou entre as falas (respiro).",
        "- \"fora\": clipes que não devem entrar de jeito nenhum (repetidos, ruins, fora do assunto), com o motivo.",
        "- Cada clipe aparece no máximo uma vez (num bloco ou em \"fora\").",
        "- A ordem do catálogo e a data de gravação ajudam a contar a história em ordem; mude se o assunto pedir.",
        "- duracao_alvo_sec de cada bloco: quanto do vídeo final ele deve ocupar; a soma perto da duração total.",
        "- Use só IDs de clipes da lista.",
        "",
        "Clipes (ID · papel · duração · fala · idioma · data de gravação · ordem do catálogo) e resumos:",
        *(clip_entry(c, compact) for c in clips),
        "",
        'Formato: {"blocos": [{"id": "B01", "titulo": "...", "assunto": "...", "clipes": ["P001", "D002"], '
        '"duracao_alvo_sec": 60}], "fora": [{"clipe": "P009", "motivo": "..."}], "duracao_total_alvo_sec": 480, '
        '"notas": "..."}',
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)


# --- E11: passada 2 (detalhe de um bloco) ---


@dataclass
class DetailPromptInput:
    unit_id: str
    block_id: str
    title: str
    subject: str
    context: EditorContext
    target_total_sec: Optional[float]
    block_target_sec: Optional[float]
    used_before_sec: float
    part: int = 1
    parts: int = 1
    previous_sentences: list[str] = field(default_factory=list)  # "S010 — texto"
    sections: list[str] = field(default_factory=list)
    library_lines: list[str] = field(default_factory=list)


def build_detail_prompt(data: DetailPromptInput, errors: Optional[list[str]] = None) -> str:
    title = f' ("{data.title}")' if data.title else ""
    lines = [
        f"Você é o editor de um vídeo e agora monta em detalhe o bloco {data.block_id}{title}. "
        "Responda só com o JSON pedido, em português, usando só IDs que aparecem abaixo.",
        "",
        *_request_lines(data.context),
    ]
    if data.subject:
        lines.append(f"Assunto do bloco: {data.subject}")
    lines.append(
        f"Duração alvo do vídeo: {_sec(data.target_total_sec)}; deste bloco: {_sec(data.block_target_sec)}; "
        f"já usado pelos blocos anteriores: {_sec(data.used_before_sec)}."
    )
    if data.parts > 1:
        lines.append(
            f"Pedaço {data.part} de {data.parts} do bloco (o bloco foi dividido para caber): escolha só entre os "
            "clipes e momentos deste pedaço; o resto do bloco vem em seguida."
        )
    if data.previous_sentences:
        lines.append("Continuidade: as últimas frases mantidas antes deste trecho são:")
        lines += [f"  {s}" for s in data.previous_sentences]
    else:
        lines.append("Continuidade: este é o começo do vídeo.")
    lines += [
        "",
        "O que responder:",
        '- manter: a faixa principal, NA ORDEM do vídeo. Cada item é {"de": "S012", "ate": "S015"} (frases '
        'seguidas do mesmo clipe; todas entre as duas entram) OU {"momento": "P012.M02", "duracao_sec": 4} (trecho '
        "de um clipe deste bloco com o som da própria imagem; sem duracao_sec = o momento inteiro). O que não "
        "estiver em manter fica fora do vídeo.",
        '- coberturas: imagem de outro clipe POR CIMA de frases mantidas (a fala continua): {"de": "S013", "ate": '
        '"S014", "momentos": ["D002.M05"], "evidencia_fala": "palavras da fala", "evidencia_imagem": "palavras da '
        'descrição, lugar ou tags do momento", "motivo": "..."}. Nas evidências, copie palavras que estão de verdade '
        "no texto da fala e na descrição do momento. Vários momentos dividem o tempo na ordem dada.",
        '- respiros: um momento inserido entre as falas, com o som dele: {"depois_de": "S015" (ou o ID de um '
        'momento de manter), "momento": "B004.M01", "duracao_sec": 3, "motivo": "..."}.',
        "- notas: curto, o que você cortou e por quê.",
        "",
        "Regras de estilo:",
        "- Conforto auditivo: frases completas; não emende tons opostos (a não ser numa piada); não corte no meio "
        "de um raciocínio.",
        "- Corte repetição, recomeço e fala fraca.",
        "- Só cubra quando a imagem mostra o que é dito ou o lugar da cena; não tape quem fala para a câmera sem "
        "motivo forte.",
        "- Imagem ruim (problemas entre colchetes): cubra se houver algo que combine; senão, mantenha.",
        "- Respiros com moderação, em mudanças de assunto ou de lugar.",
        "- Respeite a duração alvo.",
        "- Cada frase e cada momento entram no máximo uma vez no vídeo todo; a cobertura não usa o clipe que está "
        "falando.",
        "",
        "Roteiro dos clipes do bloco:",
        "",
        *(s.strip() + "\n" for s in data.sections),
        "Acervo (momentos de outros clipes para cobertura e respiro: ID · duração · lugar · tags — descrição):",
        *(data.library_lines or ["(nenhum)"]),
        "",
        'Formato: {"manter": [{"de": "S012", "ate": "S015"}, {"momento": "P012.M02"}], "coberturas": [], '
        '"respiros": [], "notas": "..."}',
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)
