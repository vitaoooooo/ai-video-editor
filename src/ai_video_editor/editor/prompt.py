"""Pedidos das três passadas do Editor de IA. As regras editoriais ficam AQUI, no texto para a IA, e não no
código: o código só confere segurança (IDs, sobreposição, evidência, completude) e converte IDs em tempos.

Cada passada tem a sua versão: mudar o pedido de uma passada não invalida o cache das outras.

Passadas 1 e 2 SEM nenhum ID de exemplo: o formato de cada código é descrito em palavras ("frase = o código
S… que aparece antes do texto") e a IA copia os códigos da entrada, que já os mostra exatamente como devem
voltar. Medido no vlog real: o exemplo `S012` (detalhe-v1) era copiado como escolha; o exemplo fictício
`X01.S900` (detalhe-v2) ensinou o formato CLIPE.Sxxx (`P009.S025`, `X018.S078`), que os dois modelos
(qwen2.5:7b e gemini) passaram a escrever. O código ainda normaliza o que vier fora do formato
(`editor/ids.py`) e conta as trocas no relatório. O pedido do resumo (E9) fica como está para não refazer os
resumos em cache; os IDs que ele copia do exemplo são retirados pelo código (`EditorService._clean_summary`).
"""

from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.schema import COVERAGE_PLACE, COVERAGE_SAYS, ClipSummary, SummaryNote
from ai_video_editor.roteiro.report import clock

# Sobem quando o pedido ou o schema da resposta da passada mudam (invalida só o cache daquela passada).
SUMMARY_PROMPT_VERSION = "resumo-v1"
STRUCTURE_PROMPT_VERSION = "estrutura-v3"
DETAIL_PROMPT_VERSION = "detalhe-v3"

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


# Como citar cada código (em palavras, sem ID de exemplo: um exemplo é copiado como se fosse escolha).
CLIP_CODES = "clipe = o código que abre a linha do clipe (ou o título ## no roteiro)"
SENTENCE_CODES = (
    "frase = só o código S… que aparece antes do texto da frase, sem o clipe na frente (a frase já é única no "
    "vídeo todo)"
)
MOMENT_CODES = "momento = o código completo que abre o título do momento (### …) ou a linha nas listas"


def _codes_line(*rules: str) -> str:
    return f"Códigos: responda copiando os códigos exatamente como aparecem na entrada; {'; '.join(rules)}."


def _retry_lines(errors: list[str]) -> list[str]:
    if not errors:
        return []
    return [
        "",
        "Sua resposta anterior foi recusada pelo programa por estes problemas:",
        *(f"- {e}" for e in errors[:20]),
        "Corrija e responda de novo com o JSON completo.",
    ]


# --- E9: resumo de um clipe (sem mudança: os resumos em cache continuam valendo) ---


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
    """Um clipe na lista da passada 1: a linha (ID, papel, duração...), o tamanho no detalhe e o resumo."""

    clip_id: str
    line: str
    summary: Optional[ClipSummary] = None
    tokens: Optional[int] = None  # tamanho da seção do clipe no pedido do detalhe


def _notes(notes: list[SummaryNote]) -> str:
    return "; ".join(f"{', '.join(n.ids) or '—'} ({n.motivo})" if n.motivo else ", ".join(n.ids) for n in notes)


def clip_entry(clip: StructureClip, compact: bool = False) -> str:
    size = f" · ~{clip.tokens} tokens" if clip.tokens is not None else ""
    lines = [f"- {clip.line}{size}"]
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
    context: EditorContext,
    clips: list[StructureClip],
    compact: bool = False,
    errors: Optional[list[str]] = None,
    block_tokens: Optional[int] = None,
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
        "- Um bloco = um trecho da história (ex.: chegada, hotel, trilha, praia, volta), não um tema abstrato; "
        "dentro dele, os clipes na ordem em que devem aparecer.",
        "- Vlog ou viagem: a ordem dos blocos e dos clipes é a cronológica (a data de gravação quando houver; "
        "senão, a ordem do catálogo), a não ser que o pedido peça outra.",
        "- TODO clipe \"com fala\" vai num bloco ou em \"fora\" (com o motivo). Nenhum clipe com fala pode ficar "
        "esquecido.",
        "- \"fora\": clipes que não devem entrar de jeito nenhum (repetidos, ruins, fora do assunto), sempre com "
        "o motivo.",
        "- Clipes \"sem fala\" podem ir num bloco (um drone pode abrir o vídeo; um vídeo sem fala pode ser só de "
        "imagens) ou ficar de fora da lista: os que você não puser em nenhum bloco nem em \"fora\" ficam no "
        "acervo, e os momentos deles podem aparecer por cima da fala (cobertura) ou entre as falas (respiro).",
        "- Cada clipe aparece no máximo uma vez (num bloco ou em \"fora\").",
        "- duracao_alvo_sec de cada bloco: quanto do vídeo final ele deve ocupar; a soma perto da duração total.",
    ]
    if block_tokens is not None:
        lines.append(
            f"- Tamanho: cada bloco é detalhado depois num pedido só, que cabe ~{block_tokens} tokens de roteiro. "
            "Prefira blocos cuja soma dos tokens dos clipes caiba nisso; um bloco maior é dividido em pedaços, e "
            "cada pedaço é editado sem ver o resto do bloco."
        )
    lines += [
        "- " + _codes_line(CLIP_CODES) + " Use só clipes da lista abaixo.",
        "",
        "Clipes (código · papel · duração · fala · idioma · data de gravação · ordem do catálogo · tamanho) e resumos:",
        *(clip_entry(c, compact) for c in clips),
        "",
        'Formato: {"blocos": [{"titulo": "...", "assunto": "...", "clipes": ["código do clipe", "código do clipe"], '
        '"duracao_alvo_sec": 60}], "fora": [{"clipe": "código do clipe", "motivo": "..."}], '
        '"duracao_total_alvo_sec": 300, "notas": "..."}',
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
    silent_lines: list[str] = field(default_factory=list)  # momentos sem fala do pedaço (os que `incluir` aceita)
    available_sec: float = 0.0  # duração das frases livres do pedaço
    unit_target_sec: Optional[float] = None  # alvo deste pedaço (o do bloco, proporcional se dividido)
    clip_ids: list[str] = field(default_factory=list)


def duration_retry_line(result_sec: float, target_sec: float, low: float, high: float) -> str:
    """O erro da nova tentativa por duração: os números e o que ajustar."""
    hint = "corte mais frases" if result_sec > high else "corte menos (ou inclua momentos sem fala)"
    return (
        f"duração: ficou {round(result_sec)} s, o alvo deste bloco é {round(target_sec)} s (aceito de {round(low)} a "
        f"{round(high)} s); ajuste os cortes: {hint}"
    )


def build_detail_prompt(data: DetailPromptInput, errors: Optional[list[str]] = None) -> str:
    title = f' ("{data.title}")' if data.title else ""
    lines = [
        f"Você é o editor de um vídeo e agora monta em detalhe o bloco {data.block_id}{title}. "
        "Responda só com o JSON pedido, em português, usando só códigos que aparecem no roteiro e nas listas abaixo.",
        _codes_line(SENTENCE_CODES, MOMENT_CODES, CLIP_CODES),
        "",
        *_request_lines(data.context),
    ]
    if data.subject:
        lines.append(f"Assunto do bloco: {data.subject}")
    if data.clip_ids:
        lines.append(f"Clipes do bloco, na ordem da estrutura: {', '.join(data.clip_ids)}")
    lines += [
        f"Duração alvo do vídeo: {_sec(data.target_total_sec)}; já usado pelos blocos anteriores: "
        f"{_sec(data.used_before_sec)}.",
        f"Duração total das frases deste bloco{' (deste pedaço)' if data.parts > 1 else ''}: "
        f"{_sec(data.available_sec)}; alvo deste bloco: {_sec(data.block_target_sec)}"
        + (f"; deste pedaço: {_sec(data.unit_target_sec)}" if data.parts > 1 else "")
        + ".",
    ]
    if data.parts > 1:
        lines.append(
            f"Pedaço {data.part} de {data.parts} do bloco (o bloco foi dividido para caber): decida só sobre os "
            "clipes e momentos deste pedaço; o resto do bloco vem em seguida."
        )
    if data.previous_sentences:
        lines.append("Continuidade: as últimas frases antes deste trecho são:")
        lines += [f"  {s}" for s in data.previous_sentences]
    else:
        lines.append("Continuidade: este é o começo do vídeo.")
    lines += [
        "",
        "Como funciona: POR PADRÃO TODAS AS FRASES DO BLOCO FICAM, na ordem do roteiro. Você diz o que sai e o "
        "que entra a mais.",
        "",
        "O que responder:",
        '- cortar: frases que saem, cada item {"de": "código da 1ª frase", "ate": "código da última frase", '
        '"motivo": "curto"} (frases seguidas do mesmo clipe; uma frase só: de = ate). O que não estiver em cortar '
        "fica no vídeo.",
        '- incluir: momentos SEM fala (só os da lista "Momentos sem fala") que entram na faixa principal com o som '
        'deles: {"momento": "código do momento", "depois_de": "código da frase que ficou, ou de outro momento '
        'incluído; vazio = no começo do bloco", "duracao_sec": 4 (opcional; sem ela, o momento inteiro)}.',
        '- ordem: opcional, os códigos dos clipes do bloco na ordem em que entram; sem ela, a ordem da estrutura.',
        '- coberturas: imagem de outro clipe POR CIMA de frases que ficaram (a fala continua): {"de": "código da '
        '1ª frase", "ate": "código da última frase", "momentos": ["código do momento"], '
        f'"tipo": "{COVERAGE_SAYS}" ou "{COVERAGE_PLACE}", '
        '"evidencia_fala": "até 6 palavras", "evidencia_imagem": "até 6 palavras", "motivo": "..."}. '
        "evidencia_fala: até 6 palavras COPIADAS das frases cobertas. evidencia_imagem: até 6 palavras COPIADAS da "
        "descrição, do lugar ou das tags do momento usado (não de outro momento). "
        f'"{COVERAGE_SAYS}": a imagem mostra a coisa falada (as duas evidências têm uma palavra em comum, ex.: '
        f'fala "o ônibus chegou", imagem "ônibus na estrada"). "{COVERAGE_PLACE}": a imagem mostra o lugar onde a '
        "fala acontece (o lugar do momento usado é o mesmo do momento da fala). Vários momentos dividem o tempo "
        "na ordem dada.",
        '- respiros: um momento inserido entre as falas, com o som dele: {"depois_de": "código da frase que ficou '
        'ou do momento incluído", "momento": "código do momento", "duracao_sec": 3, "motivo": "..."}.',
        "- notas: curto, o que você cortou e por quê.",
        "",
        "Regras de estilo:",
        "- Chegue perto do alvo deste bloco: se as frases passam do alvo, corte; se faltam, corte pouco e use "
        "momentos sem fala.",
        "- Corte repetição, recomeço (a mesma frase dita de novo: fica a melhor, em geral a última) e fala fraca.",
        "- Conforto auditivo: frases completas; não emende tons opostos (a não ser numa piada); não corte no meio "
        "de um raciocínio.",
        "- Só cubra quando a imagem mostra o que é dito ou o lugar da cena; não tape quem fala para a câmera sem "
        "motivo forte.",
        "- Imagem ruim (problemas entre colchetes): cubra se houver algo que combine; senão, mantenha.",
        "- Respiros com moderação, em mudanças de assunto ou de lugar.",
        "- Cada frase e cada momento entram no máximo uma vez no vídeo todo; a cobertura não usa o clipe que está "
        "falando.",
        "",
        "Roteiro dos clipes do bloco:",
        "",
        *(s.strip() + "\n" for s in data.sections),
        "Momentos sem fala deste bloco (os únicos que incluir aceita: código · duração · lugar · tags — descrição):",
        *(data.silent_lines or ["(nenhum)"]),
        "",
        "Acervo (momentos de outros clipes para cobertura e respiro: código · duração · lugar · tags — descrição):",
        *(data.library_lines or ["(nenhum)"]),
        "",
        'Formato: {"cortar": [{"de": "código da frase", "ate": "código da frase", "motivo": "recomeço"}], '
        '"incluir": [], "ordem": [], "coberturas": [], "respiros": [], "notas": "..."}',
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)
