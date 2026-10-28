"""Pedidos das passadas do Editor de IA. As regras editoriais ficam AQUI, no texto para a IA, e não no
código: o código só confere segurança (IDs, sobreposição, evidência, completude), mede e converte.

Cada passada tem a sua versão: mudar o pedido de uma passada não invalida o cache das outras.

A passada 2 são três perguntas pequenas por bloco, cada uma de um só tipo (medido no vlog real: a resposta
única com cortar/incluir/coberturas/respiros misturava IDs, encadeava inclusões e ignorava a duração):

- 2a CORTES, um pedido por clipe com fala: as frases do clipe e a meta do clipe; resposta = IDs de frases
  que saem, um por item.
- 2b INCLUSÕES E RESPIROS, um pedido por bloco: os trechos mantidos numerados, a sobra de tempo do bloco, as
  posições medidas e até 6 candidatos sem fala com letra; resposta = número do trecho + letra (o código
  converte em IDs). Sem sobra (bloco com fala), não há pergunta.
- 2c COBERTURAS, um pedido por bloco: só os trechos ditos para a câmera, cada um com até 4 candidatos
  medidos com letra; resposta = trecho + letra (ou "nenhuma") + a palavra em comum copiada e/ou a
  correspondência explicada (o que a imagem mostra e por que ilustra a fala).

O pedido do usuário abre todas as perguntas (menos o resumo), com `RESPECT_REQUEST`. Na 2b e na 2c,
"nenhum" é a primeira opção e a resposta vazia é válida; nenhum exemplo mostra um respiro ou uma cobertura
escolhida (medido: os modelos copiam a quantidade dos exemplos).

Pedidos SEM nenhum ID de exemplo: o formato de cada código é descrito em palavras e a IA copia os códigos
da entrada (medido: um exemplo é copiado como se fosse escolha). O código ainda normaliza o que vier fora do
formato (`editor/ids.py`) e conta as trocas no relatório. O pedido do resumo (E9) fica como está para não
refazer os resumos em cache; os IDs que ele copia do exemplo são retirados pelo código
(`EditorService._clean_summary`).
"""

from dataclasses import dataclass, field
from typing import Optional

from ai_video_editor.editor.schema import COVERAGE_PLACE, COVERAGE_RELATED, COVERAGE_SAYS
from ai_video_editor.roteiro.report import clock

# Sobem quando o pedido ou o schema da resposta da passada mudam (invalida só o cache daquela passada).
SUMMARY_PROMPT_VERSION = "resumo-v1"
STRUCTURE_PROMPT_VERSION = "estrutura-v8"
CUTS_PROMPT_VERSION = "cortes-v6"
INSERTS_PROMPT_VERSION = "insercoes-v2"
COVERAGE_PROMPT_VERSION = "coberturas-v3"

# Logo abaixo do pedido, em todas as perguntas (menos o resumo).
RESPECT_REQUEST = "Faça o que o pedido pede; não acrescente nada que ele não peça; na dúvida, não acrescente."

# Resumo de cada clipe na lista da estrutura: até tantas palavras (a lista inteira sempre cabe).
STRUCTURE_SUMMARY_WORDS = 25
# Texto de um trecho nas listas da 2b e da 2c: até tantas palavras.
TRECHO_WORDS = 30
# Resumo de cada clipe não usado na nova tentativa da estrutura por falta de duração: até tantas palavras.
SHORTFALL_SUMMARY_WORDS = 8
# Contexto de um pedaço/parte de clipe grande (só nesses casos: a decisão depende do que veio antes): o resumo
# do pedaço anterior e as frases vizinhas, cortados nestes tamanhos. `CONTEXT_MAX_CHARS` é o que o orçamento
# reserva para esse contexto ao dividir o clipe.
CONTEXT_SUMMARY_WORDS = 50
NEIGHBOR_WORDS = 30
CONTEXT_MAX_CHARS = 700
# Marca, no roteiro da 2a, da frase que continua na seguinte (par contínuo: corta ou fica junto).
CONTINUES_MARK = "(continua na próxima)"


def _sec(seconds: Optional[float]) -> str:
    if seconds is None:
        return "livre"
    return f"{clock(seconds)} ({round(seconds)} s)"


def short_text(text: str, limit: int) -> str:
    """As primeiras `limit` palavras (com "…" se cortou)."""
    words = text.split()
    return " ".join(words) if len(words) <= limit else " ".join(words[:limit]) + "…"


@dataclass
class EditorContext:
    """O pedido do usuário como chega ao editor (do briefing, se houver; senão da CLI)."""

    request: str = ""
    target_sec: Optional[float] = None
    director_instructions: str = ""
    coverage_instructions: str = ""


# Como citar cada código (em palavras, sem ID de exemplo: um exemplo é copiado como se fosse escolha).
CLIP_CODES = "clipe = o código que abre a linha do clipe"
SENTENCE_CODES = (
    "frase = só o código S… que aparece antes do texto da frase, sem o clipe na frente e sem # ou * em volta "
    "(a frase já é única no vídeo todo)"
)


def _codes_line(*rules: str) -> str:
    return f"Códigos: responda copiando os códigos exatamente como aparecem na entrada; {'; '.join(rules)}."


# A lista de erros da nova tentativa tem no máximo tantos caracteres (o orçamento reserva esse espaço:
# `budget.RETRY_TOKENS`), cada erro no máximo `RETRY_ERROR_CHARS`.
RETRY_MAX_CHARS = 1500
RETRY_ERROR_CHARS = 500


def _retry_lines(errors: list[str]) -> list[str]:
    if not errors:
        return []
    lines, used = [], 0
    for n, error in enumerate(errors[:20]):
        line = f"- {error if len(error) <= RETRY_ERROR_CHARS else error[: RETRY_ERROR_CHARS - 1] + '…'}"
        if used + len(line) > RETRY_MAX_CHARS:
            lines.append(f"- (mais {len(errors[:20]) - n} problema(s))")
            break
        lines.append(line)
        used += len(line) + 1
    return [
        "",
        "Sua resposta anterior foi recusada pelo programa por estes problemas:",
        *lines,
        "Corrija e responda de novo com o JSON completo.",
    ]


# --- E9: resumo de um clipe (sem mudança: os resumos em cache continuam valendo) ---


def build_summary_prompt(
    section: str, part: int = 1, parts: int = 1, errors: Optional[list[str]] = None, previous: str = "",
) -> str:
    """`previous` (só do 2º pedaço em diante de um clipe grande): o resumo do pedaço anterior e a última frase
    dele, para o resumo deste pedaço continuar a história em vez de recomeçar do zero."""
    lines = [
        "Você ajuda um editor de vídeo. Resuma UM clipe do material bruto a partir do roteiro dele: momentos "
        "(trechos com a mesma imagem, descritos por uma IA de visão, com problemas entre colchetes) e frases "
        "(fala transcrita, com ID e marcas de ritmo). Responda só com o JSON pedido, em português.",
    ]
    if parts > 1:
        lines.append(f"Este é o pedaço {part} de {parts} do clipe (o clipe é grande): resuma só o que está aqui.")
        if previous:
            lines.append(f"Contexto do pedaço anterior (só para entender este; não resuma de novo): {previous}")
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


def _request_lines(context: EditorContext) -> list[str]:
    """O pedido do usuário e a instrução de respeitá-lo (abrem o texto de toda pergunta)."""
    lines = []
    if context.request.strip():
        lines.append(f'Pedido do usuário: "{context.request.strip()}"')
    else:
        lines.append("Sem pedido do usuário: escolha você o que fazer com o material.")
    lines.append(RESPECT_REQUEST)
    if context.director_instructions.strip():
        lines.append(f"Instruções para os cortes da fala: {context.director_instructions.strip()}")
    if context.coverage_instructions.strip():
        lines.append(f"Instruções para a cobertura: {context.coverage_instructions.strip()}")
    return lines


@dataclass
class StructureShortfall:
    """Falta de duração da estrutura (informação para a nova tentativa; a IA pode manter a resposta)."""

    target_sec: float
    speech_sec: float  # fala disponível dos clipes escolhidos (nos blocos)
    unused_lines: list[str] = field(default_factory=list)  # clipes com fala fora dos blocos
    library_lines: list[str] = field(default_factory=list)  # clipes sem fala fora dos blocos

    @property
    def missing_sec(self) -> float:
        return max(0.0, self.target_sec - self.speech_sec)


def _capped(lines: list[str], limit: int, what: str) -> list[str]:
    """As linhas que cabem em `limit` caracteres (o resto vira "(mais N …)")."""
    result, used = [], 0
    for n, line in enumerate(lines):
        if used + len(line) > limit:
            result.append(f"- (mais {len(lines) - n} {what})")
            break
        result.append(line)
        used += len(line) + 1
    return result


def _shortfall_lines(shortfall: StructureShortfall) -> list[str]:
    lines = [
        "",
        f"Sua resposta anterior foi lida. Duração: o pedido é de {round(shortfall.target_sec)} s; os clipes escolhidos "
        f"têm {round(shortfall.speech_sec)} s de fala; faltam {round(shortfall.missing_sec)} s.",
    ]
    budget = RETRY_MAX_CHARS
    if shortfall.unused_lines:
        lines.append("Clipes com fala que não estão em nenhum bloco (código · segundos de fala — resumo):")
        shown = _capped(shortfall.unused_lines, budget, "clipe(s) com fala")
        lines += shown
        budget -= sum(len(line) + 1 for line in shown)
    if shortfall.library_lines:
        lines.append("Acervo (clipes sem fala que não estão em nenhum bloco):")
        lines += _capped(shortfall.library_lines, max(budget, 200), "clipe(s) do acervo")
    lines.append(
        "Você pode incluir alguns desses clipes nos blocos (na ordem certa) ou manter a estrutura: responder de "
        "novo a mesma estrutura também vale. Responda com o JSON completo."
    )
    return lines


def build_structure_prompt(
    context: EditorContext,
    clip_lines: list[str],
    errors: Optional[list[str]] = None,
    shortfall: Optional[StructureShortfall] = None,
) -> str:
    """`clip_lines`: uma linha por clipe (`EditorMaterial.structure_line`), sempre todos os clipes.
    `shortfall`: a nova tentativa por falta de duração (os números e o que não foi usado)."""
    if context.target_sec is not None:
        target = f"Duração alvo do vídeo final: {_sec(context.target_sec)}. Use esse valor em duracao_total_alvo_sec."
    else:
        target = "Sem duração alvo: escolha você a duração do vídeo final e diga qual em duracao_total_alvo_sec."
    lines = [
        *_request_lines(context),
        "",
        "Você é o editor de um vídeo. Monte a ESTRUTURA do vídeo final a partir dos clipes do material bruto: "
        "blocos na ordem em que vão aparecer, cada um com os clipes que contam aquela parte. "
        "Responda só com o JSON pedido, em português.",
        target,
        "",
        "Regras de estilo:",
        "- Um bloco = um trecho da história (ex.: chegada, hotel, trilha, praia, volta), não um tema abstrato; "
        "dentro dele, os clipes na ordem em que devem aparecer.",
        "- Cada bloco tem uma função na história: diga em \"assunto\" o que ele conta e o que ele acrescenta ao "
        "vídeo. Um bloco sem função própria se junta a outro.",
        "- Continuidade: cada bloco continua o anterior (o espectador entende por que passou de um ao outro); "
        "evite repetir num bloco o que outro já contou, a não ser que o pedido peça.",
        "- Ordem: a cronológica (a data de gravação quando houver; senão, a ordem da lista) quando ela ajuda a "
        "contar a história, como num vlog ou numa viagem; se o pedido permitir uma montagem por tema, os clipes "
        "podem ser agrupados por assunto.",
        "- Use só os clipes pertinentes ao pedido. Não precisa usar todos: um clipe com \"fala sim\" que você não "
        "puser em nenhum bloco fica fora do vídeo.",
        "- \"fora\": clipes que não devem entrar (repetidos, ruins, fora do assunto), com o motivo.",
        "- Clipes com \"fala não\" vão num bloco só quando o pedido pede essas imagens ou o material não tem fala; "
        "os que você não puser em nenhum bloco nem em \"fora\" ficam no acervo (o programa pode oferecê-los "
        "depois, por cima da fala ou entre as falas).",
        "- Cada clipe aparece no máximo uma vez (num bloco ou em \"fora\").",
        "- duracao_alvo_sec de cada bloco: quanto do vídeo final ele deve ocupar; a soma perto da duração total. "
        "Um bloco não passa da fala que ele tem (os segundos de frases de cada clipe estão na lista), a não ser "
        "com imagens sem fala.",
        "- " + _codes_line(CLIP_CODES) + " Use só clipes da lista abaixo.",
        "",
        "Clipes (código · papel · duração · fala e segundos de frases · data de gravação — resumo):",
        *clip_lines,
        "",
        'Formato: {"blocos": [{"titulo": "...", "assunto": "...", "clipes": ["código do clipe", "código do clipe"], '
        '"duracao_alvo_sec": <segundos>}], "fora": [{"clipe": "código do clipe", "motivo": "..."}], '
        '"duracao_total_alvo_sec": <segundos>, "notas": "..."}',
    ]
    lines += _retry_lines(errors or [])
    if shortfall is not None:
        lines += _shortfall_lines(shortfall)
    return "\n".join(lines)


# --- 2a: cortes de um clipe ---


@dataclass
class BlockHeader:
    """O que as três perguntas de um bloco repetem: o bloco, o pedido e as durações."""

    block_id: str
    title: str
    subject: str
    context: EditorContext
    target_total_sec: Optional[float] = None
    block_target_sec: Optional[float] = None
    used_before_sec: float = 0.0


def _header_lines(header: BlockHeader, what: str) -> list[str]:
    title = f' ("{header.title}")' if header.title else ""
    lines = [
        *_request_lines(header.context),
        "",
        f"Você é o editor de um vídeo e está montando o bloco {header.block_id}{title}. Nesta pergunta, decida "
        f"só {what}. Responda só com o JSON pedido, em português.",
    ]
    if header.subject:
        lines.append(f"Assunto do bloco: {header.subject}")
    lines.append(
        f"Duração alvo do vídeo: {_sec(header.target_total_sec)}; já usado pelos blocos anteriores: "
        f"{_sec(header.used_before_sec)}; alvo deste bloco: {_sec(header.block_target_sec)}."
    )
    return lines


@dataclass
class CutsPromptInput:
    header: BlockHeader
    clip_id: str
    section: str  # roteiro do clipe (ou da parte): momentos e frases com ritmo
    available_sec: float  # duração das frases deste clipe (ou parte) ainda livres
    target_sec: Optional[float]  # meta do clipe (a do bloco repartida pela fala)
    part: int = 1
    parts: int = 1
    block_clips: int = 1  # clipes com fala no bloco
    continues: list[str] = field(default_factory=list)  # frases que continuam na seguinte (pares contínuos)
    before: str = ""  # parte de clipe grande: a frase logo antes desta parte (só contexto, sem código)
    after: str = ""  # e a frase logo depois


def mark_continuous(section: str, continues: list[str]) -> str:
    """Marca `CONTINUES_MARK` no fim da linha de cada frase de `continues` (`- S012 — texto`)."""
    if not continues:
        return section
    wanted = set(continues)
    lines = []
    for line in section.splitlines():
        if line.startswith("- S"):
            code = line[2:].split(" ", 1)[0]
            if code in wanted:
                line = f"{line} {CONTINUES_MARK}"
        lines.append(line)
    return "\n".join(lines)


def duration_retry_line(result_sec: float, target_sec: float, low: float, high: float) -> str:
    """O erro da nova tentativa por duração: os números, quantos segundos cortar ou devolver e a saída de
    quando a meta não dá sem prejudicar a fala (o sentido vem antes da duração)."""
    if result_sec > high:
        hint = f"corte cerca de {round(result_sec - target_sec)} s"
        harm = "se cortar mais prejudica o sentido da fala"
    else:
        hint = f"devolva cerca de {round(target_sec - result_sec)} s (tire frases da lista cortar)"
        harm = "se devolver frases prejudica a fala"
    return (
        f"duração: ficou {round(result_sec)} s, a meta deste clipe é {round(target_sec)} s (aceito de {round(low)} a "
        f"{round(high)} s): {hint}; {harm}, ajuste só o que não prejudica e explique em limite_meta"
    )


def sentence_retry_lines(above: bool, items: list[tuple[str, float, str]]) -> list[str]:
    """As frases com a duração para a nova tentativa por duração (código, segundos, texto): acima da meta, as
    que ficaram (para escolher o que cortar); abaixo, as cortadas (para escolher o que devolver)."""
    if not items:
        return []
    if above:
        lines = ["frases que ficaram (escolha quais cortar):"]
        word = "mantida"
    else:
        lines = ["frases que você cortou (para devolver uma, tire-a da lista cortar):"]
        word = "cortada"
    lines += [f"{word} {sid} ({round(sec)} s): {short_text(text, 10)}" for sid, sec, text in items]
    return lines


def build_cuts_prompt(data: CutsPromptInput, errors: Optional[list[str]] = None) -> str:
    part = f" (parte {data.part} de {data.parts} do clipe: decida só sobre as frases desta parte)" if data.parts > 1 else ""
    lines = _header_lines(data.header, f"QUAIS FRASES DO CLIPE {data.clip_id} SAEM do vídeo{part}")
    if data.before or data.after:
        lines.append("Frases vizinhas desta parte, de outras partes do clipe (só contexto; não decida sobre elas):")
        if data.before:
            lines.append(f"- logo antes: \"{data.before}\"")
        if data.after:
            lines.append(f"- logo depois: \"{data.after}\"")
    if data.block_clips > 1:
        lines.append(f"O bloco tem {data.block_clips} clipes com fala; cada um é decidido numa pergunta como esta.")
    if data.target_sec is not None:
        what = "Esta parte do clipe" if data.parts > 1 else "Este clipe"
        meta = (
            f"{what} tem {round(data.available_sec)} s de fala; a meta é {round(data.target_sec)} s; corte cerca de "
            f"{max(0, round(data.available_sec - data.target_sec))} s de frases (a meta do bloco repartida entre os "
            "clipes pela fala de cada um; o resto fica)."
        )
    else:
        meta = f"Frases deste clipe: {_sec(data.available_sec)} no total. Sem meta de duração: corte só o que não serve."
    lines += [
        meta,
        "A meta é uma restrição, mas o sentido da fala vem antes: chegue perto dela sem deixar a fala confusa, "
        "incompleta ou sem a informação que importa. Se não der para chegar à meta sem prejudicar a fala, corte só "
        "o que não prejudica e diga por quê em limite_meta (uma frase curta); dando, deixe limite_meta vazio.",
        "",
        "Como funciona: POR PADRÃO TODAS AS FRASES FICAM, na ordem do roteiro. Você lista só as que saem.",
        "",
        "O que cortar (regras de estilo):",
        "- Corte o que você, como editor, tiraria deste clipe para atender ao pedido e à meta; o resto fica.",
        "- A mesma frase dita mais de uma vez: fica a melhor, em geral a última.",
        "- Conforto auditivo: não corte no meio de um raciocínio; não deixe uma pergunta sem a resposta.",
        "- Imagem ruim (problemas entre colchetes nos momentos) NÃO é motivo para cortar a fala: a imagem pode "
        "ser coberta depois.",
        f"- Frase marcada {CONTINUES_MARK} termina no meio da fala e segue na frase seguinte: corte as duas ou "
        "mantenha as duas (cortar só uma deixa a fala começando ou acabando pela metade; o programa junta pela "
        "maior parte).",
        f"- Se {'esta parte do clipe' if data.parts > 1 else 'o clipe'} inteiro não serve para o pedido, responda "
        "fora: true (sai do bloco) em vez de cortar frase por frase.",
        "",
        "- " + _codes_line(SENTENCE_CODES) + " Uma frase por item: para cortar várias frases seguidas, um item "
        "para cada uma.",
        "",
        f"Roteiro do clipe {data.clip_id} (momentos com a descrição da imagem; frases com o código, o texto e o ritmo):",
        "",
        mark_continuous(data.section.strip(), data.continues),
        "",
        "- motivo: escreva com as suas palavras, numa frase curta, por que AQUELA frase sai (não há lista de "
        "motivos para escolher).",
        "",
        'Formato: {"cortar": [{"frase": "código da frase", "motivo": "por que esta frase sai"}], "fora": false, '
        '"limite_meta": "", "notas": "..."}',
        "Nada a cortar: {\"cortar\": [], \"fora\": false, \"limite_meta\": \"\", \"notas\": \"...\"}",
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)


# --- 2b: inclusões e respiros de um bloco ---


@dataclass
class InsertsPromptInput:
    header: BlockHeader
    trecho_lines: list[str]  # "1: S012–S015 (P006, 6 s): texto…"
    candidate_lines: list[str]  # "A: descrição · lugar: lago · 9,0s · B004 (acervo)"
    current_sec: float  # duração do bloco com os cortes da 2a (só a fala)
    library_left: int = 0  # candidatos do acervo que não couberam no pedido
    spare_sec: Optional[float] = None  # meta do bloco − fala mantida (None = sem meta)
    own_positions: list[int] = field(default_factory=list)  # onde pode entrar imagem do bloco
    breath_position_lines: list[str] = field(default_factory=list)  # "2 (troca de lugar: quarto → rua)"
    breath_max_sec: float = 6.0  # duração máxima de um respiro (`validate.RESPIRO_MAX_SEC`)


def build_inserts_prompt(data: InsertsPromptInput, errors: Optional[list[str]] = None) -> str:
    lines = _header_lines(data.header, "SE ENTRA ALGUMA IMAGEM SEM FALA ENTRE OS TRECHOS DE FALA (em geral, nenhuma)")
    lines.append(f"Duração atual do bloco (só a fala que ficou): {_sec(data.current_sec)}.")
    if data.spare_sec is not None:
        lines.append(
            f"Sobram {round(data.spare_sec)} s neste bloco até o alvo: as imagens escolhidas somam no máximo isso "
            "(o programa recusa as que passam)."
        )
    else:
        lines.append("Sem alvo para este bloco.")
    lines += [
        "",
        "Como funciona: a fala do bloco já está decidida, em trechos numerados na ordem do vídeo. Você PODE pôr "
        "imagens sem fala (com o som delas) nas posições listadas abaixo; cada imagem tem uma letra. A primeira "
        "opção é \"nenhum\": a resposta vazia é válida e esperada quando nada for necessário.",
        "",
        "Regras de estilo:",
        "- Na dúvida, nenhuma imagem.",
        f"- Respiro (imagem do acervo) é uma pausa curta entre falas, de no máximo {data.breath_max_sec:g} s: só onde "
        "o lugar muda (as posições listadas), nunca dois seguidos.",
        "- Bloco sem fala: as imagens SÃO o bloco; escolha as que contam aquela parte, na ordem.",
        "",
        "Trechos de fala (número: frases · clipe · duração: texto):",
        *(data.trecho_lines or ["(nenhum: o bloco não tem fala; use depois_do_trecho = 0)"]),
        "",
        "Opções (letra: descrição · lugar · duração · clipe · de onde):",
        "- nenhum: não incluir nada (resposta com a lista inserir vazia)",
        *data.candidate_lines,
    ]
    if data.library_left:
        lines.append(f"(mais {data.library_left} imagem(ns) do acervo não couberam nesta lista)")
    lines += ["", "Onde cada imagem pode entrar (depois_do_trecho; 0 = antes do 1º trecho):"]
    own = ", ".join(str(p) for p in data.own_positions) or "nenhuma"
    lines.append(f"- imagens \"do bloco\": {own}")
    if data.breath_position_lines:
        lines.append(f"- respiros (\"acervo\"): {'; '.join(data.breath_position_lines)}")
    else:
        lines.append("- respiros (\"acervo\"): nenhuma posição")
    lines += [
        "",
        "Responda a lista inserir: um item por imagem que entra, com depois_do_trecho = uma posição da lista "
        "acima para aquele tipo de imagem, letra = a letra da imagem e, opcional, duracao_sec (menor que a da "
        "imagem). Cada letra no máximo uma vez. Várias imagens na mesma posição entram na ordem da lista.",
        "",
        'Formato: {"inserir": [], "notas": "curto"} (lista vazia = nenhuma imagem; cada item: {"depois_do_trecho": '
        '<número>, "letra": "<letra>", "duracao_sec": <segundos ou nada>})',
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)


# --- 2c: coberturas de um bloco ---


@dataclass
class CoverageQuestion:
    """Um trecho perguntado na 2c e os seus candidatos já formatados."""

    trecho_line: str  # "Trecho 3 (S012–S015 · P006 · 6 s; imagem de quem fala: …): texto"
    candidate_lines: list[str] = field(default_factory=list)  # "C: … — em comum: lago"


@dataclass
class CoveragePromptInput:
    header: BlockHeader
    questions: list[CoverageQuestion]
    part: int = 1
    parts: int = 1


def build_coverage_prompt(data: CoveragePromptInput, errors: Optional[list[str]] = None) -> str:
    lines = _header_lines(data.header, "QUE IMAGEM APARECE POR CIMA DE CADA TRECHO DE FALA (cobertura)")
    if data.parts > 1:
        lines.append(f"Parte {data.part} de {data.parts} dos trechos deste bloco.")
    lines += [
        "",
        "Como funciona: os trechos abaixo são de alguém falando para a câmera. A fala continua; a imagem de outro "
        "clipe pode aparecer por cima dela, a partir da frase que ela ilustra. Para cada trecho, a primeira opção é "
        "\"nenhuma\" (a imagem de quem fala continua); depois, até 4 imagens candidatas com letra: "
        f"\"{COVERAGE_SAYS}\" = uma palavra da fala aparece na descrição da imagem; \"{COVERAGE_PLACE}\" = a "
        f"imagem é do mesmo lugar onde a fala acontece; \"{COVERAGE_RELATED}\" = só as tags da imagem lembram a "
        "fala (as tags podem estar erradas: julgue pela descrição).",
        "",
        "Evidência (obrigatória para escolher uma letra): a imagem precisa mostrar de verdade o que é dito ou o lugar.",
        "- Diga em mostra o que a imagem realmente mostra, com as palavras da descrição dela, e em porque, em "
        "poucas palavras, que parte da fala ela ilustra (cite a palavra da fala) e por quê.",
        "- Em palavra, copie a palavra em comum da linha da letra quando ela existir; quando a imagem ilustra a "
        "fala com outras palavras (o mesmo objeto, ação ou lugar dito de outro jeito), deixe palavra vazia: mostra "
        "e porque são a evidência.",
        "- Sem evidência suficiente, \"nenhuma\". Uma palavra em comum sozinha não basta.",
        "",
        "Regras de estilo:",
        "- \"nenhuma\" em todos os trechos é uma resposta válida e esperada quando nenhuma imagem mostra o que é "
        "dito; na dúvida, \"nenhuma\".",
        "- Imagem ruim de quem fala (problemas entre colchetes) é motivo para cobrir.",
        "- Cada letra no máximo uma vez.",
        "",
    ]
    for question in data.questions:
        lines.append(question.trecho_line)
        lines.append("  nenhuma: a imagem de quem fala continua")
        lines += [f"  {c}" for c in question.candidate_lines]
        lines.append("")
    lines += [
        "Responda a lista coberturas, um item por trecho acima: trecho = o número do trecho, letra = \"nenhuma\" "
        "ou a letra escolhida; com \"nenhuma\", palavra, mostra e porque vazios.",
        "",
        'Formato: {"coberturas": [{"trecho": <número>, "letra": "nenhuma", "palavra": "", "mostra": "", '
        '"porque": ""}], "notas": "curto"}',
    ]
    lines += _retry_lines(errors or [])
    return "\n".join(lines)
