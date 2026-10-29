"""Schemas Pydantic do Editor de IA: as respostas das passadas (só IDs, letras e números) e o que o código registra.

As respostas são tolerantes de propósito (campos opcionais, listas cortadas no limite): um item ruim é
descartado sozinho pela validação do código (`editor/validate.py`), em vez de derrubar a resposta inteira.
Os nomes dos campos das respostas são os do pedido à IA (em português); os do código, em inglês.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from ai_video_editor.llm.schema import ModelSwitch

# Sobe quando o formato do editor_run.json / editor_plano.json muda.
EDITOR_SCHEMA_VERSION = 9

MAX_HIGHLIGHTS = 3
MAX_PROBLEMS = 3
# Motivo de um corte da 2a: texto livre, cortado neste tamanho (não há lista de motivos).
MAX_REASON_CHARS = 120

# ia: resposta validada do modelo; cache: a mesma resposta, de uma execução anterior; fallback: o código
# montou o resultado neutro porque a IA falhou de vez (aviso no relatório).
Origin = Literal["ia", "cache", "fallback"]


def _text(value) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _id(value) -> Optional[str]:
    """`s012` / ` S012 ` → `S012`; vazio → None (a validação diz que falta)."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _optional_number(value):
    """Número da IA ou None ("", "auto", texto): duração opcional não derruba o item."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ".").replace("s", "").strip())
    except ValueError:
        return None


# --- E9: resumo de um clipe ---


class SummaryNote(BaseModel):
    """Destaque ou problema de um clipe: IDs de frases/momentos e o porquê."""

    ids: list[str] = Field(default_factory=list, description="IDs de frases (S012) ou momentos (P007.M04)")
    motivo: str = Field(default="", description="Por que é forte (destaque) ou o que está ruim (problema)")

    @field_validator("ids", mode="before")
    @classmethod
    def _ids(cls, value):
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, list):
            return value
        return [i for i in (_id(v) for v in value) if i]

    @field_validator("motivo", mode="before")
    @classmethod
    def _motivo(cls, value):
        return _text(value)


class ClipSummaryAnswer(BaseModel):
    """Resposta da IA para um clipe (ou uma parte de um clipe grande)."""

    resumo: str = Field(..., min_length=1, description="2 a 4 frases: do que trata, onde, quem")
    destaques: list[SummaryNote] = Field(default_factory=list, description=f"Até {MAX_HIGHLIGHTS} trechos fortes")
    problemas: list[SummaryNote] = Field(default_factory=list, description=f"Até {MAX_PROBLEMS} problemas")
    tom: str = Field(default="", description="Curto, ex.: divertido, informativo, contemplativo")
    tem_fala_para_camera: bool = Field(default=False, description="Alguém fala olhando para a câmera")

    @field_validator("resumo", "tom", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)

    @field_validator("destaques", mode="after")
    @classmethod
    def _highlights(cls, value):
        return value[:MAX_HIGHLIGHTS]

    @field_validator("problemas", mode="after")
    @classmethod
    def _problems(cls, value):
        return value[:MAX_PROBLEMS]


class ClipSummary(BaseModel):
    """Resumo de um clipe como o código guarda: a resposta (juntada, se o clipe foi em partes) e de onde veio."""

    clip_id: str
    answer: Optional[ClipSummaryAnswer] = None
    origin: Literal["ia", "cache", "falha"] = "ia"
    parts: int = Field(default=1, ge=1, description="Chamadas em que o clipe foi dividido (seção maior que o orçamento)")
    cached_parts: int = 0
    error: str = ""
    warnings: list[str] = Field(default_factory=list)
    normalized_ids: list[str] = Field(
        default_factory=list, description="IDs da resposta aceita trocados pelo formato do roteiro (\"bruto → ID\")"
    )


# --- E10: passada 1 (estrutura) ---


class StructureBlock(BaseModel):
    id: str = Field(default="", description="B01, B02... (o código renumera na ordem final)")
    titulo: str = ""
    assunto: str = ""
    clipes: list[str] = Field(default_factory=list, description="IDs de clipes, em ordem")
    duracao_alvo_sec: Optional[float] = None

    @field_validator("clipes", mode="before")
    @classmethod
    def _clips(cls, value):
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, list):
            return value
        return [i for i in (_id(v) for v in value) if i]

    @field_validator("titulo", "assunto", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)

    @field_validator("duracao_alvo_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)


class OutClip(BaseModel):
    clipe: str = ""
    motivo: str = ""

    @field_validator("clipe", mode="before")
    @classmethod
    def _clip(cls, value):
        return _id(value) or ""

    @field_validator("motivo", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


class StructureAnswer(BaseModel):
    """Resposta da passada 1: blocos na ordem final, clipes fora e a duração alvo."""

    blocos: list[StructureBlock] = Field(default_factory=list)
    fora: list[OutClip] = Field(default_factory=list)
    duracao_total_alvo_sec: Optional[float] = None
    notas: str = ""

    @field_validator("duracao_total_alvo_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)

    @field_validator("notas", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


# --- E11: passada 2 em três perguntas por bloco (2a cortes, 2b inclusões e respiros, 2c coberturas) ---

# Palavras que a IA pode mandar no lugar de uma letra para dizer "nenhuma".
_NONE_TOKENS = {"NENHUMA", "NENHUM", "NADA", "NAO", "NÃO", "NONE", "-", "—", "X"}
# Tokens que a IA pode mandar em `depois_do_trecho` para dizer "no começo do bloco".
_START_TOKENS = {"INICIO", "INÍCIO", "COMECO", "COMEÇO", "ANTES"}


def _letter(value) -> Optional[str]:
    """`"a"`, `"A)"`, `"**B**"`, `"letra C"` → a letra; vazio → None (a validação diz que falta)."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text in _NONE_TOKENS:
        return text
    found = re.findall(r"[A-ZÀ-Ý]+", text)
    if not found:
        return text
    last = found[-1]
    if last in _NONE_TOKENS:
        return last
    return last if len(last) <= 2 else text


def _number(value):
    """`3`, `"3"`, `"trecho 3"`, `"início"` → int; outro texto fica como veio (a validação recusa)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    text = str(value).strip().upper()
    if not text or text in _START_TOKENS:
        return 0
    digits = re.findall(r"\d+", text)
    return int(digits[0]) if len(digits) == 1 else None


class CutItem(BaseModel):
    """Corte da 2a: UMA frase (o ID) e o motivo (texto livre curto, sem lista fechada)."""

    frase: Optional[str] = None
    motivo: str = ""

    @model_validator(mode="before")
    @classmethod
    def _plain_id(cls, value):
        """`"S012"` sozinho = cortar essa frase; `{"id": ...}` / `{"de": ...}` também viram `frase`."""
        if isinstance(value, str):
            return {"frase": value}
        if isinstance(value, dict) and not value.get("frase"):
            for name in ("id", "de", "sentence"):
                if value.get(name):
                    value = {**value, "frase": value[name]}
                    break
        return value

    @field_validator("frase", mode="before")
    @classmethod
    def _sentence(cls, value):
        return _id(value)

    @field_validator("motivo", mode="before")
    @classmethod
    def _strip(cls, value):
        text = _text(value)
        return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1].rstrip() + "…"


class CutsAnswer(BaseModel):
    """Resposta da 2a para um clipe (ou parte): as frases que saem; o resto fica. `fora` = o clipe (ou a
    parte) inteiro sai do bloco (decisão da IA). `limite_meta` = a IA diz por que não dá para chegar à meta
    sem prejudicar a fala (vazio = não há limite)."""

    cortar: list[CutItem] = Field(default_factory=list)
    fora: bool = False
    limite_meta: str = ""
    notas: str = ""

    @field_validator("limite_meta", mode="before")
    @classmethod
    def _limit(cls, value):
        text = _text(value)
        if text.lower() in {"", "-", "nenhum", "nenhuma", "não", "nao", "none", "null"}:
            return ""
        return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1].rstrip() + "…"

    @field_validator("fora", mode="before")
    @classmethod
    def _out(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "sim", "yes", "1", "verdadeiro"}
        return bool(value) if value is not None else False

    @field_validator("notas", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


class InsertItem(BaseModel):
    """Inclusão da 2b: depois de que trecho numerado (0 = começo) entra o candidato da letra."""

    depois_do_trecho: Optional[float] = None
    letra: Optional[str] = None
    duracao_sec: Optional[float] = None

    @field_validator("depois_do_trecho", mode="before")
    @classmethod
    def _after(cls, value):
        return _number(value)

    @field_validator("letra", mode="before")
    @classmethod
    def _letter(cls, value):
        return _letter(value)

    @field_validator("duracao_sec", mode="before")
    @classmethod
    def _duration(cls, value):
        return _optional_number(value)


class InsertsAnswer(BaseModel):
    """Resposta da 2b para um bloco: candidatos sem fala que entram entre os trechos."""

    inserir: list[InsertItem] = Field(default_factory=list)
    notas: str = ""

    @field_validator("notas", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


class CoverageItem(BaseModel):
    """Cobertura da 2c: o trecho, a letra escolhida (ou "nenhuma"), a palavra em comum copiada e, para a
    correspondência sem palavra em comum, o que a imagem mostra e por que ela ilustra a fala."""

    trecho: Optional[float] = None
    letra: Optional[str] = None
    palavra: str = ""
    mostra: str = ""
    porque: str = ""

    @field_validator("mostra", "porque", mode="before")
    @classmethod
    def _explain(cls, value):
        text = _text(value)
        return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1].rstrip() + "…"

    @field_validator("trecho", mode="before")
    @classmethod
    def _trecho(cls, value):
        return _number(value)

    @field_validator("letra", mode="before")
    @classmethod
    def _letter(cls, value):
        return _letter(value)

    @field_validator("palavra", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)

    @property
    def none(self) -> bool:
        return self.letra is None or self.letra in _NONE_TOKENS


class CoveragesAnswer(BaseModel):
    """Resposta da 2c para um bloco (ou parte dos trechos dele)."""

    coberturas: list[CoverageItem] = Field(default_factory=list)
    notas: str = ""

    @field_validator("notas", mode="before")
    @classmethod
    def _strip(cls, value):
        return _text(value)


# Tipos de cobertura (do candidato medido pelo código; aparecem no pedido e no relatório). RELATED: as tags da
# imagem dividem palavra com a fala, mas a descrição não (as tags só indicam a candidata; a evidência é a
# correspondência explicada pela IA e conferida contra a descrição e a fala).
COVERAGE_SAYS = "mostra o que é dito"
COVERAGE_PLACE = "mostra o lugar"
COVERAGE_RELATED = "relacionada pelas tags"
CoverageKind = Literal["mostra o que é dito", "mostra o lugar", "relacionada pelas tags"]


# --- o que o código registra ---


class CallStats(BaseModel):
    """Chamadas e tokens de uma etapa (somando novas tentativas e correções)."""

    calls: int = 0
    cached: int = 0
    api_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    seconds: float = 0.0

    def add(self, other: "CallStats") -> None:
        self.calls += other.calls
        self.cached += other.cached
        self.api_calls += other.api_calls
        for name in ("input_tokens", "output_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, (getattr(self, name) or 0) + value)
        self.seconds = round(self.seconds + other.seconds, 3)


PassKind = Literal["resumo", "estrutura", "cortes", "insercoes", "coberturas"]


class RequestRecord(BaseModel):
    """Um pedido à IA (ou ao cache): tokens estimados pelo orçamento × reais informados pelo provedor."""

    kind: PassKind
    unit: str = Field(..., description="Clipe, bloco ou parte (P007, B03, B03.P009b...)")
    attempt: int = 1
    cached: bool = False
    estimated_tokens: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class Discarded(BaseModel):
    """Item da resposta recusado pela validação (só segurança), com o motivo."""

    kind: str = Field(..., description="bloco, fora, cortar, inserir, cobertura")
    item: str = Field(..., description="O item como a IA mandou, resumido")
    reason: str


class Block(BaseModel):
    """Bloco da estrutura já validado (IDs que existem, cada clipe uma vez)."""

    id: str = Field(..., description="B01, B02... na ordem final")
    title: str = ""
    subject: str = ""
    clip_ids: list[str]
    target_sec: Optional[float] = Field(default=None, description="Meta do bloco (a da IA reescalada para o alvo)")
    ai_target_sec: Optional[float] = Field(default=None, description="Meta que a IA deu ao bloco, antes da reescala")
    ai_block: str = Field(default="", description="Bloco da IA de onde veio, quando foi dividido pela ordem de gravação")


# pedido: duração do briefing/CLI; texto: lida do texto do pedido; ia: escolhida pela estrutura.
TargetOrigin = Literal["pedido", "texto", "ia", "nenhuma"]


class StructureRun(BaseModel):
    origin: Origin
    blocks: list[Block] = Field(default_factory=list)
    out: list[OutClip] = Field(default_factory=list)
    library_clip_ids: list[str] = Field(default_factory=list, description="Acervo: clipes nem em bloco nem fora")
    target_total_sec: Optional[float] = None
    target_origin: TargetOrigin = "nenhuma"
    target_note: str = Field(default="", description="De onde veio o alvo (o trecho lido do pedido, o limite pela fala)")
    target_requested_sec: Optional[float] = Field(
        default=None, description="O alvo pedido, quando passou da fala disponível no material (o alvo virou ela)"
    )
    notes: str = ""
    discarded: list[Discarded] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list, description="Erro de cada tentativa recusada")
    warnings: list[str] = Field(default_factory=list)
    stats: CallStats = Field(default_factory=CallStats)
    estimated_tokens: int = 0
    normalized_ids: list[str] = Field(
        default_factory=list, description="IDs da resposta aceita trocados pelo formato do roteiro (\"bruto → ID\")"
    )
    speech_sec: Optional[float] = Field(default=None, description="Fala disponível dos clipes nos blocos (aceita)")
    speech_sec_first: Optional[float] = Field(
        default=None, description="A mesma soma na 1ª resposta, quando houve nova tentativa por falta de duração"
    )
    shortfall: str = Field(default="", description="O aviso de falta de duração dado à IA na nova tentativa")
    added_clip_ids: list[str] = Field(default_factory=list, description="Clipes incluídos em blocos pela nova tentativa")
    order: str = Field(default="", description="Ordem dos blocos: de gravação (como foi medida) ou a da IA (e por quê)")


class KeepRange(BaseModel):
    """Item aceito da faixa principal: frases seguidas de um clipe (um trecho) ou um momento sem fala."""

    kind: Literal["frases", "momento"]
    clip_id: str
    from_sentence: Optional[str] = None
    to_sentence: Optional[str] = None
    sentence_ids: list[str] = Field(default_factory=list)
    moment_id: Optional[str] = None
    duration_sec: Optional[float] = Field(default=None, description="Pedida pela IA (momento); None = inteiro")
    start_sec: Optional[float] = Field(default=None, description="Início no clipe (respiro centrado); None = início do momento")
    breath: bool = Field(default=False, description="Momento do acervo entre as falas (respiro: som mais baixo)")

    @property
    def key(self) -> str:
        """ID do item: a frase final ou o momento."""
        return self.to_sentence if self.kind == "frases" else self.moment_id


class CutChoice(BaseModel):
    """Corte aceito (2a): a frase que sai do vídeo e o motivo da IA."""

    sentence_id: str
    reason: str = ""


class Trecho(BaseModel):
    """Trecho de fala mantido (frases seguidas de um clipe depois dos cortes), numerado no bloco a partir de 1."""

    number: int
    clip_id: str
    sentence_ids: list[str]
    seconds: float


class Candidate(BaseModel):
    """Candidato com letra (2b: momento sem fala; 2c: momento medido para um trecho)."""

    letter: str
    moment_id: str
    clip_id: str
    duration_sec: float
    source: Literal["bloco", "acervo", "outro clipe"] = "acervo"
    description: str = Field(default="", description="Descrição curta do momento (como vai no pedido)")
    place: str = ""
    kind: Optional[CoverageKind] = Field(default=None, description="2c: por que é candidato")
    words: list[str] = Field(
        default_factory=list,
        description="Palavras em comum medidas (2c: fala × imagem ou lugar × lugar; 2b: com os trechos vizinhos)",
    )

    @model_serializer(mode="wrap")
    def _drop_empty(self, handler):
        """Sem `words` vazio nem `kind` nulo no JSON (a lista vazia não informa nada)."""
        data = handler(self)
        if isinstance(data, dict):
            if not data.get("words"):
                data.pop("words", None)
            if data.get("kind") is None:
                data.pop("kind", None)
        return data


class InsertChoice(BaseModel):
    """Inclusão aceita (2b): a letra convertida em momento, depois de que trecho."""

    after_trecho: int
    letter: str
    moment_id: str
    duration_sec: Optional[float] = None
    start_sec: Optional[float] = Field(default=None, description="Respiro: início no clipe (centrado no momento)")
    breath: bool = Field(..., description="True = momento do acervo (respiro); False = do próprio bloco")


class CoverageChoice(BaseModel):
    trecho: int = 0
    letter: str = ""
    word: str = Field(default="", description="A palavra em comum que a IA copiou")
    from_sentence: str
    to_sentence: str
    sentence_ids: list[str]
    clip_id: str = Field(..., description="Clipe que está falando")
    moment_ids: list[str]
    kind: CoverageKind = COVERAGE_SAYS
    common_words: list[str] = Field(default_factory=list, description="Palavras em comum medidas pelo código")
    speech_evidence: str = ""
    image_evidence: str = ""
    reason: str = ""
    semantic: bool = Field(
        default=False, description="Aceita pela correspondência explicada (mostra/porque), sem palavra em comum"
    )


class CutAttempt(BaseModel):
    """Uma resposta aproveitável da 2a: o que ficaria com ela."""

    attempt: int
    result_sec: float
    all_cut: bool = False
    out: bool = False


class CutRun(BaseModel):
    """2a de um clipe (ou parte de um clipe grande): meta, resultado e cortes."""

    unit_id: str = Field(..., description="B03.P009, ou B03.P009a/b quando o clipe foi dividido para caber")
    clip_id: str
    moment_ids: Optional[list[str]] = Field(default=None, description="Só os momentos desta parte (clipe grande)")
    origin: Origin
    cuts: list[CutChoice] = Field(default_factory=list)
    kept_ids: list[str] = Field(default_factory=list)
    available_sec: float = 0.0
    target_sec: Optional[float] = Field(default=None, description="Meta: a do bloco repartida pela fala disponível")
    result_sec: float = 0.0
    duration_off: bool = Field(default=False, description="Fora da meta ± DURATION_TOLERANCE depois da nova tentativa")
    removed: bool = Field(default=False, description="A IA tirou o clipe (ou a parte) do bloco: fora: true ou tudo cortado")
    joined: list[str] = Field(
        default_factory=list, description="Pares contínuos que a IA cortou pela metade, ajustados pelo código"
    )
    no_question: str = Field(default="", description="Por que não houve pergunta (1 frase / fala curta): entra inteiro")
    tries: list[CutAttempt] = Field(default_factory=list, description="Resultado de cada resposta aproveitável")
    chosen_attempt: Optional[int] = Field(
        default=None, description="A tentativa usada (a mais perto da meta); 0 = o clipe inteiro, sem cortes"
    )
    kept_whole: bool = Field(default=False, description="A IA cortou tudo sem fora: true; o clipe entrou inteiro")
    target_limit: str = Field(
        default="", description="A IA disse que não dá para chegar à meta sem prejudicar a fala (o porquê)"
    )
    notes: str = ""
    discarded: list[Discarded] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: CallStats = Field(default_factory=CallStats)
    estimated_tokens: int = 0
    normalized_ids: list[str] = Field(default_factory=list)


class Skipped(BaseModel):
    """Posição de respiro (2b) ou trecho (2c) que o código NÃO perguntou, com o motivo medido."""

    number: int = Field(..., description="2b: depois_do_trecho; 2c: o número do trecho")
    reason: str


class SubpassRun(BaseModel):
    """2b ou 2c de um bloco (em um ou mais pedidos): candidatos com letra, escolhas aceitas e descartes."""

    origin: Origin
    requests: int = 0
    spare_sec: Optional[float] = Field(default=None, description="2b: meta do bloco − fala mantida (None = sem meta)")
    breath_positions: list[int] = Field(default_factory=list, description="2b: posições de respiro oferecidas")
    own_positions: list[int] = Field(default_factory=list, description="2b: posições para as imagens do bloco")
    skipped: list[Skipped] = Field(
        default_factory=list, description="2b: posições de respiro recusadas; 2c: trechos sem pergunta"
    )
    no_question: str = Field(default="", description="Por que não houve pergunta (ex.: sem sobra para respiro)")
    candidates: list[Candidate] = Field(default_factory=list)
    inserts: list[InsertChoice] = Field(default_factory=list)
    coverages: list[CoverageChoice] = Field(default_factory=list)
    nones: list[int] = Field(default_factory=list, description="2c: trechos em que a IA respondeu \"nenhuma\"")
    notes: str = ""
    discarded: list[Discarded] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: CallStats = Field(default_factory=CallStats)
    estimated_tokens: int = 0


class DetailRun(BaseModel):
    """Passada 2 de um bloco: 2a por clipe, 2b e 2c do bloco, e a faixa principal montada pelo código."""

    unit_id: str = Field(..., description="O bloco (B01...)")
    block_id: str
    clip_ids: list[str]
    origin: Origin = Field(..., description="A pior das subpassadas (fallback se alguma foi)")
    keep: list[KeepRange] = Field(default_factory=list, description="Faixa principal na ordem: trechos e inclusões")
    coverages: list[CoverageChoice] = Field(default_factory=list)
    trechos: list[Trecho] = Field(default_factory=list)
    cut_runs: list[CutRun] = Field(default_factory=list)
    inserts: Optional[SubpassRun] = None
    coverage: Optional[SubpassRun] = None
    available_sec: float = Field(default=0.0, description="Fala livre do bloco antes dos cortes")
    speech_sec: float = Field(default=0.0, description="Fala que ficou depois dos cortes")
    result_sec: float = Field(default=0.0, description="Fala que ficou + inclusões")
    target_sec: Optional[float] = None
    used_before_sec: float = Field(default=0.0, description="Duração já usada pelos blocos anteriores (estimada)")
    library_offered: list[str] = Field(default_factory=list, description="Momentos do acervo listados na 2b")
    removed_clip_ids: list[str] = Field(default_factory=list, description="Clipes que a IA tirou do bloco na 2a")
    breath_sec: float = Field(default=0.0, description="Respiros do bloco (segundos)")
    warnings: list[str] = Field(default_factory=list)

    @property
    def notes(self) -> str:
        return " · ".join(n for n in [*(c.notes for c in self.cut_runs),
                                      self.inserts.notes if self.inserts else "",
                                      self.coverage.notes if self.coverage else ""] if n)


class BudgetInfo(BaseModel):
    """Como o orçamento de contexto foi calculado (`editor/budget.py`)."""

    num_ctx: int
    source: str = Field(..., description="De onde veio o num_ctx (ollama, --editor-ctx, padrão da nuvem)")
    summary_output: int
    structure_output: int
    cuts_output: int
    inserts_output: int
    coverage_output: int
    margin: int
    token_factor: float


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EditorRun(BaseModel):
    """Conteúdo do `editor_run.json`: as passadas, avisos, chamadas e tokens."""

    schema_version: int = EDITOR_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=_now)
    request: str = ""
    target_sec: Optional[float] = None
    provider: str
    model_chain: list[str] = Field(default_factory=list)
    effective_model: Optional[str] = None
    model_switches: list[ModelSwitch] = Field(default_factory=list)
    budget: BudgetInfo
    summaries: list[ClipSummary] = Field(default_factory=list)
    structure: Optional[StructureRun] = None
    details: list[DetailRun] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary_stats: CallStats = Field(default_factory=CallStats)
    cuts_stats: CallStats = Field(default_factory=CallStats)
    inserts_stats: CallStats = Field(default_factory=CallStats)
    coverage_stats: CallStats = Field(default_factory=CallStats)
    requests: list[RequestRecord] = Field(default_factory=list)

    @property
    def total_stats(self) -> CallStats:
        total = CallStats()
        structure = self.structure.stats if self.structure else CallStats()
        for stats in (self.summary_stats, structure, self.cuts_stats, self.inserts_stats, self.coverage_stats):
            total.add(stats)
        return total

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EditorRun":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# --- E12: o plano em IDs convertido em tempos (`editor_plano.json`) ---


class PlanPiece(BaseModel):
    """Um item da Timeline (base ou overlay) e de onde ele veio."""

    kind: Literal["frases", "momento", "respiro", "cobertura"]
    clip_id: str
    ids: list[str] = Field(default_factory=list, description="Frases ou momento de origem")
    source_in: float
    source_out: float
    timeline_start: float
    timeline_end: float


class PlanCoverage(BaseModel):
    from_sentence: str
    to_sentence: str
    moment_ids: list[str]
    speech_evidence: str
    image_evidence: str
    reason: str = ""
    timeline_start: float
    timeline_end: float
    pieces: list[PlanPiece] = Field(default_factory=list)
    uncovered_sec: float = 0.0


class PlanBlock(BaseModel):
    id: str
    title: str = ""
    clip_ids: list[str]
    origin: str = Field(..., description="ia, cache, fallback (a pior das subpassadas)")
    timeline_start: float = 0.0
    timeline_end: float = 0.0
    pieces: list[PlanPiece] = Field(default_factory=list)
    coverages: list[PlanCoverage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EditorPlan(BaseModel):
    """Conteúdo do `editor_plano.json`: blocos na ordem, cada item com os IDs e os tempos."""

    schema_version: int = EDITOR_SCHEMA_VERSION
    project_name: str
    fps: int
    resolution: tuple[int, int]
    duration_sec: float
    target_sec: Optional[float] = None
    blocks: list[PlanBlock] = Field(default_factory=list)
    out: list[OutClip] = Field(default_factory=list)
    library_clip_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EditorPlan":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
