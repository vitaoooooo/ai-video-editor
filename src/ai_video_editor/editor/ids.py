"""IDs como a IA escreve → IDs do roteiro, antes da validação (todas as passadas), e a dica do formato certo
para os recusados. Só segurança: nada é adivinhado.

- Frase: `P009.S025`, `p009 . s025`, `P009S025`, `S25` → `S025` SOMENTE se a frase existe e, havendo prefixo,
  ele é o clipe a que ela pertence. Prefixo de outro clipe ou de clipe inexistente (`X018.S078`) fica como
  veio (a validação recusa, com a dica `use S078, não X018.S078`).
- Momento: só caixa, espaços e zeros à esquerda (`p9.m3` → `P009.M03`), e só se o momento existe. Sem o
  clipe (`M03`) não vale: seria adivinhar o clipe.
- Clipe: caixa, espaços e zeros à esquerda (`p9` → `P009`), se existe.
- Lixo de markdown ou pontuação em volta do ID (`### P009.M06`, `**S012**`, `"S012".`, `[P007]`): sai antes de
  tudo (a IA copia o título `###` do roteiro ou o negrito).
- Faixa num campo só (`"P007.S021–P007.S023"`): reconhecida por `split_range` (a 2a recusa faixas: uma frase
  por item).

Cada troca é registrada (`IdResolver.fixed`, "P009.S025 → S025"): o relatório conta quantos IDs foram
normalizados por passada, o sinal de que o pedido ainda induz o formato errado.
"""

import re
from typing import Optional

from ai_video_editor.editor.material import EditorMaterial
from ai_video_editor.editor.schema import CutsAnswer, StructureAnswer

_CLIP = re.compile(r"^([A-Z]+)0*(\d+)$")
_SENTENCE = re.compile(r"^(?:([A-Z]+)0*(\d+)[.:_/-]?)?S0*(\d+)$")
_MOMENT = re.compile(r"^([A-Z]+)0*(\d+)[.:_/-]?M0*(\d+)$")
# Frase dentro de um texto maior (faixa num campo só).
_SENTENCE_TOKEN = re.compile(r"(?:[A-Z]+\d+[.:_/]?)?S\d+")
# O que pode separar os dois lados de uma faixa escrita num campo só.
_RANGE_GLUE = re.compile(r"^(?:[-–—~>/,;]|\.\.|A|ATE|ATÉ|AO)+$")
# Lixo em volta do ID: tudo o que não é letra/dígito nas pontas (###, **, aspas, colchetes, ponto final).
_EDGES = re.compile(r"^[^A-Z0-9]+|[^A-Z0-9]+$")


def _compact(raw: str) -> str:
    return _EDGES.sub("", re.sub(r"\s+", "", raw.upper()))


class IdResolver:
    """Normaliza os IDs de uma resposta contra o material; `fixed` = trocas feitas ("bruto → ID")."""

    def __init__(self, material: EditorMaterial):
        self.material = material
        self.fixed: list[str] = []
        self._clips: dict[tuple[str, int], str] = {}
        for clip_id in material.clips:
            m = _CLIP.match(clip_id)
            if m:
                self._clips[(m[1], int(m[2]))] = clip_id
        self._sentences: dict[int, str] = {}
        for sid in material.sentences:
            m = _SENTENCE.match(sid)
            if m and m[1] is None:
                self._sentences[int(m[3])] = sid
        self._moments: dict[tuple[str, int, int], str] = {}
        for mid in material.moments:
            m = _MOMENT.match(mid)
            if m:
                self._moments[(m[1], int(m[2]), int(m[3]))] = mid

    def _fix(self, raw: str, value: str) -> str:
        if value != raw:
            self.fixed.append(f"{raw} → {value}")
        return value

    def _clip_of(self, letters: str, number: str) -> Optional[str]:
        return self._clips.get((letters, int(number)))

    def clip(self, raw: Optional[str]) -> Optional[str]:
        if not raw or raw in self.material.clips:
            return raw
        m = _CLIP.match(_compact(raw))
        found = self._clip_of(m[1], m[2]) if m else None
        return self._fix(raw, found) if found else raw

    def sentence(self, raw: Optional[str]) -> Optional[str]:
        if not raw or raw in self.material.sentences:
            return raw
        m = _SENTENCE.match(_compact(raw))
        if not m:
            return raw
        sid = self._sentences.get(int(m[3]))
        if sid is None:
            return raw
        if m[1] is not None and self._clip_of(m[1], m[2]) != self.material.sentences[sid].clip_id:
            return raw  # prefixo de outro clipe ou inexistente: não adivinha
        return self._fix(raw, sid)

    def moment(self, raw: Optional[str]) -> Optional[str]:
        if not raw or raw in self.material.moments:
            return raw
        m = _MOMENT.match(_compact(raw))
        found = self._moments.get((m[1], int(m[2]), int(m[3]))) if m else None
        return self._fix(raw, found) if found else raw

    def anchor(self, raw: Optional[str]) -> Optional[str]:
        """Frase ou momento (destaques e problemas dos resumos)."""
        value = self.sentence(raw)
        return self.moment(raw) if value == raw else value


def sentence_tokens(text: Optional[str]) -> list[str]:
    """Frases citadas num texto (`"S170, S171"` → duas): a 2a pede uma por item."""
    return _SENTENCE_TOKEN.findall(_compact(text)) if text else []


def split_range(text: Optional[str]) -> Optional[tuple[str, str]]:
    """`"P007.S021–P007.S023"` / `"S021 a S023"` → os dois lados; None se não é uma faixa num campo só."""
    if not text:
        return None
    compact = _compact(text)
    tokens = _SENTENCE_TOKEN.findall(compact)
    if len(tokens) != 2:
        return None
    rest = compact
    for token in tokens:
        rest = rest.replace(token, "", 1)
    return (tokens[0], tokens[1]) if _RANGE_GLUE.match(rest) else None


def normalize_structure(answer: StructureAnswer, material: EditorMaterial) -> tuple[StructureAnswer, list[str]]:
    """Cópia da resposta da passada 1 com os clipes normalizados, e as trocas feitas."""
    ids = IdResolver(material)
    answer = answer.model_copy(deep=True)
    for block in answer.blocos:
        block.clipes = [ids.clip(c) for c in block.clipes]
    for out in answer.fora:
        out.clipe = ids.clip(out.clipe) or ""
    return answer, ids.fixed


def normalize_cuts(answer: CutsAnswer, material: EditorMaterial) -> tuple[CutsAnswer, list[str]]:
    """Cópia da resposta da 2a com as frases normalizadas (faixas ficam como vieram: a validação recusa)."""
    ids = IdResolver(material)
    answer = answer.model_copy(deep=True)
    for cut in answer.cortar:
        if cut.frase and split_range(cut.frase) is None:
            cut.frase = ids.sentence(cut.frase)
    return answer, ids.fixed


# --- dicas do formato certo (vão na mensagem do item recusado e, daí, na nova tentativa) ---

SENTENCE_FORMAT = "frase = só o código S… que aparece antes do texto da frase, sem o clipe na frente"
MOMENT_FORMAT = "momento = o código completo do título do momento (clipe, ponto e M…), copiado da entrada"
CLIP_FORMAT = "clipe = o código que abre a linha do clipe, copiado da entrada"


def sentence_hint(raw: str, material: EditorMaterial) -> str:
    """Por que `raw` não é uma frase e como escrever certo."""
    compact = _compact(raw)
    m = _SENTENCE.match(compact)
    if m:
        sid = IdResolver(material)._sentences.get(int(m[3]))
        if sid is not None and m[1] is not None:
            return f"use {sid}, não {raw}: {sid} é do clipe {material.sentences[sid].clip_id}"
        return f"{SENTENCE_FORMAT}; copie um código que exista no roteiro"
    if _MOMENT.match(compact):
        return f"{raw} é um momento, não uma frase; {SENTENCE_FORMAT}"
    return SENTENCE_FORMAT


def moment_hint(raw: str, material: EditorMaterial) -> str:
    """Por que `raw` não é um momento e como escrever certo."""
    compact = _compact(raw)
    if _SENTENCE.match(compact):
        return f"{raw} é uma frase, não um momento; {MOMENT_FORMAT}"
    m = _MOMENT.match(compact)
    if m and IdResolver(material)._clip_of(m[1], m[2]) is None:
        return f"o clipe {re.match(r'[A-Z]+[0-9]+', compact)[0]} não existe; {MOMENT_FORMAT}"
    return MOMENT_FORMAT


def clip_hint(raw: str, material: EditorMaterial) -> str:
    return CLIP_FORMAT if raw not in material.clips else ""
