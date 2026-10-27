"""Orçamento de contexto do Editor de IA (segurança: o que não cabe é dividido, nunca cortado em silêncio).

Conta em tokens estimados: caracteres/4 (`roteiro.report.estimate_tokens`, a mesma régua do roteiro.md)
vezes `TOKEN_FACTOR`, porque o tokenizador dos modelos locais conta mais tokens que isso em português.

    disponível para o pedido = num_ctx − saída esperada da passada − MARGIN_TOKENS
    orçamento do conteúdo    = disponível − tokens(pedido montado sem as seções e sem o acervo)

`num_ctx`: `--editor-ctx`; senão o `num_ctx` do provedor Ollama (o menor, se a cadeia tiver vários) ou,
na nuvem, `CLOUD_CONTEXT_TOKENS` (bem abaixo do limite real: pedidos menores respondem melhor).
"""

import math
from dataclasses import dataclass
from typing import Optional

from ai_video_editor.editor.schema import BudgetInfo
from ai_video_editor.roteiro.report import estimate_tokens

# Estimativa caracteres/4 vezes isto (o Qwen conta ~1,2× em português; a calibrar no material real).
TOKEN_FACTOR = 1.25
# Saída reservada de cada passada (o JSON pedido, com folga para o modelo escrever motivos).
SUMMARY_OUTPUT_TOKENS = 600
STRUCTURE_OUTPUT_TOKENS = 1500
DETAIL_OUTPUT_TOKENS = 2000
# Folga para o template de chat do servidor e para o schema que o Ollama acrescenta no modo texto.
MARGIN_TOKENS = 600
# Contexto usado com provedores de nuvem (o limite real é maior; o editor prefere pedidos menores).
CLOUD_CONTEXT_TOKENS = 32768
# Ao dividir um bloco, sobra pelo menos isso para a pré-seleção do acervo.
LIBRARY_MIN_TOKENS = 500


def tokens(text: str) -> int:
    return math.ceil(estimate_tokens(text) * TOKEN_FACTOR)


@dataclass(frozen=True)
class Budget:
    num_ctx: int
    source: str

    def available(self, output_tokens: int) -> int:
        return self.num_ctx - output_tokens - MARGIN_TOKENS

    @property
    def summary_available(self) -> int:
        return self.available(SUMMARY_OUTPUT_TOKENS)

    @property
    def structure_available(self) -> int:
        return self.available(STRUCTURE_OUTPUT_TOKENS)

    @property
    def detail_available(self) -> int:
        return self.available(DETAIL_OUTPUT_TOKENS)

    def info(self) -> BudgetInfo:
        return BudgetInfo(
            num_ctx=self.num_ctx, source=self.source, summary_output=SUMMARY_OUTPUT_TOKENS,
            structure_output=STRUCTURE_OUTPUT_TOKENS, detail_output=DETAIL_OUTPUT_TOKENS, margin=MARGIN_TOKENS,
            token_factor=TOKEN_FACTOR,
        )


def resolve_budget(providers: list[str], override: Optional[int] = None, clients: Optional[dict] = None) -> Budget:
    """Contexto do Editor: `override` (`--editor-ctx`) > `num_ctx` do Ollama da cadeia > nuvem.

    `clients` (provedor → cliente já criado) evita criar outro cliente só para ler o `num_ctx`.
    """
    if override is not None:
        return Budget(num_ctx=int(override), source="--editor-ctx")
    if "ollama" in providers:
        client = (clients or {}).get("ollama")
        if client is None:
            from ai_video_editor.llm.providers.ollama import resolve_num_ctx

            num_ctx = resolve_num_ctx()
        else:
            num_ctx = int(getattr(client, "num_ctx"))
        return Budget(num_ctx=num_ctx, source="num_ctx do ollama")
    return Budget(num_ctx=CLOUD_CONTEXT_TOKENS, source="padrão da nuvem")
