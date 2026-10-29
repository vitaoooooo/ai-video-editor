"""Orçamento de contexto do Editor de IA (segurança: o que não cabe é dividido, nunca cortado em silêncio).

Conta em tokens estimados: caracteres/4 (`roteiro.report.estimate_tokens`, a mesma régua do roteiro.md)
vezes `TOKEN_FACTOR`, medido no tokenizador do modelo local padrão.

Calibração (vlog real "trilha La Islita", 41 clipes, pedidos em português com fala em espanhol e português;
`prompt_eval_count` do qwen2.5:7b no Ollama para cada pedido reconstruído, contra caracteres/4):

    passada                  pedidos   real/estimado: mín    máx
    resumos (E9)               41                     1,355   1,674
    estrutura v3 (compacta)     2                     1,539   1,539
    detalhe v3 (antigo)        25                     1,355   1,480
    B07 (erro real, v3)         1                     1,537 (7932 tokens reais para ~5160 estimados)
    estrutura v4                2                     1,505   1,507
    2a cortes                  25                     1,372   1,622
    2b inclusões               25                     1,432   1,532
    2c coberturas              17                     1,333   1,450

O tokenizador do Qwen conta cada dígito como um token: IDs, tempos e números pesam mais que o texto
(os resumos, com mais tempos por caractere, ficam no topo). Não há diferença que justifique fator por
passada ou idioma: vale um fator só, `TOKEN_FACTOR` = 1,70, acima do maior medido (1,674). O template de
chat do servidor (~30 tokens medidos) e o schema que o Ollama acrescenta no modo texto (130 a 250
tokens estimados, conforme a passada) ficam em `MARGIN_TOKENS`. Com 1,70, o pedido do B07 (~5160 × 1,70
= 8772 tokens) passa do disponível em 8192 e é dividido; cada parte cabe mesmo no pior real/estimado
medido (teste `test_b07_case_fits_after_the_calibration_when_split`). O relatório mostra estimados ×
reais de cada pedido para acompanhar a calibração em outros materiais e modelos.

    disponível para o pedido = num_ctx − saída esperada da passada − MARGIN_TOKENS − RETRY_TOKENS
    orçamento do conteúdo    = disponível − tokens(pedido montado sem as partes variáveis)

`num_ctx`: `--editor-ctx`; senão o `num_ctx` do provedor Ollama (o menor, se a cadeia tiver vários) ou,
na nuvem, `CLOUD_CONTEXT_TOKENS` (bem abaixo do limite real: pedidos menores respondem melhor).
"""

import math
from dataclasses import dataclass
from typing import Optional

from ai_video_editor.editor.schema import BudgetInfo
from ai_video_editor.roteiro.report import estimate_tokens

# Caracteres/4 vezes isto: o maior real/estimado medido no vlog real (1,674), arredondado para cima.
TOKEN_FACTOR = 1.70
# Saída reservada de cada passada (o JSON pedido, com folga para o modelo escrever motivos).
SUMMARY_OUTPUT_TOKENS = 600
STRUCTURE_OUTPUT_TOKENS = 1500
CUTS_OUTPUT_TOKENS = 1200
INSERTS_OUTPUT_TOKENS = 600
COVERAGE_OUTPUT_TOKENS = 600
# Limite da RESPOSTA da 2c (`num_predict` no Ollama), pelo número de trechos perguntados: a parte fixa (chaves e
# notas) + tantos por trecho (um item com palavra, mostra e porque cabe em ~70 tokens do Qwen). Medido no vlog real
# (fotos a cada 1 s): 2 trechos perguntados e 128 itens na resposta (trecho 3 a 128, quase todos "nenhuma").
COVERAGE_OUTPUT_BASE_TOKENS = 100
COVERAGE_OUTPUT_PER_TRECHO_TOKENS = 120
# Folga para o template de chat do servidor e para o schema que o Ollama acrescenta no modo texto.
MARGIN_TOKENS = 600
# Espaço da lista de erros da nova tentativa (`prompt.RETRY_MAX_CHARS` = 1500 caracteres + as duas linhas
# fixas, ~1650 caracteres × 1,70 / 4 ≈ 702): o pedido da nova tentativa cabe onde o primeiro coube.
RETRY_TOKENS = 720
# Contexto usado com provedores de nuvem (o limite real é maior; o editor prefere pedidos menores).
CLOUD_CONTEXT_TOKENS = 32768


def tokens(text: str) -> int:
    return math.ceil(estimate_tokens(text) * TOKEN_FACTOR)


def coverage_output_limit(trechos: int) -> int:
    """Tokens da resposta de um pedido da 2c com `trechos` trechos perguntados."""
    return COVERAGE_OUTPUT_BASE_TOKENS + COVERAGE_OUTPUT_PER_TRECHO_TOKENS * max(1, trechos)


@dataclass(frozen=True)
class Budget:
    num_ctx: int
    source: str

    def available(self, output_tokens: int) -> int:
        return self.num_ctx - output_tokens - MARGIN_TOKENS - RETRY_TOKENS

    @property
    def summary_available(self) -> int:
        return self.available(SUMMARY_OUTPUT_TOKENS)

    @property
    def structure_available(self) -> int:
        return self.available(STRUCTURE_OUTPUT_TOKENS)

    @property
    def cuts_available(self) -> int:
        return self.available(CUTS_OUTPUT_TOKENS)

    @property
    def inserts_available(self) -> int:
        return self.available(INSERTS_OUTPUT_TOKENS)

    @property
    def coverage_available(self) -> int:
        return self.available(COVERAGE_OUTPUT_TOKENS)

    def info(self) -> BudgetInfo:
        return BudgetInfo(
            num_ctx=self.num_ctx, source=self.source, summary_output=SUMMARY_OUTPUT_TOKENS,
            structure_output=STRUCTURE_OUTPUT_TOKENS, cuts_output=CUTS_OUTPUT_TOKENS,
            inserts_output=INSERTS_OUTPUT_TOKENS, coverage_output=COVERAGE_OUTPUT_TOKENS, margin=MARGIN_TOKENS,
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
