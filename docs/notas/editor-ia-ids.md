# Notas da branch `editor-ia-ids` (2026-10-27)

O Editor de IA (receita `editor_ia`) passa a entender os IDs que os modelos realmente escrevem, e os pedidos das
passadas 1 e 2 deixam de induzir o formato errado. Este arquivo junta o que iria para `docs/HISTORICO.md`,
`docs/REFERENCIA.md`, `docs/ARQUITETURA.md` e `README.md`; a consolidação nos documentos é feita no fim do dia.

## HISTORICO

### Editor de IA: IDs como os modelos escrevem (CLIPE.Sxxx), pedidos sem IDs de exemplo (2026-10-27)

**Sintoma** (vlog real, 41 clipes, 175 frases, `edit --receita editor_ia --so-plano`, com o `editor-ia-ajustes`):

- `qwen2.5:7b` (ollama): 0 coberturas e 0 respiros; itens descartados por ID: `P006.S005`, `P007.S021–P007.S023`,
  `P009.S035`, e cópias do exemplo do pedido: `X018.S078`, `X019.S113`, `X02.M90`, `X018.M01` (e `ordem` com
  `X018, X019, X020`). Mensagens: "não é uma faixa de frases do mesmo clipe, em ordem", "frase X018.S078 não
  existe", "depois_de P006.S008 não é uma frase que ficou".
- gemini (3.8-flash 503 → 3.6-flash cota 429 → 3.5-flash-lite): 5 blocos, B02 e B03 no fallback neutro por
  "frase P009.S025 não existe", "frase P015.S053 não existe", "frase P017.S062 não existe".

**Causa medida** (respostas em `output/projects/_cache/editor_ia/` e os dois `editor_relatorio.md` do vlog):

| Pedido | Exemplo de frase no pedido | Frases simples nas respostas | Frases CLIPE.Sxxx |
| --- | --- | --- | --- |
| resumos `resumo-v1` (123 respostas) | `S012` | 288 | **0** |
| detalhe `detalhe-v1` (3 respostas, `manter`) | `S012` | 9 | **0** |
| detalhe `detalhe-v2` (7 respostas, `cortar`) | **`X01.S900`** | 13 | **17** (em 3 das 7) |

O formato CLIPE.Sxxx veio do **exemplo fictício do próprio pedido** (`X01.S900`, `X02.M90`, introduzido no
`editor-ia-ajustes` para que IDs de exemplo não passassem por escolhas reais): ele ensinou que uma frase se
escreve "clipe + ponto + S + número". Os dois modelos aprenderam o padrão e preencheram com o clipe real
(`P009.S025`); o 7B às vezes copiou também a letra do exemplo (`X018.S078` = `X01` + o número real da frase de
P018). A entrada não induz sozinha: o roteiro mostra a frase como `- S025 — texto` sob o título do momento
`### P009.M03 · …` desde o `detalhe-v1`, que teve 0 IDs prefixados. O schema também não (as descrições dos
campos do detalhe não têm ID). Agravante: `X` é um prefixo real do catálogo (papel `outro`), então `X01` nem
era "impossível".

**Correção:**

- ✅ **Pedidos das passadas 1 e 2 sem nenhum ID de exemplo** (`STRUCTURE_PROMPT_VERSION` estrutura-v2 →
  **estrutura-v3**, `DETAIL_PROMPT_VERSION` detalhe-v2 → **detalhe-v3**). O formato é descrito em palavras numa
  linha "Códigos: responda copiando os códigos exatamente como aparecem na entrada; frase = só o código S… que
  aparece antes do texto da frase, sem o clipe na frente (a frase já é única no vídeo todo); momento = o código
  completo que abre o título do momento (### …) ou a linha nas listas; clipe = o código que abre a linha do
  clipe (ou o título ## no roteiro)". Os JSON de formato usam textos (`"de": "código da 1ª frase"`,
  `"momentos": ["código do momento"]`, `"clipes": ["código do clipe", …]`); a passada 1 não mostra mais
  `"id": "B01"` (o código renumera). Listas de momentos com o cabeçalho "código · duração · …".
- ✅ **A entrada já mostra cada ID como deve voltar** (frase `S025` antes do texto, momento `P009.M03` no título,
  clipe `P009` na linha/título, continuidade `S010 — texto`): sem mudança, de propósito — a seção do clipe é a
  mesma do `roteiro.md` e do pedido dos resumos, e mudá-la refaria os 41 resumos em cache.
- ✅ **Normalização segura antes da validação, em todas as passadas** (`editor/ids.py`, `IdResolver`):
  - frase: `P009.S025`, `p009 . s025`, `P009S025` → `S025` **somente se P009 é o clipe da frase S025**; `S25`,
    `S0025` → `S025` (pelo número, contra as frases que existem: vale para qualquer quantidade de dígitos);
    prefixo de outro clipe ou de clipe inexistente (`X018.S078`) fica como veio → recusado;
  - momento: só caixa, espaços e zeros à esquerda (`d1.m2` → `D001.M02`) e só se existe; `M03` sem clipe não vale
    (seria adivinhar o clipe);
  - clipe (passada 1, `ordem`): caixa, espaços, zeros à esquerda (`p1` → `P001`), se existe;
  - faixa: `de`/`ate` normalizados um a um (um lado prefixado e o outro não funciona); faixa escrita num campo
    só (`"P007.S021–P007.S023"`, `"S021 a S023"`, com o outro lado vazio ou igual) vira os dois lados;
  - resumos (E9): os IDs dos destaques/problemas são normalizados antes de filtrar os que não são do clipe (o
    pedido dos resumos não mudou; os 41 em cache continuam valendo).
- ✅ **Mensagem do ID recusado com o formato certo** (vai para a nova tentativa): "frase X018.S078 não existe
  (use S078, não X018.S078: S078 é do clipe P018)"; "frase S999 não existe (frase = só o código S… que aparece
  antes do texto da frase, sem o clipe na frente; copie um código que exista no roteiro)"; "P025.M01 é um
  momento, não uma frase"; "momento X02.M90 não existe (o clipe X02 não existe; momento = o código completo do
  título do momento…)"; "S001 é uma frase, não um momento"; `depois_de` e `ordem` com a mesma dica; clipe
  inexistente na passada 1: "clipe = o código que abre a linha do clipe, copiado da entrada".
- ✅ **Relatório conta os IDs normalizados por passada** (coluna "IDs normalizados" na tabela de chamadas;
  por bloco e na estrutura, "IDs normalizados: N (P009.S025 → S025; …)"). Muitos = o pedido ainda induz.
  Só contam as respostas aceitas (o fallback não conta).
- 🔁 `EDITOR_SCHEMA_VERSION` 2 → 3 (`normalized_ids` em `ClipSummary`, `StructureRun`, `DetailRun`);
  `EDITOR_FORMAT` editor-v3, `REPORT_FORMAT` editor-report-v3. Pedido dos resumos sem mudança (`resumo-v1`).
- 📊 **Reprocessado offline com o roteiro real** (sem IA): dos IDs recusados nos dois relatórios, 25 frases
  prefixadas pelo próprio clipe (`P006.S005`, `P007.S021`, `P009.S025`, `P015.S053`, `P017.S062`, `P020.S117`…)
  passam a ser reconhecidas (o item ainda passa pelas outras conferências); as 8 `X018/X019.Sxxx` e os momentos `X018.M01`, `X019.M03…05`, `X02.M90` continuam
  recusados, com a dica. Nas 7 respostas `detalhe-v2` em cache: 7 trocas (todas `P006/P007/P009.Sxxx`). Os itens
  que levaram B02/B03 do gemini ao fallback deixam de ser recusados por ID (não refeito com a IA).
- ✅ `tests/test_reference_fala.py` passa sem regenerar; `fala`/`vlog`/`vlog_cobertura` sem mudança.
- 📊 Testes: 795 → 804 (`tests/test_editor_ia.py`: pedidos das passadas 1 e 2 sem nenhum ID de exemplo nem no
  formato do roteiro (só o código do bloco), e com material real só IDs da entrada e nenhuma frase com clipe na
  frente; `P001.S002` aceito quando P001 é o clipe da frase; recusado com prefixo de outro clipe e com `X018`, com
  o formato certo na mensagem; `S2`/`s0003` → `S002`/`S003`; faixa com um lado prefixado e faixa num campo só;
  momento inexistente, de clipe inexistente e frase no lugar de momento recusados; clipes da passada 1
  normalizados; nova tentativa com "use S002, não X018.S002"; contagem por passada no relatório). Suíte
  completa: 804 passaram em 4min 46s (`-n auto`).
- ❌ **Não medido:** o novo plano com `qwen2.5:7b`/gemini no vlog real (reaproveita os 41 resumos):

  ```bat
  python -m ai_video_editor.cli edit "%VLOG%" --receita editor_ia --so-plano --language pt --pedido "vlog da trilha, uns 8 minutos"
  ```
  Anotar: "IDs normalizados" por passada (perto de 0 = o pedido não induz mais), IDs recusados com dica e se a
  nova tentativa os corrige, blocos no fallback, coberturas e respiros aceitos.

## REFERENCIA

Sem opção de CLI nova ou alterada. Mudou o contrato das passadas 1 e 2 do Editor de IA (`editor/`):

- **Módulo novo `editor/ids.py`**: `IdResolver(material)` (`clip`, `sentence`, `moment`, `anchor` = frase ou
  momento, `sentence_range`; `fixed` = trocas "bruto → ID"), `split_range`, `normalize_structure` e
  `normalize_detail` (cópia da resposta + trocas), dicas `sentence_hint`, `moment_hint`, `anchor_hint`,
  `clip_hint` e os textos `SENTENCE_FORMAT`, `MOMENT_FORMAT`, `CLIP_FORMAT`.
- **Validação** (`editor/validate.py`): `validate_structure` e `validate_detail` normalizam antes de conferir
  (a resposta original não muda; o cache guarda a resposta da IA como veio); `StructureCheck.normalized` e
  `DetailCheck.normalized`. ID inexistente: a mensagem leva a dica entre parênteses.
- **Resumos** (`EditorService._clean_summary`): IDs normalizados por `IdResolver.anchor` antes do filtro do clipe.
- **Registro** (`editor_run.json`, `EDITOR_SCHEMA_VERSION = 3`): `normalized_ids` em `ClipSummary`,
  `StructureRun` e `DetailRun` (só a resposta aceita; fallback = vazio).
- **Pedidos** (`editor/prompt.py`): `STRUCTURE_PROMPT_VERSION = "estrutura-v3"`,
  `DETAIL_PROMPT_VERSION = "detalhe-v3"`, `SUMMARY_PROMPT_VERSION = "resumo-v1"` (igual). Constantes
  `CLIP_CODES`, `SENTENCE_CODES`, `MOMENT_CODES` e `_codes_line`.
- **Relatório (`editor_relatorio.md`)**: tabela "Chamadas e tokens" com a coluna "IDs normalizados" por passada
  e no total; "IDs normalizados: N (exemplos)" na estrutura e em cada bloco que teve trocas.
- **Fingerprints**: `EDITOR_FORMAT = "editor-v3"`, `REPORT_FORMAT = "editor-report-v3"`
  (`pipeline/editor_ia.py`).

## ARQUITETURA

- **Regra para todo pedido novo do editor (substitui a dos IDs fictícios):** nenhum ID de exemplo — nem no formato
  real, nem fictício. Um exemplo é copiado: no formato real vira escolha falsa; fictício, ensina um formato que
  não existe (`X01.S900` → `P009.S025`). O formato é descrito em palavras e a IA copia os códigos da entrada,
  que mostra cada ID exatamente como deve voltar.
- **Normalização ≠ adivinhação:** o código aceita uma variação de escrita só quando ela aponta, sem ambiguidade,
  para um ID que existe (frase com o prefixo do PRÓPRIO clipe; caixa, espaços, zeros). Prefixo de outro clipe,
  clipe inexistente ou momento sem clipe continuam inválidos. Isso fica antes da validação e não toma decisão
  editorial.
- **Sinal no relatório:** a contagem de IDs normalizados por passada mede se o pedido ainda induz o formato
  errado; a normalização não esconde o problema.

## README

Para o usuário do `edit --receita editor_ia`: o editor deixa de perder cortes, coberturas e respiros quando o
modelo escreve as frases com o clipe na frente (`P009.S025`), e o relatório mostra quantos códigos precisaram
ser corrigidos em cada passada. Nenhuma opção nova.
