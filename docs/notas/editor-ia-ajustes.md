# Notas da branch `editor-ia-ajustes` (2026-10-27)

Ajustes do Editor de IA (receita `editor_ia`) depois do primeiro plano real com `qwen2.5:7b`. Este arquivo
junta o que iria para `docs/HISTORICO.md`, `docs/REFERENCIA.md`, `docs/ARQUITETURA.md` e `README.md`; a
consolidação nos documentos é feita no fim do dia.

## HISTORICO

### Editor de IA: passada 2 "por padrão fica tudo", duração informada, completude da estrutura e evidência por tipo (2026-10-27)

**Medido no vlog real** (41 clipes, 23:35, 175 frases; `edit --receita editor_ia --so-plano --pedido "vlog da
trilha, uns 8 minutos"`, `qwen2.5:7b`, `num_ctx` 8192, RTX 2060): 49 pedidos em 8min58s (resumos 41 em 6min22s,
estrutura 42s, detalhe 7 em 1min54s); 71.172 tokens de entrada e 9.251 de saída (detalhe: ~200 tokens de saída
por bloco). Alvo 480 s (a IA leu "8 minutos"); **plano com 216 s**.

| Sintoma no `editor_plano.md`/`editor_relatorio.md` | Causa | Correção |
| --- | --- | --- |
| a) IDs dos exemplos nas respostas (`S012–S015`, `de S013 até S014`, `D002.M05`, `B004.M01`), recusados como "fora do bloco"/"não está no acervo" | os exemplos do pedido tinham IDs no formato real; o 7B copia o exemplo | exemplos das passadas 1 e 2 só com IDs impossíveis e visivelmente fictícios (`X01`, `X09`, `X01.S900`, `X02.M90`); teste que falha se um ID de exemplo casar com o formato real |
| b) "manter" minúsculo: B02 (6 clipes, alvo 180 s) ficou com 3 frases + 1 momento = 30 s; B03a (~80 frases) manteve 4 frases + 1 momento | a passada 2 pedia a lista do que FICA; o modelo devolve listas curtas (~200 tokens de saída) e ignora a meta | **por padrão fica tudo**: a IA devolve só `cortar` (com motivo); as frases não cortadas ficam, na ordem do roteiro |
| (b) meta do bloco ignorada | a duração ia só como texto no pedido, e o código não conferia | a entrada diz a duração das frases do bloco, o alvo do bloco (e do pedaço) e o já usado; o código calcula a duração resultante e, fora de alvo ± 35%, faz UMA nova tentativa mostrando os números; se ainda ficar fora, aceita e registra |
| c) clipes com fala sem decisão (P001, P021) viraram acervo | a estrutura não conferia completude | todo clipe com fala vai num bloco ou em "fora" com motivo; esquecido = item inválido → nova tentativa com a lista; persistindo, bloco próprio na ordem do catálogo, com aviso |
| d) blocos por tema, cronologia misturada (B02 = hotel + trilha + tronco + touro; B03 com 15 clipes, dividido pelo orçamento) | o pedido não dava a regra de ordem nem o tamanho que cabe | regras no pedido da passada 1: vlog/viagem em ordem cronológica (data de gravação, senão a ordem do catálogo) salvo pedido; blocos = trechos da história, não temas; a entrada mostra os tokens de cada clipe e o orçamento por bloco |
| e) cobertura S002 (terminal de ônibus) ← D001.M01 (lago do drone), `evidencia_imagem` tirada da imagem do próprio P002; passou porque "montanhas" também estava em D001.M01 | bastava UMA palavra da evidência aparecer em algum lugar; não ligava fala e imagem | evidências de até 6 palavras COPIADAS; TODAS as palavras de conteúdo da evidência da imagem têm de estar no momento usado (o aviso diz de qual momento vieram); tipo obrigatório: "mostra o que é dito" exige palavra em comum entre as duas evidências, "mostra o lugar" exige lugar igual/parecido ao do momento da fala |
| f) B03b no fallback: "manter momento" em momentos com frases já mantidas; `evidencia_fala` com texto da imagem | `manter` aceitava momento com fala; evidência conferida por 1 palavra; resposta com metade inválida virava fallback | `incluir` só aceita momento SEM fala (listados à parte); todas as palavras da `evidencia_fala` têm de estar nas frases cobertas; depois da nova tentativa os itens válidos são aceitos (fallback só sem nenhum item válido ou sem JSON) |

- ✅ **Passada 2 (detalhe) com novo schema** (`BlockAnswer`): `cortar` (faixas `de/ate` ou IDs de frases, com
  `motivo`), `incluir` (`momento` sem fala, `depois_de` = frase que ficou/momento incluído antes/vazio = começo
  do bloco, `duracao_sec` opcional), `ordem` (opcional, clipes do bloco), `coberturas` (com `tipo`), `respiros`,
  `notas`. A faixa principal (`KeepRange`) agora é **calculada pelo código**: frases livres não cortadas na
  ordem dos clipes e do roteiro, quebrada onde há corte ou incluído. Montagem (`assemble.py`) sem mudança.
- ✅ **Duração como informação:** `DURATION_TOLERANCE = 0.35` (`editor/service.py`). Resultante = frases que
  ficaram + incluídos + respiros. Alvo do pedaço = alvo do bloco (dividido: proporcional à duração das
  frases; sem fala, dos momentos). Sem nova tentativa quando está abaixo do alvo, nada foi cortado e não há
  momento sem fala livre (não há o que ajustar). O relatório mostra disponível → resultante → alvo e marca
  "fora do alvo".
- ✅ **Resposta parcialmente inválida:** a nova tentativa continua quando ≥ 50% dos itens é inválido (ou nada
  fica, ou a duração está fora); depois dela, aceita a última resposta aproveitável com os itens válidos e
  aviso. Fallback neutro só sem resposta aproveitável (erro da API, fora do formato, nenhum item válido,
  nada no bloco). Fallback neutro do detalhe = nada cortado (os clipes sem fala entram com os momentos).
  Estrutura: a última resposta com bloco válido vale se a nova tentativa não resolver a completude.
- ✅ **Regra de evidência** (`validate.py` + `material.same_word`, só segurança): palavras sem acento e sem
  caixa, sem palavras vazias; mesma palavra = igual sem o plural (ões/ães → ão, ais → al, eis → el, ns → m,
  res/zes/ses, s), uma começa com a outra (≥ 4 letras) ou uma letra a mais com ≥ 6 letras (ônibus/ómnibus).
  Exemplos: fala "Pegamos o ómnibus" + imagem "ônibus na estrada" → aceita (em comum: omnibus); "cachorros
  correndo" × "cachorro na praia" → aceita (cachorros); "ómnibus" × "cachorro na praia" → recusada (sem
  palavra em comum); imagem "ônibus rua carros" num drone que só tem ônibus → recusada ("rua, carros" vêm
  de P002.M01); "mostra o lugar" com lugar "terminal rodoviário" × fala em "terminal de ônibus" → aceita
  (terminal), com lugar "praia" → momento retirado.
- 🔁 `STRUCTURE_PROMPT_VERSION` estrutura-v1 → **estrutura-v2**, `DETAIL_PROMPT_VERSION` detalhe-v1 →
  **detalhe-v2**: só as passadas 1 e 2 são refeitas. **Pedido dos resumos (E9) sem mudança** (`resumo-v1`):
  as 41 respostas em cache continuam valendo. Por isso o exemplo do resumo ainda tem `S012`/`P007.M04`
  (decisão: não refazer os resumos); os IDs que ele copiar são retirados pelo código (`_clean_summary`, como
  antes) e não chegam às passadas 1 e 2. `EDITOR_SCHEMA_VERSION` 1 → 2 (`editor_run.json` ganhou campos);
  `EDITOR_FORMAT` editor-v2 e `REPORT_FORMAT` editor-report-v2.
- ✅ `tests/test_reference_fala.py` passa sem regenerar; `fala`/`vlog`/`vlog_cobertura` sem mudança.
- 📊 Testes: 786 → 795 (`tests/test_editor_ia.py`: ID de exemplo no formato real; cortar por faixa/ID com o
  resto em ordem e `ordem` opcional; incluir momento sem fala (no começo e encadeado) e recusa de momento com
  fala; duração fora do alvo → uma nova tentativa com os números → aceita e registrada (e sem nova tentativa
  dentro de ± 35%); completude da passada 1 (esquecido → nova tentativa → bloco próprio com aviso) e posição
  do bloco do esquecido; evidência "mostra o que é dito" (ônibus/ómnibus, plural) e recusa sem palavra em
  comum, sem tipo e com fala tirada da imagem; "mostra o lugar"; `evidencia_imagem` de outro momento
  recusada; itens válidos aceitos depois da nova tentativa). Suíte completa: 795 passaram em 6min 55s
  (`-n auto`; a anterior foi 3min 59s — sem teste lento novo, a diferença parece ser carga da máquina).
- ❌ **Não medido:** o novo plano com o `qwen2.5:7b` no vlog real. Para medir (reaproveita os 41 resumos):

  ```bat
  python -m ai_video_editor.cli edit "%VLOG%" --receita editor_ia --so-plano --language pt --pedido "vlog da trilha, uns 8 minutos"
  ```
  Anotar: duração do plano × 480 s, blocos "fora do alvo", clipes esquecidos, IDs fictícios (`X01...`) nas
  respostas, coberturas aceitas/recusadas por tipo e quantas respostas aceitas só com os itens válidos.

## REFERENCIA

Sem opção de CLI nova ou alterada. Mudou o contrato das passadas do Editor de IA (`editor/`):

- **Passada 1 (`StructureAnswer`)** — mesmo schema (`blocos`, `fora`, `duracao_total_alvo_sec`, `notas`).
  Validação nova: `fora` exige `motivo`; todo clipe com fala num bloco ou em `fora` (`StructureCheck.forgotten`,
  `Discarded.kind = "faltando"`); `validate.place_forgotten` põe o esquecido num bloco próprio depois do bloco
  com o clipe anterior no catálogo. Entrada: cada clipe com `~N tokens` (a seção dele no detalhe) e o
  orçamento por bloco (`EditorService._content_budget`).
- **Passada 2 (`BlockAnswer`)** — `{"cortar": [{"de", "ate", "motivo"} | "S012"], "incluir": [{"momento",
  "depois_de", "duracao_sec"}], "ordem": ["P002", "P001"], "coberturas": [{"de", "ate", "momentos", "tipo",
  "evidencia_fala", "evidencia_imagem", "motivo"}], "respiros": [{"depois_de", "momento", "duracao_sec",
  "motivo"}], "notas"}`. `tipo` = `"mostra o que é dito"` ou `"mostra o lugar"` (texto com "dito"/"lugar"
  é aceito). `depois_de` vazio/"início" = começo do bloco. O schema antigo (`manter`) saiu.
- **Registro (`DetailRun`)**: `cuts` (`CutChoice`: frases + motivo), `clip_order`, `available_sec`,
  `result_sec`, `target_sec`, `duration_off`; `CoverageChoice` ganhou `kind` e `common_words`.
- **Constantes:** `editor/service.py: DURATION_TOLERANCE = 0.35`; `editor/validate.py: EVIDENCE_MAX_WORDS = 6`;
  `editor/material.py: MIN_INSERTION_CHARS = 6` (+ `stem`, `same_word`, `common_words`). As três entram em
  `EDITOR_RULES` (fingerprint da etapa `editor`).
- **Relatório (`editor_relatorio.md`)**: por bloco, "Duração: frases disponíveis → resultante → alvo" (e
  **fora do alvo**), ordem dos clipes se a IA mudou, "incluiu", "cortou: S012–S014 · motivo" com as frases, e
  cada cobertura com o tipo e as palavras em comum.

## ARQUITETURA

- A passada 2 inverteu o padrão: em vez da IA listar o que fica (e esquecer quase tudo), **fica tudo** e a IA
  lista o que sai, o que entra a mais (momentos sem fala), a ordem dos clipes, coberturas e respiros. O código
  calcula a faixa principal a partir disso (nenhuma decisão editorial: só aplica as listas da IA na ordem do
  roteiro).
- Duração: o código **mede e informa** (entrada: frases disponíveis, alvo, já usado; depois: resultante) e
  pede UMA revisão se ficar fora de ± 35%; nunca corta sozinho para chegar no alvo.
- Validação em dois níveis: `refused` (pede de novo: ≥ 50% inválidos, nada no bloco, esquecidos na estrutura)
  e `unusable` (fallback: nada aproveitável). Entre os dois, a resposta é aceita com os itens válidos.
- Cobertura com tipo: o código confere que a ligação declarada pela IA existe em palavras (fala × imagem, ou
  lugar × lugar), sem julgar se a imagem "combina".
- Exemplos dos pedidos com IDs fictícios (`X01.S900`) — regra para todo pedido novo do editor.

## README

Para o usuário do `edit --receita editor_ia`: o plano deixa de sair muito mais curto que o pedido (por padrão
o editor mantém a fala e corta o que não serve, mirando a duração de cada bloco), nenhum clipe com fala fica
sem decisão, a ordem de vlog/viagem segue a gravação, e o relatório mostra por bloco quanto havia, quanto ficou
e o alvo, o que foi cortado e por quê, e por que cada cobertura foi aceita (tipo e palavras em comum). Nenhuma
opção nova.
