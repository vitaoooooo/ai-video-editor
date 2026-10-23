# Arquitetura — AI Video Editor

Para onde o editor está indo e como as partes se encaixam. Opções e comandos estão na [referência técnica](REFERENCIA.md); o porquê de cada decisão, no [histórico](HISTORICO.md).

## 1. Princípio (intocável)

**A IA aponta IDs; o código e o FFmpeg executam.** Nenhum modelo produz timestamps, comandos ou cortes: ele devolve dados estruturados (Pydantic) que referenciam IDs que o código criou (frases da transcrição, clipes do catálogo), e o código valida e executa de forma determinística. Uma resposta inválida é recusada, refeita ou ignorada, mas nunca corrompe o vídeo.

## 2. Hoje: vídeo falado e a espinha falada de uma pasta

O comando `edit <vídeo>` edita um vídeo falado (aula, tutorial, pessoa falando):

```text
probe → silences → transcription → director → plan → render → subtitles → report
```

Cada etapa grava um artefato e um fingerprint em `project.json`; na execução seguinte, etapas com entradas iguais vêm do cache. O Diretor de IA recebe frases numeradas e devolve só os IDs a remover. O `revise` cria novas versões a partir de pedidos em linguagem natural, com a mesma regra.

O comando `edit <pasta>` (receita `vlog`, rodada 2) monta a **espinha falada** de um projeto com vários clipes:

```text
catalog → silences (por clipe) → transcription (por clipe) → director → timeline → render → subtitles → report
```

Os clipes `principal` com fala entram na ordem do catálogo; silêncios e transcrição têm cache por clipe (fora do projeto, chave = hash do arquivo + parâmetros); as frases ganham IDs globais com o `clip_id` (uma frase nunca atravessa dois clipes); o Diretor vê os clipes como seções e ganha a categoria `retake`; os trechos mantidos de cada clipe (mesmo corte do caminho de um vídeo) viram itens de `video_main` numa Timeline, renderizada pelo render da Timeline. O caminho de um vídeo continua no `EditPlan`; a unificação dos dois caminhos fica para depois.

Com a análise visual feita (`visual <pasta>`), a receita `vlog_cobertura` (rodada 3; padrão de `edit <pasta>` quando há `visual.json` e transcrição) acrescenta a **cobertura**:

```text
… → timeline → visual (lê visual.json) → coverage → render → subtitles → report
```

A cobertura é **guiada pelo conteúdo da fala** (seção 4): o **Diretor de cobertura** (IA, duas chamadas) primeiro divide as frases em blocos de assunto e depois escolhe, só por IDs e com **evidências** (o trecho da fala e a tag/descrição da tomada que mostram a mesma coisa), coberturas (frases `S…`–`S…` + tomadas `D001.T05`) e respiros entre os blocos; o código confere as evidências, valida item a item e calcula os segundos. Imagem ruim de verdade (medida em segundos) vira uma sugestão de cobrir, não uma obrigação. Depois o código converte IDs em tempos: coberturas viram `video_overlay` (o áudio da fala continua, com o som ambiente da tomada baixinho em `audio_ambient`) e respiros viram itens de `video_main` inseridos na espinha. Render, legendas, transcrição comparativa, linha do tempo e relatório usam essa Timeline final; `--so-plano` para antes do render e mostra o plano por bloco.

## 3. Peças e receitas

- **Peça** = uma etapa que o código sabe executar (`recipes/schema.py: PIECES`): `probe`, `silences`, `transcription`, `director`, `plan`, `render`, `subtitles`, `report`. Cada peça declara o que sempre exige antes dela (ex.: `director` precisa de `transcription`).
- **Receita** = uma combinação validada de peças (`Recipe`: nome, descrição, etapas em ordem e dependências). Validar uma receita recusa etapa desconhecida, etapa repetida, dependência faltando e ordem impossível (inclusive circular), com uma mensagem em português dizendo o que corrigir.
- Peças: `catalog`, `probe`, `silences`, `transcription`, `director`, `plan`, `timeline`, `visual`, `coverage`, `render`, `subtitles`, `report`. O que só vale em uma receita (ex.: `render` depois de `plan` na `fala`, depois de `timeline` na `vlog`) fica nos `requires` da receita. O campo `multi_clip` diz se a receita trabalha com vários clipes de uma pasta (o Produtor e a CLI usam para casar receita e material).
- O `EditPipeline` recebe a receita (padrão `fala`) e executa as etapas na ordem dela; o `SpinePipeline` faz o mesmo com a `vlog` sobre uma pasta; a numeração `[4/8] Diretor` também vem dela. A receita `fala` é exatamente o `edit` de hoje, com os mesmos fingerprints, cache e saída. Um teste de referência (`tests/test_reference_fala.py`) garante isso.
- Receitas são **dados**: quem escolhe uma receita, seja o usuário (`edit --receita`) ou o Produtor, nunca inventa um processo. Ele só escolhe receitas existentes e ajusta os parâmetros das peças.

## 4. Visão: todo trabalho é um projeto com clipes

O editor deixa de ser só de vídeos falados e passa a editar qualquer tipo de vídeo.

- **Projeto com clipes.** Um vídeo único é um projeto de 1 clipe; uma pasta é um projeto de N clipes (ex.: vlog com subpastas `principal/`, `drone/`, `pov/`, `broll/`). O **catálogo** (`catalog <pasta>`) é a porta de entrada: cada clipe ganha um papel (pela subpasta), um ID estável (`P001`, `D001`, `V001`, `B001`, `X001`), dados técnicos, data de gravação com origem, detecção de fala, sinais simples de cor e miniaturas. É o vocabulário de IDs que as IAs vão usar.
- **Pedido do usuário.** O usuário descreve o que quer (`--pedido`).
- **Produtor (IA; comando `briefing`, rodada 2).** Lê o pedido e um resumo do catálogo (por papel: quantidade, duração, fala, formatos; nunca a lista de clipes) e devolve um **briefing**: tipo de vídeo, uma receita registrada, "Entendi: ...", metas (duração, intensidade, ritmo), instruções para o Diretor de cortes e para o de cobertura, papéis a usar. O código valida o briefing contra o registro de receitas (lido na hora) e o material (receita de 1 vídeo com vários clipes falados, receita de fala sem fala, papéis e duração alvo); inválido → nova tentativa com o erro → fallback determinístico registrado. O Produtor escolhe e ajusta, nunca cria etapas. **Implementado** como comando (`briefing <arquivo-ou-pasta>`, salva `briefing.json`) e no `edit`: com `--pedido`, o `edit` reaproveita o `briefing.json` do mesmo pedido e material ou roda o Produtor; receita, intensidade, ritmo, instruções ao Diretor e duração alvo vêm do briefing quando a CLI não as dá (CLI > briefing > padrão, origem registrada no `project.json` e no relatório); `--sem-briefing` desliga.
- **Análise visual (`visual`, rodada 2).** Cada clipe vira **tomadas** (`P007.T03`: mudança de cena ou janela de até 8 s), com 3 miniaturas (`P007.T03.F1..F3`) e medidas sem IA (desfoque, brilho, movimento, tela uniforme). Uma IA multimodal descreve as tomadas em lotes vendo só miniaturas rotuladas com IDs: descrição, tags, rosto, problemas, se é utilizável, o melhor trecho (por IDs de miniatura) e a beleza como cobertura. Os tempos ficam no código. É o vocabulário visual dos Diretores de cobertura, e a seção "Atenção" já aponta a fala com imagem ruim.
- **Diretores especializados.** O Diretor de fala decide por IDs de frases; o **Diretor de cobertura** (rodada 3, módulo `coverage/`) decide por IDs de frases e tomadas, com evidências, qual B-roll/drone/POV cobre qual trecho da fala e onde entram respiros. Os próximos (música, transições) seguem o mesmo molde: o código cria os IDs, valida e converte em tempos.
- **Cobertura pelo conteúdo da fala.** A imagem de cobertura existe para **mostrar o que é dito** (fala do lago → imagem do lago) ou o lugar onde a cena acontece; sem imagem que combine, a pessoa continua na tela. Na primeira versão (rodada 3) a cobertura nascia dos defeitos de imagem ("janelas obrigatórias") e da beleza das tomadas, e no vlog real isso cobriu de drone a mulher falando no ônibus e o tour inteiro pelo quarto do hotel (59 s) quando o defeito eram ~14 s. Agora:
  1. **A IA entende o vídeo** (1ª chamada): do que ele trata e os **blocos de assunto** (frases seguidas no mesmo lugar/momento: chegada de ônibus, quarto do hotel, trilha até o lago). O código valida o mapa inteiro (contíguo, sem buraco nem sobreposição); inválido, um bloco por clipe.
  2. **A IA escolhe com evidência** (2ª chamada, vendo as frases dentro dos blocos, o que está na tela, quem fala para a câmera e as candidatas com descrição e tags): cada cobertura cita um trecho **literal** da fala e uma **tag ou trecho da descrição** da tomada, e as duas precisam falar da mesma coisa. O código confere tudo (sem diferenciar acentos, com plural e uma tabela pequena de sinônimos) e descarta o item sem evidência, com o motivo. Na dúvida, não cobre.
  3. **O código decide os segundos:** a imagem fica perto de quando a coisa é dita (encurtar, nunca completar com tomada que a IA não escolheu); a pessoa falando para a câmera só é tapada com evidência forte (a coisa falada é tag da tomada); a cobertura soma no máximo 30% do vídeo.
  4. **Imagem ruim vira sugestão:** o código mede os segundos ruins de verdade (parte mantida da fala ∩ tomadas graves, com margem pequena) e diz à IA "prefira cobrir aqui se houver imagem que combine"; a cobertura ali fica limitada a esses segundos, e sem imagem que combine a imagem original fica, com aviso. Só a tela uniforme (parede, câmera tapada) é tapada sem evidência de fala, com tomada relacionada (gravada perto ou com tags em comum) e só pelo tempo dela.
  5. **Respiros só entre blocos de assunto**, com uma tomada que tenha relação com o bloco vizinho.
  6. **Legível:** uma linha por bloco (`S034–S041 · trilha até o lago · coberto 2× (D001.T10 'lago' ↔ 'lago escuro entre montanhas')`, `S000–S012 · chegada de ônibus · sem cobertura (nenhuma imagem de ônibus)`) no terminal, no relatório e no `coverage_plan.md`; `edit --so-plano` mostra o plano sem renderizar.
- **Timeline com trilhas.** Substitui o `EditPlan` (lista única de trechos) por trilhas: fala principal, cobertura de vídeo, áudio. Já existe (`core/timeline.py`, executada pelo comando `render`); o `edit <pasta>` já a gera (espinha em `video_main`; com cobertura, também `video_overlay`, respiros na base e ambiente); o `edit <vídeo>` ainda usa o `EditPlan`. O áudio da Timeline sai de um único FFmpeg, qualquer que seja o número de cortes.

```mermaid
flowchart LR
    U[Pedido do usuário] --> P[Produtor - IA]
    C[Catálogo: clipes com IDs] --> P
    C --> V[Análise visual: tomadas com IDs]
    V --> D
    P -->|briefing validado| R[Receitas e peças]
    R --> D[Diretores - IA: só IDs]
    D -->|IDs validados| T[Timeline com trilhas]
    T --> F[Render FFmpeg]
```

## 5. Plano de rodadas

Cada rodada tem tarefas em paralelo (um worktree por tarefa):

1. **Catálogo + receitas** ∥ **Timeline + render.** Catálogo de pastas de clipes, receitas como dados (`fala` = o `edit` atual), teste de referência do perfil falado; em paralelo, a Timeline com trilhas e o render dela.
2. **Produtor** ∥ **espinha falada multiclipe** ∥ **análise visual.** Briefing a partir do pedido + catálogo; a receita de fala sobre vários clipes principais em sequência (IDs de frase passam a incluir o clipe); análise visual dos clipes (tomadas, miniaturas, medidas e descrição por IA — feita: comando `visual`) para os Diretores visuais. ✅ Espinha pronta: `edit <pasta>` (receita `vlog`), com os principais sem fala listados como disponíveis para a cobertura.
3. **Diretor de cobertura** + **render com cobertura.** B-roll/drone/POV cobrindo trechos da fala, por IDs de clipes e de frases. ✅ Feita: receita `vlog_cobertura` (peças `visual` e `coverage`), Diretor de cobertura (gemini/openai/mock) com validação, nova tentativa e fallback, coberturas em `video_overlay`, respiros na base e ambiente; legendas e relatórios na Timeline final. Depois, cobertura pelo conteúdo da fala (mapa de assuntos, evidências conferidas, imagem ruim como sugestão, `--so-plano`).
4. **`revise` multitrilha** + **relatórios.** Pedidos em linguagem natural sobre a Timeline inteira, e relatórios/linha do tempo por trilha.
