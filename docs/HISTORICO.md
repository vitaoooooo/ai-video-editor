# Histórico do projeto — AI Video Editor

Linha do tempo das decisões, features que entraram, o que saiu ou mudou, e **por quê**.
Formato: cada entrada traz o que foi feito, o motivo e (quando houver) o resultado medido.

> Legenda: ✅ entrou · 🔁 mudou · ❌ saiu / foi adiado · 📊 resultado medido · 🔄 em andamento · 🔮 futuro

---

## Fase 0 — A ideia

- **Editor de vídeo com IA de propósito geral**: o usuário descreve a edição em linguagem natural, envia os vídeos, o sistema analisa, gera um relatório e renderiza.
- ✅ **Princípio de arquitetura (o mais importante do projeto):** *a IA decide, um motor determinístico (FFmpeg) executa.* A IA nunca corta o vídeo diretamente; ela produz dados estruturados validados (Pydantic), e o código executa.
  - **Por quê:** IA erra e é imprevisível. Um plano em JSON pode ser validado, corrigido e refeito sem nunca corromper o vídeo.

## Fase 1 — Motor de edição (Core/Renderer)

- ✅ `EditPlan` / `ClipSegment` (schemas Pydantic), compilador FFmpeg sem shell injection, executor com validação via ffprobe.
- ✅ Comandos `quick-cut` e `render`.

## Fase 2 — Análise técnica (Analyzer)

- ✅ Metadados com ffprobe e extração de frames com timestamps rastreáveis.

## Fase 3 — Análise visual com IA (Intelligence)

- ✅ Interface `MultimodalAIProvider` com provedores **Gemini** e **Mock** (testes offline e sem custo).
- ✅ Relatório Markdown da análise visual (`report-ai`) e métricas de execução (latência, tokens).
- 📊 49 → 53 testes automatizados.

---

## Mudança de rumo: V1 focada em vídeos falados

- 🔁 **Escopo da V1:** de "qualquer tipo de vídeo" para **tutorial, aula e pessoa falando**.
  - **Por quê:** cada tipo de vídeo depende de um sinal diferente. Em vídeo falado, o que importa está na **fala** (texto), que é o ponto forte dos modelos de linguagem. Gameplay e vlog dependem de imagem e "gosto", e são bem mais difíceis de editar bem.
- ❌ **Análise visual por frames vira opcional** na V1 (volta na V2, para gameplay/vlog).
  - **Por quê:** com poucos frames por minuto, a IA fica cega para quase tudo; para quem fala para a câmera, a imagem quase não ajuda a decidir cortes.
- ✅ **Arquitetura preparada para "perfis de edição"** (V1 = TalkingHead; V2 = Gameplay, Vlog).
- 🔁 **Ferramentas:** orquestração com **Orca** (agentes em paralelo, um *git worktree* por tarefa); agente principal passa de Antigravity para **Claude Code**. Criado o `CLAUDE.md` (contexto do projeto para os agentes) e as skills copiadas para `.claude/skills/`.

## As peças da V1 (em paralelo)

- 🔁 **Renderer reescrito** (`renderer-refactor`): um FFmpeg por trecho, em paralelo, + concatenação final; micro *fades* de áudio em cada corte.
  - **Por quê:** o renderer antigo abria o arquivo uma vez **por trecho** — com centenas de cortes de uma aula, ficaria lentíssimo e estouraria o limite de linha de comando do Windows. Sem *fades*, cada corte gerava um "clique" audível.
  - ❌ Saíram `has_audio_stream` / `get_video_duration` (substituídos por `probe_source`, uma chamada ao ffprobe por arquivo).
- ✅ **Detecção de silêncio** (`audio-silence`) com o filtro `silencedetect` do FFmpeg — determinística, sem IA, sem custo.
- ✅ **Transcrição** (`transcription`) com faster-whisper (local, gratuito) + Mock; cada trecho com **ID** estável.
- ✅ **Comando `edit`** (`pipeline`): pasta de projeto, etapas com **cache** (só refaz o que mudou), relatório `edit_report.md`.
  - 📊 Primeiro teste real (aula do MIT, 5 min): **5:00 → 4:20, 46 cortes, nenhum conteúdo perdido.**
- 🔁 **Correção da transcrição** (`transcription-fix`): o áudio passou a ser decodificado pelo FFmpeg em vez do PyAV.
  - **Por quê:** incompatibilidade entre versões do faster-whisper e do PyAV no Windows quebrava a transcrição. Também foi ligado o filtro de voz (evita texto inventado em silêncios) e a CLI passou a mostrar o erro real.

## O Diretor de IA e a revisão

- ✅ **Diretor de IA** (`director`): lê a transcrição e decide quais **frases** remover (repetições, frases recomeçadas, enrolação), com categoria e motivo.
  - **Por quê dos IDs:** a IA aponta **frases pelo número**, nunca timestamps. Os tempos das palavras do Whisper não são confiáveis (ele "engole" pausas dentro das palavras); os cortes são ajustados para dentro dos silêncios medidos.
  - 📊 Aula do MIT: removeu "And this is the first lecture, lecture one." (repetição) e frases recomeçadas — todas defensáveis.
- ✅ **CI no GitHub Actions**: testes a cada push.
- ✅ **Legendas `.srt`** sincronizadas com o vídeo **editado** (`subtitles` + `subtitles-integration`), usando a mesma conta de frames do renderer para não "escorregar".
- 🔁 **Gemini 2.5 → 3.8-flash** (`director-tuning`).
  - **Por quê:** a família 2.5 deixou de estar disponível para contas novas.
- 🔁 **Raciocínio ("thinking") configurável** e timeout realmente repassado ao SDK.
  - 📊 Latência do Diretor: **158s → 5,7s** (28× mais rápido), sem perder qualidade.
- 🔁 **Regra das legendas** (`subtitles-fix`): de "ponto médio da palavra" para "sobreposição com o trecho mantido".
  - **Por quê:** palavras como "first" e "rectangular" sumiam da legenda (mas continuavam no áudio), porque o Whisper inflava a duração delas até dentro de uma pausa cortada.
- 🔁 **Nível de raciocínio com subida automática** (`thinking-fallback`); padrão `low`.
  - **Por quê:** o gemini-3.8-flash recusa o nível mínimo.
- ✅ **Ciclo de revisão** (`revision`): `revise <projeto> "pedido em português"` gera uma nova **versão** (v1, v2, v3…), sem refazer as etapas caras. Comando `versions` mostra o histórico.
  - 📊 "tira a parte em que ele fala do site do curso" → a IA achou sozinha as 2 frases certas (**4:11 → 3:45**). "corta menos, deixa o ritmo mais natural" → restaurou frases, mudou o ritmo das pausas e **manteve o pedido anterior** (o site continuou fora).

## Teste com vídeo longo (palestra de 57 min) e o que ele revelou

- 🔁 **Testes 6× mais rápidos** (`test-speed`): marcador `slow`, fixtures reaproveitadas, execução paralela.
  - 📊 Suíte completa: **9min20s → 1min33s**; testes rápidos: 41s.
  - **Por quê:** no Windows, abrir um processo custa ~0,8s, e cada teste do pipeline abria ~10 (FFmpeg/ffprobe).
- ✅ **Tempo de cada etapa** no terminal e no relatório (`stage-timings`).
- ✅ **Novas tentativas automáticas** nas chamadas à IA (`api-retry`): esperas crescentes (2s, 4s, 8s, 16s), sem repetir erros permanentes; chaves removidas das mensagens de erro.
  - **Por quê:** o Gemini falhava com 503/504 (servidor sobrecarregado) e o editor desistia na primeira.
- 🔁 **Limite de silêncio automático por vídeo** (`adaptive-silence`): mede o ruído de fundo e o nível da fala e escolhe o limite entre os dois.
  - **Por quê:** com o limite fixo de -35 dB, a palestra (sala com ruído) teve só **9 silêncios em 57 min**.
  - 📊 Palestra: **9 → 458 pausas detectadas**. Aula do MIT: limite escolhido -37,2 dB, resultado praticamente igual ao aprovado (40 vs 44 cortes).
- ✅ **Transcrição na GPU** (`gpu-transcription`), com volta automática para a CPU.
  - 📊 RTX 2060: **85,6s → 13,2s** na aula de 5 min (**6,5× mais rápido**), mesmo texto.
- 🔁 **Frases com tamanho máximo** (`sentence-units`): 12s / 40 palavras.
  - **Por quê:** em fala corrida o Whisper pontua pouco, e a palestra virou só 125 "frases" de ~27s — grandes demais para o Diretor cortar só a enrolação.
  - 📊 Palestra: **125 → 411 frases** (média 7,4s).
- ✅ **Provedores de IA plugáveis** (`llm-providers`): camada `llm/`, Gemini + **OpenAI**, e **troca automática de modelo** (3.8-flash → 3.6-flash → 3.5-flash-lite, e opcionalmente entre provedores) com aviso no terminal.
  - **Por quê:** quando um modelo fica sobrecarregado, outro costuma estar livre; e a arquitetura prova na prática que trocar de IA não exige reescrever o editor.
  - 📊 312 testes automatizados.
- ✅ **Transcrição comparativa + status de aplicação** (`edit-transcript`): `edit_transcript.md` com a fala original inteira, em ordem cronológica — frases mantidas com `[tempo no original → tempo no vídeo editado]` (mesmo remapeamento das legendas), removidas riscadas com quem removeu, categoria e motivo, pausas removidas de 1s ou mais, e nas revisões as frases restauradas (✅) e removidas agora (🆕). O mesmo conteúdo sai estruturado em `edit_diff.json`. Cada remoção pedida é marcada como **aplicada** (≥ 90% saiu), **parcial** (segundos e %) ou **não aplicada** (com o motivo), na tabela do Diretor, na seção Revisão e num resumo no topo do relatório.
  - **Por quê:** na palestra, o relatório dizia que a palavra "anymore" tinha sido removida, mas a trava de segurança das folgas das frases vizinhas anulou o corte — nada saiu do vídeo. E de "By" só 0,19s de 0,74s foi cortado. O relatório precisa refletir o que realmente aconteceu no plano final, não só o que a IA pediu.
  - 🔁 **Regra do pedaço de palavra:** se o corte deixaria uma sobra da frase menor que 0,25s, ele é estendido até o limite da frase; se isso invadir uma frase mantida, o corte é desfeito. Nunca fica meia palavra no vídeo.
  - 📊 320 testes automatizados.

- ✅ **Andamento ao vivo no terminal** (`live-progress`): no `edit` e no `revise`, a linha da etapa em curso é reescrita a cada segundo com o tempo decorrido; a transcrição mostra a fração do áudio já transcrita, o render os trechos prontos (`118/258 trechos`), e o Diretor/revisor o modelo atual e a tentativa. Tempo restante (`faltam ~3min`) só a partir de 10% concluídos. Avisos saem em linha própria acima da linha ao vivo. Fora de um terminal, ou com `--no-live`, a saída é a mesma de antes.
  - **Por quê:** na palestra de 57 min, a transcrição e o render ficavam minutos sem nenhuma saída, e não dava para saber se o editor estava andando ou travado.
  - 🔁 O progresso vem de callbacks injetáveis (`transcribe(..., on_progress)`, `RenderExecutor.execute(plan, on_progress)`, `on_stage_progress` no pipeline); só a CLI sabe desenhar a linha. Os avisos de GPU indisponível do faster-whisper passaram a sair pelo callback do pipeline (antes iam direto para o stderr e quebrariam a linha ao vivo).
- ✅ **Normalização de volume** (`loudnorm`): o áudio final sai num nível padrão (EBU R128) com o filtro `loudnorm` do FFmpeg em **duas passadas** no concat final: a primeira mede o áudio concatenado dos trechos e a segunda aplica o ganho com os valores medidos e `linear=true` (vídeo continua com `-c:v copy`). Presets `--loudness voz` (padrão, -16 LUFS), `youtube` (-14), `broadcast` (-23) e `off`, no `edit` e no `revise`. O `EditPlan` ganhou o campo opcional `loudness` (padrão: sem normalização), então `render` e `quick-cut` não mudam. O relatório ganhou a seção "Volume" (I, TP e LRA medidos, alvo aplicado) e avisa quando o original já chega a 0 dBTP de true peak. Áudio todo em silêncio ou vídeo sem áudio: pula com aviso.
  - **Por quê:** na palestra o áudio estava "estourado" (picos em 0 dBFS); gravações caseiras costumam ter o problema oposto (volume baixo). Duas passadas porque a medição prévia permite um ganho único e previsível, em vez do modo dinâmico de uma passada, que muda o volume ao longo do vídeo.
  - 📊 Tom a -34,8 LUFS → -16,0 LUFS; tom a -4,3 LUFS (pico -0,6 dBTP) → -16,1 LUFS. Custo da passada de medição: **17,7s para 8min20s de áudio** (trechos de um vídeo 720p de 10 min; o render inteiro foi de 95s para 108s). Quase todo o tempo é o próprio loudnorm (reamostra para 192 kHz para medir o true peak): decodificar o áudio leva 1,5s, o `ebur128` mediria em 3s, mas não calcula o `target_offset`. Na palestra de 57 min, a estimativa é de ~2 min a mais.
  - 📊 358 testes automatizados (somando os do `live-progress`).

- ✅ **Benchmark de transcrição** (`benchmark transcription`): mede tempo, fator de tempo real e **WER** (distância de edição por palavras, com S/D/I e as 10 substituições mais frequentes) de configurações `modelo:dispositivo:compute_type` do faster-whisper contra a transcrição oficial do MIT. Aceita `--reference` opcional (sem ela, só velocidade; WER aparece como "—") e `--manifest` com vários itens (tabela por item + média geral). Saída em `benchmark/results/<data>-<nome>/` (`results.json` validado por Pydantic e `results.md`). O WER foi implementado no projeto, sem dependência nova; o `pypdf` foi instalado só temporariamente para extrair a referência.
  - 🔁 `FasterWhisperProvider` ganhou `preload(config)`: o modelo é carregado antes e reaproveitado pelo `transcribe`. **Por quê:** o benchmark precisa medir a carga (inclui download na primeira vez) separada da transcrição. O comportamento do `edit` não muda.
  - 🔁 **Fonte do áudio:** o link do archive.org da tarefa (`MIT18.06S05_MP4/01.mp4`) está fora do ar (404, inclusive na página do OCW); o áudio veio do YouTube oficial do MIT OCW (`J7DzL2_Na80`, mesma aula e mesmo ID do PDF), baixado com yt-dlp instalado fora do projeto. Áudio em `benchmark/data/` (não versionado); a referência (`mit_1806_lecture01_reference.txt`) é versionada com fonte e licença.
  - 📊 Aula 1 de Álgebra Linear (39min 49s, RTX 2060, `--language en`; tempo sem a carga do modelo):

    | Configuração | Carga | Tempo | Tempo real | WER | S | D | I |
    |---|---|---|---|---|---|---|---|
    | `small:cpu:int8` | 2,0s | 11min 42s | 0,294x | 9,2% | 227 | 98 | 130 |
    | `small:cuda:float16` | 1,0s | 1min 45s | 0,044x | 10,3% | 218 | 98 | 195 |
    | `large-v3-turbo:cuda:float16` | 58,2s | 1min 24s | 0,035x | 8,9% | 118 | 70 | 254 |

  - 📊 Boa parte do "erro" do `small` é de grafia, não de escuta: dígitos no lugar de números por extenso ("1" × "one", "2" × "two", "0" × "zero") e "ok" × "okay" respondem por mais de 130 substituições. O `large-v3-turbo` escreve mais por extenso (as substituições caem de ~220 para 118), mas insere mais palavras (254). A GPU é ~7x mais rápida que a CPU com o `small`, com WER parecido; o turbo é o mais rápido e o mais preciso. A referência termina no meio de uma frase, então parte das inserções pode ser o fim da aula que a referência não cobre (não verificado).

- ✅ **WER normalizado, `benchmark rescore`, cauda da referência e modelo `auto`** (2026-10-20):
  - ✅ **WER normalizado** (`benchmark/normalize.py`, sem dependência nova): nos dois textos, números por extenso → dígitos (0–999 e milhares simples), okay → ok, x's → xs. Só `en` por enquanto, com estrutura (`NORMALIZERS`) para `pt`. Tabela e `results.json` trazem WER bruto e normalizado, cada um com S/D/I. **Por quê:** mais de 130 "substituições" do `small` eram só grafia.
  - ✅ **Transcrições salvas e `benchmark rescore <pasta>`**: cada configuração grava `hypotheses/<config>.txt`; o `rescore` recalcula as métricas a partir delas, sem transcrever, mantendo os tempos medidos. A execução anterior não tinha os textos, então foi rodada de novo (autorizado) com as mesmas três configurações.
  - ✅ **Cauda após o fim da referência**: palavras que a transcrição tem a mais depois do último trecho casado (mínimo 5) saem das inserções e do WER e aparecem à parte ("Após a ref."). **Por quê:** a referência do MIT termina no meio de uma frase. Confirmado, mas pequeno: 14–19 palavras.
  - 🔁 **`TranscriptionConfig.model_size` agora é `auto`** (CLI `--whisper-model auto|<modelo>`): `large-v3-turbo` na GPU, `small` na CPU; explícito é respeitado; se a GPU falhar e cair para a CPU, o `auto` usa `small`. O modelo efetivo entra no fingerprint e em `model_used`. **Por quê:** o turbo foi o mais rápido na GPU e o `small` é o razoável na CPU. Efeito: em máquinas com GPU os projetos existentes são transcritos de novo na próxima execução; na CPU o cache continua válido.
  - 📊 Mesma aula, nova execução (WER bruto agora já sem a cauda nas inserções):

    | Configuração | Tempo | WER bruto | WER norm. | Após a ref. |
    |---|---|---|---|---|
    | `small:cpu:int8` | 11min 40s | 8,9% | 5,6% | 14 |
    | `small:cuda:float16` | 1min 36s | 9,9% | 6,8% | 18 |
    | `large-v3-turbo:cuda:float16` | 1min 22s | 8,5% | 7,4% | 19 |

  - 📊 Com a normalização o ranking inverte: o `small` na CPU é o menor WER (5,6%) e o turbo o maior (7,4%), porque o turbo escreve mais palavras que a referência omite (235 inserções; o `small` na CPU, 116). Não é a cauda. Fica a investigar se uma referência mais literal (com "uh") muda isso.

📊 **Palestra de 57 min, resultado atual:** 57:07 → 53:43, 259 cortes, processada em **8min48s** (0,15× o tempo real, com transcrição em cache).

## Skills fora do repositório

- ❌ As pastas `.claude/` e `.agents/` e o `skills-lock.json` saíram do repositório (continuam no disco, ignorados pelo `.gitignore`).
- 🔁 As skills do Claude Code foram para a pasta de usuário (`C:\Users\vitor\.claude\skills\`).
  - **Por quê:** na pasta de usuário elas valem para todos os worktrees sem cópia por pasta, e deixam de poluir o repositório.
- ❌ O Antigravity deixou de ser usado no projeto, então a cópia `.agents/skills/` e a regra das "duas cópias" perderam o sentido.

## README de portfólio e linha do tempo dos cortes (2026-10-21)

- ✅ **Linha do tempo dos cortes** (`edit_timeline.svg`, `pipeline/edit_timeline.py`): desenhada a partir do `EditDiff` na etapa do relatório, junto com o `edit_transcript.md`, na raiz do projeto e em `versions/vN/` (também no `revise`), e citada no `edit_report.md`. Uma barra representa o vídeo original, com cores para fala mantida, pausas removidas, frases removidas pelo Diretor/revisão, remoções parciais/não aplicadas e trechos sem fala; marcas de tempo, legenda e título com duração original → final e % removido. Vídeos de mais de 10 min quebram em uma linha por 10 min; cada item do diff é um grupo SVG com o texto e o motivo como dica. SVG escrito pelo próprio código (sem dependência nova), com fundo próprio para ficar legível em páginas claras e escuras.
  - **Por quê:** a transcrição comparativa diz o que saiu, mas não mostra *onde* nem *quanto*; uma imagem da edição inteira serve ao usuário e ao README.
  - 🔁 O formato do relatório mudou (`REPORT_FORMAT = report-v4-timeline`): na próxima execução do `edit`, projetos existentes refazem só o relatório; uma versão já registrada sem o SVG o recebe junto com o relatório refeito.
  - 📊 Gerados a partir dos `edit_diff.json` existentes da aula do MIT e da palestra: `docs/img/timeline-aula-mit.svg` e `docs/img/timeline-palestra.svg` (6 linhas de 10 min).
- 🔁 **README reescrito como README de portfólio** (problema, pipeline em Mermaid, demonstração com saída real, resultados medidos, benchmark, histórias de engenharia, limitações e como o projeto foi construído). Todos os números vêm deste histórico, de `benchmark/results/` e dos `edit_report.md` dos projetos.
  - 🔁 A referência técnica detalhada (todas as opções, artefatos, regras e módulos) foi para `docs/REFERENCIA.md`, com as seções de instalação e testes atualizadas e os links `file:///` trocados por caminhos relativos.
  - **Por quê:** o README antigo era um manual de opções; quem chega ao repositório (recrutadores e pessoas técnicas) precisa ver primeiro o que o editor faz, o resultado e as decisões.
  - 📊 Palestra, relatório atual (v5): 57:07 → 53:40, 259 cortes, 9min 59s (transcrição em cache; Diretor 1min 14s, render 8min 45s).
  - 📊 413 testes automatizados.

## Projetos com clipes, rodada 1: receitas e catálogo (2026-10-21)

- 🔁 **Mudança de rumo: de "só vídeos falados" para qualquer tipo de vídeo.** Todo trabalho passa a ser um **projeto com clipes** (um vídeo = 1 clipe; uma pasta = N clipes, ex.: vlog com `principal/`, `drone/`, `pov/`, `broll/`). Um Produtor de IA (rodada 2) vai ler o pedido + o catálogo e devolver um briefing validado; por dentro, **peças** (etapas) e **receitas** (combinações validadas); uma Timeline com trilhas substituirá o `EditPlan`. O princípio continua: a IA aponta IDs, código e FFmpeg executam. Visão e plano de rodadas em [ARQUITETURA.md](ARQUITETURA.md).
  - **Por quê:** a V1 de vídeo falado está pronta e medida; o próximo uso real é um vlog com várias câmeras, que precisa de um catálogo de clipes e de um jeito de combinar etapas sem que a IA invente processos.
- ✅ **Teste de referência do perfil falado** (`tests/test_reference_fala.py`, commitado antes de qualquer outra mudança): roda o `edit` completo num vídeo sintético (5 falas, uma repetida) com transcrição e Diretor mock e compara com `tests/data/reference_fala/` (frases removidas, trechos do plano com 3 casas, duração final e legendas). Tolerância de ±2 ms nos tempos (decodificador AAC de outra versão do FFmpeg, ex.: CI) e ±50 ms na duração do render; regenerar só com `UPDATE_REFERENCE=1`.
  - **Por quê:** a refatoração do pipeline em receitas não podia mudar nada do `edit`; o teste passou antes e depois, sem regenerar o snapshot.
- ✅ **Receitas como dados** (`recipes/`): catálogo de peças `PIECES` (com o que cada peça sempre exige), modelo `Recipe` validado (etapa desconhecida, repetida, dependência faltando, ordem impossível ou circular → erro em português) e registro com a receita `fala` = etapas do `edit`. O `EditPipeline.run` virou um laço sobre `recipe.steps`, uma etapa por método com o estado compartilhado em `_EditState`; a numeração `[n/N]` da CLI vem da receita. CLI: `edit --receita fala`.
  - 📊 Fingerprints das 8 etapas idênticos antes e depois, e um projeto criado pelo código antigo saiu 100% do cache com o novo.
- ✅ **Comando `catalog <pasta>`** (`catalog/`): lê a pasta sem modificá-la e gera `catalog/` no projeto `<pasta>-<hash>` (hash da listagem: caminho relativo + tamanho + mtime, sem ler os vídeos). Papel pela subpasta (sinônimos, acentos, plural; `--papel`), IDs `P001/D001/V001/B001/X001` por data de gravação, com origem registrada (tags QuickTime/container/trilha → nome do arquivo → mtime pouco confiável) e `--ajuste-relogio papel=+3h`. ffprobe (com `probe_rotation`), miniaturas a 10/50/90% e `signalstats` em paralelo; fala pelo VAD Silero do faster-whisper (áudio a 16 kHz em blocos de 60s; sem ele, `desconhecido` com aviso); suspeita de perfil log (baixo contraste + baixa saturação); folha de contato por papel com o ID escrito (Pillow, já era dependência). Saídas: `catalog.json` (`ProjectCatalog`, `schema_version` 1), `catalog.md` (resumo por papel, avisos, tabela por clipe) e `contact_sheet_<papel>.jpg`. Arquivo corrompido ou sem vídeo vira aviso.
  - 🔁 **Cache por clipe fora do projeto** (`output/projects/_cache/clips/`, chave = caminho absoluto + tamanho + mtime). **Por quê:** o hash do projeto muda quando entra um clipe; com o cache dentro do projeto, um clipe novo numa pasta de 200 refaria o VAD de todos.
  - 🔁 `StageProgress` ganhou a unidade `clips` (`12/40 clipes (30%)` na linha ao vivo).
  - 📊 Pasta sintética de 6 arquivos (5 clipes + 1 corrompido): 11,9s na primeira execução, mídia e fala do cache na segunda.
  - 📊 413 → 463 testes automatizados (referência, receitas e catálogo).
- ❌ **Fora desta tarefa** (de propósito): renderer, `core/schema.py` e o formato dos IDs das frases (outra tarefa mexe em `core/`/`renderer/`; os IDs de frase com clipe vêm na rodada 2), proxies e análise visual com IA, e o próprio Produtor (`--pedido`).

---

## 🔄 Em andamento

- **Timeline com trilhas + render** (worktree `timeline-render`, rodada 1 em paralelo com este catálogo).

## 🔮 Próximos passos (V1)

- **Ponto de corte de palavras curtas pela energia do áudio**: na palestra de 57 min, fillers como "Um" e "By" ficaram no vídeo. O tempo do Whisper para essas palavras inclui a pausa ao redor, e o corte só tirou o silêncio (nenhuma palavra foi cortada pela metade). Refinar o início/fim do corte pela energia do áudio dentro do intervalo da palavra.
- **Status "não aplicada (só a pausa saiu)"**: quando a parte removida de uma frase for apenas silêncio, marcar assim em vez de "parcial" (caso do "By": 0,19s de 0,74s saiu, mas era só pausa).
- **Benchmark de transcrição com Gemini**: incluir o provedor (e custo) na comparação; a estrutura aceita novos provedores (`register_benchmark_provider`).
- **Provedor de transcrição Gemini** (alternativa para quem não tem GPU).
- **Render na GPU (NVENC)**: o render é hoje a etapa mais lenta em vídeos longos.
- **Teste com vídeo em português** com mais enrolação.
- Pequenos ajustes: cópia de vídeo por versão ocupa o dobro de disco; motivo do Diretor às vezes truncado.

## 🔮 Projetos com clipes (próximas rodadas; ver [ARQUITETURA.md](ARQUITETURA.md))

- **Rodada 2:** Produtor (pedido + catálogo → briefing validado) ∥ espinha falada multiclipe (IDs de frase com o clipe) ∥ análise visual dos clipes (proxies; a análise visual da Fase 3 volta aqui).
- **Rodada 3:** Diretor de cobertura + render com cobertura.
- **Rodada 4:** `revise` multitrilha + relatórios.

## 🔮 V2 (futuro)

- **Perfis de edição**: Gameplay (picos de áudio + imagem), Vlog (imagem + música), como receitas.
- **Aplicação web**: upload, prévia, chat de ajustes, fila de processamento.
