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

## Timeline com trilhas e render multicâmera (2026-10-21)

- ✅ **`Timeline`** (`core/timeline.py`): edição montada a partir de vários clipes de várias câmeras. Trilhas `video_main` (base contígua; a duração é a soma dos itens; áudio do próprio item ou silêncio com `use_audio: false`), `video_overlay` (substitui a imagem numa janela, o áudio da base continua) e `audio_ambient` (áudio posicionado no tempo da timeline, com volume 0–2 e fades próprios). Itens com `clip_id`/`label` livres para relatórios. Validação em português: tempos invertidos, item com menos de 1 frame, overlay ou ambiente além da duração, overlays sobrepostos, buraco/sobreposição na base (quando o item traz `timeline_start`), resolução ímpar, campos desconhecidos.
  - Todos os tempos são quantizados em frames do fps de saída com a mesma regra do EditPlan (`round(segundos × fps)`), e as amostras de áudio (48 kHz) derivam das posições em frames, para áudio e vídeo não escorregarem.
  - `timeline_from_edit_plan` converte um EditPlan em `video_main` com o áudio de cada item; o EditPlan não mudou.
- ✅ **Achatamento** (`flatten`, função pura, sem FFmpeg): recorta a base onde há overlay e devolve pedaços de vídeo em sequência (fonte, in, out, frames) + plano de áudio (um pedaço por item da base e os itens de ambiente com posição absoluta em amostras).
  - **Por quê:** achatar permite reaproveitar o render segmentado que já existe (um FFmpeg curto por pedaço, em paralelo, e concat demuxer com `-c:v copy`), em vez de montar um filtergraph de overlays que cresce com a timeline, é difícil de depurar e gasta memória. A lógica difícil (onde cada pedaço começa e termina) fica numa função pura testada com casos de borda e com 300 timelines aleatórias.
- ✅ **Render da Timeline** (`renderer/timeline_compiler.py`, `renderer/timeline_executor.py`), ao lado do caminho do EditPlan: pedaços de vídeo sem áudio e padronizados (escala preservando a proporção + letterbox/pillarbox, fps de saída, `setsar=1`, `yuv420p` 8 bits); áudio principal em PCM 48 kHz estéreo com os micro fades em cada item; ambiente posicionado com `adelay` em amostras e mixado com `amix` sem normalizar volumes; mux com loudnorm em duas passadas (reaproveita `renderer/loudness.py`, com uma variante da medição para arquivo de áudio). A saída é validada por ffprobe (duração do vídeo com tolerância de 1 frame e fps).
  - **Rotação:** o autorotate do FFmpeg já aplica o metadado de rotação na decodificação, inclusive com `filter_complex`, e a saída sai sem o metadado; conferido em teste com pixels (vídeo vertical vira pillarbox).
  - **HDR:** fontes PQ/HLG são convertidas para 8 bits sem tonemapping e geram aviso (tonemapping de verdade ficou de fora). 10 bits SDR é convertido sem aviso.
  - **Sem áudio:** origem sem trilha de áudio ou item com `use_audio: false` vira silêncio gerado do tamanho exato do item; a saída sempre tem áudio. Se nenhum item tem áudio, o loudnorm é pulado com motivo.
  - Antes do render, cada origem é inspecionada uma vez; item que passa do fim da origem, base/overlay sem vídeo e ambiente sem áudio são recusados com o item identificado (ex.: `video_overlay[0] (D003)`).
- 🔁 **`render`** aceita JSON de EditPlan (como antes) ou de Timeline (detectado por `video_main`/`schema_version`).
- 🔄 O `edit` continua no EditPlan; a troca do pipeline para a Timeline fica para a rodada 2 (em paralelo, o worktree `multiclip-catalog` mexe em `pipeline/` e no catálogo de clipes).
- 📊 Equivalência: o mesmo EditPlan (5 segmentos de 4 formatos) renderizado pelo caminho antigo e pela Timeline tem o mesmo número de frames e a mesma duração.
- 📊 Render dos testes (isolado): Timeline mista de 4 s com 4 formatos, 2 overlays e ambiente ≈ 10 s; quase todo o tempo é a partida dos processos FFmpeg (~1,2 s cada nesta máquina Windows), não a codificação.
- 📊 455 testes automatizados (42 novos); suíte completa em 4min 31s.

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
  - 📊 413 → 463 testes automatizados (referência, receitas e catálogo); depois do merge com a Timeline, 505 testes, suíte completa em 4min 14s.
- ❌ **Fora desta tarefa** (de propósito): renderer, `core/schema.py` e o formato dos IDs das frases (outra tarefa mexe em `core/`/`renderer/`; os IDs de frase com clipe vêm na rodada 2), proxies e análise visual com IA, e o próprio Produtor (`--pedido`).

## Projetos com clipes, rodada 2: Produtor e correções do catálogo (2026-10-22)

- 🔁 **Catálogo, papel por palavra no nome da pasta:** além do nome exato, um sinônimo como palavra (ou duas palavras vizinhas juntas) dá o papel, com origem `contém`: `Osmo Pocket - PRINCIPAL` → principal, `Fx30 - BROLL` → broll, `Go3S - POV` → pov, `Imagens de Drone` → drone. Dois papéis no mesmo nome → `outro` com aviso pedindo `--papel`.
  - **Por quê:** no material real (vlog com 41 clipes), as pastas têm o nome da câmera junto do papel; todas caíam em `outro`.
- 🔁 **Catálogo, datas de exportação:** numa mesma subpasta, um clipe que "começa" antes de o anterior terminar (data + duração) marca as datas do papel como não confiáveis, com aviso, e o papel passa a ser ordenado pela ordem natural do nome (`(4)` < `(15)` < `(27)`, origem `order_source = "nome"`). A ordem natural também virou o desempate geral (antes, alfabético: `(15)` vinha antes de `(4)`).
  - **Por quê:** o material real foi reexportado e o `creation_time` é a hora da exportação (ex.: `Drone (1)` tem 2:02 e "começa" às 13:39:56; `Drone (2)` "começa" às 13:40:45), então a ordem por data embaralhava os clipes. A comparação é por subpasta porque duas câmeras no mesmo papel podem gravar juntas.
- 🔁 **Catálogo, perfil log:** a suspeita só sai com contraste **e** saturação baixos, com limites mais rígidos e nomeados (`LOG_MAX_LUMA_SPREAD = 0,30`, `LOG_MAX_SATURATION = 0,04`, antes 0,45 e 0,08). `log_suspect` passou a ser calculado dos valores (vale para o cache sem refazer nada); saíram os avisos isolados de "baixo contraste"/"baixa saturação" do `catalog.md`, que agora mostra os valores medidos por clipe para calibrar. `catalog.json` foi para `schema_version` 2.
  - **Por quê:** no material real quase todo clipe, inclusive com cor normal, saía como suspeito: cena clara ou pouco colorida já ficava abaixo dos dois limites antigos. Os novos limites são uma estimativa inicial, a calibrar com os valores do `catalog.md`.
- ✅ **Produtor e comando `briefing`** (`producer/`): pedido + resumo compacto do material + receitas registradas (lidas do registro na hora) → `Briefing` (tipo de vídeo, receita, "Entendi: ...", duração alvo, intensidade, ritmo, instruções para o Diretor e para a cobertura, papéis, avisos). Pela `ModelChain`, com as opções do Diretor (`--director-provider/model/fallback/thinking/timeout`), provedor `mock` e `none` (só a regra). Validação pelo código (receita existe e combina com o material, papéis, duração alvo), nova tentativa com o erro e **fallback determinístico** (1 clipe → `fala`; pasta com principal falado → `vlog` se registrada, senão `fala` com aviso), com a origem registrada. Pedido vazio + 1 vídeo falado não chama a IA. `--receita` força a receita. Para pasta, reaproveita o catálogo se a listagem não mudou (senão roda o `catalog`, com o cache por clipe); para arquivo, resumo mínimo (ffprobe + fala). Imprime "Entendi", receita, metas, papéis e uma estimativa sem preços (clipes, áudio a transcrever, chamadas de IA) e salva `briefing.json` na pasta do projeto.
  - **Por quê:** é o primeiro passo do fluxo "pedido em linguagem natural → plano": o usuário vê o que o sistema entendeu e quanto trabalho vem antes de gastar transcrição e IA. Falha de comunicação com a IA também cai no fallback (com aviso), porque o briefing sem IA ainda é útil.
  - 🔁 **Receitas de vários clipes se declaram com o atributo `multi_clip = True`** (lido com `getattr`; sem ele, a receita é de 1 vídeo). **Por quê:** `recipes/` é de outra tarefa desta rodada; o atributo evita fixar nomes de receitas no Produtor. A receita `vlog` precisa declará-lo para o Produtor aceitá-la em pastas com vários clipes falados.
  - 📊 Pasta sintética de 4 clipes (sem fala): catálogo em 3,5s na primeira execução; o briefing caiu no fallback com aviso, como esperado.
  - 📊 505 → 547 testes automatizados (catálogo: 22 novos; Produtor: 20); suíte completa em 5min 08s.
- ❌ **Fora desta tarefa:** o `edit` lendo o briefing (integração pendente: receita, metas e instruções ainda vêm das opções do `edit`), o Diretor de cobertura (só recebe o texto em `coverage_instructions`), `--papel`/`--ajuste-relogio` no `briefing` (rodar `catalog` antes) e a calibração dos limites de log com o material real.

## Projetos com clipes, rodada 2: análise visual das tomadas (2026-10-22)

- ✅ **Imagens na camada `llm/`** (aditivo): `generate(..., images=[ImageInput])`, com o prompt seguido de `Imagem <ID>:` + os bytes inline de cada miniatura — Gemini com `types.Part.from_bytes` (e `media_resolution` quando o detalhe não é `auto`), OpenAI com `input_text`/`input_image` (data URL base64, `detail`), mock com `image_calls` e respostas por função. Mesmas novas tentativas, mapeamento de erros, `ModelChain` e métricas (`images`).
  - **Por quê aditivo:** o Diretor e o revisor não podem mudar; sem imagens, a chamada ao SDK continua sendo o prompt em texto puro (testado nos dois provedores) e nada do fingerprint muda. O teste de referência do `edit` continua passando sem regenerar.
- ✅ **Comando `visual <pasta>`** (`visual/`): lê o catálogo (ou o gera pela API do `catalog/`), divide cada clipe em **tomadas** e descreve cada uma com IA a partir de 3 miniaturas.
  - **Tomadas sem IA:** uma passada do FFmpeg por clipe (480 px) faz ao mesmo tempo a detecção de cena (`select` com `scene` ≥ 0,30) e as medidas a 2 quadros/s (`blurdetect` e `signalstats`: desfoque, brilho, movimento por `YDIF`, tela uniforme pela faixa de luma), gravando cada ramo num arquivo de metadados. Tomadas acima de 8 s viram janelas.
  - **Por quê janelas:** no material real há um clipe de 1:13 com a câmera na parede amarela enquanto a pessoa fala, sem corte de cena; sem fatiar, o clipe inteiro vira uma tomada "talvez boa". Com janelas, o trecho ruim aparece por ID e vai para a seção **Atenção** do `visual.md` (fala com imagem obstruída/borrão = candidato a cobertura obrigatória na rodada 3).
  - **Por quê medidas no prompt:** são baratas e pegam o óbvio (parede, borrão), mas erram em céu liso ou água; vão como dica e a IA pode discordar.
  - **IA em lotes** de até 8 tomadas / 24 miniaturas rotuladas com IDs (`P007.T03.F2`); a IA nunca vê tempos. O código valida (tomada respondida exatamente uma vez, IDs existentes, `best_frames` da própria tomada e em ordem) e refaz o pedido só com as tomadas recusadas, avisando o erro; persistindo, a tomada fica `sem_analise` sem derrubar o resto, e é pedida de novo na próxima execução.
  - **Cache** por clipe em `_cache/clips/visual-<chave>/` (tomadas, miniaturas e análises da IA por fingerprint da cadeia/raciocínio/detalhe/prompt): rodar de novo não paga a IA de novo, nem com `--force`.
  - `--so-estimar` mostra tomadas, imagens, chamadas e tokens (constante por imagem da documentação do Gemini 3: 1120 no padrão, 280 em `--detalhe-imagem low`), sem preços.
  - 🔁 Miniaturas: as 3 de uma tomada saem num único FFmpeg (3 entradas com busca), porque no Windows a partida do processo pesa mais que a extração; os testes do módulo caíram de ~1min 40s para ~1 min.
- 🔁 **`intelligence/` vira legado** (docstring e referência): `analyze`/`report-ai` continuam usando; a análise visual nova está em `visual/`.
- 📊 Vídeo sintético (padrão nítido 4 s → parede amarela 10 s → padrão desfocado 4 s): cortes em 4,0 e 14,0 s, 4 tomadas (a parede em 2 janelas), parede marcada como uniforme e o trecho final como borrão alto (blurdetect ≈ 5 nítido, ≈ 20 desfocado).
- 📊 505 → 537 testes automatizados (10 de imagens no `llm/`, 22 do `visual`; o teste de referência do `edit` passa sem regenerar); suíte completa em 3min 43s.
- ❌ **Fora desta tarefa:** usar a análise visual nas decisões (Diretor de cobertura, rodada 3), estabilização/tremor por `vidstabdetect` (só a diferença entre quadros), calibração das constantes no material real, preços, e `--papel`/`--ajuste-relogio` no `visual` (rodar o `catalog` antes).

## Projetos com clipes, rodada 2: espinha falada multiclipe (2026-10-22)

- ✅ **`edit <pasta>` monta a espinha falada** (receita `vlog`, `pipeline/spine.py`): os clipes `principal` com fala, na ordem do catálogo, cada um sem pausas e sem frases ruins, num único vídeo renderizado pela Timeline. Etapas: catálogo → silêncios por clipe → transcrição por clipe → Diretor → Timeline → render → legendas → relatório. `edit <arquivo>` continua exatamente igual, pelo EditPlan (`tests/test_reference_fala.py` passa sem regenerar o snapshot). Cobertura (drone/b-roll por cima) fica para a rodada 3.
  - **Projeto = o do catálogo** (`<pasta>-<hash da listagem>`): um `catalog.json` existente é usado como está (preserva `--papel`/`--ajuste-relogio`); se falta, é gerado pela API pública de `catalog/`, sem mudar o módulo. `project.json` com `kind="multiclip"`.
  - **Clipes elegíveis:** principal, sem erro, com áudio e fala em ≥ 20% do tempo **ou** ≥ 3 s de fala (`MIN_SPEECH_RATIO`/`MIN_SPEECH_SEC`). **Por quê o "ou":** um clipe longo de paisagem com uma frase curta ainda conta a história; um clipe curto sem fala não. Os principais sem fala aparecem no relatório como "disponíveis para cobertura (rodada 3)".
  - **Cache por clipe fora do projeto** (`_cache/edit_clips/<hash do arquivo>/silences-<fp>.json`, `transcript-<fp>.json`). **Por quê:** trocar um clipe muda a listagem e, portanto, o projeto; com o cache dentro do projeto, todos os clipes seriam refeitos (mesma lição do cache do catálogo). Transcrição sequencial (um modelo na GPU), com barra somando os clipes; silêncios em paralelo (até 4 FFmpeg).
- ✅ **Diretor multiclipe:** `Sentence` ganhou `clip_id` opcional; frases montadas por clipe (uma frase nunca atravessa dois clipes; limites de 12 s / 40 palavras mantidos) com **IDs globais** sequenciais. Mesmo `LLMDirector`, mesma validação/nova tentativa/reserva; o prompt mostra os clipes como seções (`Clipe P006 (0:34) — arquivo`). Categoria nova **`retake`** (tomada repetida no mesmo clipe ou em clipes seguidos; fica a última/melhor). Validação nova: não pode remover todas as frases de um clipe, a menos que todas sejam `retake`/`off_topic`. O prompt de um vídeo só ficou byte a byte igual (conferido contra a versão anterior).
- ✅ **Plano → Timeline** (`planning/timeline_plan.py`): por clipe, os trechos do mesmo `plan_keep_ranges` do caminho de um vídeo (reaproveitado, não copiado) viram itens de `video_main` com `clip_id`; `timeline.json` salvo no projeto e na versão. Saída 1920x1080 e o fps mais comum dos clipes **arredondado** (29,97 → 30), porque a Timeline exige fps inteiro; o filtro `fps` do render converte cada clipe e o relatório avisa.
- ✅ **Legendas e relatórios por clipe:** legendas de cada clipe com a regra de um vídeo (sobreposição palavra × trecho), deslocadas para o tempo final; `edit_report.md` com resumo por clipe (original → final), clipes não usados e o que saiu de cada clipe (categoria, motivo, aplicada/parcial/não aplicada); `edit_transcript.md`/`edit_diff.json` (`SpineDiff`) por clipe; `edit_timeline.svg` com uma trilha colorida por clipe. `versions/v1` como no `edit`; **`revise` recusa projeto de pasta** ("chega na rodada 4").
- ✅ **`Recipe.multi_clip`** (combinado com a tarefa producer-briefing): "receita que trabalha com vários clipes de uma pasta"; `vlog=True`, `fala=False`. O Produtor usa para validar se a receita combina com o material; a CLI usa para recusar `--receita vlog` com um arquivo (e `fala` com uma pasta).
- 🔁 **Peças:** novas `catalog` e `timeline`; `render` e `subtitles` deixaram de exigir `plan` na peça (a `fala` declara isso em `requires`; a `vlog` exige `timeline`). `silences`/`transcription` exigem `probe` só na `fala`. CLI: `--receita` padrão automático (`fala` para vídeo, `vlog` para pasta) e `--pedido` como sinônimo de `--instructions`.
- 🔁 **Áudio da Timeline num só processo FFmpeg** (`renderer/timeline_compiler.py`). Antes: um FFmpeg por item do áudio principal + concat; com centenas de cortes, centenas de processos (cada partida custa ~1 s nesta máquina Windows). Agora: um processo para o áudio inteiro, qualquer que seja o número de itens. Os itens de cada origem são agrupados em sequências crescentes; cada sequência entra uma vez e `asegment` a divide com precisão de amostra (cada amostra vai para um único ramo), com os micro fades e a contagem de amostras derivada dos frames; `concat` junta e o ambiente é mixado no mesmo grafo, que vai num arquivo (`-/filter_complex`) por causa do limite de linha de comando do Windows. Itens muito fora de ordem na mesma origem caem em `asplit` + `atrim` (sem multiplicar entradas). O processo de áudio roda em paralelo com os pedaços de vídeo. O caminho do EditPlan não mudou.
  - **Por quê `asegment` e não `asplit`:** a primeira versão (uma entrada por origem + `asplit` + `atrim` por item) já era 1 processo, mas cada quadro de áudio passava por todos os ramos: 8,3 s para 300 itens. Com `asegment`, 3,3 s.
  - 📊 Timeline sintética de 300 itens (2 origens de 5 min, 1080p): **301 processos / 15,4 s → 1 processo / 3,3 s**; amostras exatas (`frames × 48000 / fps`) e o trecho certo em cada item (teste com tom/silêncio alternados).
- 📊 **Material real** (vlog "Trilha La Islita", 41 clipes, 28 principais Osmo Pocket 29,97 fps; Diretor `mock`, transcrição `large-v3-turbo` na GPU, catálogo existente): 22 de 28 principais na espinha (6 com pouca fala + 13 de drone/POV/b-roll para cobertura), **13min 27s → 11min 22s**, 138 trechos, 221 blocos de legenda, 11 tomadas repetidas removidas pelo mock. Tempo total **16min 13s** (1,21× o tempo real): silêncios 1min 28s (sequencial; depois de paralelizar, 19,9 s nos mesmos 22 clipes, com o disco já em cache), transcrição 3min 45s, render **10min 56s** (quase tudo é a codificação dos 138 pedaços de vídeo 1080p na CPU; o áudio agora é um processo só).
  - Observado: em clipes curtos com fala contínua (P001, P002), o limite `auto` fica no teto (-20 dB) e nada é cortado; é o comportamento conhecido de "detecção pouco confiável" sem pausas, não um erro.
- 📊 Testes novos: `tests/test_spine.py` (13: receita, IDs globais, frase não cruza clipe, prompt por seções, retake, clipe inteiro removido, escolha dos clipes, Timeline, legendas no tempo final, pipeline com cache por clipe, revise recusado, render com duração esperada) e 2 de processos/duração do áudio em `test_timeline_render.py`.
- 🔁 **Integração no merge com o Produtor e o catálogo v2:** o `edit <pasta>` passou a ler o catálogo pela mesma regra do Produtor (`catalog.builder.current_catalog`: só reaproveita o `catalog.json` com a listagem e o `schema_version` atuais; um catálogo v1, gerado com as regras antigas de papel e ordem, é refeito). Teste novo com a receita `vlog` real no Produtor: pasta com vários clipes falados é aceita como `vlog` vinda da IA, sem fallback. 📊 Depois do merge: 596 testes, suíte completa em 4min 35s.
- ❌ **Fora desta tarefa:** cobertura (rodada 3), `revise` multiclipe (rodada 4), reordenar clipes, unificar o caminho de um vídeo com a Timeline, render de vídeo na GPU, rodar o Diretor Gemini no material real (custo de API; o comando está na referência).

## Correção: `visual` em tomada contínua (2026-10-22)

- 🔁 **Bug:** `visual <pasta>` falhava em 40 dos 41 clipes do material real com `Could not open encoder before EOF ... error code: -22`. **Causa:** na passada de análise (`visual/shots.py: scan_clip`), o ramo `select='gt(scene,0.30)'` só deixa passar quadros com mudança de cena; num clipe sem nenhuma (tomada contínua, o caso comum do vlog) a saída `-map [o1] -f null -` não recebia quadro algum e o FFmpeg aborta. **Por que os testes não pegaram:** o único vídeo sintético ponta a ponta tinha duas trocas de cena de propósito.
- ✅ **Correção:** `select='eq(n,0)+gt(scene,0.30)'` — o 1º quadro sempre passa e o 1º bloco do arquivo de metadados é descartado ao montar os cortes. **Por quê e não `scdet`:** muda uma expressão e mantém o mesmo `scene` score; conferido que o resultado (cortes e amostras) do vídeo sintético com cenas é idêntico ao de antes. Sem mudança no cache: um clipe com erro nunca foi gravado (o `shots.json` só é escrito no fim), e o resultado dos que funcionavam não mudou, então `shot_params` continua igual.
- 🔁 **Mesmo tipo de problema nas miniaturas:** a duração do clipe é a do contêiner, que passa da do vídeo quando o áudio é mais longo; a miniatura F3 (fim − 10%, limitada a duração − 0,05 s) podia cair depois do último quadro (ex.: 10 fps) e o `extract_frames` falhava. Agora essa miniatura vira o último quadro do vídeo (`-sseof` + `-update 1`). O ramo de medidas (`fps=2`) já emite o 1º quadro mesmo em clipe < 0,5 s (testado).
- 📊 Testes novos: tomada contínua de 18 s (3 janelas de 6 s, 12 amostras cada, miniaturas), clipe de 0,4 s, parede de cor sólida sem cortes e miniatura depois do último quadro. 596 → 600 testes; suíte completa em 4min 20s.
- 📊 **Material real** (vlog "Trilha La Islita", 41 clipes, `--visual-provider none`): **41 sem erro, 200 tomadas** (principal 122 em 28 clipes, drone 54 em 2, POV 3 em 1, b-roll 21 em 10; 1 troca de cena, no drone) em **3min 16s** de tomadas e medidas (4min 05s com o catálogo). `--so-estimar`: 200 tomadas, 600 imagens, 25 chamadas, ~706.500 tokens de entrada (detalhe `auto`) ou ~202.500 (`low`) e ~30.000 de saída.
  - **As constantes não acertam ainda (calibração pendente):** P007 (parede amarela, 1:13): a parede fica em ~44–52 s, com faixa de luma 0,09–0,12 — só 2 amostras abaixo de `UNIFORM_SPREAD = 0,10`, e a janela T07 (43,6–50,8 s, mediana 0,11) **não** foi marcada como uniforme. P023 (borrão de movimento): o desfoque do `blurdetect` fica em 4,6–5,8 (nítido ≈ 5; `BLUR_HIGH = 9`), então o borrão de movimento **não** aparece como desfocado; aparece só como diferença entre quadros alta em ~10–12 s (T02 marcada como tremida). `MOTION_HIGH = 0,08` marca 95 das 200 tomadas como tremidas (câmera na mão), alto demais para ser útil. Nenhuma tomada saiu escura, clara demais, uniforme ou desfocada.

## Calibração das medidas visuais com o material real (2026-10-22)

- 🔁 **Tela uniforme:** `UNIFORM_SPREAD` 0,10 → **0,14**. **Por quê:** a parede amarela real do P007 (~44–52 s) tem faixa de luma entre 0,09 e 0,12 (textura e ruído do sensor); só 2 amostras ficavam abaixo de 0,10. Imagem com conteúdo fica acima de 0,25 no material.
- 🔁 **Tremor relativo ao projeto:** `MOTION_HIGH` absoluto saiu; `movimento alto` agora é movimento acima do **percentil 90 das tomadas do projeto** (`MOTION_PERCENTILE = 90`, interpolado, estritamente acima: se todas se movem igual, nenhuma se destaca), com piso `MOTION_FLOOR = 0,08`. **Por quê:** o vlog é todo de câmera na mão; com 0,08 fixo, 95 das 200 tomadas saíam tremidas e a marca não informava nada. Mesmo raciocínio do limite de silêncio `auto`. Calculado depois de medir todos os clipes (`mark_shaky`); o limite vai em `visual.json` (`motion_threshold`) e no `visual.md`.
- 🔁 **Cache guarda as amostras brutas** (`shots.json` com `samples`; `VISUAL_CACHE_VERSION` 1 → 2) e as marcas são recalculadas a cada execução. Os limites das marcas saíram de `shot_params` (a chave do cache): calibrar não refaz o FFmpeg nem perde as análises da IA guardadas na mesma pasta. Na 1ª execução depois da mudança, o cache antigo é refeito uma vez.
- ✅ **Desfoque mantido** (`BLUR_HIGH = 9`), documentado que o `blurdetect` não detecta borrão de movimento (P023 ficou em 4,6–5,8, igual a imagem nítida); isso fica a cargo da IA (`borrao_movimento`).
- 🔁 **Prompt da IA:** a linha `medidas:` só aparece quando alguma medida foi marcada (antes: `nada a notar` em todas as tomadas), sem valores brutos; o texto geral não lista mais as medidas. **Por quê:** não induzir a IA. `PROMPT_VERSION` 1 → 2 (as análises em cache de antes não valem mais).
- 📊 **Material real** (41 clipes, `--visual-provider none`): limite de tremor do projeto **0,1166**; **20 tremidas** (antes 95: principal 12, drone 5, POV 1, b-roll 2), **1 uniforme** (antes 0), 0 desfocadas. P007: só **T07** (43,6–50,8 s, mediana 0,11) sai uniforme; a parede termina ~52,5 s, pouco dentro da T08. P023: **T02 não sai tremida** (movimento 0,088 < 0,1166) — o chacoalhão dura ~2 s (10–12 s) numa janela de 7,2 s e a mediana dilui; fica para a IA (`borrao_movimento`/`tremida`). Tempo: 1ª execução 3min 09s (refaz o cache no formato novo), 2ª **1,1 s** (tomadas e medidas do cache bruto em 0,3 s).
- 📊 Testes novos: percentil (projeto com uma tomada destoante, todas iguais, tomada única e par, piso), parede com textura (0,09–0,12) uniforme e prompt sem a linha `medidas:` quando nada foi marcado. 600 → 605 testes; suíte completa em 3min 53s.

## Lotes da análise visual em paralelo (2026-10-22)

- 📊 **Problema medido:** `visual` com IA no material real (200 tomadas, 25 lotes, `--detalhe-imagem low`) levou ~5 min para 3 lotes (~100 s por lote): os 25 lotes, um depois do outro, iriam a ~40 min. Os lotes rodavam em sequência em `_describe`.
- ✅ **`--visual-paralelo N`** (padrão 3; 1 = como antes): pool de threads com até N lotes ao mesmo tempo, cada um chamando `describe_batch` com a mesma validação/nova tentativa por tomada. O resultado não depende da ordem de término (testado: idêntico ao sequencial). **Por quê:** a demora é a espera pela resposta do provedor, não CPU; com 3 lotes ao mesmo tempo o tempo da etapa cai para ~1/3, mantendo o pedido pequeno para não estourar o limite por minuto.
- ✅ **Cache gravado a cada lote**, com trava por clipe e escrita por arquivo temporário + troca. **Por quê:** antes um Ctrl+C perdia só o lote em curso, e com lotes em paralelo dois lotes do mesmo clipe podiam escrever o mesmo arquivo juntos (o teste sem a trava reproduz o `WinError 32` do Windows). No Ctrl+C, a fila é cancelada e os lotes em andamento terminam e gravam.
- ✅ **Limite do provedor:** lote que falha por cota esgotada ou 429 corta o paralelismo pela metade (mínimo 1) para os seguintes, com um aviso. A regra de parar depois de 2 lotes seguidos com erro da API continua (contada pela ordem de término).
- ✅ **Medição por lote** em `visual.json` (`batch_stats`: espera para começar, duração, chamadas, correções, trocas de modelo, modelo) e resumo no `visual.md` e no terminal (média/mín/máx por lote, novas tentativas, trocas). **Por quê:** separar lentidão do provedor (lote demorado) de lentidão da fila (lotes rápidos, etapa longa).
- 🔁 **Camada `llm/` segura para threads:** métricas da última chamada por thread (`llm/per_thread.py`: cliente, cadeia e etapa), chamadas à API contadas na própria chamada (não em `RetryPolicy.attempts_made`, compartilhado) e troca de modelo uma vez só sob trava (duas threads que falham juntas no mesmo modelo não pulam o seguinte).
- 📊 Testes novos (sem rede, IA simulada com atraso de 0,2 s, sem FFmpeg): 6 lotes em paralelo 3 bem mais rápidos que em sequência e com resultado e cache idênticos; cache gravado a cada lote e mantido num Ctrl+C no 3º lote; 429 reduz o paralelismo de 4 para 2 com um aviso; trava do cache com dois lotes do mesmo clipe; 2 lotes seguidos com erro ainda param o resto; cadeia troca uma vez só com duas threads falhando juntas; opção da CLI.

## Correção: corrida na criação do cliente do SDK com lotes em paralelo (2026-10-22)

- 🐛 **Sintoma (material real, vlog, 200 tomadas, `visual --visual-provider gemini --detalhe-imagem low --visual-thinking off`, paralelo 3):** só o 1º lote (P001–P004) foi analisado; os outros falharam com "Cannot send a request, as the client has been closed." e, depois de 2 falhas seguidas, os lotes restantes não foram enviados (5 chamadas, 192 tomadas sem análise).
- 🔍 **Causa (confirmada no código e reproduzida em teste):** `StructuredLLMClient.get_client` criava o cliente do SDK sob demanda sem trava. As 3 threads viam `_client` vazio, criavam um `genai.Client` cada e sobrescreviam o atributo; o SDK fecha a conexão HTTP no `__del__` do cliente, e o cliente descartado era coletado enquanto a thread dele ainda chamava `.models.generate_content`. O SDK da OpenAI faz o mesmo. Os testes não pegaram porque injetavam o cliente pronto.
- ✅ **`get_client` seguro para threads** na base (trava com verificação dupla: o cliente é criado uma única vez e todas as threads recebem o mesmo objeto); nenhum provedor mudou. `ModelChain.client_for` tinha a mesma corrida no dicionário de clientes por provedor e agora cria sob a trava da cadeia. `thinking_label` não guarda estado e ficou como estava.
- 📊 Testes novos (sem rede; falham sem a correção): 8 threads em `get_client` com criação lenta → 1 cliente; o mesmo para `ModelChain.client_for`; `visual` com 6 lotes em paralelo 3 e cliente criado sob demanda (o falso "fecha" quando é substituído, como o SDK) → 1 cliente e todas as tomadas analisadas.

## Render na GPU NVIDIA (NVENC) com volta automática para a CPU (2026-10-23)

- 📊 **Problema medido:** no vlog real (espinha de 22 clipes, 138 pedaços 1080p, Diretor mock) o render levou **10min 56s de 16min 13s**, quase tudo codificação x264 na CPU.
- ✅ **`--encoder auto|cpu|nvenc`** no `edit` e no `render` (padrão `auto`) e **`--gpu-paralelo N`** (padrão 3). `auto` = NVENC se o FFmpeg lista o `h264_nvenc` **e** uma codificação de teste de 1 s (`testsrc` → `null`, com os parâmetros reais) funciona; senão CPU. A detecção roda uma vez por execução (cache por FFmpeg). `nvenc` explícito sem GPU falha antes das etapas. Módulo novo `renderer/encoders.py`. **Por quê a codificação de teste:** o `h264_nvenc` aparece em qualquer build "full" do FFmpeg, com ou sem placa/driver; só listar daria falso positivo.
- ✅ **Um codificador por vídeo:** os pedaços são unidos com concat `-c copy`, então todos usam o mesmo codificador e os mesmos parâmetros. Se um pedaço falha por causa do NVENC (sessões da GeForce, memória, driver), o render **inteiro** é refeito na CPU, com aviso (`cpu (NVENC falhou: ...)`). Outros erros falham como antes. Decodificação e filtros continuam na CPU (hwaccel de decodificação fica fora).
- ✅ **Limite de sessões:** no NVENC, no máximo `--gpu-paralelo` pedaços codificam ao mesmo tempo (semáforo; o processo de áudio da Timeline não conta); quem ainda espera vaga desiste depois de uma falha.
- ✅ **Onde aparece:** linha da etapa render, "Codificador de vídeo" no `edit_report.md`, métrica `encoder` no `project.json` e `Codificador:` no `render`.
- ✅ **Cache:** o codificador só entra no fingerprint do render. Na CPU o fingerprint é **o mesmo de antes** (renders existentes continuam válidos; `tests/test_reference_fala.py` passa sem regenerar); no NVENC entram o nome e os parâmetros. Render que caiu para a CPU fica registrado com o fingerprint da CPU (um novo pedido de NVENC tenta a GPU de novo).
- 🔁 **`-cq 25` em vez do ponto de partida `-cq 23`.** Parâmetros: `h264_nvenc -preset p5 -tune hq -rc vbr -cq 25 -b:v 0 -profile:v high -g 250 -bf 3 -pix_fmt yuv420p` (GOP 250 e 3 B-frames = padrões do x264). **Por quê:** medido no material real (abaixo), `-cq 23` sai com SSIM acima do x264 mas 41% maior; `-cq 25` é o primeiro valor com SSIM ≥ ao da CPU.
- 📊 **Medido nesta máquina (RTX 2060, FFmpeg 9.0.2), material real:** EditPlan com 40 trechos de 2,5 s de 4 clipes do Osmo Pocket (1080p 29,97 → 30 fps; 100 s de saída), SSIM contra uma referência x264 `-crf 8`:

  | Codificador | Render | Tamanho | SSIM (All) |
  | --- | --- | --- | --- |
  | CPU x264 `fast` `-crf 22` | 80,7 s | 92,9 MB | 0,98692 |
  | NVENC `-cq 23` | 29,1 s | 131,2 MB | 0,98975 |
  | NVENC `-cq 24` | 28,3 s | 115,9 MB | 0,98869 |
  | **NVENC `-cq 25`** | **29,0 s** | **103,2 MB** | **0,98753** |
  | NVENC `-cq 26` | 26,4 s | 91,7 MB | 0,98615 |
  | NVENC `-cq 28` | 26,6 s | 71,9 MB | 0,98271 |

  Todos com 3000 frames e 100,000 s. Paralelismo (`-cq 25`): 2 sessões 32,8 s, 3 → 29,0 s, 4 → 30,3 s, 6 → 26,5 s (ruído de ±2 s; com 3 a GPU já não é o gargalo, e sim decodificar e escalar na CPU). O padrão fica 3 por segurança com drivers GeForce antigos. Saída idêntica (mesmo SSIM) com qualquer número de sessões. Timeline real (10 trechos + overlay de b-roll do Fx30) também fecha com NVENC (750 frames). Em fonte sintética (`testsrc2`, 1080p, 20 trechos) o NVENC ganha menos: 28,9 s (CPU) vs 17,2 s.
- 📊 **Estimativa (não medida) para o vlog completo:** render ~2,8× mais rápido → **~4 min em vez de 10min 56s** (−7 min de 16min 13s).
- 📊 **A medir no vlog real** (cmd; `%VLOG%` = pasta do vlog; `%PROJ%` = pasta mostrada em `Projeto:`; `<pasta>` = nome da pasta do vlog). Rodar primeiro o NVENC: como o fingerprint da CPU não mudou, um `--encoder cpu` primeiro sairia do cache. Transcrição e Diretor mock vêm do cache; só render e relatório rodam:

  ```bat
  python -m ai_video_editor.cli edit "%VLOG%" --transcribe --director mock --encoder nvenc
  copy "%PROJ%\<pasta>_edited.mp4" "%PROJ%\vlog_nvenc.mp4"
  python -m ai_video_editor.cli edit "%VLOG%" --transcribe --director mock --encoder cpu
  copy "%PROJ%\<pasta>_edited.mp4" "%PROJ%\vlog_cpu.mp4"

  rem frames e duração (precisam ser iguais nos dois)
  ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,duration -of csv=p=0 "%PROJ%\vlog_cpu.mp4"
  ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,duration -of csv=p=0 "%PROJ%\vlog_nvenc.mp4"
  rem SSIM entre a saída CPU e a NVENC (linha "SSIM ... All:")
  ffmpeg -hide_banner -i "%PROJ%\vlog_cpu.mp4" -i "%PROJ%\vlog_nvenc.mp4" -lavfi "[0:v][1:v]ssim" -f null -
  dir "%PROJ%\vlog_*.mp4"
  ```

  | Campo | CPU | NVENC |
  | --- | --- | --- |
  | Render (linha `[6/8] Renderização`) | _ | _ |
  | Tempo total | _ | _ |
  | Codificador no relatório | _ | _ |
  | Tamanho | _ | _ |
  | Frames / duração | _ | _ |
  | SSIM CPU × NVENC (All) | — | _ |

- 📊 Testes novos (sem GPU: FFmpeg falso e NVENC simulado; o `conftest.py` faz o `auto` ver "sem GPU"): detecção (não listado → CPU; listado mas o teste falha → CPU; listado e funciona → NVENC; FFmpeg ausente; uma detecção por FFmpeg), escolha `auto/cpu/nvenc`, argumentos NVENC iguais em todos os pedaços (EditPlan e Timeline), caminho CPU idêntico (argumentos e fingerprint), limite de sessões na GPU, erro do NVENC × outros erros, volta para a CPU no meio do render refazendo **todos** os pedaços (EditPlan e Timeline), codificador só no fingerprint do render (cache por codificador e render que caiu para a CPU) e opções da CLI. 616 → 632 testes; suíte completa em 3min 34s (`-n auto`). A fixture que força a CPU é de escopo de sessão: com escopo de função, as fixtures de módulo que renderizam rodavam antes dela e usavam o NVENC de verdade numa máquina com GPU (pego por `test_revision.py`).
## Projetos com clipes, rodada 3: Diretor de cobertura e render com cobertura (2026-10-23)

- ✅ **Receita `vlog_cobertura`** (`multi_clip=True`): a `vlog` + as peças novas `visual` (lê o `visual.json`, sem IA) e `coverage`, entre `timeline` e `render`. A `vlog` não mudou: mesmas etapas, fingerprints e saída (conferido contra um snapshot da execução antes da mudança: fingerprints de todas as etapas e hash de `timeline.json`, `spine_plan.json`, `edit_diff.json`, `edit_transcript.md`, `edit_timeline.svg`, `.srt` e relatório idênticos); `tests/test_reference_fala.py` passa sem regenerar.
  - **Receita padrão:** sem `--receita`, uma pasta usa `vlog_cobertura` quando o projeto já tem `visual/visual.json` **e** há `--transcribe`; senão, `vlog`. **Por quê:** quem rodou `visual` quer cobertura, e ela escolhe frases por ID (sem transcrição não há o que cobrir). `--receita vlog` dá a espinha sem cobertura.
- ✅ **Peça `visual`:** lê `visual/visual.json` e grava `coverage_shots.json` só com o que a cobertura usa (tempos, miniaturas, `tela uniforme`, fala, análise da IA, caminho e áudio do catálogo). Sem o arquivo: erro "rode `visual <pasta>` antes (ou use --receita vlog)"; de outro catálogo (clipe que mudou de arquivo ou sumiu): recusado. **Por quê o arquivo compacto:** o fingerprint da cobertura depende só do que ela usa (não de `created_at`, tempos de lote etc.).
- ✅ **Janelas obrigatórias (código):** frase mantida com ≥ 0,3 s da parte mantida numa tomada do clipe que fala com `usable=nao`, `obstruida`, `borrao_movimento` ou medida `tela uniforme` (esta vale também em tomada `sem_analise`); frases seguidas do mesmo clipe viram uma janela `O1` com tomadas e motivos. **Por quê pela parte mantida:** uma frase aparada pelo corte de silêncio pode não ter mais nada sobre a tomada ruim.
- ✅ **Diretor de cobertura** (módulo `coverage/`, mesmo molde do `director/`): schema, prompt compacto (frases mantidas com clipe e duração no vídeo, janelas, candidatas com papel/duração/beleza/qualidade/rosto/descrição/tags, `coverage_instructions` do briefing e o `--pedido`), `LLMCoverageDirector` sobre a `ModelChain` (gemini/openai) e `MockCoverageDirector`. Candidatas: drone/b-roll/POV e tomadas **sem fala** do principal, analisadas e utilizáveis, filtradas por `roles_to_use`.
- ✅ **Validação (código):** IDs existentes, tomada candidata (a mensagem diz por que não), tomada fora dos clipes que falam no trecho, coberturas sem sobreposição, **cada tomada uma vez só**, cobertura de 1,5 a 8 s (uma frase sozinha pode passar), respiro fora de cobertura e no máximo um a cada 30 s de espinha, todas as janelas cobertas. Inválida → nova tentativa com o erro; de novo inválida → se só faltam janelas, a IA fica e o código completa (`ia+fallback`); senão, só as janelas obrigatórias (`fallback`) com a tomada livre de maior beleza que caiba.
  - **Tomada uma vez só (decisão):** recusada na validação em vez de aviso. **Por quê:** repetir a mesma imagem num vlog salta aos olhos, e o material tem tomadas de sobra (90 candidatas no real).
- ✅ **IDs → tempos (código):** frases → frames pela posição dos trechos de cada clipe na Timeline (`clip_placements`/`map_source_range`); coberturas → `video_overlay` (a fala continua) com o trecho da tomada centrado no melhor trecho e limitado à tomada; tomada curta numa janela obrigatória → o código **encadeia** outra tomada no resto; fora de janela → **encurta**; sobra < 0,75 s fica sem cobertura (não pisca); tudo com aviso. Respiros → itens de `video_main` inseridos no primeiro corte depois da frase (senão no meio do caminho até a próxima, dividindo o item; na última frase do clipe, no fim dele), com 2/3,5/5 s limitados à tomada e o som da tomada (mudo sem áudio ou com fala).
  - **Som ambiente (decisão):** sim, o áudio da tomada da cobertura a 0,12 em `audio_ambient` com fades de 0,3 s, se o clipe tem áudio e a tomada não tem fala. **Por quê:** era simples (o render da Timeline já mixa ambiente num só FFmpeg) e tira a sensação de "imagem colada" sem disputar com a voz.
- ✅ **Tudo depois usa a Timeline final** (`coverage_timeline.json`): legendas por trecho (cada bloco vai para onde o seu trecho está; nunca atravessa um respiro), `edit_diff.json`/`edit_transcript.md` no tempo final, `edit_timeline.svg` com a faixa de overlay e os respiros em cinza, `edit_report.md` com a seção Cobertura, `coverage_decision.json` (modelo efetivo, trocas, tentativas, origem, janelas e quem atendeu, onde cada item foi parar) e a versão com a Timeline renderizada.
- 🐛 **Cache pela saída da cobertura (achado no teste com o material real):** com o fingerprint do render calculado só pelas entradas da cobertura, refazer a etapa com as mesmas entradas (artefato apagado, `--force`) e outra decisão deixava render, legendas e relatório no cache da Timeline antiga. Agora seguem o hash do `coverage_timeline.json` (e da decisão, nos dois últimos). Teste de regressão.
- 🔁 **Ciclo de imports evitado:** `coverage/` não importa os pacotes `visual`/`catalog` ao carregar (catálogo → pipeline → cobertura → catálogo); os poucos literais (`analisada`, `sim/nao/parcial`) ficam repetidos em `coverage/schema.py`, com comentário.
- 🔁 Produtor: `coverage` entrou nas peças de IA da estimativa de chamadas. Dois testes do Produtor listavam as receitas registradas e passaram a incluir `vlog_cobertura`.
- 📊 **Material real** (vlog "Trilha La Islita", 41 clipes, visual.json com as 200 tomadas analisadas; Diretor `mock`, cobertura `mock`, render falso, transcrição `large-v3-turbo` na GPU): 22 clipes na espinha (11:22), **10 janelas obrigatórias, todas cobertas** — P001 (obstruída), P007 (borrão, inutilizável, tela uniforme), P015.T03, P016.T01–T02, P021.T03–T04, P023.T02 (as da seção "Atenção" do `visual.md`) e mais quatro em P024/P025 (borrão de movimento sob fala transcrita) —, 15 coberturas (8 com tomadas encadeadas: há frases únicas de 12 a 26 s, mais longas que qualquer tomada de 8 s), 5 respiros (+17,5 s), 90 candidatas; final 11:39. A etapa de cobertura leva menos de 0,1 s (sem IA); transcrição 2min 10s; silêncios 7,8 s.
- 📊 Testes: 616 → 647 (`tests/test_coverage.py`: janelas obrigatórias a partir de um `ProjectVisual` sintético e só onde há fala mantida, frase aparada pelo silêncio, candidatas e `roles_to_use`, `visual.json` de outro catálogo, prompt, validação recusando cada caso, nova tentativa com o erro, fallback, `ia+fallback`, IA desligada, mock, herança das opções do Diretor, respiro no fim do clipe e no meio de um item, tomada mais curta que a janela, legendas por trecho, receitas; pipeline com render falso: erro sem `visual.json`, legendas/diff/relatório/SVG/versão no tempo final, cache e a regressão acima; CLI escolhendo a receita; render real com overlay e respiro: duração = espinha + respiros, quadros do b-roll na cobertura e no respiro, fala preservada sob o overlay e respiro mudo). Suíte completa em 6min 07s (antes 3min 53s a 4min 35s; a maior parte do aumento são os 3 testes novos que rodam FFmpeg de verdade).
- ❌ **Fora desta tarefa:** o `edit` ler o resto do briefing (receita, intensidade, ritmo, `director_instructions`), `revise` multitrilha (rodada 4), música, transições, cor, render na GPU. `renderer/` não foi alterado.

---

## 🔄 Em andamento

- (nada no momento; rodada 1 concluída: Timeline + render e catálogo + receitas estão na main)

## 🔮 Próximos passos (V1)

- **Ponto de corte de palavras curtas pela energia do áudio**: na palestra de 57 min, fillers como "Um" e "By" ficaram no vídeo. O tempo do Whisper para essas palavras inclui a pausa ao redor, e o corte só tirou o silêncio (nenhuma palavra foi cortada pela metade). Refinar o início/fim do corte pela energia do áudio dentro do intervalo da palavra.
- **Status "não aplicada (só a pausa saiu)"**: quando a parte removida de uma frase for apenas silêncio, marcar assim em vez de "parcial" (caso do "By": 0,19s de 0,74s saiu, mas era só pausa).
- **Benchmark de transcrição com Gemini**: incluir o provedor (e custo) na comparação; a estrutura aceita novos provedores (`register_benchmark_provider`).
- **Provedor de transcrição Gemini** (alternativa para quem não tem GPU).
- **Medir o NVENC no vlog completo** (comandos e campos na entrada de 2026-10-23) e, se o render continuar pesado, decodificação na GPU (hwaccel).
- **Teste com vídeo em português** com mais enrolação.
- Pequenos ajustes: cópia de vídeo por versão ocupa o dobro de disco; motivo do Diretor às vezes truncado.

## 🔮 Projetos com clipes (próximas rodadas; ver [ARQUITETURA.md](ARQUITETURA.md))

- **Rodada 2:** ~~Produtor (pedido + catálogo → briefing validado)~~ (feito: comando `briefing`; falta o `edit` ler o briefing) ∥ ~~espinha falada multiclipe~~ (feita: `edit <pasta>`, receita `vlog`) ∥ ~~análise visual dos clipes~~ (feita: comando `visual`). Análise visual calibrada no material real (2026-10-22). Em aberto: tremor curto dentro de uma janela longa (P023) não aparece na mediana; fica para a IA.
- **Calibrar os limites de perfil log** com os valores do `catalog.md` do material real.
- ~~**Rodada 3:** Diretor de cobertura + render com cobertura~~ (feita: receita `vlog_cobertura`, 2026-10-23). Em aberto: rodar o Diretor de cobertura Gemini no material real e calibrar o prompt (quantas coberturas fora das janelas, tamanho dos respiros); o `edit` ler o resto do briefing.
- **Rodada 4:** `revise` multitrilha + relatórios.

## 🔮 V2 (futuro)

- **Perfis de edição**: Gameplay (picos de áudio + imagem), Vlog (imagem + música), como receitas.
- **Aplicação web**: upload, prévia, chat de ajustes, fila de processamento.
