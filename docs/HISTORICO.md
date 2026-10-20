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

📊 **Palestra de 57 min, resultado atual:** 57:07 → 53:43, 259 cortes, processada em **8min48s** (0,15× o tempo real, com transcrição em cache).

---

## 🔄 Em andamento

- (nada no momento)

## 🔮 Próximos passos (V1)

- **Normalização de volume** (`loudnorm`): áudio num nível padrão, como YouTube/Spotify fazem.
- **Benchmark de transcrição**: CPU × GPU × Gemini, comparando com a **transcrição oficial do MIT** (taxa de erro de palavras, tempo e custo) → tabela no README.
- **Provedor de transcrição Gemini** (alternativa para quem não tem GPU).
- **Render na GPU (NVENC)**: o render é hoje a etapa mais lenta em vídeos longos.
- **Teste com vídeo em português** com mais enrolação.
- **README de portfólio**: antes/depois, números medidos, diagrama e decisões de engenharia.
- Pequenos ajustes: cópia de vídeo por versão ocupa o dobro de disco; motivo do Diretor às vezes truncado.

## 🔮 V2 (futuro)

- **Perfis de edição**: Gameplay (picos de áudio + imagem), Vlog (imagem + música) — a análise visual da Fase 3 volta aqui.
- **Aplicação web**: upload, prévia, chat de ajustes, fila de processamento.
