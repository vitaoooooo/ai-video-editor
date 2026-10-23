# Referência técnica — AI Video Editor

Documentação detalhada de instalação, comandos, opções, artefatos e módulos. A visão geral do projeto está no [README](../README.md); as decisões e os resultados medidos, no [histórico](HISTORICO.md).

---

## 1. Requisitos de Sistema

* **Python 3.11+** (testado no Python 3.14.8).
* **FFmpeg / FFprobe**: detectado automaticamente via:
  1. Variáveis de ambiente `FFMPEG_PATH` e `FFPROBE_PATH`.
  2. `PATH` do sistema operacional.
  3. Locais padrão no Windows (ex: Stremio/WinGet/Chocolatey).

---

## 2. Instalação Local (Sem dependências globais)

Criar o ambiente virtual e instalar o projeto em modo editável com dependências de desenvolvimento:

```bash
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar o ambiente (PowerShell no Windows)
.\.venv\Scripts\Activate.ps1

# 3. Instalar o pacote com as dependências de desenvolvimento e a transcrição
pip install -e ".[dev,transcription]"   # ou ".[dev,gpu]" com GPU NVIDIA; acrescente ",openai" para o provedor OpenAI
```

---

## 3. Executando os Testes Automatizados

O projeto inclui suíte completa de testes unitários e de integração utilizando **vídeos sintéticos** (FFmpeg) e **provedor Mock** para IA, sem necessidade de chamadas de rede externas ou chaves reais de API:

```bash
.\.venv\Scripts\python.exe -m pytest -q -m "not slow"   # rápidos, durante o desenvolvimento
.\.venv\Scripts\python.exe -m pytest -q -n auto         # suíte completa, em paralelo (inclui os `slow`, que renderizam vídeo)
```

O GitHub Actions roda a suíte completa a cada push e pull request para o `main` (Python 3.11 e 3.12, Ubuntu).

---

## 4. Uso via Linha de Comando (CLI)

### Opção A: Ingestão Técnica e Extração de Frames (`analyze`)
Inspeciona metadados com `ffprobe` e extrai frames representativos com timestamps rastreáveis:

```bash
# Extração por intervalo (ex: a cada 2.0s, máx 50 frames)
.\.venv\Scripts\python.exe -m ai_video_editor.cli analyze video1.mp4 video2.mp4 `
  --strategy interval --interval 2.0 --max-frames 50 --output-dir output/analysis

# Extração por contagem fixa (ex: exatamente 5 frames equidistantes)
.\.venv\Scripts\python.exe -m ai_video_editor.cli analyze video1.mp4 `
  --strategy fixed_count --count 5 --output-dir output/analysis --json
```

### Opção B: Análise Multimodal com IA e Relatório Markdown (`report-ai`)

> **Legado.** `report-ai` (e o módulo `intelligence/`) continua funcionando, mas será substituído pelo comando `visual` (opção J), que analisa tomadas de projetos com clipes pela camada `llm/` com imagens.

Interpreta os frames extraídos com IA multimodal e gera um relatório Markdown (`ai_report.md`) com linha do tempo visual e dados estruturados (`ai_analysis.json`). Cada execução também grava `ai_run_metrics.json` (provedor, modelo, frames enviados, latência, tokens quando o provedor informa, `success` e `error_type`), inclusive em caso de falha, e o relatório inclui a seção "Métricas da execução":

#### 1. Configuração Segura da API Key (Google Gemini)
Defina a chave via variável de ambiente (nunca passada em argumentos de linha de comando ou versionada no Git):
```powershell
$env:GEMINI_API_KEY = "sua_chave_do_google_ai_studio"
# Opcional, para o Diretor/revisor na OpenAI (pip install -e ".[openai]"):
$env:OPENAI_API_KEY = "sua_chave_da_openai"
```
As duas também podem ficar num `.env` local (ignorado pelo Git).

#### 2. Executando com o Google Gemini
```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli report-ai `
  --report output/analysis/<id_da_sessao>/analysis_report.json `
  --provider gemini `
  --model gemini-3.8-flash `
  --max-images 15 `
  --proxy-dim 1024
```

> **Modelos Gemini Compatíveis:**
> Os modelos recomendados suportam entrada multimodal e saídas estruturadas em JSON (ex: `gemini-3.8-flash` para custo/velocidade, ou `gemini-3.8-pro` para raciocínio visual aprofundado). Consulte os modelos multimodais atualizados na [documentação oficial do Google AI Studio](https://ai.google.dev/gemini-api/docs/models/gemini).

#### 3. Executando em Modo Simulado / Teste Offline (`mock`)
Permite validar o fluxo E2E e geração de relatórios Markdown sem gastar tokens ou depender de conexão de rede:
```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli report-ai `
  --report output/analysis/<id_da_sessao>/analysis_report.json `
  --provider mock
```

### Opção C: Corte Rápido de Dois Vídeos (`quick-cut`)
Corta trechos especificados de dois vídeos e concatena-os em um MP4 final:

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli quick-cut `
  --video1 clip1.mp4 --start1 1.0 --end1 3.0 `
  --video2 clip2.mp4 --start2 0.5 --end2 2.5 `
  --output output/merged_output.mp4
```

### Opção D: Renderização a partir de JSON (`render`): `EditPlan` ou `Timeline`
Renderiza um plano de edição completo a partir de um arquivo JSON estruturado. O formato é detectado pelo conteúdo: com `video_main` ou `schema_version` é uma **Timeline** (várias trilhas); senão, um **EditPlan** (uma trilha só, caminho de sempre).

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli render output/sample_plan.json
.\.venv\Scripts\python.exe -m ai_video_editor.cli render output/vlog_timeline.json
```

#### Formato JSON da Timeline (`core/timeline.py`)

```json
{
  "schema_version": 1,
  "title": "Vlog na praia",
  "output_path": "output/vlog.mp4",
  "resolution": [1920, 1080],
  "fps": 30,
  "audio_fade_sec": 0.02,
  "loudness": {"preset": "voz", "integrated_lufs": -16, "true_peak_dbtp": -1.5, "lra": 11},
  "video_main": [
    {"source_path": "cam/A001.mp4", "source_in": 12.0, "source_out": 20.5, "clip_id": "A001", "label": "abertura"},
    {"source_path": "cam/A002.mp4", "source_in": 3.0, "source_out": 9.0, "clip_id": "A002"},
    {"source_path": "pov/P001.mp4", "source_in": 0.0, "source_out": 4.0, "clip_id": "P001", "use_audio": false}
  ],
  "video_overlay": [
    {"source_path": "drone/D003.mp4", "source_in": 30.0, "source_out": 33.0, "timeline_start": 5.0, "clip_id": "D003"}
  ],
  "audio_ambient": [
    {"source_path": "pov/P001.mp4", "source_in": 0.0, "source_out": 4.0, "timeline_start": 14.5,
     "volume": 0.6, "fade_in_sec": 0.3, "fade_out_sec": 0.5, "clip_id": "P001"}
  ]
}
```

- **`video_main`** (obrigatória, ≥ 1 item): base contígua; a duração da timeline é a soma dos itens. Cada item usa o próprio áudio (trilha de áudio principal), com `volume` (0–2); `use_audio: false` põe silêncio no lugar. `timeline_start` é opcional na base: se vier, precisa bater com o fim do item anterior (senão: "buraco na base" / "sobreposição na base").
- **`video_overlay`**: substitui a **imagem** da base em `[timeline_start, timeline_start + duração)`; o áudio da base continua. Overlays não podem se sobrepor nem passar do fim da timeline; podem atravessar vários itens da base e ser adjacentes.
- **`audio_ambient`**: áudio posicionado no tempo da timeline e mixado por cima do principal, com `volume` (0–2) e `fade_in_sec`/`fade_out_sec` próprios. Não pode passar do fim da timeline.
- Todo item tem `source_path`, `source_in`, `source_out` (segundos da origem), `clip_id` e `label` opcionais (para relatórios).
- Padrões: `resolution` 1920x1080 (largura e altura pares), `fps` 30, `audio_fade_sec` 0,02, `loudness` nulo (sem normalização). Campos desconhecidos são recusados.
- **Quantização:** todos os tempos da timeline viram frames do fps de saída (`round(segundos × fps)`, a mesma regra do EditPlan); item com menos de 1 frame é recusado. As posições de áudio (48 kHz) são derivadas das posições em frames, então áudio e vídeo não escorregam.
- Um EditPlan vira Timeline com `timeline_from_edit_plan(plan, source_durations)` (segmentos → `video_main` com o áudio de cada item).

#### Como a Timeline é renderizada
1. **Achatamento** (`flatten`, função pura): a base é recortada onde há overlay, gerando uma lista de pedaços de vídeo em sequência (fonte, in, out, frames), mais o plano de áudio (um pedaço por item da base + itens de ambiente com posição absoluta em amostras).
2. **Pedaços de vídeo em paralelo** (um FFmpeg cada): vídeo sem áudio, padronizado (rotação de metadados aplicada pelo autorotate do FFmpeg, escala preservando a proporção + letterbox/pillarbox, fps de saída, `setsar=1`, `yuv420p` 8 bits).
3. **Áudio inteiro num só FFmpeg** (roda junto com os pedaços de vídeo; o número de processos não cresce com o número de itens): os itens da base de cada origem são agrupados em sequências com tempos crescentes e sem sobreposição (numa espinha, todos os trechos de um clipe); cada sequência entra uma vez (`-ss/-t` só no intervalo usado) e `asegment` a divide com precisão de amostra (cada amostra vai para um único ramo; os vãos vão para `anullsink`). Cada item vira PCM 48 kHz estéreo com a contagem exata de amostras (derivada dos frames) e os micro fades nos cortes (silêncio gerado se a origem não tem áudio ou o item está com `use_audio: false`); `concat` junta tudo na ordem e o ambiente é posicionado (`adelay` em amostras) e mixado com `amix` sem normalizar volumes, no mesmo grafo. Uma origem cujos itens estão muito fora de ordem (mais de 4 sequências) entra uma vez só e cada item recorta sua cópia (`asplit` + `atrim`). O grafo vai num arquivo (`-/filter_complex`), porque com centenas de itens passaria do limite da linha de comando do Windows. Medido: 300 itens = 1 processo e 3,3 s (antes, 301 processos e 15,4 s).
4. **Mux** do vídeo concatenado (`-c:v copy`) com o áudio (AAC 192k, 48 kHz), com loudnorm em duas passadas se `loudness` estiver definido. A saída é validada com ffprobe: duração da trilha de vídeo igual à da timeline (tolerância de 1 frame) e fps pedido.
- Fontes HDR (PQ/HLG) são convertidas para 8 bits **sem tonemapping** e geram um aviso (cores podem ficar lavadas). Fontes de 10 bits SDR são convertidas sem aviso.
- Antes do render, cada origem é inspecionada uma vez: item que passa do fim da origem (mais de 1 frame), overlay/base sem trilha de vídeo ou ambiente sem trilha de áudio são recusados com mensagem que identifica o item (ex.: `video_overlay[0] (D003)`).

### Opção E: Remover as pausas de um vídeo falado (`edit`)
Fluxo completo: metadados → silêncios → transcrição (opcional) → Diretor de IA (opcional) → plano de corte → render → legendas (com transcrição) → relatório. Sem `--director`, é só corte de silêncio, sem IA. As etapas vêm da **receita** `fala` (o padrão para um vídeo); veja "Receitas" abaixo. Com uma **pasta** no lugar do vídeo, o `edit` monta a espinha falada de um projeto com vários clipes (receita `vlog`; ver Opção K). A resolução e o FPS da saída são os do vídeo de origem (FPS arredondado para inteiro).

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4"

# Ajustando a detecção e as folgas, e gerando a transcrição (requer pip install -e ".[transcription]")
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4" `
  --threshold-db -40 --min-silence-sec 0.6 `
  --padding-before 0.15 --padding-after 0.25 --min-gap 0.3 --min-keep 0.1 `
  --transcribe --language pt --whisper-model small   # o padrão é auto: large-v3-turbo na GPU, small na CPU

# Com o Diretor de IA removendo frases repetidas, recomeçadas e enrolação (requer GEMINI_API_KEY)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4" `
  --transcribe --language pt --director-provider gemini --intensity light

# Diretor na OpenAI (pip install -e ".[openai]" e OPENAI_API_KEY), com o Gemini de reserva
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4" `
  --transcribe --language pt --director-provider openai --director-fallback "gemini:gemini-3.8-flash"
```

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--receita` | `fala` (vídeo) / `vlog` ou `vlog_cobertura` (pasta; ver Opção K) | Receita (combinação validada de etapas) a executar. Nome desconhecido é recusado com a lista das disponíveis; receita de pasta (`multi_clip`) com um vídeo, ou o contrário, é recusada |
| `--threshold-db` | auto | `auto` mede o ruído de cada vídeo e escolhe o limite; um número (ex: `-40`) usa esse limite fixo em dB |
| `--min-silence-sec` | 0.5 | Duração mínima de um silêncio detectado |
| `--padding-before` / `--padding-after` | 0.15 / 0.20 | Folga mantida antes/depois de cada fala |
| `--min-gap` | 0.30 | Pausas menores que isso (após o padding) não são cortadas, inclusive no início/fim |
| `--min-keep` | 0.10 | Falas mais curtas que isso são tratadas como ruído |
| `--transcribe` | desligado | Gera `transcript.json` (necessário para o Diretor) |
| `--director-provider` (ou `--director`) | `none` | Diretor de IA: `none`, `mock` (remove frases repetidas, sem rede), `gemini` (`GEMINI_API_KEY`) ou `openai` (`OPENAI_API_KEY`, extra `.[openai]`). Só roda com `--transcribe` |
| `--director-model` | o do provedor | Modelo principal: `gemini-3.8-flash` no Gemini, `gpt-6-luna` na OpenAI |
| `--director-fallback` | no Gemini: `gemini-3.6-flash`, `gemini-3.5-flash-lite` | Modelos de reserva, em ordem: `"provedor:modelo,..."` (pode misturar provedores; sem `provedor:` usa o do Diretor). `none` desliga. Veja "Troca automática de modelo" abaixo |
| `--director-thinking` | `low` | Raciocínio do Diretor: `off`, `low` ou `high`. Cada provedor traduz (Gemini: `thinking_level` MINIMAL/LOW/HIGH, ou `thinking_budget` na família 2.5; OpenAI: `reasoning.effort` none/low/high; modelos sem raciocínio ignoram e o relatório diz isso). Se a API recusar o nível, sobe sozinho (off → low → high); o nível pedido e o efetivo aparecem no relatório e no `director_decision.json` |
| `--director-timeout` | 120 | Tempo limite (s) da chamada ao Diretor |
| `--api-retries` | 4 | Novas tentativas da API de IA em falhas temporárias (503, 504, 429 por minuto, conexão), com espera de 2s, 4s, 8s, 16s + jitter (respeita `Retry-After`); `0` desliga. Erros permanentes (autenticação, 400, 404) nunca são repetidos. Cada tentativa aparece no terminal (`[4/8] Diretor: Gemini indisponível (503), nova tentativa em 4s (2/5)`), o erro final traz o detalhe original da API e o relatório registra as chamadas à API de cada tentativa. Vale também para `revise` e `report-ai` |
| `--intensity` | `light` | Quanto o Diretor corta: `light`, `medium` ou `aggressive` |
| `--target-duration` | — | Duração final desejada em segundos |
| `--instructions` (ou `--pedido`) | — | Pedido livre ao Diretor (ex: `"corte as piadas"`) |
| `--keep-fillers` | desligado | Impede o Diretor de remover frases só por serem enrolação |
| `--no-subtitles` | desligado | Não gera o `.srt` (por padrão ele é gerado quando há `--transcribe`) |
| `--subtitle-max-chars` | 42 | Máximo de caracteres por linha da legenda |
| `--transcript-min-pause` | 1.0 | Pausas removidas a partir desta duração (s) ganham marcador na transcrição comparativa |
| `--loudness` | `voz` | Normalização de volume do vídeo final (EBU R128, loudnorm em duas passadas): `voz` (-16 LUFS), `youtube` (-14 LUFS), `broadcast` (-23 LUFS), todos com true peak -1,5 dBTP e LRA 11; `off` mantém o volume original. Veja "Normalização de volume" abaixo |
| `--force` | desligado | Refaz todas as etapas, ignorando o cache |
| `--no-live` | desligado | Sem linha de andamento ao vivo: só as linhas finais de cada etapa (automático quando a saída não é um terminal, ex: redirecionada para arquivo). Vale também para `revise` |
| `--projects-dir` | `output/projects` | Diretório base dos projetos |

Cada vídeo ganha uma pasta `output/projects/<nome-do-video>-<hash>/` com:

* `project.json`: manifesto com caminho do vídeo, parâmetros e status/fingerprint/métricas de cada etapa.
* `probe.json`, `audio_analysis.json`, `transcript.json` (se pedido), `director_decision.json` (se pedido), `edit_plan.json`.
* `edit_report.md`: duração original e final, % removido, número de cortes, parâmetros, a lista dos trechos removidos e, com o Diretor, a seção "Decisões do Diretor" (tempo, texto, categoria, motivo e **aplicação** de cada frase removida, latência e tokens). Com o Diretor, o relatório abre com `Diretor: N remoções pedidas, X aplicadas, Y parciais, Z não aplicadas`. A seção "Volume" traz o loudness medido do áudio editado (I, TP, LRA), o alvo aplicado e o tempo da passada de medição.
* **Status de aplicação das remoções:** medido no plano final pela interseção da frase com o que saiu do vídeo. `aplicada` = saiu ≥ 90%; `parcial` = saiu uma parte (segundos e %); `não aplicada` = a frase continua no vídeo, com o motivo (ex.: `protegida pela folga das frases vizinhas`, quando o padding das frases mantidas coladas anula o corte).
* `edit_transcript.md` e `edit_diff.json` (só com transcrição; sem ela, aparecem como pulados no relatório): **transcrição comparativa** com a fala original inteira em ordem cronológica. Frase mantida: `` `[tempo no original → tempo no vídeo editado]` texto``; removida: ~~riscada~~ com tempo no original, quem removeu, categoria e motivo; ◐ parcial (quanto saiu); ⚠️ não aplicada (não riscada, com o motivo); ⏸ pausas removidas de 1s ou mais (`--transcript-min-pause`; as menores só entram no total do cabeçalho); em revisões, ✅ restaurada e 🆕 removida nesta revisão, com o pedido. O tempo no editado usa o mesmo remapeamento das legendas, então bate com o `.srt` e o player. O `edit_diff.json` traz o mesmo conteúdo validado pelo modelo `EditDiff`. Os dois arquivos são copiados para `versions/vN/`.
* `edit_timeline.svg` (só com transcrição): **linha do tempo dos cortes**, desenhada a partir do `edit_diff.json` (SVG escrito pelo próprio código, com fundo próprio). Uma barra representa o vídeo original, com cores para fala mantida, pausas removidas, frases removidas pelo Diretor/revisão, remoções parciais ou não aplicadas e trechos sem fala; marcas de tempo, legenda e título com a duração original → final e o % removido. Vídeos de mais de 10 minutos quebram em uma linha por 10 minutos. Cada item do diff é um grupo `<g class="item ...">` com o texto e o motivo como dica (`<title>`). Copiada para `versions/vN/` como os outros dois.
* **Tempo de processamento**: cada etapa mostra o tempo de relógio no terminal (`[3/8] Transcrição: ok — 49 frases (12min 34s)`; etapas em cache aparecem como `em cache`, sem tempo) e, ao final, `57min de vídeo processados em 21min (0,37× o tempo real)`. Os tempos ficam em `last_run` no `project.json`, em `timings` no `version.json` e na seção "Tempo de processamento" do `edit_report.md` (o `revise` também mede suas etapas; probe, silêncios e transcrição contam como cache).
* **Andamento ao vivo** (só em terminal): enquanto uma etapa roda, a mesma linha é reescrita a cada segundo com o tempo decorrido e, quando há como medir, o progresso e o tempo restante: `[3/8] Transcrição... 2min 13s — 41% (23min de 57min), faltam ~3min` (fim do último trecho transcrito / duração do áudio) e `[6/8] Renderização... 48,0s — 118/258 trechos (45%), faltam ~58s`. No Diretor (e no revisor), a linha mostra o modelo atual e a tentativa (`gemini:gemini-3.8-flash, tentativa 2/5`). A estimativa só aparece a partir de 10% concluídos. Avisos (nova tentativa, troca de modelo, GPU indisponível) saem em linha própria acima e não são apagados; ao fim, a linha vira a linha final da etapa. Funciona no cmd e no PowerShell (`\r` + espaços, sem códigos ANSI); em erro ou Ctrl+C a linha é fechada como `FALHOU`/`interrompida`.
* `<nome-do-video>_edited.mp4`: o vídeo final.
* `<nome-do-video>_edited.srt`: legenda sincronizada com o vídeo editado (só com `--transcribe`), ao lado do vídeo e com o mesmo nome base, então players como o VLC a carregam sozinhos. Usa as palavras da transcrição, os mesmos trechos e o FPS do plano final (já com os cortes do Diretor). Sem transcrição, ou com `--no-subtitles`, a etapa "Legendas" aparece como pulada e um `.srt` antigo é removido da pasta. O relatório cita o arquivo e o número de blocos.

**Diretor de IA:** a transcrição é agrupada em frases (segmentos unidos até `.`, `?`, `!` ou uma pausa maior que 1s; se passar de 12s ou 40 palavras, a frase fecha no melhor ponto: fim de segmento com vírgula/`;`/`:`, depois a maior pausa, depois qualquer fim de segmento, sem nunca dividir um segmento do Whisper. Os limites são `EditPipelineParams.sentences` e entram no fingerprint do Diretor; o relatório mostra quantidade, duração média e máxima das frases). O Diretor recebe as frases numeradas e o briefing e devolve só os IDs das frases a remover, com categoria (`repetition`, `false_start`, `filler`, `off_topic`, `other`; `retake` só em projetos de pasta, ver Opção K) e motivo em português; nunca timestamps. A decisão é validada (IDs existentes e únicos; no máximo 40% do tempo de fala, ou mais se `--target-duration` exigir). Se for inválida, o modelo recebe o erro e tenta mais uma vez; se falhar de novo, o vídeo sai só com o corte de silêncio e o relatório traz um aviso. Os limites de cada corte vão para dentro do silêncio detectado mais próximo (até 0,5s); sem silêncio por perto, usa-se o limite da frase sem invadir o padding das frases mantidas. **Pedaço de palavra:** se o corte ajustado deixar de um lado uma sobra da frase removida menor que 0,25s, o corte é estendido até o limite da frase quando esse limite não cai dentro de uma frase mantida; se cair (frases do Whisper sobrepostas), o corte é desfeito e a frase aparece como `não aplicada` — nunca fica meia palavra no vídeo.

**Limite de silêncio automático:** com `--threshold-db auto` (padrão), o áudio é medido em janelas de 50 ms (FFmpeg `astats`, pico por janela). O **piso de ruído** é o percentil 10 dos níveis e o **nível de fala** o percentil 90; o limite é `piso + 35% da distância até a fala`, nunca acima de `fala - 6 dB` e dentro de `[-60, -20] dB`. Isso faz gravações com ruído de fundo (plateia, sala) detectarem pausas que o `-35 dB` fixo nunca via. Modo, piso, fala e limite escolhido ficam no `audio_analysis.json` e no relatório. Quando o ruído está muito próximo da fala, o áudio não tem pausas ou é quase todo silêncio, o relatório traz um aviso de detecção pouco confiável (nesses casos o limite fica abaixo do ruído e poucos cortes são feitos, ou todo o áudio vira silêncio, respectivamente).

**Troca automática de modelo:** o Diretor (e o revisor) usa uma cadeia ordenada `provedor:modelo`; o padrão é `gemini:gemini-3.8-flash` → `gemini:gemini-3.6-flash` → `gemini:gemini-3.5-flash-lite`. Se o modelo atual continuar falhando depois das novas tentativas com erro temporário (503, 504, 429 por minuto, conexão) ou com a cota diária/de cobrança esgotada, a etapa passa para o próximo da cadeia e avisa no terminal (`[4/8] Diretor: gemini-3.8-flash indisponível (503) após 5 tentativas → trocando para gemini-3.6-flash`). Erros de autenticação ou de requisição inválida (400/404) não trocam de modelo. Provedores sem chave configurada são pulados com um aviso. Depois de uma troca, as chamadas seguintes da mesma execução continuam no modelo que funcionou. O relatório ("Cadeia de modelos", "Modelo usado", "Troca de modelo") e o `director_decision.json` (`model_chain`, `effective_provider`, `effective_model_id`, `model_switches`) registram o pedido, o usado e cada troca com o motivo. O cache do Diretor usa a cadeia **pedida**: uma troca de modelo durante a execução não muda o fingerprint.

**Normalização de volume:** com `--loudness` diferente de `off`, o render faz duas passadas do filtro `loudnorm` sobre o áudio já concatenado: a primeira mede (I, TP, LRA, limiar e offset, lidos do JSON que o FFmpeg imprime) e a segunda aplica o ganho no concat final com os valores medidos e `linear=true` (ganho único, sem "bombear" a fala; o vídeo continua com `-c:v copy` e o AAC é codificado uma única vez). Se o áudio original já chega a 0 dBTP ou mais de true peak, o relatório e o terminal avisam: a gravação provavelmente saiu distorcida, e a normalização só abaixa o volume, não desfaz a distorção. Áudio todo em silêncio (loudness `-inf` ou inválido) ou vídeo sem áudio: a normalização é pulada com aviso, sem falhar. A medição fica nas métricas `loudness_*` da etapa render no `project.json`. O `EditPlan` tem o campo opcional `loudness` (padrão `null` = sem normalização), então `render` e `quick-cut` continuam como antes.

**Receitas:** uma receita (`recipes/`) é um dado: nome, descrição, etapas em ordem, dependências entre elas e `multi_clip` (receita que trabalha com vários clipes de uma pasta; o Produtor e a CLI usam esse campo para casar a receita com o material). As etapas possíveis são as peças do catálogo `PIECES` (`catalog`, `probe`, `silences`, `transcription`, `director`, `plan`, `timeline`, `visual`, `coverage`, `render`, `subtitles`, `report`), e cada peça declara o que sempre exige antes dela (ex.: `director` precisa de `transcription`, `timeline` precisa de `catalog` e `silences`, `coverage` precisa de `visual` e `timeline`); o que depende da receita (ex.: `render` depois de `plan` na `fala` e depois de `timeline` na `vlog`) fica em `requires`. Receitas registradas: `fala` (um vídeo falado), `vlog` (pasta de clipes, `multi_clip=True`) e `vlog_cobertura` (a `vlog` + `visual` e `coverage` entre a Timeline e o render; Opção K). Ao criar uma receita, a validação recusa etapa desconhecida, etapa repetida, dependência faltando e ordem impossível (inclusive circular), com mensagem em português (`Receita 'x': ordem impossível: 'silences' precisa vir depois de 'probe'.`). O `EditPipeline` executa as etapas na ordem da receita, e a numeração `[n/N]` do terminal vem dela. A receita `fala` é exatamente o fluxo acima: mesmos fingerprints, cache e saída (garantido por `tests/test_reference_fala.py`, que compara o `edit` completo com um snapshot em `tests/data/reference_fala/`; para regenerá-lo depois de uma mudança intencional: `UPDATE_REFERENCE=1 python -m pytest -q tests/test_reference_fala.py`, e revisar o diff). A visão de receitas e do Produtor está em [ARQUITETURA.md](ARQUITETURA.md).

**Cache:** cada etapa é pulada se o artefato existe e os parâmetros que a afetam não mudaram. Mudar `--threshold-db` (ou alternar entre `auto` e um número) refaz silêncios, plano, render e relatório; mudar só um padding ou o `--loudness` refaz a partir do plano; sem Diretor, a transcrição nunca força novo render. As legendas dependem do plano, da transcrição e de `--subtitle-max-chars`: mudar só esse limite refaz apenas legendas e relatório, sem novo render. O Diretor é refeito quando mudam a transcrição, o briefing ou o modelo. O hash do vídeo usa o tamanho e blocos do início, meio e fim do arquivo, então é rápido mesmo para arquivos grandes.


### Opção F: Revisar uma edição com pedidos em linguagem natural (`revise` e `versions`)
Depois do `edit`, peça ajustes sem refazer as etapas caras (probe, silêncios e transcrição vêm do cache da pasta do projeto; nada é refeito). Cada edição vira uma **versão numerada**: o resultado do `edit` é a `v1`, e cada `revise` gera a próxima, sem tocar nas anteriores.

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli revise "output\projects\aula-1a2b3c4d5e" "corta menos"
.\.venv\Scripts\python.exe -m ai_video_editor.cli revise "output\projects\aula-1a2b3c4d5e" "tira a parte em que ele fala do site"
.\.venv\Scripts\python.exe -m ai_video_editor.cli revise "output\projects\aula-1a2b3c4d5e" "mantém a frase sobre o livro" --from v1
.\.venv\Scripts\python.exe -m ai_video_editor.cli versions "output\projects\aula-1a2b3c4d5e"
```

| Opção do `revise` | Padrão | Efeito |
| --- | --- | --- |
| `--from vN` | versão atual | Versão em que a revisão se baseia (as frases removidas/mantidas partem dela) |
| `--reviser-provider` (ou `--reviser`) | o do Diretor do projeto (ou `gemini`) | `gemini`, `openai` ou `mock` (determinístico, sem rede) |
| `--reviser-model` / `--reviser-fallback` / `--reviser-thinking` / `--reviser-timeout` | os do Diretor do projeto | Mesma configuração do Diretor: modelo, cadeia de reserva com troca automática, raciocínio (com subida automática de nível se a API recusar) e timeout. Trocando o provedor, o modelo e a reserva passam a ser os padrões dele |
| `--loudness` | o da versão base | Normalização de volume da versão nova: `voz`, `youtube`, `broadcast` ou `off` (fica registrada no `version.json`) |
| `--api-retries` | 4 | Novas tentativas em falhas temporárias da API (`0` desliga); mesma regra do `edit` |
| `--no-live` | desligado | Sem linha de andamento ao vivo (mesma regra do `edit`) |

* O revisor usa as frases gravadas na versão base (`director_decision.json`), não uma remontagem: projetos criados antes do limite de tamanho continuam abrindo e revisando com os mesmos IDs.
* O revisor recebe as frases numeradas com o estado atual na versão base (mantida ou removida, com categoria e motivo), o briefing original, os últimos pedidos e o pedido novo. Devolve o **novo conjunto completo** de frases removidas (pode restaurar frases removidas antes), o motivo de cada restauração, um resumo e, opcionalmente, o ritmo das pausas: `natural` (mais folga), `normal` (o do `edit`) ou `tight` (menos folga). Nunca timestamps.
* A resposta passa pela mesma validação do Diretor (IDs existentes e únicos, limite de remoção, enrolação proibida com `--keep-fillers`) mais a coerência das restaurações, com uma nova tentativa enviando o erro ao modelo. **Se o pedido for impossível, ambíguo, não mudar nada ou a resposta continuar inválida, nenhuma versão é criada** e o comando explica o motivo (código de saída 1). Se o render falhar, nada fica pela metade.
* O comando mostra as frases removidas e restauradas, o ritmo e a duração antes e depois.
* `versions` lista cada versão (`*` = atual) com duração, número de cortes e o pedido que a originou. Projetos criados antes do versionamento são migrados sozinhos: o resultado atual vira a `v1`.

Estrutura (o `project.json` guarda o histórico em `versions` e a versão em `current_version`):

```
<projeto>/
  project.json, probe.json, audio_analysis.json, transcript.json   # compartilhados por todas as versões
  edit_plan.json, <video>_edited.mp4, ...                          # resultado do último `edit` (cache)
  versions/
    v1/  director_decision.json, edit_plan.json, <video>_edited.mp4, <video>_edited.srt, edit_report.md,
         edit_transcript.md, edit_diff.json, edit_timeline.svg, version.json
    v2/  ...                                                       # version.json guarda o pedido, o resumo e as métricas
```

O `edit_report.md` de uma revisão traz a seção "Revisão": pedido, ritmo das pausas, resumo e uma tabela de mudanças (frase, texto, antes, depois, aplicação, motivo), com a contagem `Revisão vN: N remoções pedidas, X aplicadas...` no topo. A transcrição comparativa da versão marca as frases restauradas (✅) e as removidas nesta revisão (🆕, com o pedido). Rodar o `edit` de novo com os mesmos parâmetros não cria versão; com parâmetros que mudam o resultado, cria uma nova versão atual.

### Opção G: Benchmark de transcrição (`benchmark transcription`)

Mede velocidade e precisão (WER, taxa de erro de palavras) de configurações do faster-whisper contra uma transcrição de referência. Cada configuração é `modelo:dispositivo:compute_type`; o modelo é carregado uma vez antes de medir (o tempo de carga sai separado), e uma configuração que falha (ex.: sem GPU) é registrada e as demais seguem.

```bash
# Um áudio, com referência (sem --reference, mede só velocidade e o WER aparece como "—")
python -m ai_video_editor.cli benchmark transcription --audio benchmark/data/mit_1806_lecture01.webm --reference benchmark/data/mit_1806_lecture01_reference.txt --language en --configs small:cpu:int8 small:cuda:float16 large-v3-turbo:cuda:float16

# Vários itens, todas as configurações em todos os itens
python -m ai_video_editor.cli benchmark transcription --manifest benchmark/manifest.json --configs small:cpu:int8,small:cuda:float16
```

Saída em `benchmark/results/<data>-<nome>/`: `results.json` (validado por Pydantic: versão do projeto, CPU/GPU, configurações e medições), `results.md` (uma tabela por item, a tabela de **média geral** quando há mais de um item, as 10 substituições mais frequentes por configuração, as palavras após o fim da referência e as falhas) e `hypotheses/` com o texto gerado por cada configuração (`hypotheses/<config>.txt`, com `:` trocado por `_`; com vários itens, `hypotheses/<item>/<config>.txt`). Os áudios de `benchmark/data/` não são versionados; o texto de referência (`*_reference.txt`) é.

Resultado (aula 1 de Álgebra Linear do MIT, 39min 49s, RTX 2060; `benchmark/results/2026-10-20-mit-1806-lecture01/`, com o texto gerado em `hypotheses/`):

| Configuração | Carga | Tempo | Tempo real | WER bruto | S/D/I (bruto) | WER norm. | S/D/I (norm.) | Após a ref. |
|---|---|---|---|---|---|---|---|---|
| `small:cpu:int8` | 2,5s | 11min 40s | 0,293x | 8,9% | 227/98/116 | 5,6% | 61/98/116 | 14 |
| `small:cuda:float16` | 1,1s | 1min 36s | 0,040x | 9,9% | 216/96/176 | 6,8% | 62/98/178 | 18 |
| `large-v3-turbo:cuda:float16` | 2,9s | 1min 22s | 0,035x | 8,5% | 118/70/235 | 7,4% | 63/70/235 | 19 |

A normalização tira mais de 130 "substituições" do `small` que eram só grafia (one→1, okay→ok, ys→y's...): as substituições caem de ~220 para ~62. Com isso, o `large-v3-turbo` deixa de ser o mais preciso: as inserções (235 contra 116 do `small` na CPU) são palavras que ele escreve e a referência omite (artigos, "uh", repetições), então ele é mais literal, não pior. A cauda após o fim da referência existe (a referência termina em "...to a system of any size and", e a fala segue com "find out because elimination fails, find out when there isn't a solution. Okay, thanks."), mas são só 14–19 palavras e não explicam as inserções. Os tempos foram medidos de novo nesta execução (a carga do turbo caiu de 58s para 2,9s porque o modelo já estava baixado).

**Manifesto** (`--manifest arquivo.json`): objeto com a lista `items` (ou a própria lista). Caminhos relativos valem a partir da pasta do manifesto; `name` é obrigatório e único.

```json
{
  "items": [
    {
      "name": "mit-aula-1",
      "audio": "data/mit_1806_lecture01.webm",
      "reference": "data/mit_1806_lecture01_reference.txt",
      "language": "en"
    },
    { "name": "so-velocidade", "audio": "data/outro.wav" }
  ]
}
```

| Campo | Obrigatório | Significado |
|---|---|---|
| `name` | sim | Nome do item (título da tabela); único no manifesto |
| `audio` | sim | Áudio ou vídeo a transcrever |
| `reference` | não | Texto de referência (linhas iniciadas por `#` são ignoradas); sem ele, o item só mede velocidade |
| `language` | não | Código do idioma (ex.: `en`, `pt`); sem ele, detecção automática |

Como ler as colunas: *Tempo* é a transcrição (inclui decodificar o áudio com FFmpeg, não inclui a carga do modelo); *Tempo real* é tempo ÷ duração do áudio (0,1x = 10 vezes mais rápido que o áudio); S/D/I são substituições, omissões (palavras da referência que faltam) e inserções; *WER bruto* e *WER norm.* trazem cada um os seus S/D/I; *Após a ref.* são as palavras do fim da transcrição sem correspondência na referência (ver abaixo). Na média geral, o tempo real é o tempo total ÷ áudio total e o WER é agregado (erros somados ÷ palavras de referência somadas, só dos itens com referência).

**WER bruto × normalizado.** O *bruto* usa só a normalização básica: minúsculas, apóstrofos unificados, pontuação e hífens viram espaço, espaços colapsados. O *normalizado* aplica, nos dois textos (referência e transcrição), as regras do idioma do item (`benchmark/normalize.py`, sem dependência nova; hoje só `en`, e idioma sem regras mostra "—" no normalizado). Regras do inglês:

* números por extenso viram dígitos: 0–999 e milhares simples ("zero" → 0, "twenty one" → 21, "a hundred" → 100, "two hundred and five" → 205, "two thousand five" → 2005); sequências como "one two three" continuam separadas ("1 2 3"), e dígitos que já estão escritos não mudam;
* "okay" → "ok";
* apóstrofo antes do "s" final some: "x's" → "xs", "y's" → "ys" (outras contrações, como "don't", ficam);
* hífens já viram espaço na normalização básica ("twenty-one" → "twenty one" → 21) e o separador de milhar entre dígitos é removido ("1,000" → "1000").

Para outro idioma (ex.: `pt`), escreva uma função `texto -> palavras` e registre em `NORMALIZERS`.

**Palavras após o fim da referência.** A referência pode terminar antes da fala (a do MIT parece parar no meio de uma frase). Se a transcrição gerada continua além do último trecho que casa com a referência, as palavras a mais no final (pelo menos 5) são contadas à parte em *Após a ref.* e **não** entram nas inserções nem no WER (bruto ou normalizado); o `results.md` mostra por onde a cauda começa, para conferir. Cauda menor que 5 palavras continua sendo inserção comum.

**Recalcular sem transcrever** (`benchmark rescore`): a partir do `results.json` e de `hypotheses/`, recalcula os dois WERs, a cauda e as médias, mantendo os tempos medidos, e regenera `results.json` e `results.md`. Serve para mudar a normalização ou a referência sem repetir os 15 minutos de transcrição. Execuções sem `hypotheses/` (anteriores a esta versão) precisam ser rodadas de novo.

```bash
python -m ai_video_editor.cli benchmark rescore benchmark/results/2026-10-20-mit-1806-lecture01
# --reference outro.txt troca a referência (execuções de um item)
```

**Limitações do WER**: a comparação continua textual. Mesmo normalizado, "gonna" × "going to", hesitações ("uh", "um") que só um dos textos registra, símbolos falados ("x" × "ex") e erros da própria referência contam como erro. Compare configurações entre si com a mesma referência; o valor absoluto tem esse piso de ruído.

Para incluir outro provedor de transcrição (ex.: Gemini), registre uma fábrica com `register_benchmark_provider("nome", fabrica)` em `benchmark/runner.py` e use `nome/modelo:dispositivo:compute_type` nas configurações (o `mock/...` existe para testes).

### Opção H: Catalogar uma pasta de clipes (`catalog`)

Lê uma pasta de clipes **sem modificar nada nela** e gera o catálogo do projeto: papel e ID de cada clipe, dados técnicos, data de gravação, detecção de fala, sinais de cor e miniaturas. É a porta de entrada dos projetos com vários clipes (ver [ARQUITETURA.md](ARQUITETURA.md)).

```bash
# Pasta com subpastas por câmera: Principal/, Drone/, GoPro/, B-Roll/...
.\.venv\Scripts\python.exe -m ai_video_editor.cli catalog "D:\Vlog Viagem"

# Corrigindo o papel de uma pasta e o relógio do drone (3h atrasado)
.\.venv\Scripts\python.exe -m ai_video_editor.cli catalog "D:\Vlog Viagem" `
  --papel "Celular=broll" --ajuste-relogio drone=+3h
```

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--papel PASTA=PAPEL` | — | Papel de uma subpasta de primeiro nível (repetível). Papéis: `principal`, `drone`, `pov`, `broll`, `outro` (aceita os mesmos sinônimos das pastas). Pasta inexistente vira aviso |
| `--ajuste-relogio PAPEL=AJUSTE` | — | Soma um ajuste às datas de um papel antes de ordenar (repetível): `+3h`, `-1h30m`, `+45s`. Fica registrado no catálogo |
| `--force` | desligado | Refaz a análise de todos os clipes, ignorando o cache |
| `--projects-dir` | `output/projects` | Diretório base dos projetos |
| `--no-live` | desligado | Sem linha de andamento ao vivo |

* **Projeto:** `<projects-dir>/<nome-da-pasta>-<hash>/catalog/`. O hash vem da lista de vídeos (caminho relativo, tamanho e data de modificação); o conteúdo dos vídeos não é lido para isso. Acrescentar, remover ou alterar um vídeo cria outro projeto.
* **Arquivos:** só extensões de vídeo (`.mp4`, `.mov`, `.mkv`, `.mts`, `.mxf`, `.insv`...), em qualquer profundidade; o resto (inclusive `.LRV`/`.THM` das câmeras, ocultos e `._*` do macOS) é ignorado e contado.
* **Papel** pela subpasta de primeiro nível, sem diferenciar maiúsculas, acentos, hífens e plural: `principal`/`main`/`a-roll`/`a-cam` → `principal`; `drone` → `drone`; `pov`/`gopro` → `pov`; `broll`/`b-roll`/`b-cam` → `broll`; o resto → `outro`. Arquivos na raiz da pasta → `principal`. A origem fica em `roles[].source`:
  * `padrão`: o nome inteiro é um sinônimo (`Drone`, `B-Roll`, `GoPros`);
  * `contém`: um sinônimo aparece como **palavra** do nome, ou como duas palavras vizinhas juntas: `Osmo Pocket - PRINCIPAL` → principal, `Fx30 - BROLL` → broll, `Go3S - POV` → pov, `Imagens de Drone` → drone, `Fx30 - B Roll` → broll. Pedaço de palavra não conta (`Dronezinho`, `Povo` → outro); `outro` só vale pelo nome exato;
  * nome que cita **dois papéis diferentes** (`Drone e POV`) → `outro`, com aviso pedindo `--papel`;
  * `--papel` e `raiz`.
* **IDs** por papel, na ordem de gravação: `P001` (principal), `D001` (drone), `V001` (pov), `B001` (broll), `X001` (outro). Desempate pela **ordem natural** do nome do arquivo (`(4)` < `(15)` < `(27)`). Arquivos com problema não ganham ID.
* **Datas de exportação:** se, numa mesma subpasta (uma câmera), um clipe "começa" antes de o anterior terminar (data + duração, com 1s de folga), as datas parecem de exportação, não de gravação (ex.: arquivos reexportados, em que `creation_time` é a hora da exportação). As datas do papel inteiro viram não confiáveis (`recorded_at.reliable = false`), o papel é ordenado pela **ordem natural do nome do arquivo** (`order_source = "nome"`) e sai um aviso. Câmeras diferentes no mesmo papel podem gravar ao mesmo tempo, por isso a comparação é por subpasta. Datas não confiáveis ficam fora da verificação de relógio suspeito.
* **Dados técnicos** (ffprobe, em paralelo): duração, resolução (já girada pela rotação, via `probe_rotation`), fps, codec, bit depth e `pix_fmt`, rotação, áudio (codec), bitrate e tamanho.
* **Data de gravação**, com a origem em `recorded_at.source`: tags do arquivo (`com.apple.quicktime.creationdate` → `creation_time` do container → da trilha de vídeo; relógios zerados como 1904/1970 são ignorados) → padrão no nome (`DJI_20240115123045`, `VID_20240115_123045`, `PXL_...`, `IMG_...`, `2024-01-15 12-30-45`) → data de modificação do arquivo (`mtime`, marcada como pouco confiável, porque muda ao copiar). Tudo é guardado em UTC: tags sem fuso são UTC; datas do nome são lidas no fuso desta máquina.
* **Fala** (clipes com áudio): VAD Silero do faster-whisper (extra `.[transcription]`, import tardio) sobre o áudio decodificado pelo FFmpeg a 16 kHz mono, lido em blocos de 60s. Resultado: proporção da duração com fala, segundos e trechos. Sem faster-whisper: `desconhecido` e um aviso (na próxima execução com ele instalado, só a fala é refeita). Sem áudio: `sem_audio`.
* **Sinais de cor** (sem IA, rotulados como suspeita): `signalstats` nos três frames das miniaturas: faixa entre os percentis 10 e 90 da luma (`luma_spread`) e saturação média (`saturation`), normalizadas 0–1. **Suspeita de perfil log** só quando os dois estão baixos ao mesmo tempo: `luma_spread < LOG_MAX_LUMA_SPREAD` (0,30) **e** `saturation < LOG_MAX_SATURATION` (0,04), constantes em `catalog/schema.py`. `log_suspect` é calculado dos valores (mudar os limites vale para o cache). O `catalog.md` mostra os valores medidos em cada clipe (`luma 0.42 · sat 0.061`) para calibrar os limites com material real; contraste ou saturação baixos sozinhos não geram aviso.
* **Miniaturas:** 3 por clipe (10%, 50% e 90% da duração), 320px de largura, em `catalog/thumbs/<ID>_<pct>.jpg`; e uma **folha de contato por papel** (`contact_sheet_<papel>.jpg`, Pillow): a miniatura do meio de cada clipe, em ordem, com `ID · duração` escrito.
* **Cache por clipe** em `<projects-dir>/_cache/clips/<chave>/` (análise + miniaturas), com a chave = caminho absoluto + tamanho + data de modificação. Fica fora da pasta do projeto: acrescentar um clipe muda o hash do projeto, mas os outros clipes não são reprocessados. Papéis, IDs e ajustes de relógio não entram no cache (são recalculados a cada execução, em milissegundos).
* **Arquivo com problema** (corrompido, sem trilha de vídeo) vira aviso e uma linha em "Arquivos com problema", nunca erro fatal.
* **Saídas** em `catalog/`: `catalog.json` (`ProjectCatalog` com `schema_version` 2, papéis, ajustes, clipes `CatalogClip`, avisos, tempos), `catalog.md` (resumo por papel: quantidade, duração, resoluções/fps/codecs, intervalo de datas e proporção com fala; avisos; tabela por clipe; arquivos com problema), as folhas de contato e `thumbs/`.
* **Avisos:** arquivos com problema; relógio suspeito (data no futuro, ou um papel sem nenhuma data no intervalo do principal ±30 min, com a sugestão de `--ajuste-relogio`); datas de exportação (ordem pelo nome); datas vindas de `mtime`; formatos misturados num papel (resoluções, fps, codecs) e fps diferentes no projeto; clipes sem áudio; suspeita de perfil log.
* **Terminal:** `[1/4] Varredura`, `[2/4] Metadados e miniaturas` e `[3/4] Detecção de fala` (com `12/40 clipes (30%)` ao vivo) e `[4/4] Catálogo`; no fim, o resumo por papel, os avisos e o tempo total.

### Opção I: Briefing do Produtor (`briefing`)

O **Produtor** lê o pedido e um resumo do material e escreve o **briefing** da edição: tipo de vídeo, receita (uma das registradas), "Entendi: ...", metas (duração, intensidade, ritmo), instruções para o Diretor de cortes e para a cobertura, papéis a usar e avisos. Não edita nada; hoje o `edit` ainda não lê o briefing (integração pendente).

```bash
# Pasta (cataloga se faltar o catálogo ou se a pasta mudou; reaproveita o cache por clipe)
.\.venv\Scripts\python.exe -m ai_video_editor.cli briefing "D:\Vlog Viagem" --pedido "vlog de 8 minutos, abre com drone, ritmo rápido"

# Um vídeo, sem pedido (1 vídeo falado: nem chama a IA)
.\.venv\Scripts\python.exe -m ai_video_editor.cli briefing aula.mp4

# Forçando a receita: a IA só preenche o resto
.\.venv\Scripts\python.exe -m ai_video_editor.cli briefing aula.mp4 --receita fala --pedido "corta as piadas"
```

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--pedido` | vazio | Pedido livre do usuário |
| `--receita NOME` | — | Força a receita (precisa estar registrada); se ela não combinar com o material, vira aviso |
| `--director-provider` | `gemini` | Provedor do Produtor (mesmas opções e chaves do Diretor). `mock` = simulado; `none` = só a regra sem IA |
| `--director-model`, `--director-fallback`, `--director-thinking`, `--director-timeout`, `--api-retries` | como no `edit` | Modelo, cadeia de reserva, raciocínio, tempo limite e novas tentativas |
| `--projects-dir` | `output/projects` | Diretório base dos projetos |
| `--no-live` | desligado | Sem linha de andamento ao vivo (quando o catálogo precisa rodar) |

* **Entrada da IA:** o pedido, um **resumo compacto do material** (por papel: quantidade, duração, % do tempo com fala e clipes com fala, formatos, suspeitas de log; nunca a lista de clipes nem caminhos) e o nome, a descrição e as características das **receitas registradas**, lidas do registro na hora (receitas novas aparecem sozinhas).
* **Características da receita:** `multi_clip` (edita vários clipes) vem do atributo `multi_clip` da receita, se ela declarar (sem ele = receita de 1 vídeo, como a `fala`); `needs_speech` vem do atributo `needs_speech`, ou, sem ele, da etapa `transcription`.
* **Validação pelo código** (além do Pydantic): a receita existe; combina com o material (receita de 1 vídeo com vários clipes com fala → inválido; receita que precisa de fala sem nenhum clipe com fala → inválido; fala desconhecida conta como possível fala); papéis citados existem no material; duração alvo maior que zero e menor que a duração dos papéis a usar; tipo e "Entendi" preenchidos. Ajustes sem decisão: prefixo `Entendi: `, papéis vazios = todos.
* **Nova tentativa e fallback:** resposta inválida → nova chamada com o erro; inválida de novo, ou falha de comunicação (sem chave, cota, timeout) → **fallback determinístico**: `--receita`, se houver; 1 clipe → `fala`; pasta com papel principal com fala → `vlog` se estiver registrada, senão `fala` com aviso; sem fala no principal → `fala` com aviso. O pedido vai inteiro em `director_instructions`. `origin` (`ia`/`fallback`) e `origin_reason` registram de onde veio.
* **Atalho:** pedido vazio + 1 vídeo falado (ou com fala não analisada) → fallback direto, sem chamar a IA (registrado em `origin_reason`).
* **Material:** pasta → catálogo atual (mesmo hash da listagem e `schema_version`) ou um novo (`catalog` com cache por clipe); arquivo → resumo mínimo (ffprobe + detecção de fala, `catalog.builder.describe_clip`, sem miniaturas).
* **Estimativa** (sem preços): clipes nos papéis a usar, áudio a transcrever (duração dos clipes com fala ou fala desconhecida nesses papéis, se a receita transcreve), chamadas de IA previstas na edição (etapas de IA da receita, hoje `director`; até o dobro com novas tentativas) e chamadas já feitas pelo Produtor.
* **Saída:** `briefing.json` (`Briefing`, `schema_version` 1: o conteúdo + pedido, receita forçada, origem, resumo do material, estimativa, cadeia de modelos, modelo efetivo, trocas e tentativas) na pasta do projeto: a do catálogo (pasta) ou a que o `edit` usaria (arquivo: `<nome>-<hash do arquivo>`). No terminal: "Entendi: ...", receita e origem, tipo, metas, papéis, instruções, estimativa e avisos.

### Opção J: Análise visual das tomadas (`visual`)

Divide cada clipe do projeto em **tomadas**, mede a imagem sem IA (FFmpeg) e pede à IA uma descrição de cada tomada a partir de 3 miniaturas. É o material dos futuros Diretores visuais (cobertura com B-roll/drone/POV). A IA vê só IDs (`P007.T03`, `P007.T03.F2`), nunca tempos; o código guarda os tempos e valida as respostas.

```bash
# 1. Quanto vai custar? Gera tomadas e medidas (sem IA) e mostra tomadas, imagens, chamadas e tokens
.\.venv\Scripts\python.exe -m ai_video_editor.cli visual "D:\Vlog Viagem" --so-estimar

# 2. Análise com o Gemini (GEMINI_API_KEY), só principal e b-roll, miniaturas em baixa resolução
.\.venv\Scripts\python.exe -m ai_video_editor.cli visual "D:\Vlog Viagem" --papeis principal,broll --detalhe-imagem low

# 3. Material grande: 4 lotes ao mesmo tempo (o padrão é 3; 1 = um por vez)
.\.venv\Scripts\python.exe -m ai_video_editor.cli visual "D:\Vlog Viagem" --detalhe-imagem low --visual-paralelo 4
```

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--papeis LISTA` | todos | Papéis analisados, separados por vírgula (aceita os sinônimos das pastas, ex.: `b-roll`) |
| `--so-estimar` | desligado | Para depois das tomadas e medidas e mostra tomadas a analisar, imagens, chamadas e tokens estimados; não chama a IA e não grava `visual/` |
| `--visual-provider` | `gemini` | `gemini`, `openai`, `mock` (respostas simuladas, sem rede) ou `none` (só tomadas e medidas) |
| `--visual-model` / `--visual-fallback` | os do provedor | Modelo e cadeia de reserva, como no Diretor (`provedor:modelo,...`; `none` desliga). Padrão no Gemini: `gemini-3.8-flash` → `gemini-3.6-flash` → `gemini-3.5-flash-lite` |
| `--visual-thinking` | `low` | Raciocínio `off`/`low`/`high` (cada provedor traduz) |
| `--visual-timeout` | `180` | Tempo limite de cada chamada, em segundos |
| `--detalhe-imagem` | `auto` | Resolução com que a IA lê as miniaturas: `low` (Gemini 3: 280 tokens/imagem; OpenAI `detail: low`), `auto` (padrão do provedor; Gemini 3: 1120) ou `high` |
| `--visual-paralelo N` | `3` | Lotes enviados à IA ao mesmo tempo (`1` = um por vez, como antes). Se um lote falhar por cota esgotada ou limite de requisições (429), cai pela metade (mínimo 1) para os lotes seguintes, com um aviso |
| `--api-retries N` | `4` | Novas tentativas em falhas temporárias |
| `--force` | desligado | Refaz tomadas e medidas (as análises da IA em cache continuam valendo se as tomadas não mudarem) |
| `--projects-dir` / `--no-live` | `output/projects` / desligado | Como no `catalog` |

* **Pasta:** a pasta de clipes (a mesma do `catalog`) ou a pasta do projeto. Se o catálogo não existir, ele é gerado pela API do `catalog/` com as opções padrão (para `--papel`/`--ajuste-relogio`, rode o `catalog` antes).
* **Tomadas** (`visual/shots.py`, uma passada do FFmpeg por clipe, em `ANALYSIS_WIDTH` = 480 px): mudança de cena pelo `scene` score do filtro `select` ≥ `SCENE_THRESHOLD` (0,30), ignorando cortes a menos de `MIN_SHOT_SEC` (1 s) do anterior ou do fim. Tomadas mais longas que `MAX_SHOT_SEC` (8 s) viram janelas iguais, para achar o trecho ruim dentro de um clipe longo. `boundary`: `inicio`, `cena` ou `janela`. IDs `<clipe>.T01`, `T02`...
* **Miniaturas:** 3 por tomada (início+10%, meio, fim−10%), 512 px de largura, JPEG `-q:v 5`, IDs `<tomada>.F1..F3`, em `visual/frames/<ID>.jpg`. O tempo de cada uma fica só no `visual.json`.
* **Medidas** (na mesma passada, a `MEASURE_FPS` = 2 quadros/s; são dicas): desfoque = mediana do `blurdetect` (≥ 9 = `borrão alto`; quadro sem bordas não tem valor; pega foco errado, **não** borrão de movimento, que fica a cargo da IA), brilho = luma média (< 0,15 escura, > 0,88 estourada), movimento = mediana do `YDIF` do `signalstats` entre amostras (a amostra que atravessa um corte é ignorada; `movimento alto` = acima do **percentil 90 das tomadas do projeto** analisadas na execução, `MOTION_PERCENTILE`, e nunca abaixo do piso `MOTION_FLOOR` = 0,08; o limite usado vai em `visual.json` `motion_threshold` e no `visual.md`) e **tela uniforme** = 60% das amostras com faixa de luma (YHIGH − YLOW) < 0,14 (parede, câmera tapada; a parede real do material de teste fica entre 0,09 e 0,12). Fala por tomada: sobreposição com os trechos de fala do catálogo (≥ 1 s ou 30% da tomada).
* **IA em lotes** (`visual/describer.py`): até 8 tomadas / 24 miniaturas por chamada, cada imagem precedida do seu ID. Resposta por tomada: `description`, `tags`, `face_to_camera`, `people_count`, `issues` (`obstruida`, `desfocada`, `tremida`, `escura`, `estourada`, `borrao_movimento`, `nenhuma`), `usable` (`sim`/`nao`/`parcial`), `best_frames` (IDs F1..F3 da própria tomada, em ordem), `beauty` (1–5) e `summary_for_editor`. Só as medidas marcadas vão no prompt (linha `medidas:`, omitida se nada foi marcado; nunca valores brutos), como dica; a IA pode discordar.
* **Validação:** cada tomada do lote respondida exatamente uma vez, IDs existentes, `best_frames` da própria tomada, em ordem e não vazio se a tomada é utilizável. As tomadas recusadas voltam num novo pedido com o erro (até 3 tentativas por lote); persistindo, ficam `sem_analise` sem derrubar o resto. Erro da API (fora autenticação) marca o lote como `sem_analise`; depois de 2 lotes seguidos com erro (pela ordem de término), os restantes não são enviados. Sem chave = erro na hora.
* **Lotes em paralelo** (`--visual-paralelo`, `VisualBuilder(ai_parallel=...)`; a classe usa 1 se não for dito): um pool de threads chama `describe_batch` para até N lotes ao mesmo tempo; validação e novas tentativas por tomada são as mesmas. O resultado não depende da ordem de término (cada tomada recebe a sua análise; contagens são somas; `batch_stats` sai na ordem dos lotes; modelo efetivo = o do último lote na ordem). Cada lote, ao terminar, grava `ai-<fingerprint>.json` dos clipes que tocou, sob trava por clipe (dois lotes do mesmo clipe nunca escrevem juntos) e por arquivo temporário + troca; no Ctrl+C, os lotes da fila não começam e os em andamento terminam e gravam. Limite do provedor (cota esgotada ou 429 que persistiu): paralelismo pela metade, mínimo 1, aviso uma vez. A cadeia de modelos é compartilhada entre as threads: troca uma vez só sob trava, e as métricas da última chamada (`last_metrics`, tokens, chamadas, trocas) são por thread (`llm/per_thread.py`). Progresso ao vivo: `lotes 7/25 · 3 em andamento · <modelo>`.
* **Medição por lote** (`VisualAIRun.batch_stats`, `VisualBatchStat`): espera desde o início da etapa até o lote começar (`started_sec`), duração do envio à resposta final (`duration_sec`), chamadas (`api_calls`, com novas tentativas e correções), correções por resposta recusada, trocas de modelo, modelo, tomadas sem análise e `rate_limited`; mais `parallel`/`parallel_final` e `wall_sec`. Resumo no `visual.md` e no terminal: `Lotes: 25 em 14min 10s (paralelo 3); por lote: média 1min 40s, mín …, máx …; novas tentativas N, trocas de modelo N`. Lote demorado = provedor; lotes rápidos com etapa longa = fila.
* **Cache por clipe** em `<projects-dir>/_cache/clips/visual-<chave>/` (chave = caminho + tamanho + mtime + parâmetros da passada, das tomadas e das miniaturas): `shots.json` (tomadas e **amostras brutas**; as marcas são recalculadas a cada execução, então mudar um limite não refaz o FFmpeg), miniaturas e `ai-<fingerprint>.json` (fingerprint = cadeia pedida + raciocínio + detalhe + versão do prompt). Tomadas analisadas não são pedidas de novo; as `sem_analise`, sim.
* **Estimativa** (`--so-estimar`): chamadas = lotes (sem novas tentativas); tokens de entrada = imagens × `IMAGE_TOKENS` (Gemini 3, documentação "Media resolution": 1120 no padrão, 280 em `low`) + 900 por chamada + 60 por tomada; saída ≈ 150 por tomada. Não há preços: converta pela tabela do provedor.
* **Saídas** em `visual/`: `visual.json` (`ProjectVisual` com `schema_version`, clipes `ClipVisual`, tomadas `Shot` com `ShotFrame`, `ShotMeasures` e `ShotAnalysis`, e `VisualAIRun` com modelo efetivo, lotes, chamadas, imagens, tokens, trocas de modelo, paralelismo e medição por lote), `visual.md` e `frames/`.
* **`visual.md`:** resumo (tomadas por papel, % utilizável, com rosto, com problema), **Atenção** (trechos do principal com fala — ou fala desconhecida — e imagem obstruída ou com borrão de movimento, juntando tomadas seguidas; sem análise, valem as medidas `tela uniforme`/`borrão alto`: candidatos a cobertura obrigatória) e, por clipe, a tabela de tomadas com tempo, links das miniaturas, descrição, tags, problemas (da IA e das medidas), utilizável/melhor trecho e nota de beleza.
* **Terminal:** `[1/4] Catálogo`, `[2/4] Tomadas e medidas` (clipes ao vivo), `[3/4] Descrição por IA` (`lote 3/25 · gemini:gemini-3.8-flash`, novas tentativas e trocas de modelo), `[4/4] Relatório`; no fim, modelo efetivo, lotes, chamadas, imagens, tokens e tempo total.

### Opção K: Espinha falada de uma pasta de clipes (`edit <pasta>`, receitas `vlog` e `vlog_cobertura`)

Com uma pasta no lugar do vídeo, o `edit` junta os clipes de papel `principal` que têm fala, **na ordem do catálogo**, cada um sem pausas e sem frases ruins, num único vídeo renderizado pela Timeline. Com a análise visual feita (`visual <pasta>`), a receita `vlog_cobertura` acrescenta a **cobertura**: drone/b-roll/POV por cima da fala e respiros entre as falas (ver "Cobertura" abaixo). O caminho de um vídeo (`edit <arquivo>`) não muda.

```bash
# Corte de silêncio por clipe (sem IA)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem"

# Espinha completa: transcrição por clipe + Diretor de IA (retake, repetição, enrolação)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" `
  --transcribe --language pt --director gemini --intensity light --pedido "foque na trilha"
```

Etapas (`[n/8]` no terminal): catálogo → silêncios por clipe → transcrição por clipe → Diretor → Timeline → render → legendas → relatório. As opções são as do `edit` (Opção E).

* **Projeto = o do catálogo:** `<projects-dir>/<pasta>-<hash da listagem>/`. Se o `catalog/catalog.json` já existe (de um `catalog` anterior, com `--papel`/`--ajuste-relogio`), é usado como está; se falta, o catálogo é gerado na hora (API pública de `catalog/`, mesmas regras do comando `catalog`). O manifesto `project.json` leva `kind: "multiclip"`.
* **Clipes da espinha** (`spine_clips.json`): papel `principal`, sem erro, com áudio e com fala detectada em pelo menos 20% do tempo **ou** pelo menos 3 s de fala (`MIN_SPEECH_RATIO`/`MIN_SPEECH_SEC` em `pipeline/clips.py`). Fala `desconhecido` (VAD indisponível) entra com aviso. Os principais sem fala e os clipes dos outros papéis aparecem no relatório como **disponíveis para cobertura (rodada 3)**. Ordem = ordem dos IDs do catálogo (data de gravação); nada é reordenado.
* **Silêncios e transcrição por clipe**, reaproveitando `audio/` e `transcription/`, com **cache por clipe fora do projeto**: `<projects-dir>/_cache/edit_clips/<hash do arquivo>/silences-<fp>.json` e `transcript-<fp>.json` (chave: hash do arquivo + parâmetros da etapa). Trocar ou acrescentar um clipe cria outro projeto (a listagem mudou), mas só o clipe novo é processado. A transcrição roda um clipe por vez (a GPU fica com um modelo só), com a barra de progresso somando os clipes e o clipe atual na linha (`P006 (6/23)`). O limite de silêncio `auto` é medido em cada clipe; a detecção de silêncio roda até 4 clipes ao mesmo tempo.
* **Frases com IDs globais:** as frases de cada clipe são montadas como no caminho de um vídeo (limites de 12 s / 40 palavras) e numeradas em sequência atravessando os clipes; cada `Sentence` leva `clip_id` e tempos relativos ao clipe. Uma frase nunca atravessa dois clipes.
* **Diretor:** o mesmo `LLMDirector` (validação, nova tentativa e corte só de silêncio como reserva); o prompt mostra os clipes como seções (`Clipe P006 (0:34) — arquivo.mp4`) com as frases. Categoria nova **`retake`** (tomada repetida: a mesma fala gravada de novo, no mesmo clipe ou em clipes seguidos; mantém-se a última/melhor e as anteriores saem com o motivo). A validação recusa remover **todas** as frases de um clipe, a menos que todas essas remoções sejam `retake` ou `off_topic`. O `--director mock` remove as tomadas anteriores de falas repetidas como `retake`.
* **Timeline** (`timeline.json`, `spine_plan.json`): por clipe, os trechos mantidos saem do mesmo cálculo do caminho de um vídeo (`plan_keep_ranges`: silêncios + frases removidas ajustadas aos silêncios); viram itens de `video_main` com `clip_id` e rótulo `P006 #3`, clipe após clipe. Saída **1920x1080** (letterbox/pillarbox pelo render) e o fps mais comum dos clipes **arredondado para inteiro**, porque a Timeline exige fps inteiro: 29,97 vira 30 (o filtro `fps` do render converte cada clipe; o relatório avisa). Trechos com menos de 1 frame são descartados. Um clipe cujas frases foram todas removidas (retake/off_topic) ou sem fala nos silêncios fica fora, com o motivo no relatório.
* **Render** pela Timeline (`renderer/timeline_*`, áudio num só processo; ver "Como a Timeline é renderizada"). Saída: `<pasta>_edited.mp4`.
* **Legendas** (`<pasta>_edited.srt`): a fala de cada clipe com a regra de um vídeo (palavra vai para o trecho onde está a maior parte dela; bloco nunca atravessa corte), deslocada para onde o clipe começa no vídeo final.
* **Relatórios:** `edit_report.md` com resumo (clipes usados, duração dos clipes → final, cortes, trechos, saída), **resumo por clipe** (original → final, removido, trechos, frases removidas, limite de silêncio, situação), **clipes não usados** com o motivo, parâmetros, volume, Diretor e, por clipe, o que saiu (categoria, motivo, status aplicada/parcial/não aplicada). `edit_transcript.md` com uma seção por clipe (`[tempo no clipe → tempo no vídeo final]`) e `edit_diff.json` (`SpineDiff`: um `EditDiff` por clipe com os tempos editados já no vídeo final). `edit_timeline.svg`: uma trilha do vídeo final com os trechos na cor de cada clipe (fresta branca = corte) e legenda com a duração original → final de cada clipe.
* **Versões:** `versions/v1/` como no `edit` de um vídeo (vídeo, `timeline.json`, relatórios, legenda, decisão do Diretor); `versions <projeto>` lista. **`revise` em projeto de pasta é recusado** ("revisão de projetos com vários clipes chega na rodada 4"); por enquanto, rode `edit <pasta>` de novo com outro `--pedido`/`--intensity` (silêncios e transcrição vêm do cache).

#### Cobertura (receita `vlog_cobertura`)

```bash
# 1. Análise visual da pasta (uma vez; ver Opção J)
.\.venv\Scripts\python.exe -m ai_video_editor.cli visual "D:\Vlog Viagem" --detalhe-imagem low

# 2. Edição com cobertura (escolhida sozinha quando a pasta tem visual.json e há --transcribe)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" `
  --transcribe --language pt --director gemini --pedido "mostre bastante o lago"

# Sem IA na cobertura: só as janelas obrigatórias, escolhidas pelo código
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" --transcribe --director gemini --coverage-provider none

# Espinha sem cobertura, mesmo com visual.json
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" --transcribe --director gemini --receita vlog
```

Etapas (`[n/10]`): catálogo → silêncios → transcrição → Diretor → Timeline → **análise visual** → **cobertura** → render → legendas → relatório. A IA da cobertura só aponta IDs que o código criou (frases `S012`, tomadas `D001.T05`, janelas `O1`); nunca tempos.

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--coverage-provider` (ou `--coverage`) | o do Diretor | `none` (só as janelas obrigatórias, pelo código, sem IA), `mock` (simulado: cobre as janelas e põe um respiro nas primeiras 5 trocas de clipe), `gemini` ou `openai` |
| `--coverage-model` / `--coverage-fallback` | os do Diretor, se o provedor for o mesmo | Modelo e reservas (`provedor:modelo,...`; `none` desliga), como no Diretor |
| `--coverage-thinking` | o do Diretor | Raciocínio `off`/`low`/`high` |
| `--coverage-timeout` | o do Diretor | Tempo limite da chamada, em segundos |

* **Receita padrão:** sem `--receita`, uma pasta usa `vlog_cobertura` quando o projeto já tem `visual/visual.json` **e** há `--transcribe` (a cobertura escolhe frases por ID); senão, `vlog`. O terminal diz qual foi (`Receita: vlog_cobertura (a pasta tem análise visual)`). `--receita vlog` continua dando a espinha sem cobertura, com os mesmos fingerprints e a mesma saída de antes. Com `--receita vlog_cobertura` e sem transcrição, a etapa de cobertura é pulada e o vídeo sai como a espinha.
* **Análise visual (peça `visual`, sem IA):** lê `visual/visual.json` do projeto e grava `coverage_shots.json` (só o que a cobertura usa de cada tomada: tempos, miniaturas, medida `tela uniforme`, fala, análise da IA, caminho e áudio do clipe, do catálogo). Sem `visual.json`: erro "rode `visual <pasta>` antes (ou use --receita vlog)". Um `visual.json` de outro catálogo (clipe que mudou de arquivo ou sumiu) também é recusado. Clipe da espinha sem análise: aviso (a imagem ruim dele não é detectada).
* **Janelas obrigatórias (código):** frases mantidas (as que aparecem na espinha, já aparadas pelos cortes de silêncio) com pelo menos `MIN_BAD_OVERLAP_SEC` (0,3 s) da parte mantida numa tomada do clipe que está falando com `usable=nao`, problema `obstruida` ou `borrao_movimento`, ou medida `tela uniforme` (esta vale também para tomadas `sem_analise`). Frases seguidas do mesmo clipe viram uma janela (`O1: S045–S047, tela uniforme`), com as tomadas ruins e os motivos.
* **Candidatas a cobertura:** tomadas analisadas, com `usable` diferente de `nao`, dos papéis `drone`, `broll`, `pov` e `principal` (do principal, só tomadas **sem fala**, inclusive de clipes da espinha), filtradas por `roles_to_use` do `briefing.json`, se houver. Tomadas `sem_analise` nunca são cobertura.
* **Prompt (compacto):** regras, instruções (`coverage_instructions` do `briefing.json` e o `--pedido`), janelas obrigatórias, frases mantidas em ordem (`[S012] P006 2.3s texto curto`) e candidatas (`D001.T05 drone, 6.2s, beleza 4, boa — descrição [tags]`). Resposta: `coverages` (`from_sentence`, `to_sentence`, `shot_id`, `reason`, `windows`), `breaths` (`after_sentence`, `shot_id`, `size` short/medium/long, `reason`) e `summary`.
* **Validação (código):** IDs existentes (frase mantida, tomada, janela); tomada candidata (a mensagem diz por que não: sem análise, inutilizável, papel não permitido, principal com fala); tomada de outro clipe que não os que falam no trecho; coberturas sem sobreposição; **cada tomada uma vez só** (cobertura ou respiro); cobertura entre `MIN_COVERAGE_SEC` (1,5 s) e `MAX_COVERAGE_SEC` (8 s) no vídeo (uma frase sozinha pode passar do máximo); respiro fora de uma cobertura, um por frase, tomada com pelo menos `MIN_BREATH_SHOT_SEC` (1 s), no máximo um a cada `BREATH_EVERY_SEC` (30 s) de espinha; janela citada que a cobertura não toca; **todas as janelas obrigatórias cobertas**. Inválida → nova tentativa com o erro. Inválida de novo: se só faltam janelas, a decisão da IA fica e o código completa (`ia+fallback`); senão, **fallback**: só as janelas obrigatórias (`fallback`). Erros de comunicação sobem, como no Diretor.
* **Preenchimento das janelas (fallback):** as frases descobertas de cada janela são divididas em coberturas de até 8 s (uma curta demais pega frases vizinhas livres até 1,5 s, sem atravessar um respiro) e cada uma recebe a **tomada livre de maior beleza que caiba** (se nenhuma couber, a de maior beleza, e o resto é encadeado na montagem), fora dos clipes que estão falando. Janela sem tomada livre: aviso.
* **IDs → tempos (código, `coverage/assemble.py`):** frases → frames pela posição dos trechos de cada clipe na Timeline (`clip_id` + `source_in/out`; uma frase aparada pelo corte de silêncio começa no primeiro instante mantido). **Respiros** viram itens de `video_main` inseridos na espinha: no primeiro corte entre o fim da frase e o início da próxima do mesmo clipe, ou no meio do caminho (o item é dividido), ou no fim do clipe na última frase; tamanho `BREATH_SECONDS` (curto 2 s, médio 3,5 s, longo 5 s) limitado à tomada; som da tomada (`BREATH_VOLUME` 1,0), mudo se o clipe não tem áudio (drone) ou a tomada tem fala. **Coberturas** viram `video_overlay` do início da primeira frase ao fim da última, no tempo final (o áudio da fala continua); o trecho da tomada é do tamanho da janela, centrado no melhor trecho (`best_frames`) e limitado à tomada (a última tomada de um clipe perde 0,1 s no fim, que pode passar do último quadro). Tomada mais curta que a janela: numa janela obrigatória, o código **encadeia** a tomada livre de maior beleza no resto; fora dela, a cobertura é **encurtada**; sobra menor que `MIN_CHAIN_SEC` (0,75 s) fica sem cobertura (não pisca); tudo com aviso. **Som ambiente** da cobertura: o áudio da tomada em `audio_ambient` a `COVERAGE_AMBIENT_VOLUME` (0,12) com fades de 0,3 s, se o clipe tem áudio e a tomada não tem fala.
* **Tudo depois usa a Timeline final** (`coverage_timeline.json`, a que é renderizada): os respiros deslocam o que vem depois. Legendas: cada bloco vai para onde o seu trecho está (um bloco nunca atravessa um corte, logo nunca atravessa um respiro); `edit_transcript.md`/`edit_diff.json` com os tempos do vídeo final; `edit_timeline.svg` com a faixa de cobertura (overlay) acima da trilha da fala e os respiros em cinza; `edit_report.md` com a seção **Cobertura** (modelo, origem `ia`/`ia+fallback`/`fallback`, janelas obrigatórias e quem atendeu, cada cobertura com frases, tomada(s), posição no vídeo, motivo, se era obrigatória e origem, e cada respiro).
* **`coverage_decision.json`** (`CoverageRun`): cadeia pedida, modelo efetivo, trocas, tentativas, origem e motivo, instruções, papéis, janelas, quem atendeu cada janela, decisão aceita (IDs, com a origem de cada item), onde cada item foi parar (tempos e trechos de tomada) e avisos.
* **Cache:** `visual` = hash do `visual.json` + do catálogo + clipes da espinha; `coverage` = Timeline (que já inclui o Diretor) + `coverage_shots.json` + instruções e papéis do briefing + `--pedido` + IA (provedor, modelo, raciocínio, cadeia pedida) + versão do prompt + as constantes acima. Render, legendas e relatório seguem o **resultado** da cobertura (hash do `coverage_timeline.json` e, nos dois últimos, do `coverage_decision.json`), não só as entradas: se a etapa for refeita com as mesmas entradas (artefato apagado, `--force`) e a IA decidir diferente, o que vem depois acompanha. Trocar só a IA da cobertura refaz cobertura, render, legendas e relatório. Na `vlog`, nenhum fingerprint mudou.
* **Versões:** `versions/vN/` leva `coverage_timeline.json` (apontando para o vídeo da versão), a espinha `timeline.json` e `coverage_decision.json`.

### Usando a GPU
A transcrição usa a GPU NVIDIA automaticamente quando ela existe e as bibliotecas CUDA estão instaladas (`--whisper-device auto`, o padrão); senão, volta para a CPU com um aviso. Para instalar as bibliotecas (cuBLAS para CUDA 12 e cuDNN 9, compatíveis com o ctranslate2 4.x do faster-whisper) no Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -e ".[gpu]"
```

Requer driver NVIDIA com suporte a CUDA 12. As pastas `bin` desses pacotes são registradas pelo próprio programa (não precisa mexer no PATH).

- `--whisper-device {auto,cpu,cuda}`: `cuda` falha com instruções se a GPU não puder ser usada; `cpu` ignora a GPU.
- `--whisper-compute-type`: `auto` = `float16` na GPU e `int8` na CPU; aceita valores explícitos (ex.: `int8_float16`).
- `--whisper-model {auto,<modelo>}`: `auto` (padrão) escolhe pelo dispositivo efetivo: `large-v3-turbo` na GPU (CUDA) e `small` na CPU; um nome explícito (`small`, `medium`...) é respeitado. Se a GPU falhar e o programa voltar para a CPU, o `auto` passa a usar o `small`.
- O modelo, o dispositivo e o compute_type efetivos aparecem no `transcript.json` (`model_used`, ex.: `faster-whisper:large-v3-turbo (cuda, float16)`), no relatório e entram no cache da transcrição.
- **Atenção ao atualizar:** o padrão mudou de `small` para `auto`. Em máquina com GPU, projetos já transcritos serão transcritos de novo na próxima execução do `edit` (o modelo efetivo mudou e faz parte do cache), com o download único do `large-v3-turbo`. Na CPU o `auto` resolve para `small`, igual ao padrão antigo, e o cache continua válido. Para manter o cache de antes numa máquina com GPU, passe `--whisper-model small`.

---

## 5. Estrutura dos Módulos

### Módulo Core (Fase 1)
* [`src/ai_video_editor/core/schema.py`](../src/ai_video_editor/core/schema.py): Modelos `ClipSegment`, `EditPlan` e `LoudnessTarget` (presets de normalização).
* [`src/ai_video_editor/core/timeline.py`](../src/ai_video_editor/core/timeline.py): `Timeline` com trilhas (`video_main`, `video_overlay`, `audio_ambient`; áudio principal = áudio dos itens da base), validação quantizada em frames, `timeline_from_edit_plan` e `flatten` (achatamento puro em pedaços de vídeo + plano de áudio).
* [`src/ai_video_editor/renderer/timeline_compiler.py`](../src/ai_video_editor/renderer/timeline_compiler.py) e [`timeline_executor.py`](../src/ai_video_editor/renderer/timeline_executor.py): render da Timeline (inspeção das origens, pedaços padronizados em paralelo, áudio 48 kHz com ambiente, mux, loudnorm e validação de duração/fps).
* [`src/ai_video_editor/renderer/compiler.py`](../src/ai_video_editor/renderer/compiler.py): Compilação segura de filtros FFmpeg sem shell injection.
* [`src/ai_video_editor/renderer/executor.py`](../src/ai_video_editor/renderer/executor.py): Execução isolada via `subprocess` e validação com FFprobe.
* [`src/ai_video_editor/renderer/loudness.py`](../src/ai_video_editor/renderer/loudness.py): Normalização de loudness em duas passadas (medição com loudnorm, parse do JSON, filtro da segunda passada).
* [`src/ai_video_editor/renderer/ffmpeg_locator.py`](../src/ai_video_editor/renderer/ffmpeg_locator.py): Descoberta resiliente de executáveis FFmpeg/FFprobe.

### Módulo Analyzer (Fase 2)
* [`src/ai_video_editor/analyzer/schema.py`](../src/ai_video_editor/analyzer/schema.py): Schemas Pydantic (`VideoMetadata`, `VideoStreamInfo`, `AudioStreamInfo`, `ExtractedFrame`, `VideoAnalysisResult`, `BatchAnalysisResult`).
* [`src/ai_video_editor/analyzer/exceptions.py`](../src/ai_video_editor/analyzer/exceptions.py): Exceções de domínio (`VideoNotFoundError`, `InvalidMediaFormatError`, `NoVideoStreamError`, `FrameExtractionError`).
* [`src/ai_video_editor/analyzer/probe.py`](../src/ai_video_editor/analyzer/probe.py): Extração de metadados e conversão segura de FPS fracionário via `fractions.Fraction`.
* [`src/ai_video_editor/analyzer/extractor.py`](../src/ai_video_editor/analyzer/extractor.py): Extração determinística de frames com timestamps rastreáveis.
* [`src/ai_video_editor/analyzer/service.py`](../src/ai_video_editor/analyzer/service.py): Orquestrador de análise técnica em lote.

### Módulo Intelligence (Fase 3) — legado
Usado por `analyze`/`report-ai`; a análise visual passa para `visual/`. As exceções e o `retry.py` continuam sendo a base da camada `llm/`.
* [`src/ai_video_editor/intelligence/schema.py`](../src/ai_video_editor/intelligence/schema.py): Modelos Pydantic (`VisualObservation`, `SceneTransitionHypothesis`, `VisualAnalysisResponse`, `AIAnalysisConfig`, `CompleteAIReport`).
* [`src/ai_video_editor/intelligence/exceptions.py`](../src/ai_video_editor/intelligence/exceptions.py): Exceções padronizadas de IA (`AIAuthenticationError`, `AIRateLimitError`, `AITimeoutError`, `AIUnavailableError`, `AIResponseValidationError`).
* [`src/ai_video_editor/intelligence/retry.py`](../src/ai_video_editor/intelligence/retry.py): `RetryPolicy` e `call_with_retry` (nova tentativa com espera exponencial, jitter e `Retry-After`; `sleep` injetável).
* [`src/ai_video_editor/intelligence/provider.py`](../src/ai_video_editor/intelligence/provider.py): Interface abstrata `MultimodalAIProvider`.
* [`src/ai_video_editor/intelligence/image_proxy.py`](../src/ai_video_editor/intelligence/image_proxy.py): Otimizador e downscaler de imagens 4K para proxies leves (JPEG / 1024px) com Pillow.
* [`src/ai_video_editor/intelligence/reporter.py`](../src/ai_video_editor/intelligence/reporter.py): Compilador de relatório Markdown rico com caminhos relativos e ressalvas anti-alucinação.
* [`src/ai_video_editor/intelligence/providers/gemini.py`](../src/ai_video_editor/intelligence/providers/gemini.py): Adaptador oficial do Google Gemini via SDK `google-genai`.
* [`src/ai_video_editor/intelligence/providers/mock.py`](../src/ai_video_editor/intelligence/providers/mock.py): Provedor simulado para testes offline rápidos e determinísticos.
* [`src/ai_video_editor/intelligence/service.py`](../src/ai_video_editor/intelligence/service.py): Orquestrador de análise multimodal e geração de relatórios.

### Módulo Planning (corte de silêncio)
* [`src/ai_video_editor/planning/schema.py`](../src/ai_video_editor/planning/schema.py): `SilenceCutParams` (paddings, `min_gap_to_cut_sec`, `min_keep_sec`), `TimeRange` e os presets de pausa (`PausePreset`, `apply_pause_preset`).
* [`src/ai_video_editor/planning/silence_cut.py`](../src/ai_video_editor/planning/silence_cut.py): `plan_silence_cuts` (trechos a manter a partir da `AudioAnalysisResult`), `removed_ranges` e `build_edit_plan` (lista de trechos → `EditPlan` na resolução/FPS da origem).
* [`src/ai_video_editor/planning/sentence_cut.py`](../src/ai_video_editor/planning/sentence_cut.py): `apply_sentence_cuts` (subtrai as frases removidas pelo Diretor dos trechos mantidos e devolve o destino de cada corte, `SentenceCutResult`: aplicado, estendido, protegido ou desfeito), `snap_cut` (leva os limites do corte para dentro dos silêncios), a regra do pedaço de palavra (`MIN_FRAGMENT_SEC`) e `subtract_ranges`.
* [`src/ai_video_editor/planning/exceptions.py`](../src/ai_video_editor/planning/exceptions.py): `PlanningError`, `NoSpeechDetectedError`.

### Módulo Director (Diretor de IA)
* [`src/ai_video_editor/director/schema.py`](../src/ai_video_editor/director/schema.py): `Sentence`, `UserBrief`, `DirectorContext`, `SentenceRemoval`, `DirectorDecision` e `DirectorRun` (conteúdo do `director_decision.json`).
* [`src/ai_video_editor/director/sentences.py`](../src/ai_video_editor/director/sentences.py): `build_sentences` (transcrição → frases) e `split_by_removal` (frases → intervalos a cortar e a proteger).
* [`src/ai_video_editor/director/provider.py`](../src/ai_video_editor/director/provider.py): Interface `EditDirector` (`decide(sentences, brief, context)`).
* [`src/ai_video_editor/director/service.py`](../src/ai_video_editor/director/service.py): `validate_decision` e `run_director` (nova tentativa com o erro, fallback e métricas `AIRunMetrics`).
* [`src/ai_video_editor/director/prompt.py`](../src/ai_video_editor/director/prompt.py): `build_director_prompt` (independente do provedor).
* [`src/ai_video_editor/director/providers/llm.py`](../src/ai_video_editor/director/providers/llm.py): `LLMDirector` (pede `DirectorDecision` a uma `ModelChain`); `providers/gemini.py` tem o atalho `GeminiDirector` (um único modelo Gemini).
* [`src/ai_video_editor/director/providers/mock.py`](../src/ai_video_editor/director/providers/mock.py): `MockDirector` determinístico (remoções fixas, sequência de respostas ou frases repetidas).

### Módulo Revision (Revisor de IA)
* [`src/ai_video_editor/revision/schema.py`](../src/ai_video_editor/revision/schema.py): `SentenceState`, `RevisionContext`, `RevisionDecision` (conjunto completo de remoções, restaurações com motivo, ritmo, `feasible`/`problem`) e `RevisionRun`.
* [`src/ai_video_editor/revision/provider.py`](../src/ai_video_editor/revision/provider.py): Interface `EditReviser` (`revise(states, brief, context)`).
* [`src/ai_video_editor/revision/service.py`](../src/ai_video_editor/revision/service.py): `validate_revision` (regras do Diretor + restaurações) e `run_reviser` (nova tentativa; sem versão se falhar).
* [`src/ai_video_editor/revision/prompt.py`](../src/ai_video_editor/revision/prompt.py): `build_reviser_prompt`.
* [`src/ai_video_editor/revision/providers/`](../src/ai_video_editor/revision/providers): `LLMReviser` (cadeia de modelos, como o Diretor), o atalho `GeminiReviser` e `MockReviser` determinístico.

### Módulo LLM (provedores de IA com saída estruturada)
* [`src/ai_video_editor/llm/client.py`](../src/ai_video_editor/llm/client.py): interface `StructuredLLMClient` (`generate(prompt, schema, config, retry, images=None)` → objeto validado + `LLMCallMetrics`), com novas tentativas, subida do nível de raciocínio e métricas já implementadas.
* **Imagens** (opcionais): lista de `ImageInput` (`llm/schema.py`: caminho do JPEG, `label` = ID citado no prompt, `mime_type`, `detail` `low`/`auto`/`high`). O provedor envia o prompt e, para cada imagem, o texto `Imagem <label>:` seguido dos bytes inline — Gemini: `types.Part.from_bytes` numa `Content` do usuário, com `media_resolution` quando o detalhe não é `auto`; OpenAI: itens `input_text` e `input_image` (data URL base64, `detail`). Mesmas novas tentativas, mapeamento de erros e `ModelChain`; `LLMCallMetrics.images` registra quantas foram. Sem imagens, a chamada ao SDK é a mesma de antes (texto puro), assim como os fingerprints. O `MockLLMClient` guarda as imagens em `image_calls` e aceita funções `(prompt, imagens, schema) → resposta` na lista de respostas.
* [`src/ai_video_editor/llm/registry.py`](../src/ai_video_editor/llm/registry.py): `register_provider`/`create_client` (`gemini`, `openai`, `mock`, com import tardio).
* [`src/ai_video_editor/llm/chain.py`](../src/ai_video_editor/llm/chain.py): `ModelChain` (troca automática de modelo) e `resolve_chain`/`parse_chain` (`provedor:modelo,...`).
* [`src/ai_video_editor/llm/providers/`](../src/ai_video_editor/llm/providers): `GeminiLLMClient` (`google-genai`, `response_schema`), `OpenAILLMClient` (SDK `openai`, Responses API com `text_format` / JSON schema estrito; as novas tentativas do SDK ficam desligadas) e `MockLLMClient`.
* [`src/ai_video_editor/llm/keys.py`](../src/ai_video_editor/llm/keys.py) e [`errors.py`](../src/ai_video_editor/llm/errors.py): chaves por variável de ambiente/`.env` e utilitários de erro (remove chaves `AIza…`/`sk-…` das mensagens).

### Módulo Benchmark (comando `benchmark transcription`)
* [`src/ai_video_editor/benchmark/wer.py`](../src/ai_video_editor/benchmark/wer.py): normalização básica e WER por distância de edição entre palavras (S/D/I, substituições mais frequentes e palavras após o fim da referência), sem dependências.
* [`src/ai_video_editor/benchmark/normalize.py`](../src/ai_video_editor/benchmark/normalize.py): normalização por idioma (inglês: números, okay/ok, x's) para o WER normalizado.
* [`src/ai_video_editor/benchmark/rescore.py`](../src/ai_video_editor/benchmark/rescore.py): `benchmark rescore`, que recalcula as métricas a partir de `hypotheses/`.
* [`src/ai_video_editor/benchmark/runner.py`](../src/ai_video_editor/benchmark/runner.py): `run_benchmark` (configurações × itens, carga do modelo medida à parte, falhas registradas) e o registro extensível de provedores.
* [`src/ai_video_editor/benchmark/schema.py`](../src/ai_video_editor/benchmark/schema.py) e [`report.py`](../src/ai_video_editor/benchmark/report.py): modelos Pydantic (`BenchmarkRun`, manifesto) e o `results.md`.

### Módulo Recipes (receitas)
* [`src/ai_video_editor/recipes/schema.py`](../src/ai_video_editor/recipes/schema.py): catálogo de peças `PIECES` (etapas que o código executa e o que cada uma sempre exige), `Recipe` (validação de etapas, dependências e ordem) e `RecipeError`.
* [`src/ai_video_editor/recipes/registry.py`](../src/ai_video_editor/recipes/registry.py): `register_recipe`, `get_recipe`, `recipe_names` e as receitas `fala` (padrão de `edit <vídeo>`) e `vlog` (padrão de `edit <pasta>`, `multi_clip=True`).

### Módulo Catalog (comando `catalog`)
* [`src/ai_video_editor/catalog/builder.py`](../src/ai_video_editor/catalog/builder.py): `CatalogBuilder` (varredura, hash da listagem, cache por clipe, etapas em paralelo, IDs, avisos), ordem dos papéis (`order_role`, `date_conflicts`, `natural_key`), `current_catalog` (catálogo ainda válido da pasta) e `describe_clip` (resumo mínimo de um vídeo avulso).
* [`src/ai_video_editor/catalog/schema.py`](../src/ai_video_editor/catalog/schema.py): `ProjectCatalog`, `CatalogClip`, `ClipMedia`, `RecordingDate`, `SpeechDetection`, `ColorSignals` e `ClipAnalysis` (o que vai para o cache).
* [`src/ai_video_editor/catalog/roles.py`](../src/ai_video_editor/catalog/roles.py) e [`dates.py`](../src/ai_video_editor/catalog/dates.py): papéis por pasta, `--papel`, `--ajuste-relogio` e a data de gravação com origem.
* [`src/ai_video_editor/catalog/media.py`](../src/ai_video_editor/catalog/media.py): ffprobe do clipe, miniaturas e `signalstats`.
* [`src/ai_video_editor/catalog/speech.py`](../src/ai_video_editor/catalog/speech.py): interface `SpeechDetector`, `FasterWhisperVAD` e `MockSpeechDetector`.
* [`src/ai_video_editor/catalog/sheets.py`](../src/ai_video_editor/catalog/sheets.py) e [`report.py`](../src/ai_video_editor/catalog/report.py): folhas de contato e `catalog.md`.

### Módulo Producer (comando `briefing`)
* [`src/ai_video_editor/producer/schema.py`](../src/ai_video_editor/producer/schema.py): `BriefingContent` (o schema pedido à IA), `Briefing` (+ origem, material, estimativa, métricas; `briefing.json`), `MaterialSummary`/`RoleSummary`, `RecipeInfo`, `WorkEstimate`, `ProducerContext`.
* [`src/ai_video_editor/producer/summary.py`](../src/ai_video_editor/producer/summary.py): resumo do material por papel e receitas registradas (`registered_recipes`, lidas na hora).
* [`src/ai_video_editor/producer/service.py`](../src/ai_video_editor/producer/service.py): `validate_content`, `recipe_problems`, `fallback_content`, `estimate_work` e `run_producer` (nova tentativa com o erro e fallback).
* [`src/ai_video_editor/producer/material.py`](../src/ai_video_editor/producer/material.py): catálogo atual ou novo (pasta) ou resumo mínimo (arquivo) e a pasta do projeto.
* [`src/ai_video_editor/producer/provider.py`](../src/ai_video_editor/producer/provider.py), [`prompt.py`](../src/ai_video_editor/producer/prompt.py) e `providers/`: interface `BriefingProducer`, prompt, `LLMProducer` (cadeia de modelos) e `MockProducer`.

### Módulo Coverage (Diretor de cobertura, receita `vlog_cobertura`)
* [`src/ai_video_editor/coverage/schema.py`](../src/ai_video_editor/coverage/schema.py): `VisualShot`/`VisualMaterial` (`coverage_shots.json`), `CoverageSentence`, `MandatoryWindow`, `CoverageContext`, a resposta da IA (`CoverageDecision`, `CoverageChoice`, `BreathChoice`), `CoverageParams` (`inherit`: o que não foi dito vem do Diretor) e `CoverageRun` (`coverage_decision.json`).
* [`src/ai_video_editor/coverage/material.py`](../src/ai_video_editor/coverage/material.py): `build_visual_material` (visual.json + catálogo, recusa outro catálogo), `coverage_sentences` (frases mantidas → frames da espinha), `find_mandatory_windows`, `coverage_roles` e `coverage_candidates`.
* [`src/ai_video_editor/coverage/service.py`](../src/ai_video_editor/coverage/service.py): `coverage_problems` (validação), `fill_windows` (tomada livre de maior beleza), `window_cover` e `run_coverage` (nova tentativa, `ia+fallback` e fallback), com as constantes nomeadas.
* [`src/ai_video_editor/coverage/assemble.py`](../src/ai_video_editor/coverage/assemble.py): `assemble_coverage` (IDs → tempos: respiros na base, overlays, ambiente) e `shot_segment`.
* [`src/ai_video_editor/coverage/prompt.py`](../src/ai_video_editor/coverage/prompt.py), [`provider.py`](../src/ai_video_editor/coverage/provider.py) e `providers/`: prompt (`COVERAGE_PROMPT_VERSION`), interface `CoverageDirector`, `LLMCoverageDirector` (cadeia de modelos) e `MockCoverageDirector`.

### Módulo Visual (comando `visual`)
* [`src/ai_video_editor/visual/builder.py`](../src/ai_video_editor/visual/builder.py): `VisualBuilder` (catálogo existente ou gerado, tomadas por clipe em paralelo com cache, IA em lotes paralelos com cache por fingerprint gravado a cada lote, estimativa, saídas).
* [`src/ai_video_editor/visual/shots.py`](../src/ai_video_editor/visual/shots.py): passada do FFmpeg (`scan_clip`: cena + `blurdetect`/`signalstats`), `split_shots`, `measure_shot`, `frame_times` e `extract_frames`, com as constantes nomeadas.
* [`src/ai_video_editor/visual/describer.py`](../src/ai_video_editor/visual/describer.py): `VisualDescriber` (sobre `ModelChain`), prompt, lotes, `validate_batch`, estimativa de tokens e `mock_visual_responder`.
* [`src/ai_video_editor/visual/schema.py`](../src/ai_video_editor/visual/schema.py) e [`report.py`](../src/ai_video_editor/visual/report.py): `ProjectVisual`, `ClipVisual`, `Shot`, `ShotFrame`, `ShotMeasures`, `ShotAnalysis`, `VisualBatchResponse`, `VisualAIRun` e o `visual.md`.

### Módulo Pipeline (comandos `edit` e `revise`)
* [`src/ai_video_editor/pipeline/runner.py`](../src/ai_video_editor/pipeline/runner.py): `EditPipeline`, que executa as etapas da receita (uma por método, na ordem de `recipe.steps`) com cache por fingerprint e callbacks de eventos (`on_progress`) e de andamento da etapa em curso (`on_stage_progress`, com `StageProgress`); `TranscriptionProvider`, `EditDirector` e `RenderExecutor` são injetáveis. O transcriber (`transcribe(..., on_progress)`) e o executor (`execute(plan, on_progress)`) informam o progresso; o pipeline só repassa, sem depender do terminal.
* [`src/ai_video_editor/live_status.py`](../src/ai_video_editor/live_status.py): `LiveStatus`, a linha de andamento ao vivo da CLI (thread de atualização a cada 1s, avisos acima da linha, modo sem terminal só com linhas finais).
* [`src/ai_video_editor/pipeline/schema.py`](../src/ai_video_editor/pipeline/schema.py): `EditPipelineParams`, `ProjectManifest` (`project.json`), `StageRecord`, `RunTimings`/`StageTiming`, `EditPipelineResult`.
* [`src/ai_video_editor/pipeline/project.py`](../src/ai_video_editor/pipeline/project.py): Hash curto do vídeo, nome da pasta do projeto, nomes dos artefatos e fingerprints.
* [`src/ai_video_editor/pipeline/timing.py`](../src/ai_video_editor/pipeline/timing.py): `TimingRecorder` (medição com `time.perf_counter`), formatação (`8,4s`, `12min 34s`) e a seção "Tempo de processamento".
* [`src/ai_video_editor/pipeline/report.py`](../src/ai_video_editor/pipeline/report.py): Geração do `edit_report.md` (e da seção "Revisão").
* [`src/ai_video_editor/pipeline/versions.py`](../src/ai_video_editor/pipeline/versions.py): versões em `versions/vN/`, registro após o `edit` e migração de projetos antigos.
* [`src/ai_video_editor/pipeline/revise.py`](../src/ai_video_editor/pipeline/revise.py): `ProjectReviser` (revisor → plano → render → legendas → relatório numa versão nova).
* [`src/ai_video_editor/pipeline/cutting.py`](../src/ai_video_editor/pipeline/cutting.py): `plan_keep_ranges`, compartilhado por `edit` e `revise`.
* [`src/ai_video_editor/pipeline/edit_diff.py`](../src/ai_video_editor/pipeline/edit_diff.py): status de aplicação das remoções e transcrição comparativa (`build_edit_diff` → `EditDiff`/`DiffEntry`, `edit_transcript.md` e `edit_diff.json`).
* [`src/ai_video_editor/pipeline/edit_timeline.py`](../src/ai_video_editor/pipeline/edit_timeline.py): linha do tempo dos cortes (`render_edit_timeline` → `edit_timeline.svg`; `timeline_from_diff_file` desenha a partir de um `edit_diff.json` existente).
* [`src/ai_video_editor/pipeline/spine.py`](../src/ai_video_editor/pipeline/spine.py): `SpinePipeline` (`edit <pasta>`, receitas `vlog` e `vlog_cobertura`): herda do `EditPipeline` o Diretor, a transcrição e o progresso; etapas por clipe com cache por clipe; etapas `visual` e `coverage` (Diretor de cobertura injetável, `run(..., coverage=CoverageParams)`); render pela Timeline final (`TimelineRenderExecutor` injetável).
* [`src/ai_video_editor/pipeline/clips.py`](../src/ai_video_editor/pipeline/clips.py): localização/geração do catálogo (só leitura de `catalog/`), escolha dos clipes da espinha (`select_spine_clips`, mínimos de fala) e arquivos do cache por clipe.
* [`src/ai_video_editor/pipeline/spine_schema.py`](../src/ai_video_editor/pipeline/spine_schema.py): `SpineSelection`/`SpineClip`/`UnusedClip`, `SpinePlan`/`ClipPlan`, `SpineDiff`/`ClipDiff`, `SpineManifest` (`kind="multiclip"`) e `SpineResult`.
* [`src/ai_video_editor/pipeline/spine_report.py`](../src/ai_video_editor/pipeline/spine_report.py): diff por clipe, `edit_transcript.md` por clipe, `edit_timeline.svg` colorido por clipe e `edit_report.md` da espinha.
* [`src/ai_video_editor/planning/timeline_plan.py`](../src/ai_video_editor/planning/timeline_plan.py): trechos por clipe → `Timeline` (`build_spine_timeline`, `spine_fps`, `clip_spans`) e a posição de cada trecho de fala na Timeline, com respiros no meio (`clip_placements`, `map_source_range`, `concat_to_final`, `is_breath_item`).
* [`src/ai_video_editor/subtitles/clips.py`](../src/ai_video_editor/subtitles/clips.py): legendas de vários clipes deslocadas para o tempo final.
