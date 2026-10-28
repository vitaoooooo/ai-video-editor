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
pip install -e ".[dev,transcription]"   # ou ".[dev,gpu]" com GPU NVIDIA; acrescente ",openai" para o provedor OpenAI (o Ollama não precisa de extra)
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
.\.venv\Scripts\python.exe -m ai_video_editor.cli render output/vlog_timeline.json --encoder cpu
```

`--encoder auto|cpu|nvenc` (padrão `auto`) e `--gpu-paralelo N` (padrão 3) escolhem o codificador de vídeo, como no `edit` (ver "Render na GPU (NVENC)" em "Usando a GPU"); a saída do comando mostra `Codificador: nvenc` ou `cpu (motivo)`.

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
2. **Pedaços de vídeo em paralelo** (um FFmpeg cada): vídeo sem áudio, padronizado (rotação de metadados aplicada pelo autorotate do FFmpeg, escala preservando a proporção + letterbox/pillarbox, fps de saída, `setsar=1`, `yuv420p` 8 bits), todos com o mesmo codificador e os mesmos parâmetros (x264 na CPU ou NVENC; ver "Render na GPU (NVENC)").
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
| `--receita` | `fala` (vídeo) / `vlog` ou `vlog_cobertura` (pasta; ver Opção K); `editor_ia` aceita os dois (Opção N) | Receita (combinação validada de etapas) a executar. Nome desconhecido é recusado com a lista das disponíveis; receita de pasta (`multi_clip`) com um vídeo, ou o contrário, é recusada |
| `--threshold-db` | auto | `auto` mede o ruído de cada vídeo e escolhe o limite; um número (ex: `-40`) usa esse limite fixo em dB |
| `--min-silence-sec` | 0.5 | Duração mínima de um silêncio detectado |
| `--padding-before` / `--padding-after` | 0.15 / 0.20 | Folga mantida antes/depois de cada fala |
| `--min-gap` | 0.30 | Pausas menores que isso (após o padding) não são cortadas, inclusive no início/fim |
| `--min-keep` | 0.10 | Falas mais curtas que isso são tratadas como ruído |
| `--transcribe` | desligado | Gera `transcript.json` (necessário para o Diretor) |
| `--director-provider` (ou `--director`) | `none` | Diretor de IA: `none`, `mock` (remove frases repetidas, sem rede), `gemini` (`GEMINI_API_KEY`), `openai` (`OPENAI_API_KEY`, extra `.[openai]`) ou `ollama` (modelo local, sem chave; Opção L). Só roda com `--transcribe` |
| `--director-model` | o do provedor | Modelo principal: `gemini-3.8-flash` no Gemini, `gpt-6-luna` na OpenAI, `qwen2.5:7b` no Ollama |
| `--director-fallback` | no Gemini: `gemini-3.6-flash`, `gemini-3.5-flash-lite` | Modelos de reserva, em ordem: `"provedor:modelo,..."` (pode misturar provedores; sem `provedor:` usa o do Diretor). `none` desliga. Veja "Troca automática de modelo" abaixo |
| `--director-thinking` | `low` | Raciocínio do Diretor: `off`, `low` ou `high`. Cada provedor traduz (Gemini: `thinking_level` MINIMAL/LOW/HIGH, ou `thinking_budget` na família 2.5; OpenAI: `reasoning.effort` none/low/high; modelos sem raciocínio ignoram e o relatório diz isso). Se a API recusar o nível, sobe sozinho (off → low → high); o nível pedido e o efetivo aparecem no relatório e no `director_decision.json` |
| `--director-timeout` | 120 (ollama: 600) | Tempo limite (s) da chamada ao Diretor. Sem a opção, vale o padrão do provedor; a cobertura e o Produtor herdam |
| `--api-retries` | 4 | Novas tentativas da API de IA em falhas temporárias (503, 504, 429 por minuto, conexão), com espera de 2s, 4s, 8s, 16s + jitter (respeita `Retry-After`); `0` desliga. Erros permanentes (autenticação, 400, 404) nunca são repetidos. Cada tentativa aparece no terminal (`[4/8] Diretor: Gemini indisponível (503), nova tentativa em 4s (2/5)`), o erro final traz o detalhe original da API e o relatório registra as chamadas à API de cada tentativa. Vale também para `revise` e `report-ai` |
| `--pedido` | — | Pedido livre (ex: `"aula sem enrolação, ritmo rápido"`): o Produtor escreve o `briefing.json` do projeto (ou ele é reaproveitado), e o briefing dá receita, intensidade, ritmo, instruções ao Diretor e duração alvo. Ver "Briefing no `edit`" abaixo. Com `--sem-briefing`, vai direto ao Diretor (como antes) |
| `--sem-briefing` | desligado | Não usa nem gera o `briefing.json`: valores só da CLI e padrões (a cobertura também não lê o arquivo) |
| `--intensity` (ou `--intensidade`) | a do briefing, senão `light` | Quanto o Diretor corta: `light`, `medium` ou `aggressive` |
| `--ritmo` (ou `--pace`) | o do briefing, senão `normal` | Ritmo das pausas: `natural` (padding ×1,75, `min_gap` ×1,5), `normal` (os valores das opções) ou `tight` (padding ×0,5, `min_gap` ×0,6); o mesmo ajuste do `revise` |
| `--target-duration` | a do briefing, senão livre | Duração final desejada em segundos, informada ao Diretor (o código não força o corte) |
| `--instructions` | as do briefing, senão — | Instruções diretas ao Diretor (ex: `"corte as piadas"`); valem mais que as do briefing |
| `--keep-fillers` | desligado | Impede o Diretor de remover frases só por serem enrolação |
| `--no-subtitles` | desligado | Não gera o `.srt` (por padrão ele é gerado quando há `--transcribe`) |
| `--subtitle-max-chars` | 42 | Máximo de caracteres por linha da legenda |
| `--transcript-min-pause` | 1.0 | Pausas removidas a partir desta duração (s) ganham marcador na transcrição comparativa |
| `--loudness` | `voz` | Normalização de volume do vídeo final (EBU R128, loudnorm em duas passadas): `voz` (-16 LUFS), `youtube` (-14 LUFS), `broadcast` (-23 LUFS), todos com true peak -1,5 dBTP e LRA 11; `off` mantém o volume original. Veja "Normalização de volume" abaixo |
| `--encoder` | `auto` | Codificador de vídeo do render: `auto` = NVENC (GPU NVIDIA) se disponível, senão CPU; `cpu` = x264; `nvenc` = exige a GPU (falha antes das etapas se ela não puder ser usada). Ver "Render na GPU (NVENC)" |
| `--gpu-paralelo` | 3 | Pedaços codificados ao mesmo tempo no NVENC (a GeForce limita as sessões simultâneas) |
| `--force` | desligado | Refaz todas as etapas, ignorando o cache |
| `--no-live` | desligado | Sem linha de andamento ao vivo: só as linhas finais de cada etapa (automático quando a saída não é um terminal, ex: redirecionada para arquivo). Vale também para `revise` |
| `--projects-dir` | `output/projects` | Diretório base dos projetos |

**Briefing no `edit`** (`pipeline/briefing.py`):

* **Quando há briefing:** com `--pedido`, o `edit` procura o `briefing.json` do projeto (o mesmo lugar do comando `briefing`, Opção I). Se ele existe, é do mesmo pedido (sem espaços nas pontas) e do mesmo material, é **reaproveitado** (o Produtor não roda). Senão o Produtor roda (mesma API do comando `briefing`, com as opções do Diretor: `--director-provider none` = regra sem IA, `mock`, `gemini`, `openai`; `--receita` é repassada como receita forçada) e o arquivo é salvo. Sem `--pedido`, um `briefing.json` atual (ex.: feito antes pelo comando `briefing`) também é usado. Sem `--pedido` e sem `briefing.json` atual: o `edit` de antes, sem briefing. `--sem-briefing` desliga tudo.
* **Mesmo material:** num vídeo, a pasta do projeto já é a do conteúdo do arquivo (hash); numa pasta, o resumo do catálogo atual (papéis, durações, fala) tem de ser igual ao gravado no briefing. Material diferente com `--pedido` refaz o briefing (aviso `briefing refeito: o material mudou`); sem `--pedido`, o arquivo é ignorado com aviso.
* **Precedência:** opção explícita da CLI > briefing > padrão, valor a valor: receita (`--receita`), intensidade, ritmo, instruções ao Diretor (`--instructions`; `director_instructions` do briefing) e duração alvo (`--target-duration`; `target_duration_sec`, informada ao Diretor, sem forçar o corte). `coverage_instructions` e `roles_to_use` seguem lidos pela cobertura (Opção K). A receita de um briefing de **fallback** (regra sem IA) sem receita forçada não conta: vale o padrão do `edit`, que sabe da análise visual e do `--transcribe` (`vlog_cobertura`).
* **Receita do briefing que não serve** é recusada antes de qualquer etapa, com o caminho do briefing e a saída: receita de pasta para um vídeo (ou o contrário) ou `vlog_cobertura` sem `visual/visual.json` → "Erro: o briefing (…) escolheu a receita 'vlog', que edita uma pasta de clipes, mas recebido um vídeo (…). Escolha outra com --receita, refaça o briefing com outro --pedido ou use --sem-briefing."
* **Terminal:** com briefing, antes das etapas: o "Entendi: …", `Briefing: gerado|reaproveitado (IA|regra sem IA; caminho)` e cada valor efetivo com a origem (`Intensidade: medium (briefing)`, `Ritmo das pausas: tight (opção da CLI)`, `Instruções ao Diretor: — (padrão)`).
* **Registro:** `settings` no `project.json` (vídeo e pasta) com o estado do briefing (`gerado`, `reaproveitado`, `desligado`, `sem_briefing`), origem `ia`/`fallback`, pedido, "Entendi" e cada valor com `origin` = `cli`/`briefing`/`padrao`, mais os avisos. Com briefing, o `edit_report.md` ganha a seção "Briefing" (o "Entendi", o pedido e a tabela valor/efetivo/origem).
* **Cache:** só o que muda o resultado entra nos fingerprints: o texto que chega ao Diretor, a intensidade e a duração alvo (no `UserBrief`, fingerprint do Diretor), o ritmo (no corte efetivo, fingerprint do plano/Timeline) e a receita (as etapas). Um briefing igual (reaproveitado, ou reescrito pelo comando `briefing` com outra data) não refaz nada; o relatório só inclui os valores e origens quando há briefing (sem estado nem avisos). O ritmo `normal` fica fora do fingerprint do relatório: projetos de antes continuam no cache.

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

**Cache:** cada etapa é pulada se o artefato existe e os parâmetros que a afetam não mudaram. Mudar `--threshold-db` (ou alternar entre `auto` e um número) refaz silêncios, plano, render e relatório; mudar só um padding ou o `--loudness` refaz a partir do plano; sem Diretor, a transcrição nunca força novo render. As legendas dependem do plano, da transcrição e de `--subtitle-max-chars`: mudar só esse limite refaz apenas legendas e relatório, sem novo render. O Diretor é refeito quando mudam a transcrição, o briefing ou o modelo. O codificador (`--encoder`) só entra no fingerprint do render: trocar de CPU para NVENC refaz render e relatório, nada antes. O hash do vídeo usa o tamanho e blocos do início, meio e fim do arquivo, então é rápido mesmo para arquivos grandes.


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
| `--reviser-provider` (ou `--reviser`) | o do Diretor do projeto (ou `gemini`) | `gemini`, `openai`, `ollama` (local) ou `mock` (determinístico, sem rede) |
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

O **Produtor** lê o pedido e um resumo do material e escreve o **briefing** da edição: tipo de vídeo, receita (uma das registradas), "Entendi: ...", metas (duração, intensidade, ritmo), instruções para o Diretor de cortes e para a cobertura, papéis a usar e avisos. Não edita nada; o `edit` lê o `briefing.json` (receita, intensidade, ritmo, instruções e duração alvo; ver "Briefing no `edit`" na Opção E) e também o gera sozinho com `edit ... --pedido`.

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
| `--visual-provider` | `gemini` | `gemini`, `openai`, `ollama` (local, ex.: `--visual-model gemma3:4b`; Opção L), `mock` (respostas simuladas, sem rede) ou `none` (só tomadas e medidas) |
| `--visual-model` / `--visual-fallback` | os do provedor | Modelo e cadeia de reserva, como no Diretor (`provedor:modelo,...`; `none` desliga). Padrão no Gemini: `gemini-3.8-flash` → `gemini-3.6-flash` → `gemini-3.5-flash-lite` |
| `--visual-thinking` | `low` | Raciocínio `off`/`low`/`high` (cada provedor traduz) |
| `--visual-timeout` | `180` (ollama: `600`) | Tempo limite de cada chamada, em segundos |
| `--detalhe-imagem` | `auto` | Resolução com que a IA lê as miniaturas: `low` (Gemini 3: 280 tokens/imagem; OpenAI `detail: low`), `auto` (padrão do provedor; Gemini 3: 1120) ou `high`. O Ollama não tem esse parâmetro: ignora, com aviso |
| `--visual-paralelo N` | `3` (ollama: `1`) | Lotes enviados à IA ao mesmo tempo (`1` = um por vez, como antes). Se um lote falhar por cota esgotada ou limite de requisições (429), cai pela metade (mínimo 1) para os lotes seguintes, com um aviso. Com `--visual-provider ollama`, o padrão é 1 (uma GPU); pedir mais é aceito, com aviso |
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
* **`visual.md`:** resumo (tomadas por papel, % utilizável, com rosto, com problema), **Atenção** (trechos do principal com fala — ou fala desconhecida — e imagem obstruída ou com borrão de movimento, juntando tomadas seguidas; sem análise, valem as medidas `tela uniforme`/`borrão alto`: candidatos a janela com imagem ruim na cobertura) e, por clipe, a tabela de tomadas com tempo, links das miniaturas, descrição, tags, problemas (da IA e das medidas), utilizável/melhor trecho e nota de beleza.
* **Terminal:** `[1/4] Catálogo`, `[2/4] Tomadas e medidas` (clipes ao vivo), `[3/4] Descrição por IA` (`lote 3/25 · gemini:gemini-3.8-flash`, novas tentativas e trocas de modelo), `[4/4] Relatório`; no fim, modelo efetivo, lotes, chamadas, imagens, tokens e tempo total.

### Opção K: Espinha falada de uma pasta de clipes (`edit <pasta>`, receitas `vlog` e `vlog_cobertura`)

Com uma pasta no lugar do vídeo, o `edit` junta os clipes de papel `principal` que têm fala, **na ordem do catálogo**, cada um sem pausas e sem frases ruins, num único vídeo renderizado pela Timeline. Com a análise visual feita (`visual <pasta>`), a receita `vlog_cobertura` acrescenta a **cobertura**: drone/b-roll/POV por cima da fala e respiros entre as falas (ver "Cobertura" abaixo). O caminho de um vídeo (`edit <arquivo>`) não muda. Com `--pedido`, o Produtor escreve (ou reaproveita) o `briefing.json` da pasta, que dá receita, intensidade, ritmo, instruções e duração alvo (ver "Briefing no `edit`" na Opção E).

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
* **Silêncios e transcrição por clipe**, reaproveitando `audio/` e `transcription/`, com **cache por clipe fora do projeto**: `<projects-dir>/_cache/edit_clips/<hash do arquivo>/silences-<fp>.json` e `transcript-<fp>.json` (chave: hash do arquivo + parâmetros da etapa). Trocar ou acrescentar um clipe cria outro projeto (a listagem mudou), mas só o clipe novo é processado. A transcrição roda um clipe por vez (a GPU fica com um modelo só), com a barra de progresso somando os clipes e o clipe atual na linha (`P006 (6/23)`). Um erro na transcrição de um clipe diz qual (`P012 (OsmoPocket_FullHD (40).mp4): ...`); os ajustes da normalização do clipe ficam ao lado da transcrição no cache (`transcript-<fp>.adjustments.json`, só quando houve ajuste), para o aviso reaparecer quando ela vem do cache. O limite de silêncio `auto` é medido em cada clipe; a detecção de silêncio roda até 4 clipes ao mesmo tempo.
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

# 2. Só o plano da cobertura (sem render): uma linha por bloco de assunto + coverage_plan.md
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" --transcribe --director gemini --coverage gemini --so-plano

# 3. Edição com cobertura (escolhida sozinha quando a pasta tem visual.json e há --transcribe)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" `
  --transcribe --language pt --director gemini --pedido "mostre bastante o lago"

# Sem IA na cobertura: um bloco por clipe e só o tapa das telas uniformes (tomada relacionada), pelo código
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" --transcribe --director gemini --coverage-provider none

# Espinha sem cobertura, mesmo com visual.json
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "D:\Vlog Viagem" --transcribe --director gemini --receita vlog
```

Etapas (`[n/10]`): catálogo → silêncios → transcrição → Diretor → Timeline → **análise visual** → **cobertura** → render → legendas → relatório. A cobertura é **guiada pelo conteúdo da fala**: a IA entende do que o vídeo trata e o que é dito em cada trecho, e só cobre onde a imagem mostra aquilo (fala do lago → imagem do lago); sem imagem que combine, a pessoa continua na tela. A IA só aponta IDs que o código criou (frases `S012`, tomadas `D001.T05`, trechos `O1`) e **cita evidências**; o código confere as evidências, valida **item a item** e calcula **todos os tempos**.

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--coverage-provider` (ou `--coverage`) | o do Diretor | `none` (sem IA: um bloco por clipe e só o tapa das telas uniformes, pelo código), `mock` (simulado: um bloco por clipe; cobre a frase cuja palavra é tag de uma tomada livre, com essa palavra e a tag como evidências; respiro nas trocas de bloco com tomada relacionada), `gemini`, `openai` ou `ollama` |
| `--coverage-model` / `--coverage-fallback` | os do Diretor, se o provedor for o mesmo | Modelo e reservas (`provedor:modelo,...`; `none` desliga), como no Diretor |
| `--coverage-thinking` | o do Diretor | Raciocínio `off`/`low`/`high` |
| `--coverage-timeout` | o do Diretor | Tempo limite da chamada, em segundos (provedor diferente do Diretor e sem `--director-timeout`: o padrão do provedor, 600 no Ollama) |
| `--so-plano` | desligado | Roda até a cobertura e mostra/salva o plano (`coverage_plan.md`), **sem renderizar** (sem render, legendas, relatório nem versão). Exige uma receita com cobertura (`vlog_cobertura`) e `--transcribe`, ou a receita `editor_ia` (opção N) |

* **Receita padrão:** sem `--receita`, uma pasta usa `vlog_cobertura` quando o projeto já tem `visual/visual.json` **e** há `--transcribe` (a cobertura escolhe frases por ID); senão, `vlog`. O terminal diz qual foi (`Receita: vlog_cobertura (a pasta tem análise visual)`). `--receita vlog` continua dando a espinha sem cobertura, com os mesmos fingerprints e a mesma saída de antes. Com `--receita vlog_cobertura` e sem transcrição, a etapa de cobertura é pulada e o vídeo sai como a espinha.
* **Análise visual (peça `visual`, sem IA):** lê `visual/visual.json` do projeto e grava `coverage_shots.json` (só o que a cobertura usa de cada tomada: tempos, miniaturas, medidas `tela uniforme` e `borrão alto`, fala, análise da IA, e do catálogo o caminho, o áudio e a data de gravação do clipe com a indicação de que ela é confiável, ordem do papel pela data). Sem `visual.json`: erro "rode `visual <pasta>` antes (ou use --receita vlog)". Um `visual.json` de outro catálogo (clipe que mudou de arquivo ou sumiu) também é recusado. Clipe da espinha sem análise: aviso (a imagem ruim dele não é detectada).
* **O que está na tela (código):** cada frase guarda as tomadas do clipe que fala por baixo da parte mantida (`screen`, mais tempo primeiro) e se **a pessoa fala para a câmera** (`to_camera`: a tomada principal da frase tem rosto e `usable=sim`).
* **Duas chamadas à IA (decisão):** (1) **mapa de assuntos** e (2) **cobertura**. **Por quê:** no vlog real (167 frases, 90 candidatas) o prompt de uma chamada só tinha ~41 mil caracteres, ~20 mil deles de frases e tela; o mapa sai numa chamada pequena (~24 mil, sem as candidatas) focada em entender o vídeo, a 2ª (~40 mil) mostra cada frase dentro do seu bloco, e cada resposta é validada e refeita sozinha. Custo: ~50% a mais de tokens de entrada que uma chamada só.
* **Mapa de assuntos (1ª chamada, `coverage/topics.py`):** a IA divide as frases mantidas em blocos contíguos `{from_sentence, to_sentence, topic (assunto curto), place (lugar/objeto falado)}` e diz do que o vídeo trata (`summary`). O código valida o **mapa inteiro**: blocos em ordem, sem buraco nem sobreposição, cobrindo todas as frases, com assunto (`buraco: S013–S014 sem bloco`, `sobreposição: S010 já estão no bloco anterior`). Inválido → nova tentativa com os problemas (`MAX_TOPIC_ATTEMPTS` = 2) → **um bloco por trecho seguido do mesmo clipe** (`clipe P007`), com aviso. Sem IA, o mesmo mapa por clipe. Blocos validados ganham IDs `A1`, `A2`...
* **Trechos com imagem ruim (código, `find_bad_spans`):** medidos **em segundos**, não por frase: a parte mantida das frases ∩ as tomadas **graves** do clipe que fala (`is_severe`: medida `tela uniforme`, `usable=nao` ou `borrao_movimento` confirmado por `borrão alto`; "parcial" é a IA dizendo que a imagem serve e não vira trecho), mais `BAD_MARGIN_SEC` (0,25 s) de cada lado sem sair da frase; pedaços seguidos do mesmo clipe e do mesmo tipo viram um trecho (`O1`); menos de `MIN_BAD_OVERLAP_SEC` (0,3 s) de imagem ruim é ignorado. Cada trecho diz se é **tela uniforme** (parede, tampa). **Não obrigam cobertura:** vão ao prompt como "imagem ruim aqui: prefira cobrir se houver imagem que combine"; sem cobertura válida, a imagem original fica, **com aviso** (`imagem ruim mantida em O2 (S023, 6,8s, imagem inutilizável): nenhuma imagem que combine com a fala`). No vlog real: só P007.T07 (tela uniforme, 4,6 s) e P007.T08 (inutilizável, 6,8 s), em vez das 10 janelas de antes.
* **Tomadas relacionadas (`coverage/related.py`):** uma candidata é relacionada a uma tela se foi **gravada perto** — do mesmo papel e até `NEIGHBOR_CLIPS` (1) posição do clipe que fala na ordem do catálogo ("gravada logo depois de P007"), ou de outro papel com as duas datas vindas da gravação a até `NEIGHBOR_MINUTES` (15 min) — ou se tem **tags específicas em comum** com ela: pelo menos `MIN_COMMON_TAGS` (2), ou a única que a tela tiver; `GENERIC_TAGS` (pessoa, rosto, detalhe, paisagem, árvore, céu) não contam. Usada no tapa da tela uniforme e nos respiros.
* **Candidatas a cobertura:** tomadas analisadas, com `usable` diferente de `nao`, dos papéis `drone`, `broll`, `pov` e `principal` (do principal, só tomadas **sem fala**, inclusive de clipes da espinha), filtradas por `roles_to_use` do `briefing.json`, se houver (e se o `edit` usa o briefing). Tomadas `sem_analise` nunca são cobertura.
* **Prompt da cobertura (2ª chamada, `COVERAGE_PROMPT_VERSION` 3):** regras (cobrir só onde a imagem mostra o que é falado ou o lugar da cena, **na dúvida não cobrir**; evidências; pessoa para a câmera; imagem ruim; limite de 30%; ~8 s por tomada; respiros só entre blocos), "do que o vídeo trata", instruções (`coverage_instructions` do briefing e as instruções ao Diretor), as telas uniformes com as candidatas relacionadas, as **frases por bloco** (`== A3 S034–S041 · trilha até o lago (lugar/objeto: lago)`), cada uma com o texto quase inteiro (até 600 caracteres; a evidência é copiada dele), a linha `~ tela: P006.T02 descrição [tags] (pessoa falando para a câmera; parcial: obstruída)` quando a tomada muda e `! O1 imagem ruim aqui (4.6s, tela uniforme (parede/tampa), S022–S023): prefira cobrir se houver imagem que combine` onde o trecho começa; e as candidatas (`D001.T05 drone, 7.9s, beleza 4, boa — descrição [tags]`, com a proximidade de gravação **só quando as datas das duas são confiáveis**). Resposta: `coverages` (`from_sentence`, `to_sentence`, `shot_ids` em ordem, `speech_evidence`, `image_evidence`, `reason`), `breaths` (`after_sentence`, `shot_id`, `size`, `image_evidence`, `reason`) e `summary`. Respostas antigas (com `shot_id`, `windows`, `window_decisions`) continuam legíveis; sem evidência, os itens saem.
* **Evidências (código, `coverage/evidence.py`):**
  * **De fala** (`speech_evidence`): trecho **literal** (palavras seguidas) do texto das frases cobertas, podendo atravessar frases seguidas.
  * **De imagem** (`image_evidence`): uma **tag** ou trecho literal da **descrição** da tomada; conferida em **cada** tomada listada (a que não tem sai da lista, com nota; nenhuma → item descartado).
  * **Ligação:** as duas precisam ter uma palavra de conteúdo em comum (fora de `STOPWORDS`, das palavras com menos de `MIN_CONCEPT_CHARS` = 3 letras e de `GENERIC_CONCEPTS`: pessoa, rosto, detalhe, paisagem, árvore, céu, imagem, vista, lugar, coisa, dia). **Forte** quando essa palavra é uma **tag** da tomada (o assunto principal da imagem, não um detalhe da descrição).
  * **Comparação:** sem diferenciar maiúsculas, acentos e pontuação; plural simples (`lagos`, `flores`, `cães`) vale como o singular; **sinônimos** da tabela `SYNONYMS` valem como a palavra principal: lago (lagoa, laguna), montanha (cordilheira, morro, serra, montaña, cerro), cachorro (cão, cachorra, perro), ilha (isla, islita), praia (playa), ônibus (ómnibus, microônibus, bus), trilha (sendero). Pequena de propósito: o material real mistura português e espanhol; sinônimo largo demais viraria "qualquer coisa combina".
* **Validação item a item (código, `review_decision`):** o item inválido é **descartado com o motivo** e os válidos ficam. Categorias: `evidência de fala` (sem trecho ou trecho que não está no texto), `evidência de imagem`, `evidência sem relação` (as duas não falam da mesma coisa), `pessoa para a câmera`, `ID inexistente`, `tomada não candidata`, `clipe que fala`, `tomada repetida` (cada tomada uma vez só), `sobreposição` (pelos frames, não pelas frases), `curta demais`, `limite de cobertura` e os dos respiros (abaixo). Numa lista, a tomada inválida sai e o item fica (`ia-ajustada`).
* **Pessoa falando para a câmera:** uma cobertura que pega uma frase `to_camera` só fica com evidência **forte**; senão sai (`S005: a pessoa fala para a câmera; só com evidência forte`).
* **Limite de cobertura:** a cobertura de conteúdo soma no máximo `MAX_COVERAGE_RATIO` (30%) da espinha. Passando, saem primeiro as que não tocam imagem ruim, depois as de evidência fraca, depois as mais longas, **com aviso** do que ficou de fora. Os tapas de tela uniforme não contam.
* **Segundos (código; `_place`):** uma cobertura que **toca um trecho com imagem ruim fica limitada aos segundos ruins** dele (`limitada aos segundos com imagem ruim (O1, 2,3s)`); senão cobre as frases inteiras. O tempo é dividido entre as tomadas listadas (até `MAX_COVERAGE_SEC` = 8 s cada; `distribute`); se elas não bastam, a cobertura é **encurtada em volta de onde a evidência é dita** (posição pela fração das palavras da frase; a fala tem ritmo parecido) — **nunca completada com tomada que a IA não escolheu**, salvo na tela uniforme. Trecho menor que `MIN_COVERAGE_SEC` (1,5 s): cresce até o mínimo dentro das frases ou estende para a frase vizinha livre mais curta; sem como, o item sai.
* **Tela uniforme (parede, tampa):** a única imagem ruim que pode ser tapada **sem evidência de fala**: um item da IA sem evidência sobre ela fica só com as tomadas **relacionadas** à tela (as outras saem) e só pelos segundos da tela uniforme; o código completa com outras relacionadas livres se faltar. Depois da IA, o código tapa sozinho as telas uniformes ainda descobertas (`fill_uniform`) com relacionadas; sem nenhuma, a imagem fica, com aviso. Um respiro que pegaria a tomada que tapa a parede é descartado (`reservada para tela uniforme`): tapar a parede vale mais.
* **Respiros:** só **entre blocos de assunto** (depois da última frase de um bloco; troca de lugar/momento), com tomada que tenha relação com o bloco anterior ou o seguinte: `image_evidence` na tomada e ligada ao assunto/lugar do bloco, ou tomada relacionada à tela dele. Descartes: `respiro fora da troca de assunto`, `respiro sem relação`, no meio de cobertura, repetido, tomada com menos de `MIN_BREATH_SHOT_SEC` (1 s), mais que um a cada `BREATH_EVERY_SEC` (30 s) de espinha.
* **Nova tentativa (só uma, `MAX_ATTEMPTS` = 2):** quando mais de `RETRY_DISCARD_RATIO` (metade) dos itens foram descartados ou a resposta não pôde ser interpretada. Leva **só os problemas** e os itens já aceitos e pede substitutos válidos (ou nenhum).
* **Origem:** por item `ia`, `ia-ajustada` ou `fallback`; por trecho com imagem ruim `coberto`, `parcial` (sobra mais de `MAX_UNCOVERED_SEC` = 0,5 s) ou `mantido`; no geral `ia`, `ia+fallback` (o código tapou uma tela uniforme) ou `fallback` (sem IA ou nada aproveitável).
* **Plano por bloco (`coverage/plan.py`):** uma linha por bloco de assunto, no terminal (no fim do `edit` e com `--so-plano`), no `edit_report.md` e no `coverage_plan.md`: `S034–S041 · trilha até o lago · coberto 2× (D001.T10 'lago' ↔ 'lago escuro entre montanhas'; …)`, `S000–S012 · chegada de ônibus · sem cobertura (nenhuma imagem de ônibus)` (ou quantas propostas saíram e por quê), `tela uniforme tapada (P008.T01, relacionada)`, `respiro depois (D002.T01)` e `imagem ruim mantida (O2, 6,8s)`. Até `SHOWN_PER_BLOCK` (2) coberturas detalhadas por linha.
* **IDs → tempos (código, `coverage/assemble.py`):** as coberturas usam os frames da espinha que a validação decidiu (`start_frame`/`end_frame`), levados para o tempo final. **Respiros** viram itens de `video_main` inseridos na espinha: no primeiro corte entre o fim da frase e o início da próxima do mesmo clipe, ou no meio do caminho (o item é dividido), ou no fim do clipe na última frase; tamanho `BREATH_SECONDS` (curto 2 s, médio 3,5 s, longo 5 s) limitado à tomada; som da tomada (`BREATH_VOLUME` 1,0), mudo se o clipe não tem áudio (drone) ou a tomada tem fala. **Coberturas** viram `video_overlay` (o áudio da fala continua), um item por tomada, cada trecho centrado no melhor trecho (`best_frames`) e limitado à tomada. **Som ambiente** da cobertura: o áudio da tomada em `audio_ambient` a `COVERAGE_AMBIENT_VOLUME` (0,12) com fades de 0,3 s, se o clipe tem áudio e a tomada não tem fala.
* **Tudo depois usa a Timeline final** (`coverage_timeline.json`): os respiros deslocam o que vem depois. Legendas, `edit_transcript.md`/`edit_diff.json`, `edit_timeline.svg` (faixa de cobertura e respiros em cinza) e `edit_report.md` com a seção **Cobertura**: modelo, as duas chamadas (tentativas e tokens), mapa de assuntos (IA ou código) e "do que o vídeo trata", origem, itens da IA, cobertura de conteúdo em segundos e % da espinha, **plano por bloco**, **trechos com imagem ruim** (resultado e coberturas), cada cobertura com a **evidência** (`'na praia' ↔ 'praia' (forte)` ou `tela uniforme; relacionada: ...`), cada respiro com a relação, e **Ajustes e descartes do código**.
* **Terminal:** a linha da etapa (`22 blocos de assunto, 10 coberturas (64.9s), 4 respiros (+14.0s), imagem ruim: 0 coberto(s), 1 parcial(is), 1 mantido(s) de 2 (ia (6 ajustadas))`), os avisos (uma linha de resumo dos descartes, imagem ruim mantida, limite de cobertura) e, no fim, o resumo e **uma linha por bloco**.
* **`coverage_decision.json`** (`CoverageRun`): cadeia pedida, modelo efetivo, trocas, **as duas chamadas** (tentativas, respostas cruas `topic_responses` e `responses`), mapa de assuntos e origem, "do que o vídeo trata", origem e motivo, itens recebidos, descartados (tentativa, item, categoria, motivo), trechos com imagem ruim e o resultado de cada um, decisão aceita (IDs, frames da espinha, evidência conferida, origem e ajustes), onde cada item foi parar, plano por bloco, segundos de cobertura, avisos e notas.
* **Cache:** `visual` = hash do `visual.json` + do catálogo + clipes da espinha; `coverage` = Timeline + `coverage_shots.json` + instruções e papéis do briefing + instruções ao Diretor + IA (provedor, modelo, raciocínio, cadeia pedida) + versão do prompt + as constantes (`COVERAGE_RULES` em `pipeline/spine.py`: as de duração, limite, margem, relação, sinônimos, stopwords, conceitos genéricos, tentativas e plano). Formato `coverage-v3`: projetos de antes refazem a etapa `coverage` (e o que vem depois). O `coverage_plan.md` é regravado da decisão a cada execução (também do cache). Render, legendas e relatório seguem o **resultado** da cobertura (hash do `coverage_timeline.json` e do `coverage_decision.json`). Na `vlog`, nenhum fingerprint mudou.
* **Versões:** `versions/vN/` leva `coverage_timeline.json` (apontando para o vídeo da versão), a espinha `timeline.json` e `coverage_decision.json`. Com `--so-plano`, nenhuma versão é criada.

### Opção L: Modelos locais com o Ollama (`--*-provider ollama` e `ollama-check`)

Roda o Diretor, o revisor, o Produtor, a cobertura e a descrição visual num modelo local pelo [Ollama](https://ollama.com), sem chave de API. Perfil pensado para uma RTX 2060 (6 GB): `qwen2.5:7b` como Diretor e `gemma3:4b` para as imagens (o `qwen2.5vl:3b` continua aceito; ver **Modelo de visão** abaixo).

```powershell
ollama pull qwen2.5:7b
ollama pull gemma3:4b

# Diagnóstico: servidor no ar, modelos baixados, uma chamada de texto e a com imagem (miniatura sintética) nos dois
# formatos (JSON schema e JSON pelo texto), com JSON validado, tempo, tokens, carga do modelo e o formato que funcionou
.\.venv\Scripts\python.exe -m ai_video_editor.cli ollama-check
.\.venv\Scripts\python.exe -m ai_video_editor.cli ollama-check --modelo-texto qwen2.5:7b --modelo-imagem gemma3:4b --ollama-ctx 8192

# Diretor local; descrição visual local (lotes um por vez)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit aula.mp4 --transcribe --director ollama
.\.venv\Scripts\python.exe -m ai_video_editor.cli visual "D:\Vlog Viagem" --visual-provider ollama --visual-model gemma3:4b
# Cadeia: Ollama primeiro, Gemini de reserva (o modelo do Ollama tem ":"; sem provedor, a tag não é lida como provedor)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit aula.mp4 --transcribe --director ollama --director-fallback "qwen2.5:3b,gemini:gemini-3.8-flash"
```

| Opção (em `edit`, `revise`, `briefing`, `visual`, `roteiro` e `ollama-check`) | Padrão | Efeito |
| --- | --- | --- |
| `--ollama-host URL` | `OLLAMA_HOST` ou `http://localhost:11434` | Endereço do servidor (`localhost:11434` e `0.0.0.0` também valem; `0.0.0.0` vira `127.0.0.1`) |
| `--ollama-ctx TOKENS` | `OLLAMA_CONTEXT_LENGTH` ou `8192` | `num_ctx` enviado em **toda** chamada. 8192 é uma estimativa para 6 GB, a medir: no primeiro `ollama-check` (RTX 2060), `qwen2.5:7b` com 8192 ocupou 5,4 GB e rodou 21% na CPU (`ollama ps`) |
| `--ollama-keep-alive` | `OLLAMA_KEEP_ALIVE` ou o do servidor (5 min) | Quanto tempo o modelo fica carregado depois da chamada (`10m`, `3600`, `-1` = sempre) |
| `--ollama-formato` | `OLLAMA_FORMATO` ou `auto` | Como pedir o JSON: `schema` (JSON schema em `format`), `texto` (sem `format`; schema no fim do prompt e JSON lido do texto) ou `auto` (schema; na resposta zerada do Ollama, `texto` para aquele modelo até o fim do processo). Ver **Formato** abaixo |

As opções valem para o processo inteiro (gravam as variáveis acima), inclusive para as etapas internas do `edit` (Produtor, Diretor, cobertura). `ollama-check` tem também `--modelo-texto` (padrão `qwen2.5:7b`), `--modelo-imagem` (padrão `gemma3:4b`; `none` pula) e `--timeout` (600); sai com código 1 se algum item falhar. A chamada com imagem é feita nos dois formatos (`schema` e `texto`) e a linha `Formato com imagem (...)` diz qual funcionou; basta um para o diagnóstico passar.

* **Chamada** (`llm/providers/ollama.py`): `POST /api/chat` com `stream: false`, `format` = JSON schema do modelo Pydantic (`model_json_schema()`; só no formato `schema`), `options.num_ctx` e `options.temperature`, e `keep_alive` se configurado. Cliente HTTP = `urllib` da biblioteca padrão, sem novas tentativas internas (as da `RetryPolicy` valem). Sem chave (`has_credentials` sempre verdadeiro).
* **Imagens:** o prompt numa mensagem do usuário e cada imagem numa mensagem do usuário própria, com o texto `Imagem <ID>:` e a imagem em base64 em `images` (a API do Ollama não posiciona a imagem dentro do texto; assim cada rótulo fica junto da sua imagem). `detail` é ignorado e registrado em `LLMCallMetrics.notes`. Sem imagens, uma única mensagem de texto.
* **Contexto (truncamento silencioso):** o Ollama corta o começo da entrada que passa do `num_ctx` sem devolver erro. Depois de cada resposta, se `prompt_eval_count` ≥ `num_ctx` − 32 (ou entrada + resposta ≥ isso), a chamada falha com `AIContextLimitError` (não transitória, não troca de modelo): "aumente o contexto (`--ollama-ctx`) ou reduza a entrada". Com o prefixo do prompt em cache, o Ollama conta só a parte nova, então um corte pode passar despercebido (nunca o contrário).
* **Formato** (`--ollama-formato`/`OLLAMA_FORMATO`, padrão `auto`): o Ollama 0.40 (runner llama.cpp) às vezes quebra no sampler da gramática gerada do JSON schema (`server.log`: `Unexpected empty grammar stack after accepting piece`) e devolve **HTTP 200 com a resposta zerada** (`{"model": "", "created_at": "0001-01-01T00:00:00Z", "message": {"role": "", "content": ""}, "done": false}`). **Detecção:** sem conteúdo e `done` diferente de `true` (ou `model` vazio) → `AIProviderError` permanente "O Ollama interrompeu a geração do modelo '…' (provável falha do formato JSON forçado; veja server.log do Ollama)", nunca "JSON inválido" (resposta terminada e vazia continua sendo "resposta sem conteúdo"). No `auto`, a mesma chamada é refeita **na hora** no formato `texto` (não conta como tentativa de correção do describer nem da `RetryPolicy`) e o modelo (por servidor) fica em `texto` até o fim do processo; aviso uma vez só, mesmo com threads: `qwen2.5vl:3b: o Ollama falhou com o JSON schema; usando JSON pelo texto do pedido` (`LLMCallMetrics.warnings` e `notes`; no `roteiro`, linha da etapa, `vision.notes` no `roteiro.json` e "Aviso da visão" no resumo). **Formato `texto`:** sem `format`; o prompt ganha no fim `Responda só com um objeto JSON com estes campos (JSON schema; "required" = obrigatórios), sem texto antes ou depois:` + o schema compacto numa linha (`compact_schema`: só `type`, `properties`, `items`, `enum` e `required`, com `$ref` resolvidos e sem títulos, descrições e padrões); a leitura extrai o primeiro objeto JSON do texto (tolera ```` ```json ```` e texto em volta) e valida no modelo Pydantic como no `schema` (`"people": "2"` vira 2); sem objeto ou inválido → `AIResponseValidationError` (o laço de correção do `roteiro` continua valendo). Contexto e métricas iguais nos dois formatos; o formato não entra nos fingerprints (as descrições em cache continuam valendo). Resposta zerada também no `texto` → o mesmo erro, com "sem o formato JSON forçado".
* **Modelo de visão** (padrão `gemma3:4b` no `roteiro` e no `ollama-check`): medido no vlog real (41 clipes; Windows, RTX 2060 6 GB, Ollama 0.40.11), o `gemma3:4b` descreveu 644/644 fotos sem falha, 3,0 s por foto (32 min), com descrições coerentes com a fala. O `qwen2.5vl:3b` continua aceito (`--visao-model qwen2.5vl:3b`, `--modelo-imagem qwen2.5vl:3b`), mas **nessa RTX 2060** (GPU sem bf16) devolveu lixo (`@@@@`, `8888...`) de forma intermitente na GPU; a mesma foto na CPU saiu certa (79 s), descarregar o modelo pela API não recuperou e nenhuma das 9 variações do pedido testadas mudou isso. Não foi medido em outras placas: não vale como regra para toda GPU.
* **Raciocínio:** ignorado (`thinking_setting` = None; o relatório diz "ignorado"). Temperatura e timeout repassados; timeout padrão 600 s (`default_timeout_sec` do provedor; modelo local numa GPU de 6 GB é lento, e a primeira chamada inclui a carga do modelo).
* **Erros:** conexão recusada → "Ollama não está rodando em <host>: abra o Ollama ou rode `ollama serve`" (não transitório, sem troca de modelo); 404/modelo não encontrado → "rode `ollama pull <modelo>`" (não transitório); tempo esgotado, 503 (servidor ocupado), queda do processo do modelo (500 "process has terminated", visto com um erro de CUDA ao carregar) e queda de conexão → transitórios; outros 4xx/5xx (ex.: falta de memória) → permanentes, com o detalhe do servidor; JSON inválido → `AIResponseValidationError` (o Diretor e o `visual` refazem com o erro, como nos outros provedores).
* **Paralelismo:** com `--visual-provider ollama`, `--visual-paralelo` passa a 1 por padrão; pedir mais é aceito com aviso (os lotes disputam a mesma GPU).

### Opção M: Roteiro unificado para o Editor de IA (`roteiro`)

Lê **qualquer material** (um vídeo = projeto de 1 clipe; uma pasta = N clipes de qualquer papel) e escreve o **roteiro unificado**: fala, imagem e ritmo de cada clipe, tudo com IDs para o Editor de IA citar. Não edita nada. O programa mede, descreve e confere; nenhuma regra depende de papel ou de tipo de vídeo (drone, b-roll, gameplay sem voz e evento entram com momentos e descrições; o papel e "tem fala" são só informação para a IA).

```bash
# 1. Quanto vai custar? Catálogo, silêncios, transcrição e fotos; mostra fotos (antes/depois do descarte),
#    chamadas e tempo estimado da IA de visão, sem chamá-la
python -m ai_video_editor.cli roteiro D:\vlog --so-estimar

# 2. Roteiro com a IA de visão local (padrão: ollama, gemma3:4b, quando o provedor está instalado)
python -m ai_video_editor.cli roteiro D:\vlog --language pt

# Outro provedor de visão, foto a cada 3 s, 2 clipes descritos ao mesmo tempo
python -m ai_video_editor.cli roteiro D:\vlog --visao-provider gemini --intervalo 3 --visao-paralelo 2

# Um vídeo só (projeto de 1 clipe, na mesma pasta de projeto do `edit <vídeo>`)
python -m ai_video_editor.cli roteiro aula.mp4 --visao-provider mock
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--intervalo` | `2` | Uma foto a cada N segundos em todos os clipes (grade fixa 0, 2, 4...) |
| `--visao-provider` | `ollama` (se registrado) | IA que descreve as fotos: `ollama`, `gemini`, `openai`, `mock` ou `none` (sem descrição; momentos só por troca de cena e duração). Sem o provedor `ollama` instalado e sem a opção, o comando pede para escolher |
| `--visao-model` | `gemma3:4b` no ollama; o do provedor nos outros | Modelo da visão (cadeia sem reserva); `qwen2.5vl:3b` também é aceito (ver Opção L, **Modelo de visão**) |
| `--visao-thinking` / `--visao-timeout` | `off` / `120` | Raciocínio (se o modelo tiver) e tempo limite por foto |
| `--visao-paralelo` | `1` | Clipes descritos ao mesmo tempo (as fotos de um clipe vão sempre em ordem); 1 para GPU local |
| `--so-estimar` | — | Para depois das fotos e mostra a estimativa (não grava o roteiro): 3,0 s por foto (`SECONDS_PER_PHOTO_ESTIMATE`, **medido** no `gemma3:4b` numa RTX 2060) e 256 tokens por imagem (Gemma 3) + ~350 do prompt |
| `--threshold-db`, `--min-silence-sec`, `--language`, `--whisper-*` | os do `edit` | Silêncios e transcrição: com os mesmos valores do `edit`, o cache é o mesmo |
| `--force` | — | Refaz silêncios, volume, cenas e fotos; transcrição e descrições em cache continuam valendo |
| `--projects-dir`, `--no-live`, `--api-retries` | | Como nos outros comandos |
| `--ollama-host`, `--ollama-ctx`, `--ollama-keep-alive`, `--ollama-formato` | | Ollama local (Opção L) |

Etapas (`[n/8]` no terminal): catálogo → silêncios e volume → transcrição → frases e ritmo → fotos → descrição das fotos → momentos → roteiro. Saídas na pasta do projeto: `roteiro/roteiro.json` (`ProjectRoteiro`, `schema_version` 1), `roteiro/roteiro.md` e `roteiro/fotos/<ID>.jpg` (só as mantidas). Terminal: caminhos, clipes, fotos antes/depois do descarte, momentos, frases, modelo e segundos por foto da visão, tokens estimados (total e por clipe) e o tempo de cada etapa.

* **Base:** catálogo como o `edit <pasta>` (lido se atual, gerado se faltar); arquivo único: ffprobe + fala (VAD), guardado em `roteiro/clip.json` (não roda o VAD de novo). Clipe com arquivo com problema fica fora, com aviso. **Transcrição de todo clipe com fala detectada, de qualquer papel** (≥ `MIN_SPEECH_TO_TRANSCRIBE_SEC` = 1 s de fala no VAD; fala `desconhecido` também). Silêncios e transcrição usam o **cache por clipe do `edit <pasta>`** (`_cache/edit_clips/<hash>/`, mesmos fingerprints, funções comuns em `pipeline/clips.py`); num arquivo único, também o `audio_analysis.json`/`transcript.json` do projeto do `edit <vídeo>` quando o fingerprint gravado no `project.json` é o mesmo. Um projeto já transcrito não é transcrito de novo.
* **Frases:** as do Diretor (`build_clip_sentences`: IDs globais `S000`... na ordem do catálogo, `clip_id`, mesmos limites).
* **Ritmo por frase** (`roteiro/rhythm.py`; limites são **estimativa inicial, a calibrar**, e os valores medidos ficam no JSON): **volume** = percentil 75 dos picos por janela de 50 ms (`audio/levels.py`, cache `levels-*.json`) dentro da frase, comparado com a mediana das frases do mesmo clipe: ≤ −12 dB sussurro, ≤ −5 dB baixo, ≥ +5 dB alto; **velocidade** = palavras do Whisper ÷ duração: ≤ 1,8 p/s lenta, ≥ 3,6 p/s rápida (frases com menos de 3 palavras ficam sem); **pausas** antes/depois = intervalo até a frase vizinha (ou o início/fim do clipe), ou o silêncio detectado que encosta (até 0,5 s) no início/fim da frase, se maior.
* **Fotos** (`roteiro/photos.py`): grade a cada `--intervalo` + extra no início de cada frase depois de pausa ≥ `PAUSE_PHOTO_SEC` (1 s) + extra em cada troca de cena (`scan_clip` do `visual`, com cache bruto `scan-*.json`; se o `visual` já passou pelo arquivo, a passada dele é aproveitada sem decodificar de novo); extra a menos de 0,5 s de outra foto só soma o motivo. **Um FFmpeg por clipe** (grade pelo `fps`, extras por um `select` com os tempos, tempos reais pelo `metadata=print`; até 400 extras por processo), lado maior **512 px** sem ampliar, JPEG `-q:v 5`; cache em `photos-*/`. **Descarte:** cada foto é comparada com a última mantida (miniatura cinza 16x16, diferença média 0–255); até `DUPLICATE_MAX_DIFF` (6) é descartada, salvo se a última mantida está a `MAX_KEPT_GAP_SEC` (10 s) ou mais. A primeira de cada clipe sempre fica. IDs estáveis das mantidas: `P007.F023`; tempos só no JSON; as descartadas ficam contadas por clipe (`discarded`: tempo, motivo, foto parecida e diferença).
* **Descrição** (`roteiro/describer.py`; `PhotoDescriber` sobre `ModelChain`, qualquer provedor da camada `llm/`): uma chamada por foto com a imagem, a **fala daquele momento** (palavras do Whisper da foto até a próxima mantida, até 60 palavras, ou "sem fala") e a **descrição da foto mantida anterior do mesmo clipe**. Resposta `PhotoDescription` (português): `what` (uma frase), `place`, `people`, `talking_to_camera`, `action`, `tags` (1–6) e `issue` (nenhum/borrada/tremida/obstruída/escura/clara/tela uniforme; sinônimos e caixa normalizados). Inválida → nova tentativa com o erro (até 3); persistindo, a foto fica `sem_descricao` (aviso) e é pedida de novo na próxima execução. 3 fotos seguidas com erro da API param o envio das restantes. **Cache por foto** em `_cache/roteiro_fotos/<fingerprint>/<hash da imagem>.json` (fingerprint = `VISION_PROMPT_VERSION` + cadeia pedida + raciocínio), gravado assim que a foto termina (temporário + troca). Andamento `120/640 fotos`. As fotos de um clipe vão em ordem (dependem da anterior); `--visao-paralelo` descreve vários clipes ao mesmo tempo. **Ctrl+C sai na hora** (threads daemon; a foto em andamento é abandonada, as prontas ficam no cache e a próxima execução continua de onde parou).
* **Momentos** (`roteiro/moments.py`): fotos seguidas do mesmo clipe continuam o momento se não há troca de cena entre elas, o lugar é o mesmo (quando os dois dizem) e têm ≥ `MOMENT_MIN_COMMON_TAGS` (2) tags em comum (ou todas as da que tem menos); foto sem descrição não quebra. Duração máxima `MAX_MOMENT_SEC` (20 s; um momento vai da sua 1ª foto até a 1ª do seguinte, e o 1º começa em 0). ID `P007.M04`, descrição representativa (a foto com mais tags do momento), tags somadas (até 8), pessoas, falando para a câmera, problemas da IA e das medidas do intervalo (`measure_shot` sobre as amostras do `scan_clip`; `movimento alto` relativo ao projeto, como no `visual`). **Luz só pelas medidas:** o `issue` `escura`/`clara` da IA é ignorado no momento (`AI_IGNORED_ISSUES`; no material real o modelo marcava `clara` em quase toda foto, entendendo como foto nítida/de dia); `borrada`, `tremida`, `obstruída` e `tela uniforme` da IA continuam. `escura` e `clara` nunca saem juntas: as medidas usam a luma média do momento (`escura` < 0,15, `clara` > 0,88), e se as duas marcas viessem ligadas valeria o lado da luma média (< 0,5 = `escura`). **Tags do pedido filtradas** (`PROMPT_TAGS` + IDs, só ao montar o momento; a descrição no cache não muda): os papéis do catálogo (`principal`, `drone`, `pov`, `broll`, `outro`, `b-roll`, `papel <papel>`), `papel`, `clipe`, `contexto`, `sem fala`, `com fala`, `silêncio`, `foto`, `imagem`, `foto anterior` e IDs de clipe/foto/momento (`P007`, `P007.F023`, `P007.M04`); comparação sem acentos e sem caixa. Saem antes de comparar fotos (não juntam momentos) e de somar as tags. Clipe sem fotos = um momento. Cada frase é ligada aos momentos que atravessa (`moment_ids`, e `sentence_ids` no momento).
* **`roteiro.md`:** índice (clipes, papéis, duração, momentos, frases, fotos e tamanho estimado em tokens = caracteres/4, total e por clipe), "Como ler" e, por clipe (`## P001 · principal · 1:05 · com fala · gravado ...`; data só se confiável, nunca a de modificação), os momentos (`### P001.M01 · 0:00–0:12 · descrição [problemas]`, lugar, pessoas, tags, troca de cena) e, embaixo de cada um, as frases que começam nele: `- S012 (alto, rápida, ⏸1,4s antes) — texto`. O ritmo só aparece quando foge do normal (a pausa "depois" fica só no JSON: é a "antes" da frase seguinte).

### Opção N: Editor de IA em passadas (`edit --receita editor_ia`)

Edita **qualquer material** (um vídeo = 1 clipe; uma pasta = N clipes de qualquer papel) a partir do roteiro unificado (Opção M). **O programa mede, descreve, confere e executa; a IA decide tudo e responde só com IDs** (clipes `P001`/`D001`, frases `S012`, momentos `P007.M04`). Nenhuma regra editorial no código: ele valida a segurança de cada ID e converte IDs em tempos; quando a IA falha de vez, entra um **fallback neutro** (nada de "bom gosto" do código), com aviso no terminal e no relatório. Padrão: Ollama local com `qwen2.5:7b`; `mock`, `gemini` e `openai` pela camada `llm/`.

```bat
:: Só o plano (sem render): roteiro (com a IA de visão local) + resumos + estrutura + detalhe + Timeline + relatório
python -m ai_video_editor.cli edit "%VLOG%" --receita editor_ia --so-plano --language pt --pedido "vlog da trilha, uns 8 minutos"

:: Edição completa (render + legendas), editor na nuvem, contexto maior
python -m ai_video_editor.cli edit "%VLOG%" --receita editor_ia --editor-provider gemini --editor-ctx 32768 --pedido "..."

:: Um vídeo só, sem rede (IA simulada no editor e na visão)
python -m ai_video_editor.cli edit aula.mp4 --receita editor_ia --editor-provider mock --visao-provider mock
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--receita editor_ia` | — | Liga o Editor de IA (a receita não é oferecida ao Produtor: `producer=False`; o briefing, se houver, só dá pedido, duração alvo e instruções) |
| `--pedido` | — | O que o usuário quer (vai para as três passadas); "uns 8 minutos" vira duração alvo pelo briefing |
| `--editor-provider` / `--editor-model` / `--editor-fallback` | `ollama` / `qwen2.5:7b` (o do provedor nos outros) / reservas do provedor | IA do editor (`mock` = simulada, sem rede) |
| `--editor-thinking` / `--editor-timeout` | `low` / 600 s no ollama, 120 s na nuvem | Raciocínio (o Ollama ignora) e tempo limite por pedido |
| `--editor-ctx` | `num_ctx` do Ollama (`--ollama-ctx`, padrão 8192); 32768 na nuvem | Contexto que o editor considera para dividir os pedidos |
| `--so-plano` | — | Para depois da montagem: `editor_plano.json/.md`, `timeline.json` e o relatório, sem render nem legendas |
| `--intervalo`, `--visao-provider`, `--visao-model`, `--visao-thinking`, `--visao-timeout`, `--visao-paralelo` | os do `roteiro` | Repassadas ao roteiro (Opção M) |
| `--ollama-host`, `--ollama-ctx`, `--ollama-keep-alive`, `--ollama-formato` | | Ollama local (Opção L), para o editor e a visão |
| `--encoder`, `--gpu-paralelo`, `--threshold-db`, `--language`, `--whisper-*`, `--force`, `--no-live`, `--api-retries` | os do `edit` | Render, silêncios, transcrição e cache |

Etapas (`[n/7]`): catálogo → roteiro → Editor de IA → Timeline → render → legendas → relatório. Trabalho em `<projeto>/editor_ia/`: `project.json` (`kind="editor_ia"`), `editor_run.json` (`EditorRun`: tudo o que a IA respondeu, tentativas recusadas, descartes, chamadas e tokens), `editor_plano.json` (`EditorPlan`: blocos → pedaços com IDs **e** tempos de origem e da Timeline), `editor_plano.md`, `timeline.json`, `<nome>_editor.mp4`/`.srt` e `editor_relatorio.md`. Ctrl+C a qualquer momento: cada resposta aceita já está no cache.

* **E9 — resumos por clipe** (um pedido por clipe, com a seção do clipe no `roteiro.md`): `ClipSummaryAnswer` = `resumo`, até 3 `destaques` e 3 `problemas` (`{ids, motivo}`), `tom`, `tem_fala_para_camera`. IDs que não são do clipe são retirados. Clipe que não cabe no contexto é resumido em pedaços de momentos, juntados pelo código (resumos em sequência, os 3 primeiros destaques/problemas).
* **E10 — estrutura (passada 1)**, um pedido com o pedido do usuário + os resumos + uma linha por clipe (papel, duração, fala): `StructureAnswer` = `blocos` (`id`, `titulo`, `assunto`, `clipes` em ordem, `duracao_alvo_sec`), `fora` (`{clipe, motivo}`), `duracao_total_alvo_sec`, `notas`. Clipe que não está em bloco nem em `fora` vira **acervo** (cobertura e respiros da passada 2). Se não couber, o pedido vai **compacto** (resumo cortado em 180 caracteres); se nem assim, fallback.
* **E11 — detalhe (passada 2)**, um pedido por bloco: as seções dos clipes do bloco, o fim do que já está no vídeo (2 frases, continuidade), o alvo do bloco e o tempo já usado, e o **acervo pré-selecionado** (momentos dos clipes do acervo ordenados por tags em comum com o bloco, +2 se o lugar é o mesmo; empate = ordem do catálogo) até encher o orçamento. `BlockAnswer` = `manter` (`{de, ate}` faixa de frases, ou `{momento, duracao_sec}`), `coberturas` (`{de, ate, momentos[], evidencia_fala, evidencia_imagem, motivo}`), `respiros` (`{depois_de, momento, duracao_sec, motivo}`), `notas`. Respostas tolerantes: IDs em minúsculas/com espaço, listas em texto ("S001, S002"), números com vírgula ou "3,5s".
* **Orçamento de contexto** (`editor/budget.py`): tokens = ⌈caracteres/4 × 1,25⌉ (a régua do `roteiro.md`, com folga porque o tokenizador local conta mais em português); disponível = `num_ctx` − saída da passada (resumo 600, estrutura 1500, detalhe 2000) − margem 600. No detalhe, orçamento das seções = disponível − pedido montado sem seções nem acervo − 500 (mínimo do acervo). Bloco maior que isso é dividido em pedaços em ordem (`B03a`, `B03b`...: grupos de clipes; clipe gigante, grupos de momentos), cada um com "Pedaço k de n" e a continuidade do anterior. Nada é cortado em silêncio (o provedor Ollama também recusa entrada que encosta no `num_ctx`).
* **Validação item a item** (`editor/validate.py`; o item inválido sai com o motivo, o resto fica): ID inexistente; frase ou momento fora do bloco (ou do pedaço); faixa fora de ordem ou de outro clipe; frase/momento já usado no vídeo; sobreposição com trecho já no vídeo (tolerância 0,25 s); `duracao_sec` maior que o momento (+0,05 s); respiro com momento repetido; cobertura sobre frase não mantida neste bloco, não contígua, com respiro no meio, ou com momento do próprio clipe que fala. **Evidência:** `evidencia_fala` precisa de ≥ 1 palavra de conteúdo (≥ 3 letras, fora das palavras vazias pt/es/en; sem acento e caixa; mesmo radical de ≥ 4 letras vale) nas frases cobertas; cada momento precisa de ≥ 1 palavra de `evidencia_imagem` na sua descrição/lugar/tags — momento sem evidência é retirado da cobertura (aviso); cobertura sem momento válido sai. **Recusa da resposta inteira** (nova tentativa com a lista de erros): nada válido em `manter`, ou ≥ 50% dos itens inválidos.
* **Novas tentativas e fallbacks:** até 2 pedidos por passada/bloco (o 2º com os erros). Resumo que falha = sem resumo (a estrutura segue). Estrutura que falha = um bloco por clipe com fala, na ordem do catálogo (sem fala: um bloco por clipe), sem `fora`. Bloco que falha = **todas as frases livres** dos clipes do bloco e, nos clipes sem fala, todos os momentos; sem cobertura nem respiro. Depois de 3 erros seguidos da API, a IA não é mais chamada nesta execução (o resto pelos fallbacks); erro de autenticação interrompe.
* **Cache por passada** (`<projects>/_cache/editor_ia/<resumos|estrutura|detalhe>/<fingerprint>.json`, gravação atômica, só respostas aceitas): fingerprint = pedido exato + versão do pedido da passada (`resumo-v1`, `estrutura-v1`, `detalhe-v1`) + provedor, cadeia e raciocínio. Mudar o pedido do detalhe não refaz os resumos; mudar o pedido do usuário refaz estrutura e detalhe, não os resumos.
* **E12 — montagem** (`editor/assemble.py`, pura): faixa de frases = o corte do `edit` (`plan_keep_ranges`: silêncios do clipe + as outras frases do clipe removidas, cortes dentro dos silêncios, sem pedaço de palavra), no máximo 0,5 s antes da 1ª e depois da última frase; momento = item da base com o som da imagem (duração pedida ou o momento inteiro); respiro = item depois do indicado (no meio de uma faixa, a faixa é dividida entre as duas frases); cobertura = `video_overlay` do início da 1ª frase ao fim da última, momentos dividindo o tempo na ordem dada sem passar da duração de cada um (o que falta fica sem cobertura, com aviso; nunca estica nem repete), som da imagem baixinho em `audio_ambient`. Início/fim de cobertura a menos de 0,5 s da borda do item da base vai para a borda (regra técnica: pedaço de poucos frames encadeia mal no render). 1920x1080, fps inteiro dos clipes usados, loudnorm.
* **E13 — render, legendas e relatório:** o render da Timeline (Opção K), legendas pelas transcrições dos itens da base, e o `editor_relatorio.md`: pedido, alvo e de onde veio, IA (cadeia, modelo que respondeu, trocas), contexto, estrutura (blocos, alvo × no plano, fora, acervo), por bloco o que ficou/cortou, coberturas com a evidência achada, respiros e tentativas recusadas, resumos, avisos, chamadas e tokens por passada e os tempos.
* **Andamento:** `[3/7] Editor de IA ... 12/54 pedidos (22%)`; avisos (recusa, divisão de bloco, fallback) acima da linha.

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
- **Saída fora de ordem do Whisper** (`transcription/normalize.py`, `normalize_segments`): antes de montar o `Transcript`, a saída crua (formato do faster-whisper) é normalizada, lida na ordem em que o provedor a emitiu. Segmento que começa antes do início do anterior é descartado se repete o texto dele (ignorando caixa e pontuação) e se sobrepõe; senão o início vai para o **fim** do anterior, e as palavras que ficaram antes dele saem (o texto é refeito com as que restaram). Segmento que fica sem duração, ou sem nenhuma das palavras que tinha, é descartado. Palavras de cada segmento em ordem de início e dentro dele (1 ms de tolerância no fim). IDs refeitos 0..n-1. Saída já em ordem não muda (conferido em 834 segmentos reais). O que mudou (`SegmentAdjustments`) fica em `provider.last_adjustments` e aparece no `edit` como aviso: na linha da etapa, nos avisos do fim e como "Atenção" no `edit_report.md` (em pasta, por clipe). Não entra no `Transcript` nem no fingerprint: transcrições já gravadas continuam no cache.
- **Atenção ao atualizar:** o padrão mudou de `small` para `auto`. Em máquina com GPU, projetos já transcritos serão transcritos de novo na próxima execução do `edit` (o modelo efetivo mudou e faz parte do cache), com o download único do `large-v3-turbo`. Na CPU o `auto` resolve para `small`, igual ao padrão antigo, e o cache continua válido. Para manter o cache de antes numa máquina com GPU, passe `--whisper-model small`.

#### Render na GPU (NVENC)
Os pedaços do render podem ser codificados no NVENC da placa NVIDIA (`--encoder`, no `edit` e no `render`). Decodificação e filtros continuam na CPU; só a codificação H.264 vai para a GPU.

- **`auto` (padrão):** usa o NVENC se o FFmpeg listar o `h264_nvenc` (`ffmpeg -encoders`) **e** uma codificação de teste de 1 s (`testsrc` → muxer `null`, com os mesmos parâmetros do render) funcionar (driver e placa presentes); senão, CPU. A detecção roda uma vez por execução. `cpu` não detecta nada; `nvenc` exige a GPU e falha com o motivo antes das etapas.
- **Parâmetros** (`renderer/encoders.py`): `h264_nvenc -preset p5 -tune hq -rc vbr -cq 25 -b:v 0 -profile:v high -g 250 -bf 3 -pix_fmt yuv420p`, iguais em todos os pedaços. Na CPU continua `libx264 -preset fast -crf 22 -pix_fmt yuv420p`. Como os pedaços são unidos com concat `-c copy`, um vídeo nunca mistura codificadores nem parâmetros.
- **Paralelismo:** no NVENC, no máximo `--gpu-paralelo` (padrão 3) pedaços codificam ao mesmo tempo; o processo de áudio da Timeline não conta. Na CPU, o limite de sempre (`min(8, núcleos)`).
- **Volta para a CPU:** se um pedaço falhar por causa do NVENC (sessões, memória, driver; mensagem com `nvenc`/`cuda`), o render **inteiro** é refeito na CPU, com aviso no terminal e no relatório. Outros erros (origem corrompida etc.) falham como antes.
- **Onde aparece:** a linha da etapa render (`45,10s, 102,3 MB, nvenc`), a linha "Codificador de vídeo" do `edit_report.md` e a métrica `encoder` da etapa render no `project.json`: `nvenc`, `cpu (--encoder cpu)`, `cpu (sem GPU: <motivo>)` ou `cpu (NVENC falhou: <erro>)`.
- **Cache:** na CPU o fingerprint do render é o de antes (renders antigos continuam válidos); no NVENC entram o nome e os parâmetros. Um render que caiu para a CPU fica registrado com o fingerprint da CPU, então um novo pedido de NVENC tenta a GPU de novo.
- **Testes:** a suíte não usa a GPU (o `conftest.py` faz o `auto` ver "sem GPU"); a detecção e a volta para a CPU são testadas com FFmpeg falso.

---

## 5. Estrutura dos Módulos

### Módulo Core (Fase 1)
* [`src/ai_video_editor/core/schema.py`](../src/ai_video_editor/core/schema.py): Modelos `ClipSegment`, `EditPlan` e `LoudnessTarget` (presets de normalização).
* [`src/ai_video_editor/core/timeline.py`](../src/ai_video_editor/core/timeline.py): `Timeline` com trilhas (`video_main`, `video_overlay`, `audio_ambient`; áudio principal = áudio dos itens da base), validação quantizada em frames, `timeline_from_edit_plan` e `flatten` (achatamento puro em pedaços de vídeo + plano de áudio).
* [`src/ai_video_editor/renderer/timeline_compiler.py`](../src/ai_video_editor/renderer/timeline_compiler.py) e [`timeline_executor.py`](../src/ai_video_editor/renderer/timeline_executor.py): render da Timeline (inspeção das origens, pedaços padronizados em paralelo, áudio 48 kHz com ambiente, mux, loudnorm e validação de duração/fps).
* [`src/ai_video_editor/renderer/compiler.py`](../src/ai_video_editor/renderer/compiler.py): Compilação segura de filtros FFmpeg sem shell injection.
* [`src/ai_video_editor/renderer/executor.py`](../src/ai_video_editor/renderer/executor.py): Execução isolada via `subprocess` e validação com FFprobe.
* [`src/ai_video_editor/renderer/encoders.py`](../src/ai_video_editor/renderer/encoders.py): Codificador do render (x264 ou NVENC): detecção do NVENC com cache, parâmetros, escolha `auto/cpu/nvenc` e o rótulo com o motivo.
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
* [`src/ai_video_editor/llm/registry.py`](../src/ai_video_editor/llm/registry.py): `register_provider`/`create_client` (`gemini`, `openai`, `ollama`, `mock`, com import tardio) e `default_timeout` (tempo limite padrão do provedor, `StructuredLLMClient.default_timeout_sec`; usado pela CLI quando o usuário não passa o timeout).
* [`src/ai_video_editor/llm/chain.py`](../src/ai_video_editor/llm/chain.py): `ModelChain` (troca automática de modelo) e `resolve_chain`/`parse_chain` (`provedor:modelo,...`). Modelos com `:` no nome (Ollama, `model_ids_have_colon`): `ollama:qwen2.5:7b` separa no primeiro `:`, e numa reserva sem provedor (`qwen2.5:3b` com o Diretor no Ollama) o texto inteiro é o modelo.
* [`src/ai_video_editor/llm/providers/`](../src/ai_video_editor/llm/providers): `GeminiLLMClient` (`google-genai`, `response_schema`), `OpenAILLMClient` (SDK `openai`, Responses API com `text_format` / JSON schema estrito; as novas tentativas do SDK ficam desligadas), `OllamaLLMClient` (local, `/api/chat` com `format` = JSON schema e `num_ctx` explícito; Opção L) e `MockLLMClient`. `LLMCallMetrics.notes`: observações do provedor (contexto usado, parâmetro ignorado, carga do modelo).
* [`src/ai_video_editor/llm/ollama_check.py`](../src/ai_video_editor/llm/ollama_check.py): diagnóstico do comando `ollama-check` (`run_ollama_check`, `OllamaCheckReport`).
* [`src/ai_video_editor/llm/keys.py`](../src/ai_video_editor/llm/keys.py) e [`errors.py`](../src/ai_video_editor/llm/errors.py): chaves por variável de ambiente/`.env` e utilitários de erro (remove chaves `AIza…`/`sk-…` das mensagens).

### Módulo Benchmark (comando `benchmark transcription`)
* [`src/ai_video_editor/benchmark/wer.py`](../src/ai_video_editor/benchmark/wer.py): normalização básica e WER por distância de edição entre palavras (S/D/I, substituições mais frequentes e palavras após o fim da referência), sem dependências.
* [`src/ai_video_editor/benchmark/normalize.py`](../src/ai_video_editor/benchmark/normalize.py): normalização por idioma (inglês: números, okay/ok, x's) para o WER normalizado.
* [`src/ai_video_editor/benchmark/rescore.py`](../src/ai_video_editor/benchmark/rescore.py): `benchmark rescore`, que recalcula as métricas a partir de `hypotheses/`.
* [`src/ai_video_editor/benchmark/runner.py`](../src/ai_video_editor/benchmark/runner.py): `run_benchmark` (configurações × itens, carga do modelo medida à parte, falhas registradas) e o registro extensível de provedores.
* [`src/ai_video_editor/benchmark/schema.py`](../src/ai_video_editor/benchmark/schema.py) e [`report.py`](../src/ai_video_editor/benchmark/report.py): modelos Pydantic (`BenchmarkRun`, manifesto) e o `results.md`.

### Módulo Recipes (receitas)
* [`src/ai_video_editor/recipes/schema.py`](../src/ai_video_editor/recipes/schema.py): catálogo de peças `PIECES` (etapas que o código executa e o que cada uma sempre exige), `Recipe` (validação de etapas, dependências e ordem) e `RecipeError`.
* [`src/ai_video_editor/recipes/registry.py`](../src/ai_video_editor/recipes/registry.py): `register_recipe`, `get_recipe`, `recipe_names` e as receitas `fala` (padrão de `edit <vídeo>`), `vlog` (padrão de `edit <pasta>`, `multi_clip=True`), `vlog_cobertura` e `editor_ia` (peças `roteiro` e `editor`; `multi_clip=True` e `single_file=True` = aceita pasta e vídeo; `producer=False` = fora do Produtor). A peça `roteiro` declara `provides` (silêncios e transcrição), que contam como dependências cumpridas.

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
* [`src/ai_video_editor/coverage/schema.py`](../src/ai_video_editor/coverage/schema.py): `VisualShot`/`VisualMaterial` (`coverage_shots.json`), `CoverageSentence` (com `screen` e `to_camera`), `BadSpan` (trecho com imagem ruim em frames, `uniform`), `TopicBlock`/`TopicMap` (resposta da 1ª chamada) e `Topic` (bloco validado), `CoverageContext`, a resposta da cobertura (`CoverageDecision`, `CoverageChoice` com `shot_ids` e as evidências, `BreathChoice` com `image_evidence`), `Evidence`, `AcceptedCoverage` (frames da espinha, `kind` conteudo/tela_uniforme), `ItemNote`/`DiscardedItem`/`BadSpanOutcome`, `CoverageParams` (`inherit`) e `CoverageRun` (`coverage_decision.json`, com `origin_label` e `bad_label`).
* [`src/ai_video_editor/coverage/material.py`](../src/ai_video_editor/coverage/material.py): `build_visual_material` (visual.json + catálogo, recusa outro catálogo), `coverage_sentences` (frases mantidas → frames da espinha), `annotate_screen` (tela e pessoa para a câmera), `is_severe`, `find_bad_spans` (segundos ruins + margem), `coverage_roles` e `coverage_candidates`.
* [`src/ai_video_editor/coverage/evidence.py`](../src/ai_video_editor/coverage/evidence.py): comparação de palavras (acentos, plural, `SYNONYMS`), `find_speech` (trecho literal e posição), `image_matches`, `shared_concepts` e `is_strong`.
* [`src/ai_video_editor/coverage/topics.py`](../src/ai_video_editor/coverage/topics.py): `validate_topics` (contíguos, sem buraco nem sobreposição), `topics_by_clip` (mapa do código) e `run_topics` (1ª chamada com nova tentativa).
* [`src/ai_video_editor/coverage/related.py`](../src/ai_video_editor/coverage/related.py): relação de uma candidata com uma tela (vizinha na gravação, tags em comum) e a proximidade de gravação mostrada no prompt (só com datas confiáveis).
* [`src/ai_video_editor/coverage/service.py`](../src/ai_video_editor/coverage/service.py): `CoverageState`, `review_decision` (evidências, pessoa para a câmera, segundos, respiros entre blocos), `apply_limit`, `fill_uniform`, `span_outcomes`, `ai_digest` e `run_coverage` (as duas chamadas, nova tentativa só com os problemas), com as constantes nomeadas.
* [`src/ai_video_editor/coverage/plan.py`](../src/ai_video_editor/coverage/plan.py): `plan_lines` (uma linha por bloco) e `plan_markdown` (`coverage_plan.md`).
* [`src/ai_video_editor/coverage/assemble.py`](../src/ai_video_editor/coverage/assemble.py): `assemble_coverage` (IDs e frames → tempos: respiros na base, overlays com o tempo dividido entre as tomadas, ambiente), `distribute` e `shot_segment`.
* [`src/ai_video_editor/coverage/prompt.py`](../src/ai_video_editor/coverage/prompt.py), [`provider.py`](../src/ai_video_editor/coverage/provider.py) e `providers/`: os dois prompts (`build_topic_prompt`, `build_coverage_prompt`, `COVERAGE_PROMPT_VERSION`), interface `CoverageDirector` (`map_topics` e `decide`), `LLMCoverageDirector` (cadeia de modelos) e `MockCoverageDirector`.

### Módulo Visual (comando `visual`)
* [`src/ai_video_editor/visual/builder.py`](../src/ai_video_editor/visual/builder.py): `VisualBuilder` (catálogo existente ou gerado, tomadas por clipe em paralelo com cache, IA em lotes paralelos com cache por fingerprint gravado a cada lote, estimativa, saídas).
* [`src/ai_video_editor/visual/shots.py`](../src/ai_video_editor/visual/shots.py): passada do FFmpeg (`scan_clip`: cena + `blurdetect`/`signalstats`), `split_shots`, `measure_shot`, `frame_times` e `extract_frames`, com as constantes nomeadas.
* [`src/ai_video_editor/visual/describer.py`](../src/ai_video_editor/visual/describer.py): `VisualDescriber` (sobre `ModelChain`), prompt, lotes, `validate_batch`, estimativa de tokens e `mock_visual_responder`.
* [`src/ai_video_editor/visual/schema.py`](../src/ai_video_editor/visual/schema.py) e [`report.py`](../src/ai_video_editor/visual/report.py): `ProjectVisual`, `ClipVisual`, `Shot`, `ShotFrame`, `ShotMeasures`, `ShotAnalysis`, `VisualBatchResponse`, `VisualAIRun` e o `visual.md`.

### Módulo Roteiro (comando `roteiro`)
* [`src/ai_video_editor/roteiro/builder.py`](../src/ai_video_editor/roteiro/builder.py): `RoteiroBuilder` (etapas, caches por clipe, aproveitamento do cache do `edit` e do `visual`, visão com threads daemon, estimativa, saídas) e `RoteiroParams`.
* [`src/ai_video_editor/roteiro/photos.py`](../src/ai_video_editor/roteiro/photos.py): `plan_photos`, `extract_photos` (um FFmpeg por clipe), `select_photos` (descarte) e as constantes.
* [`src/ai_video_editor/roteiro/rhythm.py`](../src/ai_video_editor/roteiro/rhythm.py): `measure_rhythm` (volume, velocidade, pausas) e os limites.
* [`src/ai_video_editor/roteiro/describer.py`](../src/ai_video_editor/roteiro/describer.py): `PhotoDescriber`, prompt (`VISION_PROMPT_VERSION`), cache por foto, estimativa e `mock_vision_responder` (descreve pela cor média).
* [`src/ai_video_editor/roteiro/moments.py`](../src/ai_video_editor/roteiro/moments.py): `build_moments`, `link_sentences`, `finalize_issues`.
* [`src/ai_video_editor/roteiro/schema.py`](../src/ai_video_editor/roteiro/schema.py) e [`report.py`](../src/ai_video_editor/roteiro/report.py): `ProjectRoteiro`, `RoteiroClip`, `RoteiroPhoto`, `Moment`, `RoteiroSentence`/`SentenceRhythm`, `PhotoDescription`, `VisionRun`, `RoteiroEstimate` e o `roteiro.md` (com a estimativa de tokens).

### Módulo Editor (receita `editor_ia`)
* [`src/ai_video_editor/editor/schema.py`](../src/ai_video_editor/editor/schema.py): respostas tolerantes da IA (`ClipSummaryAnswer`, `StructureAnswer`, `BlockAnswer` e itens) e registros (`ClipSummary`, `StructureRun`, `DetailRun`, `KeepRange`, `CoverageChoice`, `BreathChoice`, `Discarded`, `CallStats`, `BudgetInfo`, `EditorRun`, `EditorPlan`/`PlanBlock`/`PlanPiece`/`PlanCoverage`).
* [`src/ai_video_editor/editor/material.py`](../src/ai_video_editor/editor/material.py): `EditorMaterial` (índices do roteiro, seção de um clipe, linhas compactas), evidência (`matched_words`) e `library_ranking`.
* [`src/ai_video_editor/editor/budget.py`](../src/ai_video_editor/editor/budget.py): `tokens`, `Budget`, `resolve_budget` e as constantes.
* [`src/ai_video_editor/editor/prompt.py`](../src/ai_video_editor/editor/prompt.py): pedidos das três passadas (versões, `EditorContext`, regras editoriais escritas para a IA) e a lista de erros da nova tentativa.
* [`src/ai_video_editor/editor/provider.py`](../src/ai_video_editor/editor/provider.py) e [`providers/`](../src/ai_video_editor/editor/providers/): `EditorAI`, `LLMEditorAI` (pela `ModelChain`) e `MockEditorAI` (determinístico; respostas programáveis por passada).
* [`src/ai_video_editor/editor/validate.py`](../src/ai_video_editor/editor/validate.py): `validate_structure`, `validate_detail` (item a item, `Usage`, `Unit`), `neutral_structure`, `neutral_detail`.
* [`src/ai_video_editor/editor/service.py`](../src/ai_video_editor/editor/service.py): `EditorService` (passadas, divisão pelo orçamento, novas tentativas, fallbacks) e `CallCache`.
* [`src/ai_video_editor/editor/assemble.py`](../src/ai_video_editor/editor/assemble.py) e [`report.py`](../src/ai_video_editor/editor/report.py): IDs → `Timeline` + `EditorPlan`, fontes das legendas, `editor_plano.md` e `editor_relatorio.md`.
* [`src/ai_video_editor/pipeline/editor_ia.py`](../src/ai_video_editor/pipeline/editor_ia.py): `EditorPipeline` (etapas da receita, caches, `--so-plano`), `EditorParams`, `EditorManifest`, `EditorResult`.

### Módulo Pipeline (comandos `edit` e `revise`)
* [`src/ai_video_editor/pipeline/runner.py`](../src/ai_video_editor/pipeline/runner.py): `EditPipeline`, que executa as etapas da receita (uma por método, na ordem de `recipe.steps`) com cache por fingerprint e callbacks de eventos (`on_progress`) e de andamento da etapa em curso (`on_stage_progress`, com `StageProgress`); `TranscriptionProvider`, `EditDirector` e `RenderExecutor` são injetáveis. O transcriber (`transcribe(..., on_progress)`) e o executor (`execute(plan, on_progress)`) informam o progresso; o pipeline só repassa, sem depender do terminal.
* [`src/ai_video_editor/live_status.py`](../src/ai_video_editor/live_status.py): `LiveStatus`, a linha de andamento ao vivo da CLI (thread de atualização a cada 1s, avisos acima da linha, modo sem terminal só com linhas finais).
* [`src/ai_video_editor/pipeline/schema.py`](../src/ai_video_editor/pipeline/schema.py): `EditPipelineParams` (com `pacing` e `effective_cut`), `EditSettings`/`EditSetting` (valores efetivos e origem), `ProjectManifest` (`project.json`), `StageRecord`, `RunTimings`/`StageTiming`, `EditPipelineResult`.
* [`src/ai_video_editor/pipeline/project.py`](../src/ai_video_editor/pipeline/project.py): Hash curto do vídeo, nome da pasta do projeto, nomes dos artefatos e fingerprints.
* [`src/ai_video_editor/pipeline/timing.py`](../src/ai_video_editor/pipeline/timing.py): `TimingRecorder` (medição com `time.perf_counter`), formatação (`8,4s`, `12min 34s`) e a seção "Tempo de processamento".
* [`src/ai_video_editor/pipeline/report.py`](../src/ai_video_editor/pipeline/report.py): Geração do `edit_report.md` (e da seção "Revisão").
* [`src/ai_video_editor/pipeline/versions.py`](../src/ai_video_editor/pipeline/versions.py): versões em `versions/vN/`, registro após o `edit` e migração de projetos antigos.
* [`src/ai_video_editor/pipeline/revise.py`](../src/ai_video_editor/pipeline/revise.py): `ProjectReviser` (revisor → plano → render → legendas → relatório numa versão nova).
* [`src/ai_video_editor/pipeline/cutting.py`](../src/ai_video_editor/pipeline/cutting.py): `plan_keep_ranges`, compartilhado por `edit` e `revise`.
* [`src/ai_video_editor/pipeline/edit_diff.py`](../src/ai_video_editor/pipeline/edit_diff.py): status de aplicação das remoções e transcrição comparativa (`build_edit_diff` → `EditDiff`/`DiffEntry`, `edit_transcript.md` e `edit_diff.json`).
* [`src/ai_video_editor/pipeline/edit_timeline.py`](../src/ai_video_editor/pipeline/edit_timeline.py): linha do tempo dos cortes (`render_edit_timeline` → `edit_timeline.svg`; `timeline_from_diff_file` desenha a partir de um `edit_diff.json` existente).
* [`src/ai_video_editor/pipeline/briefing.py`](../src/ai_video_editor/pipeline/briefing.py): briefing no `edit`: `prepare_briefing` (reaproveita o `briefing.json` atual — mesmo pedido e material — ou roda o Produtor e salva), `resolve_settings` (CLI > briefing > padrão → `EditSettings` com a origem de cada valor) e `briefing_location`.
* [`src/ai_video_editor/pipeline/spine.py`](../src/ai_video_editor/pipeline/spine.py): `SpinePipeline` (`edit <pasta>`, receitas `vlog` e `vlog_cobertura`): herda do `EditPipeline` o Diretor, a transcrição e o progresso; etapas por clipe com cache por clipe; etapas `visual` e `coverage` (Diretor de cobertura injetável, `run(..., coverage=CoverageParams, plan_only=...)`; `plan_only` = `--so-plano`, para depois da cobertura); render pela Timeline final (`TimelineRenderExecutor` injetável).
* [`src/ai_video_editor/pipeline/clips.py`](../src/ai_video_editor/pipeline/clips.py): localização/geração do catálogo (só leitura de `catalog/`), escolha dos clipes da espinha (`select_spine_clips`, mínimos de fala) e arquivos do cache por clipe. Também os silêncios e a transcrição por clipe comuns ao `edit <pasta>` e ao `roteiro` (`silences_fingerprint`, `load_or_detect_silences`, `transcription_fingerprint`, `transcribe_to_cache`, ajustes da normalização): mesmos fingerprints de antes.
* [`src/ai_video_editor/pipeline/spine_schema.py`](../src/ai_video_editor/pipeline/spine_schema.py): `SpineSelection`/`SpineClip`/`UnusedClip`, `SpinePlan`/`ClipPlan`, `SpineDiff`/`ClipDiff`, `SpineManifest` (`kind="multiclip"`) e `SpineResult`.
* [`src/ai_video_editor/pipeline/spine_report.py`](../src/ai_video_editor/pipeline/spine_report.py): diff por clipe, `edit_transcript.md` por clipe, `edit_timeline.svg` colorido por clipe e `edit_report.md` da espinha.
* [`src/ai_video_editor/planning/timeline_plan.py`](../src/ai_video_editor/planning/timeline_plan.py): trechos por clipe → `Timeline` (`build_spine_timeline`, `spine_fps`, `clip_spans`) e a posição de cada trecho de fala na Timeline, com respiros no meio (`clip_placements`, `map_source_range`, `concat_to_final`, `is_breath_item`).
* [`src/ai_video_editor/subtitles/clips.py`](../src/ai_video_editor/subtitles/clips.py): legendas de vários clipes deslocadas para o tempo final.
