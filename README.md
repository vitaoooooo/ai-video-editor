[![Tests](https://github.com/vitaoooooo/ai-video-editor/actions/workflows/tests.yml/badge.svg)](https://github.com/vitaoooooo/ai-video-editor/actions/workflows/tests.yml)

# AI Video Editor — Core Editing, Ingestion & Multimodal AI Analysis Engine

Motor de edição, ingestão técnica e análise visual com IA multimodal para o **AI Video Editor**, baseado em schemas Pydantic, FFmpeg determinístico e arquitetura multi-provedor (Google Gemini, Mock).

📜 [Histórico de decisões do projeto](docs/HISTORICO.md)

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

# 3. Instalar o pacote e dependências
pip install -e .
pip install pytest
```

---

## 3. Executando os Testes Automatizados

O projeto inclui suíte completa de testes unitários e de integração utilizando **vídeos sintéticos** (FFmpeg) e **provedor Mock** para IA, sem necessidade de chamadas de rede externas ou chaves reais de API:

```bash
.\.venv\Scripts\python.exe -m pytest -v
```

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

### Opção D: Renderização a partir de um `EditPlan` JSON (`render`)
Renderiza um plano de edição completo a partir de um arquivo JSON estruturado:

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli render output/sample_plan.json
```

### Opção E: Remover as pausas de um vídeo falado (`edit`)
Fluxo completo: metadados → silêncios → transcrição (opcional) → Diretor de IA (opcional) → plano de corte → render → legendas (com transcrição) → relatório. Sem `--director`, é só corte de silêncio, sem IA. A resolução e o FPS da saída são os do vídeo de origem (FPS arredondado para inteiro).

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4"

# Ajustando a detecção e as folgas, e gerando a transcrição (requer pip install -e ".[transcription]")
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4" `
  --threshold-db -40 --min-silence-sec 0.6 `
  --padding-before 0.15 --padding-after 0.25 --min-gap 0.3 --min-keep 0.1 `
  --transcribe --language pt --whisper-model small

# Com o Diretor de IA removendo frases repetidas, recomeçadas e enrolação (requer GEMINI_API_KEY)
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4" `
  --transcribe --language pt --director-provider gemini --intensity light

# Diretor na OpenAI (pip install -e ".[openai]" e OPENAI_API_KEY), com o Gemini de reserva
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit "C:\videos\aula.mp4" `
  --transcribe --language pt --director-provider openai --director-fallback "gemini:gemini-3.8-flash"
```

| Opção | Padrão | Efeito |
| --- | --- | --- |
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
| `--instructions` | — | Pedido livre ao Diretor (ex: `"corte as piadas"`) |
| `--keep-fillers` | desligado | Impede o Diretor de remover frases só por serem enrolação |
| `--no-subtitles` | desligado | Não gera o `.srt` (por padrão ele é gerado quando há `--transcribe`) |
| `--subtitle-max-chars` | 42 | Máximo de caracteres por linha da legenda |
| `--force` | desligado | Refaz todas as etapas, ignorando o cache |
| `--projects-dir` | `output/projects` | Diretório base dos projetos |

Cada vídeo ganha uma pasta `output/projects/<nome-do-video>-<hash>/` com:

* `project.json`: manifesto com caminho do vídeo, parâmetros e status/fingerprint/métricas de cada etapa.
* `probe.json`, `audio_analysis.json`, `transcript.json` (se pedido), `director_decision.json` (se pedido), `edit_plan.json`.
* `edit_report.md`: duração original e final, % removido, número de cortes, parâmetros, a lista dos trechos removidos e, com o Diretor, a seção "Decisões do Diretor" (tempo, texto, categoria e motivo de cada frase removida, latência e tokens).
* **Tempo de processamento**: cada etapa mostra o tempo de relógio no terminal (`[3/8] Transcrição: ok — 49 frases (12min 34s)`; etapas em cache aparecem como `em cache`, sem tempo) e, ao final, `57min de vídeo processados em 21min (0,37× o tempo real)`. Os tempos ficam em `last_run` no `project.json`, em `timings` no `version.json` e na seção "Tempo de processamento" do `edit_report.md` (o `revise` também mede suas etapas; probe, silêncios e transcrição contam como cache).
* `<nome-do-video>_edited.mp4`: o vídeo final.
* `<nome-do-video>_edited.srt`: legenda sincronizada com o vídeo editado (só com `--transcribe`), ao lado do vídeo e com o mesmo nome base, então players como o VLC a carregam sozinhos. Usa as palavras da transcrição, os mesmos trechos e o FPS do plano final (já com os cortes do Diretor). Sem transcrição, ou com `--no-subtitles`, a etapa "Legendas" aparece como pulada e um `.srt` antigo é removido da pasta. O relatório cita o arquivo e o número de blocos.

**Diretor de IA:** a transcrição é agrupada em frases (segmentos unidos até `.`, `?`, `!` ou uma pausa maior que 1s; se passar de 12s ou 40 palavras, a frase fecha no melhor ponto: fim de segmento com vírgula/`;`/`:`, depois a maior pausa, depois qualquer fim de segmento, sem nunca dividir um segmento do Whisper. Os limites são `EditPipelineParams.sentences` e entram no fingerprint do Diretor; o relatório mostra quantidade, duração média e máxima das frases). O Diretor recebe as frases numeradas e o briefing e devolve só os IDs das frases a remover, com categoria (`repetition`, `false_start`, `filler`, `off_topic`, `other`) e motivo em português; nunca timestamps. A decisão é validada (IDs existentes e únicos; no máximo 40% do tempo de fala, ou mais se `--target-duration` exigir). Se for inválida, o modelo recebe o erro e tenta mais uma vez; se falhar de novo, o vídeo sai só com o corte de silêncio e o relatório traz um aviso. Os limites de cada corte vão para dentro do silêncio detectado mais próximo (até 0,5s); sem silêncio por perto, usa-se o limite da frase sem invadir o padding das frases mantidas.

**Limite de silêncio automático:** com `--threshold-db auto` (padrão), o áudio é medido em janelas de 50 ms (FFmpeg `astats`, pico por janela). O **piso de ruído** é o percentil 10 dos níveis e o **nível de fala** o percentil 90; o limite é `piso + 35% da distância até a fala`, nunca acima de `fala - 6 dB` e dentro de `[-60, -20] dB`. Isso faz gravações com ruído de fundo (plateia, sala) detectarem pausas que o `-35 dB` fixo nunca via. Modo, piso, fala e limite escolhido ficam no `audio_analysis.json` e no relatório. Quando o ruído está muito próximo da fala, o áudio não tem pausas ou é quase todo silêncio, o relatório traz um aviso de detecção pouco confiável (nesses casos o limite fica abaixo do ruído e poucos cortes são feitos, ou todo o áudio vira silêncio, respectivamente).

**Troca automática de modelo:** o Diretor (e o revisor) usa uma cadeia ordenada `provedor:modelo`; o padrão é `gemini:gemini-3.8-flash` → `gemini:gemini-3.6-flash` → `gemini:gemini-3.5-flash-lite`. Se o modelo atual continuar falhando depois das novas tentativas com erro temporário (503, 504, 429 por minuto, conexão) ou com a cota diária/de cobrança esgotada, a etapa passa para o próximo da cadeia e avisa no terminal (`[4/8] Diretor: gemini-3.8-flash indisponível (503) após 5 tentativas → trocando para gemini-3.6-flash`). Erros de autenticação ou de requisição inválida (400/404) não trocam de modelo. Provedores sem chave configurada são pulados com um aviso. Depois de uma troca, as chamadas seguintes da mesma execução continuam no modelo que funcionou. O relatório ("Cadeia de modelos", "Modelo usado", "Troca de modelo") e o `director_decision.json` (`model_chain`, `effective_provider`, `effective_model_id`, `model_switches`) registram o pedido, o usado e cada troca com o motivo. O cache do Diretor usa a cadeia **pedida**: uma troca de modelo durante a execução não muda o fingerprint.

**Cache:** cada etapa é pulada se o artefato existe e os parâmetros que a afetam não mudaram. Mudar `--threshold-db` (ou alternar entre `auto` e um número) refaz silêncios, plano, render e relatório; mudar só um padding refaz a partir do plano; sem Diretor, a transcrição nunca força novo render. As legendas dependem do plano, da transcrição e de `--subtitle-max-chars`: mudar só esse limite refaz apenas legendas e relatório, sem novo render. O Diretor é refeito quando mudam a transcrição, o briefing ou o modelo. O hash do vídeo usa o tamanho e blocos do início, meio e fim do arquivo, então é rápido mesmo para arquivos grandes.


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
| `--api-retries` | 4 | Novas tentativas em falhas temporárias da API (`0` desliga); mesma regra do `edit` |

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
    v1/  director_decision.json, edit_plan.json, <video>_edited.mp4, <video>_edited.srt, edit_report.md, version.json
    v2/  ...                                                       # version.json guarda o pedido, o resumo e as métricas
```

O `edit_report.md` de uma revisão traz a seção "Revisão": pedido, ritmo das pausas, resumo e uma tabela de mudanças (frase, texto, antes, depois, motivo). Rodar o `edit` de novo com os mesmos parâmetros não cria versão; com parâmetros que mudam o resultado, cria uma nova versão atual.

### Usando a GPU
A transcrição usa a GPU NVIDIA automaticamente quando ela existe e as bibliotecas CUDA estão instaladas (`--whisper-device auto`, o padrão); senão, volta para a CPU com um aviso. Para instalar as bibliotecas (cuBLAS para CUDA 12 e cuDNN 9, compatíveis com o ctranslate2 4.x do faster-whisper) no Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -e ".[gpu]"
```

Requer driver NVIDIA com suporte a CUDA 12. As pastas `bin` desses pacotes são registradas pelo próprio programa (não precisa mexer no PATH).

- `--whisper-device {auto,cpu,cuda}`: `cuda` falha com instruções se a GPU não puder ser usada; `cpu` ignora a GPU.
- `--whisper-compute-type`: `auto` = `float16` na GPU e `int8` na CPU; aceita valores explícitos (ex.: `int8_float16`).
- O dispositivo e o compute_type efetivos aparecem no `transcript.json` (`model_used`, ex.: `faster-whisper:small (cuda, float16)`), no relatório e entram no cache da transcrição.

---

## 5. Estrutura dos Módulos

### Módulo Core (Fase 1)
* [`src/ai_video_editor/core/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/core/schema.py): Modelos `ClipSegment` e `EditPlan`.
* [`src/ai_video_editor/renderer/compiler.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/renderer/compiler.py): Compilação segura de filtros FFmpeg sem shell injection.
* [`src/ai_video_editor/renderer/executor.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/renderer/executor.py): Execução isolada via `subprocess` e validação com FFprobe.
* [`src/ai_video_editor/renderer/ffmpeg_locator.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/renderer/ffmpeg_locator.py): Descoberta resiliente de executáveis FFmpeg/FFprobe.

### Módulo Analyzer (Fase 2)
* [`src/ai_video_editor/analyzer/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/analyzer/schema.py): Schemas Pydantic (`VideoMetadata`, `VideoStreamInfo`, `AudioStreamInfo`, `ExtractedFrame`, `VideoAnalysisResult`, `BatchAnalysisResult`).
* [`src/ai_video_editor/analyzer/exceptions.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/analyzer/exceptions.py): Exceções de domínio (`VideoNotFoundError`, `InvalidMediaFormatError`, `NoVideoStreamError`, `FrameExtractionError`).
* [`src/ai_video_editor/analyzer/probe.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/analyzer/probe.py): Extração de metadados e conversão segura de FPS fracionário via `fractions.Fraction`.
* [`src/ai_video_editor/analyzer/extractor.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/analyzer/extractor.py): Extração determinística de frames com timestamps rastreáveis.
* [`src/ai_video_editor/analyzer/service.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/analyzer/service.py): Orquestrador de análise técnica em lote.

### Módulo Intelligence (Fase 3)
* [`src/ai_video_editor/intelligence/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/schema.py): Modelos Pydantic (`VisualObservation`, `SceneTransitionHypothesis`, `VisualAnalysisResponse`, `AIAnalysisConfig`, `CompleteAIReport`).
* [`src/ai_video_editor/intelligence/exceptions.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/exceptions.py): Exceções padronizadas de IA (`AIAuthenticationError`, `AIRateLimitError`, `AITimeoutError`, `AIUnavailableError`, `AIResponseValidationError`).
* [`src/ai_video_editor/intelligence/retry.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/retry.py): `RetryPolicy` e `call_with_retry` (nova tentativa com espera exponencial, jitter e `Retry-After`; `sleep` injetável).
* [`src/ai_video_editor/intelligence/provider.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/provider.py): Interface abstrata `MultimodalAIProvider`.
* [`src/ai_video_editor/intelligence/image_proxy.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/image_proxy.py): Otimizador e downscaler de imagens 4K para proxies leves (JPEG / 1024px) com Pillow.
* [`src/ai_video_editor/intelligence/reporter.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/reporter.py): Compilador de relatório Markdown rico com caminhos relativos e ressalvas anti-alucinação.
* [`src/ai_video_editor/intelligence/providers/gemini.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/providers/gemini.py): Adaptador oficial do Google Gemini via SDK `google-genai`.
* [`src/ai_video_editor/intelligence/providers/mock.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/providers/mock.py): Provedor simulado para testes offline rápidos e determinísticos.
* [`src/ai_video_editor/intelligence/service.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/service.py): Orquestrador de análise multimodal e geração de relatórios.

### Módulo Planning (corte de silêncio)
* [`src/ai_video_editor/planning/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/schema.py): `SilenceCutParams` (paddings, `min_gap_to_cut_sec`, `min_keep_sec`), `TimeRange` e os presets de pausa (`PausePreset`, `apply_pause_preset`).
* [`src/ai_video_editor/planning/silence_cut.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/silence_cut.py): `plan_silence_cuts` (trechos a manter a partir da `AudioAnalysisResult`), `removed_ranges` e `build_edit_plan` (lista de trechos → `EditPlan` na resolução/FPS da origem).
* [`src/ai_video_editor/planning/sentence_cut.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/sentence_cut.py): `apply_sentence_cuts` (subtrai as frases removidas pelo Diretor dos trechos mantidos), `snap_cut` (leva os limites do corte para dentro dos silêncios) e `subtract_ranges`.
* [`src/ai_video_editor/planning/exceptions.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/exceptions.py): `PlanningError`, `NoSpeechDetectedError`.

### Módulo Director (Diretor de IA)
* [`src/ai_video_editor/director/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/schema.py): `Sentence`, `UserBrief`, `DirectorContext`, `SentenceRemoval`, `DirectorDecision` e `DirectorRun` (conteúdo do `director_decision.json`).
* [`src/ai_video_editor/director/sentences.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/sentences.py): `build_sentences` (transcrição → frases) e `split_by_removal` (frases → intervalos a cortar e a proteger).
* [`src/ai_video_editor/director/provider.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/provider.py): Interface `EditDirector` (`decide(sentences, brief, context)`).
* [`src/ai_video_editor/director/service.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/service.py): `validate_decision` e `run_director` (nova tentativa com o erro, fallback e métricas `AIRunMetrics`).
* [`src/ai_video_editor/director/prompt.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/prompt.py): `build_director_prompt` (independente do provedor).
* [`src/ai_video_editor/director/providers/llm.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/providers/llm.py): `LLMDirector` (pede `DirectorDecision` a uma `ModelChain`); `providers/gemini.py` tem o atalho `GeminiDirector` (um único modelo Gemini).
* [`src/ai_video_editor/director/providers/mock.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/providers/mock.py): `MockDirector` determinístico (remoções fixas, sequência de respostas ou frases repetidas).

### Módulo Revision (Revisor de IA)
* [`src/ai_video_editor/revision/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/revision/schema.py): `SentenceState`, `RevisionContext`, `RevisionDecision` (conjunto completo de remoções, restaurações com motivo, ritmo, `feasible`/`problem`) e `RevisionRun`.
* [`src/ai_video_editor/revision/provider.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/revision/provider.py): Interface `EditReviser` (`revise(states, brief, context)`).
* [`src/ai_video_editor/revision/service.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/revision/service.py): `validate_revision` (regras do Diretor + restaurações) e `run_reviser` (nova tentativa; sem versão se falhar).
* [`src/ai_video_editor/revision/prompt.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/revision/prompt.py): `build_reviser_prompt`.
* [`src/ai_video_editor/revision/providers/`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/revision/providers): `LLMReviser` (cadeia de modelos, como o Diretor), o atalho `GeminiReviser` e `MockReviser` determinístico.

### Módulo LLM (provedores de IA com saída estruturada)
* [`src/ai_video_editor/llm/client.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/llm/client.py): interface `StructuredLLMClient` (`generate(prompt, schema, config, retry)` → objeto validado + `LLMCallMetrics`), com novas tentativas, subida do nível de raciocínio e métricas já implementadas.
* [`src/ai_video_editor/llm/registry.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/llm/registry.py): `register_provider`/`create_client` (`gemini`, `openai`, `mock`, com import tardio).
* [`src/ai_video_editor/llm/chain.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/llm/chain.py): `ModelChain` (troca automática de modelo) e `resolve_chain`/`parse_chain` (`provedor:modelo,...`).
* [`src/ai_video_editor/llm/providers/`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/llm/providers): `GeminiLLMClient` (`google-genai`, `response_schema`), `OpenAILLMClient` (SDK `openai`, Responses API com `text_format` / JSON schema estrito; as novas tentativas do SDK ficam desligadas) e `MockLLMClient`.
* [`src/ai_video_editor/llm/keys.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/llm/keys.py) e [`errors.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/llm/errors.py): chaves por variável de ambiente/`.env` e utilitários de erro (remove chaves `AIza…`/`sk-…` das mensagens).

### Módulo Pipeline (comandos `edit` e `revise`)
* [`src/ai_video_editor/pipeline/runner.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/runner.py): `EditPipeline`, que orquestra as etapas com cache por fingerprint e callback de progresso; `TranscriptionProvider`, `EditDirector` e `RenderExecutor` são injetáveis.
* [`src/ai_video_editor/pipeline/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/schema.py): `EditPipelineParams`, `ProjectManifest` (`project.json`), `StageRecord`, `RunTimings`/`StageTiming`, `EditPipelineResult`.
* [`src/ai_video_editor/pipeline/project.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/project.py): Hash curto do vídeo, nome da pasta do projeto, nomes dos artefatos e fingerprints.
* [`src/ai_video_editor/pipeline/timing.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/timing.py): `TimingRecorder` (medição com `time.perf_counter`), formatação (`8,4s`, `12min 34s`) e a seção "Tempo de processamento".
* [`src/ai_video_editor/pipeline/report.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/report.py): Geração do `edit_report.md` (e da seção "Revisão").
* [`src/ai_video_editor/pipeline/versions.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/versions.py): versões em `versions/vN/`, registro após o `edit` e migração de projetos antigos.
* [`src/ai_video_editor/pipeline/revise.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/revise.py): `ProjectReviser` (revisor → plano → render → legendas → relatório numa versão nova).
* [`src/ai_video_editor/pipeline/cutting.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/cutting.py): `plan_keep_ranges`, compartilhado por `edit` e `revise`.
