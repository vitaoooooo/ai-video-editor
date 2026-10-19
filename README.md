[![Tests](https://github.com/vitaoooooo/ai-video-editor/actions/workflows/tests.yml/badge.svg)](https://github.com/vitaoooooo/ai-video-editor/actions/workflows/tests.yml)

# AI Video Editor — Core Editing, Ingestion & Multimodal AI Analysis Engine

Motor de edição, ingestão técnica e análise visual com IA multimodal para o **AI Video Editor**, baseado em schemas Pydantic, FFmpeg determinístico e arquitetura multi-provedor (Google Gemini, Mock).

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
```

#### 2. Executando com o Google Gemini
```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli report-ai `
  --report output/analysis/<id_da_sessao>/analysis_report.json `
  --provider gemini `
  --model gemini-2.5-flash `
  --max-images 15 `
  --proxy-dim 1024
```

> **Modelos Gemini Compatíveis:**
> Os modelos recomendados suportam entrada multimodal e saídas estruturadas em JSON (ex: `gemini-2.5-flash` para custo/velocidade, ou `gemini-2.5-pro` para raciocínio visual aprofundado). Consulte os modelos multimodais atualizados na [documentação oficial do Google AI Studio](https://ai.google.dev/gemini-api/docs/models/gemini).

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
  --transcribe --language pt --director gemini --intensity light
```

| Opção | Padrão | Efeito |
| --- | --- | --- |
| `--threshold-db` | -35 | Volume abaixo do qual o áudio é considerado silêncio |
| `--min-silence-sec` | 0.5 | Duração mínima de um silêncio detectado |
| `--padding-before` / `--padding-after` | 0.15 / 0.20 | Folga mantida antes/depois de cada fala |
| `--min-gap` | 0.30 | Pausas menores que isso (após o padding) não são cortadas, inclusive no início/fim |
| `--min-keep` | 0.10 | Falas mais curtas que isso são tratadas como ruído |
| `--transcribe` | desligado | Gera `transcript.json` (necessário para o Diretor) |
| `--director` | `none` | Diretor de IA: `none`, `mock` (remove frases repetidas, sem rede) ou `gemini`. Só roda com `--transcribe` |
| `--director-model` | `gemini-2.5-flash` | Modelo do Diretor Gemini |
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
* `<nome-do-video>_edited.mp4`: o vídeo final.
* `<nome-do-video>_edited.srt`: legenda sincronizada com o vídeo editado (só com `--transcribe`), ao lado do vídeo e com o mesmo nome base, então players como o VLC a carregam sozinhos. Usa as palavras da transcrição, os mesmos trechos e o FPS do plano final (já com os cortes do Diretor). Sem transcrição, ou com `--no-subtitles`, a etapa "Legendas" aparece como pulada e um `.srt` antigo é removido da pasta. O relatório cita o arquivo e o número de blocos.

**Diretor de IA:** a transcrição é agrupada em frases (segmentos unidos até `.`, `?`, `!` ou uma pausa maior que 1s). O Diretor recebe as frases numeradas e o briefing e devolve só os IDs das frases a remover, com categoria (`repetition`, `false_start`, `filler`, `off_topic`, `other`) e motivo em português; nunca timestamps. A decisão é validada (IDs existentes e únicos; no máximo 40% do tempo de fala, ou mais se `--target-duration` exigir). Se for inválida, o modelo recebe o erro e tenta mais uma vez; se falhar de novo, o vídeo sai só com o corte de silêncio e o relatório traz um aviso. Os limites de cada corte vão para dentro do silêncio detectado mais próximo (até 0,5s); sem silêncio por perto, usa-se o limite da frase sem invadir o padding das frases mantidas.

**Cache:** cada etapa é pulada se o artefato existe e os parâmetros que a afetam não mudaram. Mudar `--threshold-db` refaz silêncios, plano, render e relatório; mudar só um padding refaz a partir do plano; sem Diretor, a transcrição nunca força novo render. As legendas dependem do plano, da transcrição e de `--subtitle-max-chars`: mudar só esse limite refaz apenas legendas e relatório, sem novo render. O Diretor é refeito quando mudam a transcrição, o briefing ou o modelo. O hash do vídeo usa o tamanho e blocos do início, meio e fim do arquivo, então é rápido mesmo para arquivos grandes.

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
* [`src/ai_video_editor/intelligence/provider.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/provider.py): Interface abstrata `MultimodalAIProvider`.
* [`src/ai_video_editor/intelligence/image_proxy.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/image_proxy.py): Otimizador e downscaler de imagens 4K para proxies leves (JPEG / 1024px) com Pillow.
* [`src/ai_video_editor/intelligence/reporter.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/reporter.py): Compilador de relatório Markdown rico com caminhos relativos e ressalvas anti-alucinação.
* [`src/ai_video_editor/intelligence/providers/gemini.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/providers/gemini.py): Adaptador oficial do Google Gemini via SDK `google-genai`.
* [`src/ai_video_editor/intelligence/providers/mock.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/providers/mock.py): Provedor simulado para testes offline rápidos e determinísticos.
* [`src/ai_video_editor/intelligence/service.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/intelligence/service.py): Orquestrador de análise multimodal e geração de relatórios.

### Módulo Planning (corte de silêncio)
* [`src/ai_video_editor/planning/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/schema.py): `SilenceCutParams` (paddings, `min_gap_to_cut_sec`, `min_keep_sec`) e `TimeRange`.
* [`src/ai_video_editor/planning/silence_cut.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/silence_cut.py): `plan_silence_cuts` (trechos a manter a partir da `AudioAnalysisResult`), `removed_ranges` e `build_edit_plan` (lista de trechos → `EditPlan` na resolução/FPS da origem).
* [`src/ai_video_editor/planning/sentence_cut.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/sentence_cut.py): `apply_sentence_cuts` (subtrai as frases removidas pelo Diretor dos trechos mantidos), `snap_cut` (leva os limites do corte para dentro dos silêncios) e `subtract_ranges`.
* [`src/ai_video_editor/planning/exceptions.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/planning/exceptions.py): `PlanningError`, `NoSpeechDetectedError`.

### Módulo Director (Diretor de IA)
* [`src/ai_video_editor/director/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/schema.py): `Sentence`, `UserBrief`, `DirectorContext`, `SentenceRemoval`, `DirectorDecision` e `DirectorRun` (conteúdo do `director_decision.json`).
* [`src/ai_video_editor/director/sentences.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/sentences.py): `build_sentences` (transcrição → frases) e `split_by_removal` (frases → intervalos a cortar e a proteger).
* [`src/ai_video_editor/director/provider.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/provider.py): Interface `EditDirector` (`decide(sentences, brief, context)`).
* [`src/ai_video_editor/director/service.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/service.py): `validate_decision` e `run_director` (nova tentativa com o erro, fallback e métricas `AIRunMetrics`).
* [`src/ai_video_editor/director/providers/gemini.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/providers/gemini.py): `GeminiDirector` (texto, `response_schema=DirectorDecision`) e `build_director_prompt`.
* [`src/ai_video_editor/director/providers/mock.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/director/providers/mock.py): `MockDirector` determinístico (remoções fixas, sequência de respostas ou frases repetidas).

### Módulo Pipeline (comando `edit`)
* [`src/ai_video_editor/pipeline/runner.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/runner.py): `EditPipeline`, que orquestra as etapas com cache por fingerprint e callback de progresso; `TranscriptionProvider`, `EditDirector` e `RenderExecutor` são injetáveis.
* [`src/ai_video_editor/pipeline/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/schema.py): `EditPipelineParams`, `ProjectManifest` (`project.json`), `StageRecord`, `EditPipelineResult`.
* [`src/ai_video_editor/pipeline/project.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/project.py): Hash curto do vídeo, nome da pasta do projeto, nomes dos artefatos e fingerprints.
* [`src/ai_video_editor/pipeline/report.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/pipeline/report.py): Geração do `edit_report.md`.
