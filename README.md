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
Interpreta os frames extraídos com IA multimodal e gera um relatório Markdown (`ai_report.md`) com linha do tempo visual e dados estruturados (`ai_analysis.json`):

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
