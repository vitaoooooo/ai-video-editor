# AI Video Editor — Core Editing Engine (Phase 1 PoC)

Motor de edição de vídeo programático e determinístico para o **AI Video Editor**, baseado em schemas Pydantic e compilação segura de filtros FFmpeg.

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

O projeto inclui suíte de testes unitários e de integração utilizando **vídeos sintéticos** gerados dinamicamente via FFmpeg (`testsrc` + `sine`), sem necessidade de arquivos externos:

```bash
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 4. Uso via Linha de Comando (CLI)

### Opção A: Corte Rápido de Dois Vídeos (`quick-cut`)
Corta trechos especificados de dois vídeos e concatena-os em um MP4 final:

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli quick-cut `
  --video1 output/samples/input_clip1.mp4 --start1 1.0 --end1 3.0 `
  --video2 output/samples/input_clip2.mp4 --start2 0.5 --end2 2.5 `
  --output output/merged_output.mp4
```

### Opção B: Renderização a partir de um `EditPlan` JSON (`render`)
Renderiza um plano de edição completo a partir de um arquivo JSON estruturado:

```bash
.\.venv\Scripts\python.exe -m ai_video_editor.cli render output/sample_plan.json
```

Exemplo de schema `EditPlan`:
```json
{
  "title": "Demo Two Clip Merge",
  "output_path": "output/from_json_plan.mp4",
  "target_resolution": [1280, 720],
  "target_fps": 30,
  "segments": [
    {
      "source_path": "output/samples/input_clip1.mp4",
      "start_sec": 0.0,
      "end_sec": 2.0,
      "volume": 1.0
    },
    {
      "source_path": "output/samples/input_clip2.mp4",
      "start_sec": 1.0,
      "end_sec": 3.0,
      "volume": 1.0
    }
  ]
}
```

---

## 5. Estrutura do Núcleo

* [`src/ai_video_editor/core/schema.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/core/schema.py): Modelos `ClipSegment` e `EditPlan` com validação de intervalos temporais e resoluções.
* [`src/ai_video_editor/renderer/ffmpeg_locator.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/renderer/ffmpeg_locator.py): Descoberta resiliente de executáveis FFmpeg/FFprobe.
* [`src/ai_video_editor/renderer/compiler.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/renderer/compiler.py): Compilação segura de listas de argumentos e filtergraph (`trim`, `scale`, `pad`, `concat`, `volume`) sem shell injection.
* [`src/ai_video_editor/renderer/executor.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/renderer/executor.py): Execução isolada via `subprocess`, validação de saída e extração de metadados via FFprobe.
* [`src/ai_video_editor/cli.py`](file:///C:/Users/vitor/Projects/ai-video-editor/src/ai_video_editor/cli.py): Ponto de entrada CLI para comandos `render` e `quick-cut`.
