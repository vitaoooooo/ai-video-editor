# CLAUDE.md — AI Video Editor

Editor de vídeo com IA. O usuário descreve o que quer em linguagem natural, envia vídeos, o sistema analisa, monta um plano de edição e renderiza com FFmpeg.

## Princípio de arquitetura (não violar)

A IA **nunca executa cortes diretamente**. Ela produz dados estruturados (validados com Pydantic), e um motor determinístico (FFmpeg) executa. Criatividade probabilística na decisão, execução previsível no render.

## Estado atual

- Fase 1 — Core/Renderer: `core/schema.py` (`ClipSegment`, `EditPlan`), `renderer/` (compiler, executor, ffmpeg_locator). ✅
- Fase 2 — Analyzer: `analyzer/` (ffprobe, extração de frames com timestamps, `VideoAnalysisResult`, `BatchAnalysisResult`). ✅
- Fase 3 — Intelligence: `intelligence/` (interface `MultimodalAIProvider`, providers `gemini` e `mock`, proxies de imagem, relatório Markdown, métricas de execução). ✅
- Rumo da V1: editar vídeos falados (tutorial, aula, pessoa falando). Pipeline: áudio → silêncios → transcrição → Diretor de IA (decide por IDs da transcrição, nunca inventa timestamps) → EditPlan → render. A análise visual vira opcional neste perfil. Arquitetura preparada para perfis de edição (v2: gameplay, vlog).

## Comandos

```bash
pip install -e ".[dev]"          # instalar
python -m pytest -q              # testes (devem passar todos antes de concluir)
python -m ai_video_editor.cli --help
```

Subcomandos da CLI: `analyze`, `report-ai`, `quick-cut`, `render`.

## Convenções

- Python 3.11+, src layout (`src/ai_video_editor/`), testes em `tests/`.
- Todo dado trocado com IA ou entre módulos é um modelo Pydantic.
- Novo provedor = classe que implementa a interface abstrata do módulo + um provedor `mock` equivalente para testes.
- Erros de IA usam as exceções de `intelligence/exceptions.py`; erros de mídia usam `analyzer/exceptions.py`.
- FFmpeg sempre via lista de argumentos em `subprocess` (nunca `shell=True`).
- Testes não fazem chamadas de rede: usam vídeos sintéticos gerados por FFmpeg (`tests/conftest.py`) e provedores mock.
- Chaves de API só por variável de ambiente (`GEMINI_API_KEY`) ou `.env` local. Nunca commitar, logar ou imprimir chaves.
- Docstrings e mensagens em português; nomes de código em inglês.
- Não adicionar dependências sem necessidade clara; se adicionar, atualizar `pyproject.toml`.

## Ao concluir uma tarefa

1. Rodar `python -m pytest -q` e garantir que tudo passa.
2. Atualizar o `README.md` se mudou CLI, módulo ou fluxo.
3. Resumir: arquivos alterados, decisões tomadas, o que ficou pendente.

## Skills

As skills do projeto existem em duas cópias idênticas:

- `.agents/skills/` — usada pelo Antigravity (registrada em `skills-lock.json`).
- `.claude/skills/` — usada pelo Claude Code.

Uma alteração em uma das cópias deve ser replicada na outra.
