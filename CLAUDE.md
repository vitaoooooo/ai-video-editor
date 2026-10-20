# CLAUDE.md — AI Video Editor

Editor de vídeo com IA. O usuário descreve o que quer em linguagem natural, envia vídeos, o sistema analisa, monta um plano de edição e renderiza com FFmpeg.

## Princípio de arquitetura (não violar)

A IA **nunca executa cortes diretamente**. Ela produz dados estruturados (validados com Pydantic), e um motor determinístico (FFmpeg) executa. Criatividade probabilística na decisão, execução previsível no render.

## Estado atual

- Fase 1 — Core/Renderer: `core/schema.py` (`ClipSegment`, `EditPlan`), `renderer/` (compiler, executor, ffmpeg_locator). ✅
- Fase 2 — Analyzer: `analyzer/` (ffprobe, extração de frames com timestamps, `VideoAnalysisResult`, `BatchAnalysisResult`). ✅
- Fase 3 — Intelligence: `intelligence/` (interface `MultimodalAIProvider`, providers `gemini` e `mock`, proxies de imagem, relatório Markdown, métricas de execução). ✅
- Comando `edit` (`pipeline/`): probe → silêncios (`audio/`; limite `--threshold-db auto` por padrão, medido em janelas de 50 ms por `audio/levels.py`, ou número fixo) → transcrição (`transcription/`) → Diretor (`director/`: frases, `EditDirector` gemini/mock, validação com retry e fallback) → plano (`planning/`) → render → legendas → relatório + transcrição comparativa (`pipeline/edit_diff.py`: `edit_transcript.md`/`edit_diff.json`, status de aplicação de cada remoção). ✅
- Ciclo de revisão (`revision/`, `pipeline/revise.py`, `pipeline/versions.py`): cada resultado é uma versão em `versions/vN/` (v1 = `edit`; projetos antigos migram sozinhos). `revise <pasta> "pedido" [--from vN]` reusa probe/silêncios/transcrição do cache; `EditReviser` (gemini/mock) recebe o estado das frases na versão base e devolve o conjunto completo de removidas, restaurações com motivo e ritmo de pausas (`natural`/`normal`/`tight`); mesma validação do Diretor com retry, e sem versão criada se o pedido for impossível, ambíguo, sem efeito ou inválido. `versions <pasta>` lista o histórico. ✅
- Provedores de IA plugáveis (`llm/`): interface `StructuredLLMClient` (prompt + schema Pydantic → objeto validado + métricas), registro `gemini`/`openai`/`mock` e `ModelChain` (cadeia `provedor:modelo` com troca automática em erro temporário persistente ou cota esgotada; pula provedor sem chave). Diretor (`LLMDirector`) e revisor (`LLMReviser`) só usam a interface. CLI: `--director-provider/--director-model/--director-fallback` e `--reviser-*`. ✅
- Rumo da V1: editar vídeos falados (tutorial, aula, pessoa falando). Pipeline: áudio → silêncios → transcrição → Diretor de IA (decide por IDs da transcrição, nunca inventa timestamps) → EditPlan → render. A análise visual vira opcional neste perfil. Arquitetura preparada para perfis de edição (v2: gameplay, vlog).

## Comandos

```bash
pip install -e ".[dev]"          # instalar
python -m pytest -q              # testes (devem passar todos antes de concluir)
python -m ai_video_editor.cli --help
```

Subcomandos da CLI: `analyze`, `report-ai`, `quick-cut`, `render`, `edit`, `revise`, `versions`.

## Convenções

- Python 3.11+, src layout (`src/ai_video_editor/`), testes em `tests/`.
- Todo dado trocado com IA ou entre módulos é um modelo Pydantic.
- Novo provedor = classe que implementa a interface abstrata do módulo + um provedor `mock` equivalente para testes.
- Erros de IA usam as exceções de `intelligence/exceptions.py`; erros de mídia usam `analyzer/exceptions.py`.
- FFmpeg sempre via lista de argumentos em `subprocess` (nunca `shell=True`).
- Testes não fazem chamadas de rede: usam vídeos sintéticos gerados por FFmpeg (`tests/conftest.py`) e provedores mock.
- Testes que renderizam vídeo ou rodam FFmpeg de forma pesada levam `@pytest.mark.slow`. Vídeos sintéticos reaproveitáveis são fixtures `scope="session"` (somente leitura); toda saída vai para `tmp_path`, porque a suíte roda em paralelo (pytest-xdist).
- Revisor (`revise`): reaproveita a configuração do Diretor (provedor, modelo, cadeia de reserva, raciocínio, timeout); como o Diretor, só referencia IDs de frases, nunca timestamps.
- Diretor: raciocínio genérico `low` por padrão (`--director-thinking off/low/high`; cada provedor traduz — Gemini `thinking_level`, OpenAI `reasoning.effort` — ou ignora e registra; se a API recusar o nível, sobe automaticamente off → low → high e registra o nível efetivo) e timeout de 120s repassado ao SDK.
- Cadeia de modelos: padrão `gemini:gemini-3.8-flash` → `gemini-3.6-flash` → `gemini-3.5-flash-lite`. Troca só em erro temporário após esgotar as tentativas (`transient`) ou cota esgotada (`quota_exhausted`); nunca em autenticação ou requisição inválida. O fingerprint do cache usa a cadeia PEDIDA; `director_decision.json` e o relatório registram o modelo efetivo e as trocas.
- Chaves de API só por variável de ambiente (`GEMINI_API_KEY`, `OPENAI_API_KEY`) ou `.env` local (`llm/keys.py`). Nunca commitar, logar ou imprimir chaves; o mapeamento de erros remove chaves das mensagens (`llm/errors.py: redact`).
- Docstrings e mensagens em português; nomes de código em inglês.
- Não adicionar dependências sem necessidade clara; se adicionar, atualizar `pyproject.toml`.

## Adicionar um provedor de IA (Diretor/revisor)

1. Criar `src/ai_video_editor/llm/providers/<nome>.py` com uma classe que herda `StructuredLLMClient` (`llm/client.py`): definir `provider_name`, `api_key_env`, `default_model` (só modelos citados na documentação oficial) e, se fizer sentido, `default_fallback`; implementar `_create_client` (import tardio do SDK, com as novas tentativas do SDK desligadas), `_request`, `_usage`, `_parse`, `map_error` (usar `llm/errors.py`; marcar `transient` e `quota_exhausted`) e `thinking_setting` (traduzir off/low/high ou devolver None se o modelo não tiver raciocínio). Opcional: `thinking_ladder` e `is_thinking_refused`.
2. Registrar em `llm/registry.py`: `register_provider("<nome>", "ai_video_editor.llm.providers.<nome>:<Classe>")`. A CLI passa a aceitá-lo em `--director-provider`, `--reviser-provider` e nas cadeias `<nome>:<modelo>`.
3. Se precisar de SDK, adicionar um extra opcional em `pyproject.toml` (como `openai`).
4. Testes com cliente falso (sem rede): parse da saída estruturada, tradução do raciocínio e mapeamento de erros (ver `tests/test_openai_provider.py`).

## Ao concluir uma tarefa

1. Durante o desenvolvimento, rodar só os testes rápidos e os arquivos afetados: `python -m pytest -q -m "not slow"` ou `python -m pytest -q tests/test_x.py`.
   A suíte completa (`python -m pytest -q -n auto`, inclui os `slow`) roda **uma única vez, antes do commit**, e deve passar toda.
2. Atualizar o `README.md` se mudou CLI, módulo ou fluxo.
3. Atualizar `docs/HISTORICO.md`: adicionar a entrada da tarefa na data de hoje (o que entrou, o que mudou/saiu e POR QUÊ, e resultados medidos, se houver), e mover o item de 'Em andamento'/'Próximos passos' se for o caso.
4. Resumir: arquivos alterados, decisões tomadas, o que ficou pendente.

## Skills

As skills do projeto existem em duas cópias idênticas:

- `.agents/skills/` — usada pelo Antigravity (registrada em `skills-lock.json`).
- `.claude/skills/` — usada pelo Claude Code.

Uma alteração em uma das cópias deve ser replicada na outra.
