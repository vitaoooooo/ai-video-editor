# CLAUDE.md — AI Video Editor

Editor de vídeo com IA. O usuário descreve o que quer em linguagem natural, envia vídeos, o sistema analisa, monta um plano de edição e renderiza com FFmpeg.

## Princípio de arquitetura (não violar)

A IA **nunca executa cortes diretamente**. Ela produz dados estruturados (validados com Pydantic), e um motor determinístico (FFmpeg) executa. Criatividade probabilística na decisão, execução previsível no render.

## Estado atual

- Fase 1 — Core/Renderer: `core/schema.py` (`ClipSegment`, `EditPlan`), `renderer/` (compiler, executor, ffmpeg_locator). ✅
- Fase 2 — Analyzer: `analyzer/` (ffprobe, extração de frames com timestamps, `VideoAnalysisResult`, `BatchAnalysisResult`). ✅
- Fase 3 — Intelligence: `intelligence/` (interface `MultimodalAIProvider`, providers `gemini` e `mock`, proxies de imagem, relatório Markdown, métricas de execução). ✅
- Comando `edit` (`pipeline/`): probe → silêncios (`audio/`; limite `--threshold-db auto` por padrão, medido em janelas de 50 ms por `audio/levels.py`, ou número fixo) → transcrição (`transcription/`) → Diretor (`director/`: frases, `EditDirector` gemini/mock, validação com retry e fallback) → plano (`planning/`) → render → legendas → relatório + transcrição comparativa (`pipeline/edit_diff.py`: `edit_transcript.md`/`edit_diff.json`, status de aplicação de cada remoção) + linha do tempo dos cortes (`pipeline/edit_timeline.py`: `edit_timeline.svg`). ✅
- Ciclo de revisão (`revision/`, `pipeline/revise.py`, `pipeline/versions.py`): cada resultado é uma versão em `versions/vN/` (v1 = `edit`; projetos antigos migram sozinhos). `revise <pasta> "pedido" [--from vN]` reusa probe/silêncios/transcrição do cache; `EditReviser` (gemini/mock) recebe o estado das frases na versão base e devolve o conjunto completo de removidas, restaurações com motivo e ritmo de pausas (`natural`/`normal`/`tight`); mesma validação do Diretor com retry, e sem versão criada se o pedido for impossível, ambíguo, sem efeito ou inválido. `versions <pasta>` lista o histórico. ✅
- Provedores de IA plugáveis (`llm/`): interface `StructuredLLMClient` (prompt + schema Pydantic → objeto validado + métricas), registro `gemini`/`openai`/`mock` e `ModelChain` (cadeia `provedor:modelo` com troca automática em erro temporário persistente ou cota esgotada; pula provedor sem chave). Diretor (`LLMDirector`) e revisor (`LLMReviser`) só usam a interface. CLI: `--director-provider/--director-model/--director-fallback` e `--reviser-*`. ✅
- Timeline com trilhas (`core/timeline.py`): `video_main` (base contígua, com o áudio dos itens ou silêncio), `video_overlay` (substitui a imagem numa janela; o áudio da base continua) e `audio_ambient` (áudio posicionado, com volume e fades); tempos quantizados em frames, `timeline_from_edit_plan` e `flatten` (achatamento puro em pedaços de vídeo + plano de áudio). Render em `renderer/timeline_compiler.py`/`timeline_executor.py` (pedaços padronizados em paralelo — rotação, letterbox/pillarbox, fps, yuv420p 8 bits —, áudio 48 kHz com ambiente mixado, mux e loudnorm). `render` aceita JSON de EditPlan ou Timeline. O `edit <vídeo>` ainda usa o EditPlan; o `edit <pasta>` usa a Timeline. ✅
- Receitas (`recipes/`): `Recipe` (etapas em ordem + dependências + `multi_clip`, validada contra o catálogo de peças `PIECES`) e registro com as receitas `fala` = etapas do `edit <vídeo>` e `vlog` = espinha falada de `edit <pasta>` (`multi_clip=True`). `EditPipeline` executa `recipe.steps` (uma etapa por método; numeração `[n/N]` vem da receita); CLI `edit --receita fala`. `tests/test_reference_fala.py` compara o `edit` completo com o snapshot em `tests/data/reference_fala/` (regenerar só com `UPDATE_REFERENCE=1`); não alterar o snapshot sem mudança intencional de comportamento. ✅
- Comando `catalog <pasta>` (`catalog/`): lê uma pasta de clipes sem modificá-la e gera `output/projects/<pasta>-<hash da listagem>/catalog/` (`catalog.json` `ProjectCatalog`/`CatalogClip` com `schema_version`, `catalog.md`, `contact_sheet_<papel>.jpg`, `thumbs/`). Papel pela subpasta (sinônimos; `--papel pasta=papel`), IDs `P/D/V/B/X001` por data de gravação (tags → nome → mtime, com `source`; `--ajuste-relogio papel=+3h`), ffprobe/miniaturas/signalstats em paralelo, fala pelo VAD do faster-whisper (`SpeechDetector`, mock nos testes; sem ele = `desconhecido`). Cache por clipe em `output/projects/_cache/clips/`. ✅
- Produtor e comando `briefing <arquivo-ou-pasta> [--pedido] [--receita]` (`producer/`): pedido + resumo compacto do catálogo (por papel, sem caminhos) + receitas registradas (lidas na hora) → `Briefing` (receita, "Entendi", metas, instruções para os Diretores, papéis), pela `ModelChain` com as opções do Diretor; validação pelo código, nova tentativa com o erro e fallback determinístico registrado; salva `briefing.json` na pasta do projeto. Receita de vários clipes declara `multi_clip = True`. O `edit` ainda não lê o briefing. Catálogo: papel por palavra no nome da pasta (`contém`), datas de exportação → ordem natural pelo nome, perfil log só com contraste e saturação baixos. ✅
- Imagens em `llm/`: `generate(..., images=[ImageInput])` (caminho do JPEG + rótulo/ID citado no prompt + detalhe `low/auto/high`); aditivo: sem imagens, a chamada e os fingerprints são os de antes. `intelligence/` é legado (só `analyze`/`report-ai`), a ser substituído por `visual/`. ✅
- Comando `visual <pasta>` (`visual/`): lê/gera o catálogo, divide cada clipe em tomadas (`P007.T03`: cena pelo `select`/`scene` + janelas de até 8 s), 3 miniaturas por tomada (`P007.T03.F1..F3`, tempos só no JSON), medidas FFmpeg sem IA (blurdetect, signalstats: desfoque, brilho, movimento, tela uniforme) e descrição por IA em lotes (até 8 tomadas/24 imagens; validação com nova tentativa só das tomadas recusadas; persistindo, `sem_analise`). Cache por clipe em `_cache/clips/visual-<chave>/` (inclui as análises da IA por fingerprint). Saídas `visual/visual.json` (`ProjectVisual`), `visual.md` (resumo, "Atenção": fala com imagem ruim) e `frames/`. `--so-estimar` não chama a IA. ✅
- Espinha falada (`edit <pasta>`, receita `vlog`; `pipeline/spine.py`, `pipeline/clips.py`, `pipeline/spine_report.py`, `planning/timeline_plan.py`, `subtitles/clips.py`): catálogo (lido; gerado pela API pública de `catalog/` se faltar) → silêncios e transcrição por clipe (cache por clipe em `_cache/edit_clips/`) → Diretor (frases com IDs globais e `clip_id`, clipes como seções, categoria `retake`, clipe inteiro removido só com `retake`/`off_topic`) → Timeline (`timeline.json`, 1920x1080, fps inteiro) → render da Timeline → legendas → relatórios por clipe. `project.json` com `kind="multiclip"`; `revise` recusa projeto de pasta (rodada 4). O áudio da Timeline sai de um único FFmpeg (`asegment` por origem + `concat`). ✅
- Rumo: o editor passa a editar qualquer tipo de vídeo; todo trabalho é um projeto com clipes (1 vídeo = 1 clipe). Visão, peças/receitas, Produtor, Timeline e plano de rodadas em `docs/ARQUITETURA.md`. Hoje o perfil pronto é o de vídeo falado (receita `fala`): áudio → silêncios → transcrição → Diretor de IA (decide por IDs da transcrição, nunca inventa timestamps) → EditPlan → render. A Timeline com trilhas (`core/timeline.py`) já é usada pelo `edit <pasta>`; o `edit <vídeo>` continua no EditPlan (unificação depois; `tests/test_reference_fala.py` garante o caminho de um vídeo).

## Comandos

```bash
pip install -e ".[dev]"          # instalar
python -m pytest -q              # testes (devem passar todos antes de concluir)
python -m ai_video_editor.cli --help
```

Subcomandos da CLI: `analyze`, `report-ai`, `quick-cut`, `render`, `edit` (vídeo ou pasta de clipes; ver docs/REFERENCIA.md, opções E e K), `revise`, `versions`, `catalog` (pasta de clipes; ver docs/REFERENCIA.md, opção H), `briefing` (Produtor; opção I), `visual` (tomadas e descrição por IA; opção J), `benchmark transcription` (WER e velocidade; ver docs/REFERENCIA.md, opção G).

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
5. Imagens: `_request` recebe `images` (lista de `ImageInput`, vazia nas chamadas só de texto). Sem imagens, monte exatamente a chamada de texto de antes; com imagens, uma mensagem do usuário com o prompt e, para cada imagem, `image.caption` (`Imagem <ID>:`) seguido dos bytes inline no formato oficial do SDK (`image.read_bytes()`, `image.mime_type`), traduzindo `image.detail` (`low/auto/high`; `auto` = não enviar o parâmetro). O `generate` da base já confere se os arquivos existem e conta as imagens nas métricas. Teste o payload com cliente falso e que a chamada sem imagem não mudou (ver `tests/test_llm_images.py`).

## Ao concluir uma tarefa

1. Durante o desenvolvimento, rodar só os testes rápidos e os arquivos afetados: `python -m pytest -q -m "not slow"` ou `python -m pytest -q tests/test_x.py`.
   A suíte completa (`python -m pytest -q -n auto`, inclui os `slow`) roda **uma única vez, antes do commit**, e deve passar toda.
2. Atualizar a referência técnica `docs/REFERENCIA.md` se mudou CLI, módulo ou fluxo; o `README.md` (visão geral de portfólio) só se mudou algo que ele cita (comandos principais, números, pipeline).
3. Atualizar `docs/HISTORICO.md`: adicionar a entrada da tarefa na data de hoje (o que entrou, o que mudou/saiu e POR QUÊ, e resultados medidos, se houver), e mover o item de 'Em andamento'/'Próximos passos' se for o caso.
4. Resumir: arquivos alterados, decisões tomadas, o que ficou pendente.

## Skills

As skills do Claude Code ficam na pasta de usuário (`~/.claude/skills/`; no Windows, `C:\Users\vitor\.claude\skills\`), fora do repositório. Por isso valem para todos os worktrees. Para alterar uma skill, edite lá.
