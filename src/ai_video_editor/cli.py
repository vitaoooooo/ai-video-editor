"""Interface de linha de comando (CLI) para o motor do AI Video Editor."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, get_args

from ai_video_editor.analyzer.schema import FrameExtractionConfig
from ai_video_editor.analyzer.service import VideoIngestService
from ai_video_editor.core.schema import ClipSegment, EditPlan, LoudnessPreset
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.coverage.schema import CoverageParams
from ai_video_editor.director.schema import UserBrief
from ai_video_editor.intelligence.retry import DEFAULT_API_RETRIES, RetryPolicy
from ai_video_editor.intelligence.schema import AIAnalysisConfig
from ai_video_editor.llm.chain import ModelChain, resolve_chain
from ai_video_editor.llm.providers.ollama import (
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_TIMEOUT_SEC,
    FORMAT_CHOICES as OLLAMA_FORMAT_CHOICES,
    FORMAT_ENV as OLLAMA_FORMAT_ENV,
    HOST_ENV as OLLAMA_HOST_ENV,
    KEEP_ALIVE_ENV as OLLAMA_KEEP_ALIVE_ENV,
    NUM_CTX_ENV as OLLAMA_NUM_CTX_ENV,
)
from ai_video_editor.llm.ollama_check import DEFAULT_CHECK_IMAGE_MODEL, DEFAULT_CHECK_TEXT_MODEL
from ai_video_editor.llm.registry import default_timeout, provider_names
from ai_video_editor.intelligence.service import MultimodalAnalysisService
from ai_video_editor.live_status import LiveStatus
from ai_video_editor.pipeline import (
    EditPipeline,
    EditPipelineParams,
    ProjectManifest,
    ProjectReviser,
)
from ai_video_editor.pipeline.project import MANIFEST_FILE
from ai_video_editor.pipeline.report import format_timestamp
from ai_video_editor.pipeline.schema import RunTimings, StageProgress
from ai_video_editor.pipeline.spine import SpinePipeline
from ai_video_editor.pipeline.spine_schema import (
    MULTICLIP_REVISE_MESSAGE,
    SpineManifest,
    SpineResult,
    is_multiclip_manifest,
)
from ai_video_editor.pipeline.timing import STAGE_LABELS, format_duration, summarize_run
from ai_video_editor.pipeline.versions import ensure_versions, parse_version_ref
from ai_video_editor.planning import SilenceCutParams
from ai_video_editor.visual.builder import DEFAULT_AI_PARALLEL
from ai_video_editor.recipes import (
    COVERAGE_FOLDER_RECIPE,
    DEFAULT_FOLDER_RECIPE,
    DEFAULT_RECIPE,
    Recipe,
    RecipeError,
    get_recipe,
    recipe_names,
)
from ai_video_editor.renderer.encoders import DEFAULT_GPU_WORKERS, ENCODER_CHOICES
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.renderer.timeline_executor import TimelineRenderExecutor
from ai_video_editor.revision import RevisionError
from ai_video_editor.subtitles import SubtitleParams
from ai_video_editor.transcription.schema import TranscriptionConfig

LOUDNESS_CHOICES = list(get_args(LoudnessPreset))


def is_timeline_document(data: object) -> bool:
    """JSON de Timeline (tem `video_main` ou `schema_version`) ou de EditPlan (o resto)."""
    return isinstance(data, dict) and ("video_main" in data or "schema_version" in data)


def handle_plan_render(args: argparse.Namespace) -> int:
    """Executa um EditPlan ou uma Timeline definidos em arquivo JSON (detectado pelo conteúdo)."""
    plan_path = Path(args.plan_file)
    if not plan_path.is_file():
        print(f"Erro: arquivo de plano não encontrado: {plan_path}", file=sys.stderr)
        return 1

    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if is_timeline_document(data):
        return _render_timeline(Timeline.model_validate(data), args)

    plan = EditPlan.model_validate(data)
    print(f"Plano carregado: '{plan.title}' com {len(plan.segments)} segmentos.")
    print(f"Destino: {plan.target_resolution[0]}x{plan.target_resolution[1]} @ {plan.target_fps}fps")

    executor = RenderExecutor(encoder=args.encoder, gpu_workers=args.gpu_paralelo)
    try:
        result = executor.execute(plan)
    except RuntimeError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    print("Renderização concluída com sucesso!")
    print(f"Arquivo gerado: {result.output_path}")
    print(f"Duração: {result.duration_sec:.2f}s | Resolução: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Tamanho: {result.size_bytes} bytes")
    print(f"Codificador: {result.encoder.label}")
    for warning in result.warnings:
        print(f"Aviso: {warning}")
    return 0


def _render_timeline(timeline: Timeline, args: argparse.Namespace) -> int:
    print(
        f"Timeline carregada: '{timeline.title}' com {len(timeline.video_main)} itens na base, "
        f"{len(timeline.video_overlay)} overlays e {len(timeline.audio_ambient)} áudios de ambiente "
        f"({format_duration(timeline.duration_sec)})."
    )
    print(f"Destino: {timeline.resolution[0]}x{timeline.resolution[1]} @ {timeline.fps}fps")

    try:
        result = TimelineRenderExecutor(encoder=args.encoder, gpu_workers=args.gpu_paralelo).execute(timeline)
    except RuntimeError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    print("Renderização concluída com sucesso!")
    print(f"Arquivo gerado: {result.output_path}")
    print(
        f"Duração: {result.duration_sec:.2f}s ({result.frame_count} frames) | "
        f"Resolução: {result.resolution[0]}x{result.resolution[1]}"
    )
    print(f"Tamanho: {result.size_bytes} bytes")
    print(f"Codificador: {result.encoder.label}")
    for warning in result.warnings:
        print(f"Aviso: {warning}")
    return 0


def handle_quick_cut(args: argparse.Namespace) -> int:
    """Corta trechos de dois vídeos e une-os em um MP4 de saída."""
    v1_path = Path(args.video1)
    v2_path = Path(args.video2)
    out_path = Path(args.output)

    if not v1_path.is_file():
        print(f"Erro: vídeo 1 não encontrado: {v1_path}", file=sys.stderr)
        return 1
    if not v2_path.is_file():
        print(f"Erro: vídeo 2 não encontrado: {v2_path}", file=sys.stderr)
        return 1

    plan = EditPlan(
        title="Quick Cut Two Videos",
        output_path=out_path,
        target_resolution=(args.width, args.height),
        target_fps=args.fps,
        segments=[
            ClipSegment(
                source_path=v1_path,
                start_sec=args.start1,
                end_sec=args.end1,
            ),
            ClipSegment(
                source_path=v2_path,
                start_sec=args.start2,
                end_sec=args.end2,
            ),
        ],
    )

    executor = RenderExecutor()
    result = executor.execute(plan)

    print("Renderização concluída com sucesso!")
    print(f"Arquivo gerado: {result.output_path}")
    print(f"Duração: {result.duration_sec:.2f}s | Resolução: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Tamanho: {result.size_bytes} bytes")
    return 0


def handle_analyze(args: argparse.Namespace) -> int:
    """Analisa um ou mais vídeos: extrai metadados e frames de amostragem."""
    config = FrameExtractionConfig(
        strategy=args.strategy,
        interval_sec=args.interval,
        target_count=args.count,
        max_frames=args.max_frames,
        image_format=args.format,
    )
    service = VideoIngestService(default_base_dir=Path(args.output_dir))
    paths = [Path(p) for p in args.videos]
    batch_result = service.analyze_batch(paths, config=config)

    print(f"Sessão de Análise: {batch_result.session_id}")
    print(f"Diretório da sessão: {batch_result.session_dir}")
    print(f"Total processado: {batch_result.total_requested}")
    print(f"Sucesso: {len(batch_result.successful)} | Falhas: {len(batch_result.failed)}")

    for res in batch_result.successful:
        m = res.metadata
        v = m.video_stream
        a = m.audio_stream
        audio_info = f"{a.codec_name} ({a.channels}ch, {a.sample_rate}Hz)" if a else "Sem Áudio"
        print(f"\n[OK] {m.source_path.name}")
        print(f"     Duração: {m.duration_sec:.2f}s | Tamanho: {m.size_bytes} bytes")
        print(f"     Vídeo: {v.codec_name} {v.width}x{v.height} @ {v.fps}fps (pix_fmt: {v.pixel_format})")
        print(f"     Áudio: {audio_info}")
        print(f"     Frames Extraídos: {len(res.frames)} (Estratégia: {res.sampling_strategy_used})")
        print(f"     Nota: {res.sampling_disclaimer}")

    for path_str, err in batch_result.failed.items():
        print(f"\n[FALHA] {path_str}: {err}", file=sys.stderr)

    if args.json:
        print("\n--- RELATÓRIO JSON ---")
        print(batch_result.model_dump_json(indent=2))

    return 0 if not batch_result.failed else 1


def handle_report_ai(args: argparse.Namespace) -> int:
    """Interpreta visualmente os frames amostrados e gera relatório Markdown com IA."""
    report_file = Path(args.report)
    if not report_file.is_file():
        print(f"Erro: arquivo de relatório técnico não encontrado: {report_file}", file=sys.stderr)
        return 1

    ai_config = AIAnalysisConfig(
        provider=args.provider,
        model_id=args.model,
        max_images_per_request=args.max_images,
        max_image_dimension=args.proxy_dim,
    )

    print(f"Iniciando análise com IA (Provedor: {ai_config.provider}, Modelo: {ai_config.model_id})...")
    print(f"Carregando dados de: {report_file.name}")

    try:
        provider = None
        if ai_config.provider.lower().strip() == "gemini":
            from ai_video_editor.intelligence.providers.gemini import GeminiMultimodalProvider

            provider = GeminiMultimodalProvider(
                retry=RetryPolicy(max_retries=args.api_retries, on_retry=_print_api_retry)
            )
        reports = MultimodalAnalysisService.from_report_file(report_file, config=ai_config, provider=provider)
    except Exception as e:
        print(f"Erro durante a execução da análise multimodal: {e}", file=sys.stderr)
        return 1

    print(f"\nConcluído! {len(reports)} vídeo(s) analisado(s) com sucesso:")
    for rep in reports:
        print(f"\n* Relatório Markdown: {rep.markdown_path}")
        print(f"* Dados Estruturados (JSON): {rep.json_path}")
        print(f"* Resumo: {rep.analysis_response.summary[:120]}...")

    return 0


class EditProgressPrinter:
    """Mostra o andamento de cada etapa do comando `edit` (linha ao vivo em um terminal).

    A numeração (`[4/8] Diretor`) vem da receita executada.
    """

    def __init__(self, status: LiveStatus, recipe: Optional[Recipe] = None):
        self.status = status
        self.recipe = recipe or get_recipe(DEFAULT_RECIPE)

    def __call__(self, stage: str, event: str, detail: str) -> None:
        step = f"[{self.recipe.position(stage)}/{len(self.recipe.steps)}] {STAGE_LABELS[stage]}"
        if event == "start":
            self.status.begin(step)
        elif event == "done":
            self.status.end(f"{step}: ok — {detail}")
        elif event == "cached":
            self.status.end(f"{step}: em cache")
        elif event == "skipped":
            self.status.end(f"{step}: pulada ({detail})")
        elif event == "failed":
            self.status.end(f"{step}: FALHOU")
        elif event in ("retry", "notice"):
            self.status.message(f"{step}: {detail}")

    def progress(self, stage: str, progress: StageProgress) -> None:
        self.status.apply(progress)


def _print_progress(stage: str, event: str, detail: str) -> None:
    """Uma linha por evento do `edit`, sem linha ao vivo."""
    EditProgressPrinter(LiveStatus(live=False))(stage, event, detail)


def _print_api_retry(event) -> None:
    """Mostra uma nova tentativa da API fora do pipeline (ex: `report-ai`)."""
    print(f"IA: {event.message}", flush=True)


def _llm_providers() -> list[str]:
    """Provedores reais registrados (o `mock` da camada `llm` não é oferecido na CLI)."""
    return [name for name in provider_names() if name != "mock"]


# Tempo limite padrão das chamadas aos provedores de nuvem (o do Ollama é DEFAULT_OLLAMA_TIMEOUT_SEC).
CLOUD_TIMEOUT_SEC = 120.0
VISUAL_TIMEOUT_SEC = 180.0
TIMEOUT_HELP = f"padrão: 120; ollama: {DEFAULT_OLLAMA_TIMEOUT_SEC:.0f}"


def _director_timeout(args: argparse.Namespace) -> float:
    """`--director-timeout`, ou o padrão do provedor do Diretor (o Ollama local é mais lento)."""
    return args.director_timeout or default_timeout(args.director, CLOUD_TIMEOUT_SEC)


def _coverage_timeout(args: argparse.Namespace) -> Optional[float]:
    """`--coverage-timeout`; sem ele, o do Diretor (None), salvo provedor diferente com padrão próprio."""
    if args.coverage_timeout is not None or args.coverage in (None, args.director) or args.director_timeout:
        return args.coverage_timeout
    return default_timeout(args.coverage, _director_timeout(args))


def _add_ollama_args(parser: argparse.ArgumentParser) -> None:
    """Opções do Ollama local; valem para o processo (gravam as variáveis de ambiente do provedor)."""
    group = parser.add_argument_group("Ollama (modelos locais)")
    group.add_argument(
        "--ollama-host", default=None, metavar="URL",
        help=f"Endereço do servidor Ollama (padrão: {OLLAMA_HOST_ENV} ou http://localhost:11434)",
    )
    group.add_argument(
        "--ollama-ctx", type=_ctx_tokens, default=None, metavar="TOKENS",
        help=f"Contexto (num_ctx) enviado em toda chamada ao Ollama (padrão: {OLLAMA_NUM_CTX_ENV} ou "
        f"{DEFAULT_NUM_CTX}, estimativa para 6 GB de VRAM). Entrada que encosta no limite vira erro",
    )
    group.add_argument(
        "--ollama-keep-alive", default=None, metavar="DURAÇÃO",
        help=f"Tempo que o modelo fica carregado depois da chamada (ex: 10m, 3600, -1 = sempre; "
        f"padrão: {OLLAMA_KEEP_ALIVE_ENV} ou o do servidor)",
    )
    group.add_argument(
        "--ollama-formato", choices=OLLAMA_FORMAT_CHOICES, default=None,
        help=f"Como pedir o JSON ao Ollama: schema = JSON schema em format; texto = schema no fim do prompt e "
        f"JSON lido do texto; auto = schema e, se o Ollama devolver a resposta zerada, texto para aquele modelo "
        f"até o fim (padrão: {OLLAMA_FORMAT_ENV} ou auto)",
    )


def apply_ollama_args(args: argparse.Namespace) -> None:
    """Grava `--ollama-*` nas variáveis de ambiente lidas pelo provedor (inclusive nas etapas internas)."""
    for attr, env in (
        ("ollama_host", OLLAMA_HOST_ENV), ("ollama_ctx", OLLAMA_NUM_CTX_ENV), ("ollama_keep_alive", OLLAMA_KEEP_ALIVE_ENV),
        ("ollama_formato", OLLAMA_FORMAT_ENV),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env] = str(value)


def _ctx_tokens(value: str) -> int:
    number = int(value)
    if number < 256:
        raise argparse.ArgumentTypeError("deve ser 256 tokens ou mais")
    return number


def _threshold_arg(value: str):
    """`auto` ou um número em dB."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"esperado `auto` ou um número em dB, recebido {value!r}")


def default_edit_recipe(video_path: Path, is_folder: bool, transcribe: bool, projects_dir: Path) -> str:
    """Receita sem `--receita`: `fala` para um vídeo; para uma pasta, `vlog_cobertura` se a pasta já tem a
    análise visual (`visual <pasta>`) e há transcrição (a cobertura escolhe frases por ID), senão `vlog`."""
    if not is_folder:
        return DEFAULT_RECIPE
    from ai_video_editor.pipeline.clips import catalog_location

    project_dir, _ = catalog_location(video_path.resolve(), projects_dir)
    if transcribe and (project_dir / "visual" / "visual.json").is_file():
        return COVERAGE_FOLDER_RECIPE
    return DEFAULT_FOLDER_RECIPE


def _edit_briefing(args: argparse.Namespace, video_path: Path):
    """Briefing do `edit` (`EditBriefing`): desligado por `--sem-briefing`, reaproveitado ou gerado agora."""
    from ai_video_editor.pipeline.briefing import EditBriefing, briefing_location, prepare_briefing

    if args.sem_briefing:
        return EditBriefing(None, "desligado", briefing_location(video_path, Path(args.projects_dir)))
    status = LiveStatus(live=False if args.no_live else None)
    printer = CatalogProgressPrinter(status)
    # Receita fora do Produtor (`producer=False`, ex.: editor_ia) não é forçada nele: o briefing só dá o resto.
    forced = args.receita if args.receita is None or get_recipe(args.receita).producer else None
    with status:
        return prepare_briefing(
            video_path, args.pedido, Path(args.projects_dir), lambda: _announce_producer(_build_producer(args)),
            forced_recipe=forced, on_progress=printer, on_stage_progress=printer.progress,
        )


def _announce_producer(producer):
    if producer is not None and producer.model_chain:
        print(f"[produtor] {producer.model_chain[0]}", flush=True)
    return producer


def _briefing_recipe_problem(recipe: Recipe, is_folder: bool, video_path: Path, projects_dir: Path) -> Optional[str]:
    """Por que a receita escolhida pelo briefing não serve para este material (None = serve)."""
    if not recipe.accepts(is_folder):
        kind = "uma pasta de clipes" if recipe.multi_clip else "um vídeo só"
        given = "uma pasta" if is_folder else "um vídeo"
        return f"edita {kind}, mas recebido {given} ({video_path})"
    if "visual" in recipe.steps:
        from ai_video_editor.pipeline.clips import catalog_location

        project_dir, _ = catalog_location(video_path.resolve(), projects_dir)
        if not (project_dir / "visual" / "visual.json").is_file():
            return "precisa da análise visual da pasta, e ela não existe (rode `visual <pasta>` antes)"
    return None


def _print_settings(settings, path: Optional[Path]) -> None:
    """O "Entendi" do briefing e os valores efetivos com a origem de cada um."""
    from ai_video_editor.pipeline.report import SETTING_LABELS, format_setting_value

    by = "IA" if settings.briefing_origin == "ia" else "regra sem IA"
    print(settings.understood or "")
    print(f"Briefing: {settings.briefing} ({by}; {path})")
    for name, label in SETTING_LABELS.items():
        setting = getattr(settings, name)
        print(f"  {label}: {format_setting_value(name, setting.value)} ({setting.origin_label})")


def handle_edit(args: argparse.Namespace) -> int:
    """Edita um vídeo falado (receita `fala`) ou uma pasta de clipes (receitas `vlog`/`vlog_cobertura`), com cache por etapa.

    Com `--pedido` (ou um `briefing.json` atual no projeto), o briefing do Produtor dá receita, intensidade,
    ritmo, instruções e duração alvo; opção explícita da CLI > briefing > padrão.
    """
    from ai_video_editor.pipeline.briefing import resolve_settings

    video_path = Path(args.video)
    is_folder = video_path.is_dir()
    if not is_folder and not video_path.is_file():
        print(f"Erro: vídeo ou pasta não encontrado: {video_path}", file=sys.stderr)
        return 1
    if args.receita is not None:
        try:
            get_recipe(args.receita)
        except RecipeError as e:
            print(f"Erro: {e}", file=sys.stderr)
            return 1

    try:
        edit_briefing = _edit_briefing(args, video_path)
    except Exception as e:
        print(f"Erro: briefing: {e}", file=sys.stderr)
        return 1
    # Sem briefing, o `--pedido` vai direto ao Diretor (como antes do Produtor no `edit`).
    instructions = args.instructions
    if instructions is None and edit_briefing.briefing is None:
        instructions = args.pedido
    settings = resolve_settings(
        edit_briefing,
        default_edit_recipe(video_path, is_folder, args.transcribe, Path(args.projects_dir)),
        recipe=args.receita, intensity=args.intensity, pace=args.ritmo, instructions=instructions,
        target_duration_sec=args.target_duration,
    )
    for note in settings.notes:
        print(f"Aviso: {note}", file=sys.stderr)
    try:
        recipe = get_recipe(str(settings.recipe.value))
    except RecipeError as e:
        print(f"Erro: o briefing ({edit_briefing.path}) escolheu uma receita que não existe. {e}", file=sys.stderr)
        return 1
    if settings.recipe.origin == "briefing":
        problem = _briefing_recipe_problem(recipe, is_folder, video_path, Path(args.projects_dir))
        if problem is not None:
            print(
                f"Erro: o briefing ({edit_briefing.path}) escolheu a receita '{recipe.name}', que {problem}. "
                "Escolha outra com --receita, refaça o briefing com outro --pedido ou use --sem-briefing.",
                file=sys.stderr,
            )
            return 1
    if not recipe.accepts(is_folder):
        kind = "uma pasta de clipes" if recipe.multi_clip else "um vídeo só"
        print(f"Erro: a receita '{recipe.name}' edita {kind}; recebido {video_path}.", file=sys.stderr)
        return 1

    try:
        params = EditPipelineParams(
            threshold_db=args.threshold_db,
            min_silence_sec=args.min_silence_sec,
            cut=SilenceCutParams(
                padding_before_sec=args.padding_before,
                padding_after_sec=args.padding_after,
                min_gap_to_cut_sec=args.min_gap,
                min_keep_sec=args.min_keep,
            ),
            transcribe=args.transcribe,
            transcription=TranscriptionConfig(
                model_size=args.whisper_model,
                language=args.language,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
            ),
            director=args.director,
            director_model=args.director_model,
            director_fallback=args.director_fallback,
            director_thinking=args.director_thinking,
            director_timeout_sec=_director_timeout(args),
            api_retries=args.api_retries,
            pacing=settings.pace.value,
            brief=UserBrief(
                intensity=settings.intensity.value,
                target_duration_sec=settings.target_duration_sec.value,
                remove_fillers=not args.keep_fillers,
                instructions=settings.instructions.value,
            ),
            subtitles=not args.no_subtitles,
            subtitle=SubtitleParams(max_chars_per_line=args.subtitle_max_chars),
            transcript_min_pause_sec=args.transcript_min_pause,
            loudness=args.loudness,
        )
        coverage = CoverageParams.inherit(
            params.director, params.director_model, params.director_fallback, params.director_thinking,
            params.director_timeout_sec, params.api_retries,
            provider=args.coverage, model=args.coverage_model, fallback=args.coverage_fallback,
            thinking=args.coverage_thinking, timeout_sec=_coverage_timeout(args),
        )
        if coverage.provider not in ("none", "mock"):
            coverage.chain()  # valida provedor e reservas
    except ValueError as e:
        print(f"Erro: parâmetros inválidos: {e}", file=sys.stderr)
        return 1

    if "editor" in recipe.steps:
        return _run_editor(args, video_path, recipe, params, settings, edit_briefing)
    if settings.uses_briefing:
        _print_settings(settings, edit_briefing.path)
    elif is_folder and args.receita is None:
        why = " (a pasta tem análise visual)" if recipe.name == COVERAGE_FOLDER_RECIPE else ""
        print(f"Receita: {recipe.name}{why}")
    status = LiveStatus(live=False if args.no_live else None)
    printer = EditProgressPrinter(status, recipe)
    pipeline_class = SpinePipeline if is_folder else EditPipeline
    executor = (TimelineRenderExecutor if is_folder else RenderExecutor)(
        encoder=args.encoder, gpu_workers=args.gpu_paralelo
    )
    pipeline = pipeline_class(
        projects_root=Path(args.projects_dir), on_progress=printer, on_stage_progress=printer.progress,
        recipe=recipe, render_executor=executor,
    )
    try:
        executor.resolve_encoder()  # `--encoder nvenc` sem GPU falha aqui, antes das etapas
        with status:
            if is_folder:
                result = pipeline.run(
                    video_path, params, force=args.force, coverage=coverage, settings=settings,
                    use_briefing=settings.uses_briefing,
                )
            else:
                result = pipeline.run(video_path, params, force=args.force, settings=settings)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    removed = result.original_duration_sec - result.final_duration_sec
    pct = removed / result.original_duration_sec * 100 if result.original_duration_sec > 0 else 0.0
    for warning in result.warnings:
        print(f"Aviso: {warning}", file=sys.stderr)
    print("\nEdição concluída!")
    print(f"Projeto: {result.project_dir}")
    print(f"Vídeo final: {result.output_path}")
    print(f"Relatório: {result.report_path}")
    if result.subtitles_path is not None:
        print(f"Legendas: {result.subtitles_path}")
    if result.transcript_path is not None:
        print(f"Transcrição comparativa: {result.transcript_path}")
    if isinstance(result, SpineResult):
        print(f"Clipes na espinha: {len(result.clips_used)} ({', '.join(result.clips_used)}); não usados: {result.clips_unused}")
        print(f"Timeline: {result.timeline_path}")
        if result.coverage is not None:
            run = result.coverage
            breath_sec = sum(b.duration_sec for b in run.placed_breaths)
            print(
                f"Cobertura: {len(run.placed_coverages)} coberturas, {len(run.placed_breaths)} respiros "
                f"(+{breath_sec:.1f}s), janelas com imagem ruim: {run.windows_label} — origem {run.origin_label}"
            )
        if result.version is not None:
            print(f"Versão: v{result.version}")
    elif result.version is not None:
        print(f"Versão: v{result.version} (revise com: revise \"{result.project_dir}\" \"pedido\")")
    print(
        f"Duração: {result.original_duration_sec:.2f}s -> {result.final_duration_sec:.2f}s "
        f"({pct:.1f}% removido)"
    )
    _print_run_summary(result.timings)
    return 0


def _editor_params(args: argparse.Namespace):
    from ai_video_editor.pipeline.editor_ia import DEFAULT_EDITOR_PROVIDER, EditorParams

    provider = args.editor_provider or DEFAULT_EDITOR_PROVIDER
    editor = EditorParams(
        provider=provider, model=args.editor_model, fallback=args.editor_fallback, thinking=args.editor_thinking,
        timeout_sec=args.editor_timeout or default_timeout(provider, CLOUD_TIMEOUT_SEC), api_retries=args.api_retries,
        context_tokens=args.editor_ctx, plan_only=args.so_plano, interval_sec=args.intervalo,
    )
    editor.chain()  # valida provedor e reservas
    return editor


def _editor_context(args: argparse.Namespace, settings, edit_briefing):
    """Pedido, duração alvo e instruções do Editor de IA (CLI > briefing)."""
    from ai_video_editor.editor import EditorContext

    briefing = edit_briefing.briefing
    request = (args.pedido or "").strip() or (briefing.request if briefing is not None else "")
    director = args.instructions or (briefing.director_instructions if briefing is not None else "")
    coverage = briefing.coverage_instructions if briefing is not None else ""
    return EditorContext(
        request=request, target_sec=settings.target_duration_sec.value, director_instructions=director or "",
        coverage_instructions=coverage or "",
    )


def _run_editor(args: argparse.Namespace, video_path: Path, recipe: Recipe, params, settings, edit_briefing) -> int:
    """Receita `editor_ia`: roteiro (com a IA de visão) e o Editor de IA em passadas; `--so-plano` para sem render."""
    from ai_video_editor.intelligence.exceptions import AIProviderError
    from ai_video_editor.pipeline.editor_ia import EditorPipeline

    status = LiveStatus(live=False if args.no_live else None)
    printer = EditProgressPrinter(status, recipe)
    try:
        editor = _editor_params(args)
        describer = _build_photo_describer(args, lambda stage, event, detail: printer("roteiro", "notice", detail))
    except ValueError as e:
        print(f"Erro: parâmetros inválidos: {e}", file=sys.stderr)
        return 1
    context = _editor_context(args, settings, edit_briefing)
    if settings.uses_briefing:
        _print_settings(settings, edit_briefing.path)
    print(f"Receita: {recipe.name}" + (" (só o plano, sem render)" if editor.plan_only else ""))
    if describer is not None:
        print(f"[visão] {' → '.join(describer.model_chain)}", flush=True)
    print(f"[editor] {' → '.join(spec.label for spec in editor.chain())}", flush=True)
    executor = TimelineRenderExecutor(encoder=args.encoder, gpu_workers=args.gpu_paralelo)
    pipeline = EditorPipeline(
        projects_root=Path(args.projects_dir), on_progress=printer, on_stage_progress=printer.progress, recipe=recipe,
        render_executor=executor, describer=describer, vision_parallel=args.visao_paralelo,
    )
    try:
        if not editor.plan_only:
            executor.resolve_encoder()  # `--encoder nvenc` sem GPU falha aqui, antes das etapas
        with status:
            result = pipeline.run(video_path, params, editor, context, force=args.force, settings=settings)
    except AIProviderError as e:
        print(f"Erro da IA: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Aviso: {warning}", file=sys.stderr)
    print("\nPlano do Editor de IA pronto (sem render)." if result.plan_only else "\nEdição concluída!")
    print(f"Projeto: {result.project_dir}")
    print(f"Pasta do editor: {result.work_dir}")
    print(f"Plano: {result.plan_md_path} (dados: {result.plan_path.name})")
    print(f"Relatório: {result.report_path}")
    if result.output_path is not None:
        print(f"Vídeo final: {result.output_path}")
    if result.subtitles_path is not None:
        print(f"Legendas: {result.subtitles_path}")
    print(
        f"Estrutura: {result.blocks} bloco(s), {result.coverages} cobertura(s), {result.breaths} respiro(s)"
        + (f"; FALLBACK em: {', '.join(result.fallbacks)}" if result.fallbacks else "")
    )
    tokens = ""
    if result.input_tokens is not None:
        tokens = f", {_thousands(result.input_tokens)} tokens de entrada e {_thousands(result.output_tokens)} de saída"
    print(f"IA do editor: {result.calls} pedido(s) ({result.cached_calls} do cache){tokens}")
    final = result.final_duration_sec if result.final_duration_sec is not None else result.plan_duration_sec
    print(f"Duração: {result.material_sec:.1f}s de material -> {final:.1f}s")
    _print_run_summary(result.timings)
    return 0


def _print_run_summary(timings: Optional[RunTimings]) -> None:
    """Tempo total da execução e razão em relação à duração do vídeo."""
    if timings is None:
        return
    print(summarize_run(timings) or f"Tempo total: {format_duration(timings.total_sec)}")


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class RevisionProgressPrinter:
    """Mostra o andamento do `revise`: linha ao vivo por etapa e avisos do revisor acima dela."""

    def __init__(self, status: LiveStatus):
        self.status = status
        # Modelo pedido ao revisor, mostrado enquanto ele responde.
        self.model: Optional[str] = None

    def __call__(self, step: str, detail: str) -> None:
        """Resumo de uma etapa: fecha a linha ao vivo dela (ou vira linha avulsa)."""
        line = f"[revisão] {step}: {detail}"
        if self.status.active:
            self.status.end(line)
        else:
            self.status.message(line)

    def start(self, step: str) -> None:
        self.status.begin(f"[revisão] {step}", announce=False)
        if step == "revisor" and self.model:
            self.status.update(note=self.model)

    def progress(self, step: str, progress: StageProgress) -> None:
        self.status.apply(progress)

    def retry(self, chain: ModelChain, event) -> None:
        self.status.message(f"[revisão] revisor: {event.message}")
        self.status.update(note=f"{chain.current.label}, tentativa {event.next_attempt}/{event.total_attempts}")

    def notice(self, chain: ModelChain, message: str) -> None:
        self.status.message(f"[revisão] revisor: {message}")
        if chain.switches:
            # O aviso sai antes da troca; o próximo elo é o destino da última troca registrada.
            self.status.update(note=f"{chain.switches[-1].to_model}, tentativa 1")


def _build_reviser(
    args: argparse.Namespace,
    params: EditPipelineParams,
    progress: Optional[RevisionProgressPrinter] = None,
):
    """Revisor pedido na CLI; sem opções, herda provedor, modelo e reservas do Diretor do projeto."""
    progress = progress or RevisionProgressPrinter(LiveStatus(live=False))
    provider = args.reviser or (params.director if params.director not in ("none", "mock") else "gemini")
    if provider == "mock":
        from ai_video_editor.revision.providers.mock import MockReviser

        return MockReviser()
    from ai_video_editor.revision.providers.llm import LLMReviser

    same_as_director = provider == params.director and args.reviser_model is None
    model = args.reviser_model or (params.director_model if same_as_director else None)
    fallback = args.reviser_fallback
    if fallback is None and same_as_director:
        fallback = params.director_fallback
    chain = ModelChain(
        resolve_chain(provider, model, fallback),
        thinking=args.reviser_thinking or params.director_thinking,
        timeout_sec=args.reviser_timeout or (
            params.director_timeout_sec if provider == params.director
            else default_timeout(provider, params.director_timeout_sec)
        ),
        retry=RetryPolicy(
            max_retries=args.api_retries,
            on_retry=lambda event: progress.retry(chain, event),
        ),
        on_event=lambda message: progress.notice(chain, message),
    )
    return LLMReviser(chain)


def handle_revise(args: argparse.Namespace) -> int:
    """Gera uma nova versão do projeto a partir de um pedido em linguagem natural."""
    project_dir = Path(args.project)
    try:
        from_version = parse_version_ref(args.from_version) if args.from_version else None
        manifest_path = project_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            print(f"Erro: {project_dir} não é uma pasta de projeto (falta {MANIFEST_FILE}).", file=sys.stderr)
            return 1
        if is_multiclip_manifest(manifest_path):
            print(f"Versão não criada: {MULTICLIP_REVISE_MESSAGE}", file=sys.stderr)
            return 1
        params = ProjectManifest.load(manifest_path).params
        status = LiveStatus(live=False if args.no_live else None)
        printer = RevisionProgressPrinter(status)
        reviser = _build_reviser(args, params, printer)
        printer.model = reviser.model_chain[0] if reviser.model_chain else None
        with status:
            result = ProjectReviser(
                reviser,
                on_progress=printer,
                on_step_start=printer.start,
                on_stage_progress=printer.progress,
            ).revise(project_dir, args.request, from_version, loudness=args.loudness)
    except RevisionError as e:
        print(f"Versão não criada: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Aviso: {warning}", file=sys.stderr)
    print(f"\nVersão v{result.version.number} criada (a partir da v{result.base_version}).")
    print(f"Resumo: {result.summary.strip() or '—'}")
    for label, items in (("Removidas", result.removed), ("Restauradas", result.restored)):
        if items:
            print(f"{label}:")
            for item in items:
                print(f"  [{item.id}] {_short(item.text)} — {_short(item.reason, 60)}")
    if result.pacing_before != result.pacing_after:
        print(f"Ritmo das pausas: {result.pacing_before} -> {result.pacing_after}")
    print(
        f"Duração: {result.duration_before_sec:.2f}s -> {result.duration_after_sec:.2f}s "
        f"(v{result.base_version} -> v{result.version.number})"
    )
    print(f"Vídeo final: {result.video_path}")
    print(f"Relatório: {result.report_path}")
    if result.subtitles_path is not None:
        print(f"Legendas: {result.subtitles_path}")
    if result.transcript_path is not None:
        print(f"Transcrição comparativa: {result.transcript_path}")
    _print_run_summary(result.timings)
    return 0


def _print_markdown(text: str) -> None:
    """Imprime sem quebrar em consoles que não têm "→", "—" etc. (ex.: cp1252 no Windows)."""
    print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))


def handle_benchmark_transcription(args: argparse.Namespace) -> int:
    """Mede velocidade e WER de configurações de transcrição em um ou mais áudios."""
    from datetime import date

    from ai_video_editor.benchmark import (
        BenchmarkConfig,
        BenchmarkItem,
        BenchmarkManifest,
        run_benchmark,
        save_run,
    )

    if bool(args.manifest) == bool(args.audio):
        print("Erro: informe --audio (um item) ou --manifest (vários itens), não os dois.", file=sys.stderr)
        return 1
    try:
        configs = [
            BenchmarkConfig.parse(spec)
            for chunk in args.configs
            for spec in chunk.split(",")
            if spec.strip()
        ]
        if not configs:
            raise ValueError("--configs não pode ser vazio")
        if args.manifest:
            manifest_path = Path(args.manifest)
            items = BenchmarkManifest.load(manifest_path).items
            default_name = manifest_path.stem
        else:
            audio = Path(args.audio)
            items = [
                BenchmarkItem(
                    name=audio.stem,
                    audio=audio,
                    reference=Path(args.reference) if args.reference else None,
                    language=args.language,
                )
            ]
            default_name = audio.stem
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    for item in items:
        for label, path in (("áudio", item.audio), ("referência", item.reference)):
            if path is not None and not path.is_file():
                print(f"Erro: {label} do item '{item.name}' não encontrado: {path}", file=sys.stderr)
                return 1

    name = args.name or default_name
    out_dir = Path(args.results_dir) / f"{date.today().isoformat()}-{name}"
    print(f"Benchmark '{name}': {len(configs)} configuração(ões) × {len(items)} item(ns)")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = run_benchmark(name, configs, items, on_progress=print, out_dir=out_dir)
    markdown = save_run(run, out_dir)
    _print_markdown(f"\n{markdown}")
    print(f"Resultados em: {out_dir}")
    return 0 if any(config_run.status == "ok" for config_run in run.configs) else 1


def handle_benchmark_rescore(args: argparse.Namespace) -> int:
    """Recalcula WER (bruto e normalizado) de uma execução a partir dos textos salvos, sem transcrever."""
    from ai_video_editor.benchmark import RescoreError, rescore_run, save_run

    run_dir = Path(args.run_dir)
    try:
        run = rescore_run(run_dir, Path(args.reference) if args.reference else None)
    except (RescoreError, ValueError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    _print_markdown(save_run(run, run_dir))
    print(f"Resultados recalculados em: {run_dir}")
    return 0


def handle_versions(args: argparse.Namespace) -> int:
    """Lista as versões de um projeto (migra projetos antigos para v1)."""
    project_dir = Path(args.project)
    manifest_path = project_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        print(f"Erro: {project_dir} não é uma pasta de projeto (falta {MANIFEST_FILE}).", file=sys.stderr)
        return 1
    try:
        if is_multiclip_manifest(manifest_path):
            manifest = SpineManifest.load(manifest_path)
        else:
            manifest = ensure_versions(project_dir, ProjectManifest.load(manifest_path), manifest_path)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    print(f"{'Versão':<8}{'Duração':>10}{'Cortes':>8}  Pedido")
    for v in manifest.versions:
        mark = "*" if v.number == manifest.current_version else " "
        print(
            f"{mark}v{v.number:<6}{format_timestamp(v.duration_sec):>10}{v.cuts:>8}  {_short(v.request, 80)}"
        )
    print("\n* = versão atual")
    return 0


class CatalogProgressPrinter:
    """Andamento das etapas do `catalog` (linha ao vivo em um terminal)."""

    def __init__(self, status: LiveStatus):
        self.status = status

    def __call__(self, stage: str, event: str, detail: str) -> None:
        from ai_video_editor.catalog import CATALOG_STAGE_LABELS, CATALOG_STAGES

        step = f"[{CATALOG_STAGES.index(stage) + 1}/{len(CATALOG_STAGES)}] {CATALOG_STAGE_LABELS[stage]}"
        if event == "start":
            self.status.begin(step)
        elif event == "done":
            self.status.end(f"{step}: ok — {detail}")
        elif event == "cached":
            self.status.end(f"{step}: em cache")
        elif event == "notice":
            self.status.message(f"{step}: {detail}")

    def progress(self, stage: str, progress: StageProgress) -> None:
        self.status.apply(progress)


def handle_catalog(args: argparse.Namespace) -> int:
    """Cataloga uma pasta de clipes (sem modificá-la): papéis, IDs, dados técnicos, datas, fala, miniaturas."""
    from ai_video_editor.catalog import CatalogBuilder
    from ai_video_editor.catalog.builder import CATALOG_DIR, CATALOG_MD
    from ai_video_editor.catalog.roles import parse_clock_adjustment, parse_role_override

    source = Path(args.folder)
    if not source.is_dir():
        print(f"Erro: pasta não encontrada: {source}", file=sys.stderr)
        return 1
    try:
        overrides = dict(parse_role_override(text) for text in args.papel or [])
        adjustments = dict(parse_clock_adjustment(text) for text in args.ajuste_relogio or [])
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    status = LiveStatus(live=False if args.no_live else None)
    printer = CatalogProgressPrinter(status)
    builder = CatalogBuilder(
        projects_root=Path(args.projects_dir), on_progress=printer, on_stage_progress=printer.progress
    )
    try:
        with status:
            catalog = builder.build(source, overrides, adjustments, force=args.force)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    catalog_dir = catalog.project_dir / CATALOG_DIR
    valid = [c for c in catalog.clips if c.error is None]
    print("\nCatálogo concluído!")
    print(f"Projeto: {catalog.project_dir}")
    print(f"Catálogo: {catalog_dir / CATALOG_MD}")
    counts = ", ".join(
        f"{role} {len(catalog.clips_of(role))}" for role in ("principal", "drone", "pov", "broll", "outro")
        if catalog.clips_of(role)
    )
    print(f"Clipes: {len(valid)} ({counts})")
    for role, name in catalog.contact_sheets.items():
        print(f"Folha de contato ({role}): {catalog_dir / name}")
    if catalog.warnings:
        print(f"Avisos ({len(catalog.warnings)}):")
        for warning in catalog.warnings:
            print(f"  - {warning}")
    print(f"Tempo total: {format_duration(catalog.total_sec)}")
    return 0


def _build_producer(args: argparse.Namespace):
    """Produtor com as opções do Diretor; None com `--director-provider none` (só a regra determinística)."""
    if args.director == "none":
        return None
    if args.director == "mock":
        from ai_video_editor.producer.providers.mock import MockProducer

        return MockProducer()
    from ai_video_editor.producer.providers.llm import LLMProducer

    chain = ModelChain(
        resolve_chain(args.director, args.director_model, args.director_fallback),
        thinking=args.director_thinking,
        timeout_sec=_director_timeout(args),
        retry=RetryPolicy(
            max_retries=args.api_retries,
            on_retry=lambda event: print(f"[produtor] {event.message}", flush=True),
        ),
        on_event=lambda message: print(f"[produtor] {message}", flush=True),
    )
    return LLMProducer(chain)


def _origin_label(briefing) -> str:
    if briefing.origin == "ia":
        label = f"IA ({briefing.effective_model or briefing.provider})"
    else:
        label = f"regra sem IA: {briefing.origin_reason}"
    return f"--receita; resto por {label}" if briefing.forced_recipe else label


def handle_briefing(args: argparse.Namespace) -> int:
    """Escreve o briefing (receita, metas, instruções) para um vídeo ou pasta de clipes, sem editar nada."""
    from ai_video_editor.producer import BRIEFING_FILE, run_producer
    from ai_video_editor.producer.material import load_material

    target = Path(args.target)
    if not target.exists():
        print(f"Erro: arquivo ou pasta não encontrado: {target}", file=sys.stderr)
        return 1
    if args.receita is not None:
        try:
            get_recipe(args.receita)
        except RecipeError as e:
            print(f"Erro: {e}", file=sys.stderr)
            return 1

    status = LiveStatus(live=False if args.no_live else None)
    printer = CatalogProgressPrinter(status)
    try:
        with status:
            material = load_material(
                target, Path(args.projects_dir), on_progress=printer, on_stage_progress=printer.progress
            )
        producer = _build_producer(args)
        if producer is not None and producer.model_chain:
            print(f"[produtor] {producer.model_chain[0]}", flush=True)
        briefing = run_producer(producer, args.pedido or "", material.summary, args.receita)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    path = material.project_dir / BRIEFING_FILE
    briefing.save(path)
    if material.catalog is not None:
        print(f"Catálogo ({material.source}): {material.project_dir / 'catalog'}")
    estimate = briefing.estimate
    target_text = format_duration(briefing.target_duration_sec) if briefing.target_duration_sec else "livre"
    print()
    print(briefing.understood)
    print(f"Receita: {briefing.recipe} ({_origin_label(briefing)})")
    print(f"Tipo de vídeo: {briefing.video_type}")
    print(f"Metas: duração {target_text} | intensidade {briefing.intensity} | ritmo {briefing.pace}")
    print(f"Papéis: {', '.join(briefing.roles_to_use)}")
    if briefing.director_instructions:
        print(f"Diretor de cortes: {briefing.director_instructions}")
    if briefing.coverage_instructions:
        print(f"Cobertura: {briefing.coverage_instructions}")
    steps = ", ".join(estimate.ai_steps) or "nenhuma"
    print(
        f"Estimativa: {estimate.clips} clipe(s); {format_duration(estimate.audio_to_transcribe_sec)} de áudio a "
        f"transcrever ({estimate.clips_to_transcribe} clipe(s)); {estimate.ai_calls} chamada(s) de IA prevista(s) "
        f"na edição ({steps}; até {estimate.ai_calls_max} com novas tentativas); Produtor: "
        f"{estimate.producer_calls} chamada(s)"
    )
    if briefing.warnings:
        print(f"Avisos ({len(briefing.warnings)}):")
        for warning in briefing.warnings:
            print(f"  - {warning}")
    print(f"Briefing: {path}")
    return 0


class VisualProgressPrinter:
    """Andamento das etapas do `visual` (linha ao vivo em um terminal)."""

    def __init__(self, status: LiveStatus):
        self.status = status

    def step(self, stage: str) -> str:
        from ai_video_editor.visual import VISUAL_STAGE_LABELS, VISUAL_STAGES

        return f"[{VISUAL_STAGES.index(stage) + 1}/{len(VISUAL_STAGES)}] {VISUAL_STAGE_LABELS[stage]}"

    def __call__(self, stage: str, event: str, detail: str) -> None:
        step = self.step(stage)
        if event == "start":
            self.status.begin(step)
        elif event == "done":
            self.status.end(f"{step}: ok — {detail}")
        elif event == "skipped":
            self.status.message(f"{step}: pulada ({detail})")
        elif event == "notice":
            self.status.message(f"{step}: {detail}")

    def progress(self, stage: str, progress: StageProgress) -> None:
        self.status.apply(progress)


def _parse_roles(text: Optional[str]) -> list[str]:
    from ai_video_editor.catalog.roles import role_for_folder
    from ai_video_editor.catalog.schema import ROLES

    if not text or text.strip().lower() in ("todos", "all"):
        return list(ROLES)
    roles: list[str] = []
    for item in text.split(","):
        name = item.strip().lower()
        if not name:
            continue
        role = name if name in ROLES else role_for_folder(name)
        if role not in ROLES or (name not in ROLES and role == "outro"):
            raise ValueError(f"papel desconhecido: {item.strip()!r} (papéis: {', '.join(ROLES)})")
        if role not in roles:
            roles.append(role)
    return roles


def _build_visual_describer(args: argparse.Namespace, printer: "VisualProgressPrinter"):
    """Descritor visual pedido na CLI (None com `--visual-provider none`)."""
    from ai_video_editor.visual import VisualDescriber, mock_visual_responder

    provider = args.visual_provider
    if provider == "none":
        return None
    clients = None
    if provider == "mock":
        from ai_video_editor.llm.providers.mock import MockLLMClient

        clients = {"mock": MockLLMClient(responses=[mock_visual_responder])}
    chain = ModelChain(
        resolve_chain(provider, args.visual_model, args.visual_fallback),
        thinking=args.visual_thinking,
        timeout_sec=args.visual_timeout or default_timeout(provider, VISUAL_TIMEOUT_SEC),
        retry=RetryPolicy(
            max_retries=args.api_retries,
            on_retry=lambda event: printer(
                "ai", "notice", f"{event.message} (tentativa {event.next_attempt}/{event.total_attempts})"
            ),
        ),
        on_event=lambda message: printer("ai", "notice", message),
        clients=clients,
    )
    return VisualDescriber(
        chain, detail=args.detalhe_imagem, on_retry=lambda message: printer("ai", "notice", message)
    )


def visual_parallel(args: argparse.Namespace) -> int:
    """Lotes em paralelo do `visual`: com o Ollama (uma GPU), 1 por padrão, com aviso se pedirem mais."""
    if args.visual_provider != "ollama":
        return args.visual_paralelo or DEFAULT_AI_PARALLEL
    if args.visual_paralelo is None:
        return 1
    if args.visual_paralelo > 1:
        print(
            f"Aviso: --visual-paralelo {args.visual_paralelo} com o Ollama: os lotes disputam a mesma GPU "
            "(o servidor os enfileira ou divide a memória); o padrão com ollama é 1.",
            file=sys.stderr,
        )
    return args.visual_paralelo


def _thousands(value: Optional[int]) -> str:
    return "—" if value is None else f"{value:,}".replace(",", ".")


def handle_visual(args: argparse.Namespace) -> int:
    """Tomadas, miniaturas, medidas e descrição por IA dos clipes de um projeto."""
    from ai_video_editor.intelligence.exceptions import AIProviderError
    from ai_video_editor.visual import VisualBuilder
    from ai_video_editor.visual.builder import VISUAL_DIR, VISUAL_JSON, VISUAL_MD
    from ai_video_editor.visual.describer import BATCH_MAX_IMAGES, BATCH_MAX_SHOTS, IMAGE_TOKENS
    from ai_video_editor.visual.report import batch_timing_line

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Erro: pasta não encontrada: {folder}", file=sys.stderr)
        return 1
    if args.visual_provider == "ollama" and args.detalhe_imagem != "auto":
        print(f"Aviso: --detalhe-imagem {args.detalhe_imagem} é ignorado pelo Ollama (não tem esse parâmetro).",
              file=sys.stderr)
    try:
        roles = _parse_roles(args.papeis)
        status = LiveStatus(live=False if args.no_live else None)
        printer = VisualProgressPrinter(status)
        describer = _build_visual_describer(args, printer)
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    from ai_video_editor.catalog import CatalogBuilder

    builder = VisualBuilder(
        projects_root=Path(args.projects_dir),
        describer=describer,
        detail=args.detalhe_imagem,
        on_progress=printer,
        on_stage_progress=printer.progress,
        catalog_builder=CatalogBuilder(projects_root=Path(args.projects_dir)),
        ai_parallel=visual_parallel(args),
    )
    try:
        with status:
            result = builder.build(folder, roles=roles, estimate_only=args.so_estimar, force=args.force)
    except AIProviderError as e:
        print(f"Erro da IA: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    if result.estimate is not None:
        est = result.estimate
        per_role = ", ".join(f"{role} {n}" for role, n in est.shots_by_role.items())
        print("\nEstimativa (nenhuma chamada à IA foi feita):")
        print(f"Projeto: {result.project_dir}")
        print(f"Tomadas: {sum(est.shots_by_role.values())} ({per_role})")
        if est.cached_shots:
            print(f"Já analisadas (cache, não serão enviadas): {est.cached_shots}")
        print(f"A analisar: {est.shots} tomadas, {est.images} imagens")
        print(f"Chamadas: {est.calls} (lotes de até {BATCH_MAX_SHOTS} tomadas / {BATCH_MAX_IMAGES} imagens; "
              "sem contar novas tentativas)")
        print(
            f"Tokens estimados: ~{_thousands(est.input_tokens)} de entrada "
            f"({IMAGE_TOKENS[est.image_detail]} por imagem, detalhe {est.image_detail}, referência Gemini 3) "
            f"e ~{_thousands(est.output_tokens)} de saída"
        )
        print("Consulte os preços do provedor para converter em custo.")
        return 0

    visual = result.visual
    shots = visual.all_shots()
    visual_dir = result.project_dir / VISUAL_DIR
    print("\nAnálise visual concluída!")
    print(f"Projeto: {result.project_dir}")
    print(f"Relatório: {visual_dir / VISUAL_MD}")
    print(f"Dados: {visual_dir / VISUAL_JSON}")
    analysed = sum(1 for s in shots if s.status == "analisada")
    missing = sum(1 for s in shots if s.status == "sem_analise")
    line = f"Tomadas: {len(shots)} em {len(visual.clips)} clipe(s), {analysed} analisadas"
    if missing:
        line += f", {missing} sem análise"
    print(line)
    ai = visual.ai
    if ai is not None:
        print(f"Modelo efetivo: {ai.effective_model or '— (nenhuma chamada)'}"
              + (f" (raciocínio {ai.thinking})" if ai.thinking else ""))
        print(f"IA: {ai.batches} lote(s), {ai.api_calls} chamada(s), {ai.images_sent} imagens; "
              f"tokens {_thousands(ai.input_tokens)} de entrada / {_thousands(ai.output_tokens)} de saída")
        for switch in ai.model_switches:
            print(f"Troca de modelo: {switch.from_model} → {switch.to_model} ({switch.reason})")
        timing = batch_timing_line(ai)
        if timing:
            print(timing)
    for warning in visual.warnings:
        print(f"Aviso: {warning}")
    print(f"Tempo total: {format_duration(visual.total_sec)}")
    return 0


def handle_ollama_check(args: argparse.Namespace) -> int:
    """Confere o Ollama local: servidor, modelos baixados e chamadas mínimas (texto e imagem) com JSON validado."""
    from ai_video_editor.llm.ollama_check import run_ollama_check

    try:
        report = run_ollama_check(
            text_model=args.modelo_texto,
            image_model=None if args.modelo_imagem.lower() == "none" else args.modelo_imagem,
            timeout_sec=args.timeout,
            on_line=lambda line: print(line, flush=True),
        )
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    return 0 if report.ok else 1


class RoteiroProgressPrinter:
    """Andamento das etapas do `roteiro` (linha ao vivo em um terminal)."""

    def __init__(self, status: LiveStatus):
        self.status = status

    def step(self, stage: str) -> str:
        from ai_video_editor.roteiro import ROTEIRO_STAGE_LABELS, ROTEIRO_STAGES

        return f"[{ROTEIRO_STAGES.index(stage) + 1}/{len(ROTEIRO_STAGES)}] {ROTEIRO_STAGE_LABELS[stage]}"

    def __call__(self, stage: str, event: str, detail: str) -> None:
        step = self.step(stage)
        if event == "start":
            self.status.begin(step)
        elif event == "done":
            self.status.end(f"{step}: ok — {detail}")
        elif event == "skipped":
            self.status.message(f"{step}: pulada ({detail})")
        elif event == "notice":
            self.status.message(f"{step}: {detail}")

    def progress(self, stage: str, progress: StageProgress) -> None:
        self.status.apply(progress)


# Modelo padrão da IA de visão por provedor (os outros usam o padrão do provedor).
DEFAULT_VISION_MODELS = {"ollama": DEFAULT_CHECK_IMAGE_MODEL}


def _default_vision_provider() -> Optional[str]:
    """`ollama` (Gemma 3 4B local) quando o provedor está registrado; senão é preciso escolher."""
    return "ollama" if "ollama" in provider_names() else None


def _build_photo_describer(args: argparse.Namespace, printer: "RoteiroProgressPrinter"):
    """IA de visão pedida na CLI (None com `--visao-provider none`)."""
    from ai_video_editor.roteiro import PhotoDescriber, mock_vision_responder

    provider = args.visao_provider or _default_vision_provider()
    if provider is None:
        raise ValueError(
            "o provedor ollama (IA de visão local) ainda não está disponível nesta instalação; "
            "escolha --visao-provider (gemini, openai, mock ou none)"
        )
    if provider == "none":
        return None
    clients = None
    if provider == "mock":
        from ai_video_editor.llm.providers.mock import MockLLMClient

        clients = {"mock": MockLLMClient(responses=[mock_vision_responder])}
    chain = ModelChain(
        resolve_chain(provider, args.visao_model or DEFAULT_VISION_MODELS.get(provider), "none"),
        thinking=args.visao_thinking,
        timeout_sec=args.visao_timeout,
        retry=RetryPolicy(
            max_retries=args.api_retries,
            on_retry=lambda event: printer(
                "vision", "notice", f"{event.message} (tentativa {event.next_attempt}/{event.total_attempts})"
            ),
        ),
        on_event=lambda message: printer("vision", "notice", message),
        clients=clients,
    )
    return PhotoDescriber(chain, on_retry=lambda message: printer("vision", "notice", message))


def handle_roteiro(args: argparse.Namespace) -> int:
    """Roteiro unificado (fala + imagem + ritmo) de um arquivo ou de uma pasta de clipes, com IDs."""
    from ai_video_editor.intelligence.exceptions import AIProviderError
    from ai_video_editor.roteiro import ROTEIRO_STAGE_LABELS, RoteiroBuilder, RoteiroParams
    from ai_video_editor.roteiro.describer import IMAGE_TOKENS_ESTIMATE, SECONDS_PER_PHOTO_ESTIMATE

    target = Path(args.target)
    if not target.exists():
        print(f"Erro: arquivo ou pasta não encontrado: {target}", file=sys.stderr)
        return 1
    status = LiveStatus(live=False if args.no_live else None)
    printer = RoteiroProgressPrinter(status)
    try:
        describer = _build_photo_describer(args, printer)
        params = RoteiroParams(
            interval_sec=args.intervalo,
            threshold_db=args.threshold_db,
            min_silence_sec=args.min_silence_sec,
            transcription=TranscriptionConfig(
                model_size=args.whisper_model,
                language=args.language,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
            ),
        )
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    if describer is not None:
        print(f"[visão] {' → '.join(describer.model_chain)}", flush=True)
    builder = RoteiroBuilder(
        projects_root=Path(args.projects_dir),
        describer=describer,
        on_progress=printer,
        on_stage_progress=printer.progress,
        vision_parallel=args.visao_paralelo,
    )
    try:
        with status:
            result = builder.build(target, params, estimate_only=args.so_estimar, force=args.force)
    except AIProviderError as e:
        print(f"Erro da IA: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1

    times = ", ".join(
        f"{ROTEIRO_STAGE_LABELS[t.name].lower()} {format_duration(t.elapsed_sec)}" for t in result.stage_times
    )
    if result.estimate is not None:
        est = result.estimate
        print("\nEstimativa (a IA de visão não foi chamada):")
        print(f"Projeto: {result.project_dir}")
        print(f"Fotos: {est.photos_before} extraídas, {est.photos_kept} depois do descarte das quase iguais")
        if est.cached:
            print(f"Já descritas (cache, não serão enviadas): {est.cached}")
        print(f"A descrever: {est.to_describe} fotos = {est.calls} chamadas (sem contar novas tentativas)")
        print(
            f"Tempo estimado: ~{format_duration(est.seconds)} ({SECONDS_PER_PHOTO_ESTIMATE:g}s por foto, "
            f"{args.visao_paralelo} ao mesmo tempo; medido no gemma3:4b numa RTX 2060)"
        )
        print(
            f"Tokens estimados: ~{_thousands(est.input_tokens)} de entrada (~{IMAGE_TOKENS_ESTIMATE} por imagem) "
            f"e ~{_thousands(est.output_tokens)} de saída"
        )
        print(f"Tempos: {times}")
        return 0

    roteiro = result.roteiro
    before = sum(c.photos_before for c in roteiro.clips)
    described = sum(1 for p in roteiro.photos if p.status == "descrita")
    missing = sum(1 for p in roteiro.photos if p.status == "sem_descricao")
    print("\nRoteiro concluído!")
    print(f"Projeto: {result.project_dir}")
    print(f"Roteiro: {result.md_path}")
    print(f"Dados: {result.json_path}")
    roles = ", ".join(sorted({c.role for c in roteiro.clips}))
    print(f"Clipes: {len(roteiro.clips)} ({roles}), {sum(1 for c in roteiro.clips if c.has_speech)} com fala")
    line = f"Fotos: {before} extraídas, {len(roteiro.photos)} mantidas ({before - len(roteiro.photos)} quase iguais descartadas)"
    if roteiro.vision is not None:
        line += f"; {described} descritas"
        if missing:
            line += f", {missing} sem descrição (pedidas de novo na próxima execução)"
    print(line)
    print(f"Momentos: {len(roteiro.moments)} · Frases: {len(roteiro.sentences)}")
    vision = roteiro.vision
    if vision is not None:
        speed = f", {vision.sec_per_photo:.1f}s por foto".replace(".", ",") if vision.sec_per_photo else ""
        print(
            f"Visão: {vision.effective_model or vision.model_chain[0]} — {vision.described} agora, {vision.cached} do "
            f"cache, {vision.api_calls} chamada(s){speed}; total {format_duration(vision.wall_sec)}"
        )
        for note in vision.notes:
            print(f"Aviso da visão: {note}")
    per_clip = ", ".join(f"{c.id} ~{_thousands(c.estimated_tokens)}" for c in roteiro.clips)
    print(f"Tamanho do roteiro: ~{_thousands(roteiro.estimated_tokens)} tokens (estimativa: caracteres/4; {per_clip})")
    for warning in roteiro.warnings:
        print(f"Aviso: {warning}")
    print(f"Tempos: {times}")
    print(f"Tempo total: {format_duration(roteiro.total_sec)}")
    return 0


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("deve ser 0 ou maior")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("deve ser 1 ou maior")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Constrói o analisador de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        prog="ai-video-editor",
        description="Motor de edição, ingestão técnica e análise multimodal com IA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: render
    render_parser = subparsers.add_parser(
        "render", help="Renderiza um EditPlan ou uma Timeline (com trilhas) a partir de JSON"
    )
    render_parser.add_argument("plan_file", help="Caminho do JSON de EditPlan ou de Timeline")

    # Subcomando: quick-cut
    cut_parser = subparsers.add_parser("quick-cut", help="Corta e concatena dois arquivos de vídeo")
    cut_parser.add_argument("--video1", required=True, help="Caminho do primeiro vídeo")
    cut_parser.add_argument("--start1", type=float, default=0.0, help="Início em segundos para o vídeo 1")
    cut_parser.add_argument("--end1", type=float, required=True, help="Fim em segundos para o vídeo 1")
    cut_parser.add_argument("--video2", required=True, help="Caminho do segundo vídeo")
    cut_parser.add_argument("--start2", type=float, default=0.0, help="Início em segundos para o vídeo 2")
    cut_parser.add_argument("--end2", type=float, required=True, help="Fim em segundos para o vídeo 2")
    cut_parser.add_argument("--output", "-o", required=True, help="Caminho do MP4 de saída")
    cut_parser.add_argument("--width", type=int, default=1280, help="Largura de saída (padrão: 1280)")
    cut_parser.add_argument("--height", type=int, default=720, help="Altura de saída (padrão: 720)")
    cut_parser.add_argument("--fps", type=int, default=30, help="Taxa de quadros de saída (padrão: 30)")

    # Subcomando: analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Ingere e extrai metadados e frames de vídeos"
    )
    analyze_parser.add_argument(
        "videos", nargs="+", help="Caminho(s) para um ou mais arquivos de vídeo a analisar"
    )
    analyze_parser.add_argument(
        "--strategy",
        choices=["interval", "fixed_count"],
        default="interval",
        help="Estratégia de amostragem de frames (padrão: interval)",
    )
    analyze_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Intervalo em segundos para strategy=interval (padrão: 2.0)",
    )
    analyze_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Quantidade de frames desejada para strategy=fixed_count (padrão: 5)",
    )
    analyze_parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Teto máximo de frames por vídeo (padrão: 50)",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Formato dos frames extraídos (padrão: jpg)",
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="output/analysis",
        help="Diretório base de saída (padrão: output/analysis)",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Exibe relatório JSON bruto no terminal",
    )

    # Subcomando: report-ai
    report_ai_parser = subparsers.add_parser(
        "report-ai", help="Interpreta frames com IA multimodal e gera relatório Markdown"
    )
    report_ai_parser.add_argument(
        "--report",
        required=True,
        help="Caminho do analysis_report.json produzido pela etapa de análise técnica",
    )
    report_ai_parser.add_argument(
        "--provider",
        choices=["gemini", "mock"],
        default="gemini",
        help="Provedor de IA a utilizar (padrão: gemini)",
    )
    report_ai_parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="Identificador do modelo (padrão: gemini-3.8-flash)",
    )
    report_ai_parser.add_argument(
        "--max-images",
        type=int,
        default=15,
        help="Limite máximo de imagens por requisição (padrão: 15)",
    )
    report_ai_parser.add_argument(
        "--proxy-dim",
        type=int,
        default=1024,
        help="Dimensão máxima em pixels para proxies leves (padrão: 1024)",
    )

    # Subcomando: edit
    edit_parser = subparsers.add_parser(
        "edit",
        help="Remove as pausas de um vídeo falado, ou monta a espinha falada de uma pasta de clipes "
        "(pasta de projeto com cache)",
    )
    edit_parser.add_argument(
        "video", help="Vídeo a editar, ou pasta de clipes (vlog: clipes principais com fala, em ordem)"
    )
    edit_parser.add_argument(
        "--receita",
        default=None,
        metavar="NOME",
        help=f"Receita (combinação de etapas) a executar; disponíveis: {', '.join(recipe_names())} "
        f"(padrão: {DEFAULT_RECIPE} para um vídeo, {DEFAULT_FOLDER_RECIPE} para uma pasta)",
    )
    edit_parser.add_argument(
        "--threshold-db",
        type=_threshold_arg,
        default="auto",
        help="Limiar de silêncio em dB, ou `auto` para medir o ruído do vídeo (padrão: auto)",
    )
    edit_parser.add_argument(
        "--min-silence-sec",
        type=float,
        default=0.5,
        help="Duração mínima para detectar um silêncio (padrão: 0.5)",
    )
    edit_parser.add_argument(
        "--padding-before", type=float, default=0.15, help="Folga antes de cada fala (padrão: 0.15)"
    )
    edit_parser.add_argument(
        "--padding-after", type=float, default=0.20, help="Folga depois de cada fala (padrão: 0.20)"
    )
    edit_parser.add_argument(
        "--min-gap",
        type=float,
        default=0.30,
        help="Pausas menores que isso (após o padding) não são cortadas (padrão: 0.30)",
    )
    edit_parser.add_argument(
        "--min-keep",
        type=float,
        default=0.10,
        help="Falas mais curtas que isso são tratadas como ruído (padrão: 0.10)",
    )
    edit_parser.add_argument(
        "--transcribe", action="store_true", help="Gera transcript.json com faster-whisper"
    )
    edit_parser.add_argument(
        "--language", default=None, help="Idioma da transcrição (ex: pt); padrão: automático"
    )
    edit_parser.add_argument(
        "--whisper-model",
        default="auto",
        help="Modelo do faster-whisper: auto (large-v3-turbo na GPU, small na CPU) ou um nome "
        "explícito, ex: small, medium (padrão: auto)",
    )
    edit_parser.add_argument(
        "--whisper-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo do faster-whisper: auto usa a GPU NVIDIA se houver, senão CPU (padrão: auto)",
    )
    edit_parser.add_argument(
        "--whisper-compute-type",
        default="auto",
        help="Precisão do faster-whisper: auto = float16 na GPU e int8 na CPU (padrão: auto)",
    )
    edit_parser.add_argument(
        "--director-provider",
        "--director",
        dest="director",
        choices=["none", "mock", *_llm_providers()],
        default="none",
        help=(
            "Provedor do Diretor de IA que remove frases repetidas/recomeçadas (requer --transcribe; "
            "mock = simulado, sem IA; padrão: none)"
        ),
    )
    edit_parser.add_argument(
        "--director-model",
        default=None,
        help="Modelo do Diretor (padrão: o do provedor; gemini-3.8-flash no Gemini, gpt-6-luna na OpenAI)",
    )
    edit_parser.add_argument(
        "--director-fallback",
        default=None,
        metavar="LISTA",
        help=(
            "Modelos de reserva se o principal ficar indisponível, ex: "
            "\"gemini:gemini-3.6-flash,openai:gpt-6-luna\"; \"none\" desliga "
            "(padrão no Gemini: gemini-3.6-flash, gemini-3.5-flash-lite)"
        ),
    )
    edit_parser.add_argument(
        "--director-thinking",
        choices=["off", "low", "high"],
        default="low",
        help="Raciocínio ('thinking') do Diretor Gemini: off = mínimo (nem todo modelo aceita; sobe de nível sozinho) (padrão: low)",
    )
    edit_parser.add_argument(
        "--director-timeout",
        type=float,
        default=None,
        help=f"Tempo limite em segundos da chamada ao Diretor ({TIMEOUT_HELP})",
    )
    edit_parser.add_argument(
        "--coverage-provider", "--coverage", dest="coverage", choices=["none", "mock", *_llm_providers()], default=None,
        help="Diretor de cobertura (receita vlog_cobertura): none = só as janelas obrigatórias, pelo código; "
        "mock = simulado (padrão: o do Diretor)",
    )
    edit_parser.add_argument(
        "--coverage-model", default=None, help="Modelo do Diretor de cobertura (padrão: o do Diretor, se o provedor for o mesmo)"
    )
    edit_parser.add_argument(
        "--coverage-fallback", default=None, metavar="LISTA",
        help="Reservas do Diretor de cobertura, \"provedor:modelo,...\" ou \"none\" (padrão: as do Diretor)",
    )
    edit_parser.add_argument(
        "--coverage-thinking", choices=["off", "low", "high"], default=None,
        help="Raciocínio do Diretor de cobertura (padrão: o do Diretor)",
    )
    edit_parser.add_argument(
        "--coverage-timeout", type=float, default=None,
        help="Tempo limite em segundos da chamada ao Diretor de cobertura (padrão: o do Diretor)",
    )
    edit_parser.add_argument(
        "--pedido", default=None,
        help="Pedido livre (ex: \"vlog de 8 minutos, ritmo rápido\"): o Produtor escreve o briefing.json do projeto "
        "(reaproveitado enquanto o pedido e o material não mudam), que dá receita, intensidade, ritmo, "
        "instruções e duração alvo; com --sem-briefing, vai direto ao Diretor",
    )
    edit_parser.add_argument(
        "--sem-briefing", action="store_true",
        help="Não usa nem gera o briefing.json (valores só da CLI e padrões)",
    )
    edit_parser.add_argument(
        "--intensity", "--intensidade", dest="intensity",
        choices=["light", "medium", "aggressive"],
        default=None,
        help="Quanto o Diretor deve cortar (padrão: a do briefing, senão light)",
    )
    edit_parser.add_argument(
        "--ritmo", "--pace", dest="ritmo", choices=["natural", "normal", "tight"], default=None,
        help="Ritmo das pausas: natural (mais folga), normal ou tight (menos) (padrão: o do briefing, senão normal)",
    )
    edit_parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Duração final desejada em segundos, informada ao Diretor (padrão: a do briefing, senão livre)",
    )
    edit_parser.add_argument(
        "--instructions", dest="instructions", default=None,
        help="Instruções diretas ao Diretor (ex: 'corte as piadas'); valem mais que as do briefing",
    )
    edit_parser.add_argument(
        "--keep-fillers",
        action="store_true",
        help="Impede o Diretor de remover frases só por serem enrolação",
    )
    edit_parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Não gera o .srt (por padrão sai ao lado do vídeo quando há --transcribe)",
    )
    edit_parser.add_argument(
        "--subtitle-max-chars",
        type=int,
        default=SubtitleParams().max_chars_per_line,
        help="Máximo de caracteres por linha da legenda (padrão: 42)",
    )
    edit_parser.add_argument(
        "--transcript-min-pause",
        type=float,
        default=1.0,
        help="Pausas removidas a partir desta duração (s) aparecem na transcrição comparativa (padrão: 1.0)",
    )
    edit_parser.add_argument(
        "--loudness",
        choices=LOUDNESS_CHOICES,
        default="voz",
        help=(
            "Normaliza o volume do vídeo final (EBU R128, loudnorm em duas passadas): "
            "voz = -16 LUFS (padrão), youtube = -14, broadcast = -23, off = volume original"
        ),
    )
    edit_parser.add_argument(
        "--force", action="store_true", help="Refaz todas as etapas, ignorando o cache"
    )
    editor_group = edit_parser.add_argument_group("Editor de IA (receita editor_ia)")
    editor_group.add_argument(
        "--editor-provider", choices=["mock", *_llm_providers()], default=None,
        help="IA que decide a edição em passadas (resumos, estrutura, detalhe por bloco); mock = simulada, sem rede "
        "(padrão: ollama)",
    )
    editor_group.add_argument(
        "--editor-model", default=None, help="Modelo do editor (padrão: qwen2.5:7b no ollama; o do provedor nos outros)"
    )
    editor_group.add_argument(
        "--editor-fallback", default=None, metavar="LISTA",
        help="Reservas do editor, \"provedor:modelo,...\" ou \"none\" (padrão: as do provedor)",
    )
    editor_group.add_argument(
        "--editor-thinking", choices=["off", "low", "high"], default="low",
        help="Raciocínio do editor, se o modelo tiver (padrão: low; o ollama ignora)",
    )
    editor_group.add_argument(
        "--editor-timeout", type=float, default=None, help=f"Tempo limite em segundos de cada pedido ao editor ({TIMEOUT_HELP})"
    )
    editor_group.add_argument(
        "--editor-ctx", type=_ctx_tokens, default=None, metavar="TOKENS",
        help="Contexto que o editor considera para dividir os pedidos (padrão: o num_ctx do ollama; 32768 na nuvem)",
    )
    editor_group.add_argument(
        "--so-plano", action="store_true",
        help="Para depois da montagem: plano (editor_plano.json/.md), Timeline e relatório, sem render nem legendas",
    )
    editor_group.add_argument(
        "--intervalo", type=float, default=2.0, metavar="SEG",
        help="Roteiro: uma foto a cada SEG segundos, além das extras (padrão: 2, o mesmo do comando roteiro)",
    )
    editor_group.add_argument(
        "--visao-provider", choices=["none", "mock", *_llm_providers()], default=None,
        help="Roteiro: IA que descreve as fotos; none = sem descrição (padrão: ollama, Gemma 3 4B local)",
    )
    editor_group.add_argument(
        "--visao-model", default=None, help="Roteiro: modelo da IA de visão (padrão: gemma3:4b no ollama)"
    )
    editor_group.add_argument(
        "--visao-thinking", choices=["off", "low", "high"], default="off", help="Roteiro: raciocínio da IA de visão"
    )
    editor_group.add_argument(
        "--visao-timeout", type=float, default=120.0, help="Roteiro: tempo limite em segundos de cada foto (padrão: 120)"
    )
    editor_group.add_argument(
        "--visao-paralelo", type=_positive_int, default=1, metavar="N",
        help="Roteiro: clipes descritos ao mesmo tempo (padrão 1, GPU local)",
    )
    edit_parser.add_argument(
        "--projects-dir",
        default="output/projects",
        help="Diretório base dos projetos (padrão: output/projects)",
    )

    # Subcomando: revise
    revise_parser = subparsers.add_parser(
        "revise", help="Ajusta uma edição com um pedido em linguagem natural (gera nova versão)"
    )
    revise_parser.add_argument("project", help="Pasta do projeto (criada pelo comando edit)")
    revise_parser.add_argument("request", help='Pedido, ex: "corta menos" ou "tira a parte do site"')
    revise_parser.add_argument(
        "--from", dest="from_version", default=None, metavar="vN",
        help="Versão em que a revisão se baseia (padrão: a atual)",
    )
    revise_parser.add_argument(
        "--reviser-provider", "--reviser", dest="reviser", choices=["mock", *_llm_providers()], default=None,
        help="Provedor do revisor (padrão: o do Diretor do projeto, ou gemini)",
    )
    revise_parser.add_argument(
        "--reviser-model", default=None, help="Modelo do revisor (padrão: o do Diretor do projeto)"
    )
    revise_parser.add_argument(
        "--reviser-fallback", default=None, metavar="LISTA",
        help="Modelos de reserva \"provedor:modelo,...\"; \"none\" desliga (padrão: as do Diretor do projeto)",
    )
    revise_parser.add_argument(
        "--reviser-thinking", choices=["off", "low", "high"], default=None,
        help="Raciocínio do revisor (padrão: o do Diretor do projeto)",
    )
    revise_parser.add_argument(
        "--loudness",
        choices=LOUDNESS_CHOICES,
        default=None,
        help="Normalização de volume da versão nova: voz/youtube/broadcast/off (padrão: a da versão base)",
    )
    revise_parser.add_argument(
        "--reviser-timeout", type=float, default=None,
        help="Tempo limite em segundos (padrão: o do Diretor do projeto)",
    )

    for ai_parser in (report_ai_parser, edit_parser, revise_parser):
        ai_parser.add_argument(
            "--api-retries",
            type=_non_negative_int,
            default=DEFAULT_API_RETRIES,
            metavar="N",
            help=(
                "Novas tentativas da API de IA em falhas temporárias (503, 504, 429, conexão), "
                f"com espera crescente; 0 desliga (padrão: {DEFAULT_API_RETRIES})"
            ),
        )

    for render_options in (render_parser, edit_parser):
        render_options.add_argument(
            "--encoder", choices=ENCODER_CHOICES, default="auto",
            help=(
                "Codificador de vídeo: auto = NVENC (GPU NVIDIA) se disponível, senão CPU (padrão); "
                "cpu = x264; nvenc = exige a GPU. Se o NVENC falhar no meio, o render inteiro é refeito na CPU"
            ),
        )
        render_options.add_argument(
            "--gpu-paralelo", type=_positive_int, default=DEFAULT_GPU_WORKERS, metavar="N",
            help=f"Pedaços codificados ao mesmo tempo no NVENC (a GeForce limita as sessões; padrão: {DEFAULT_GPU_WORKERS})",
        )

    for live_parser in (edit_parser, revise_parser):
        live_parser.add_argument(
            "--no-live",
            action="store_true",
            help="Sem linha de andamento ao vivo; só as linhas finais de cada etapa "
            "(automático quando a saída não é um terminal)",
        )

    # Subcomando: catalog
    catalog_parser = subparsers.add_parser(
        "catalog", help="Cataloga uma pasta de clipes (papéis, IDs, dados técnicos, datas, fala, miniaturas)"
    )
    catalog_parser.add_argument("folder", help="Pasta com os clipes (subpastas = papéis); não é modificada")
    catalog_parser.add_argument(
        "--papel",
        action="append",
        metavar="PASTA=PAPEL",
        help="Corrige o papel de uma subpasta, ex: --papel \"Celular=broll\" (repetível; papéis: "
        "principal, drone, pov, broll, outro)",
    )
    catalog_parser.add_argument(
        "--ajuste-relogio",
        action="append",
        metavar="PAPEL=AJUSTE",
        help="Corrige o relógio de uma câmera antes de ordenar, ex: drone=+3h, pov=-1h30m (repetível)",
    )
    catalog_parser.add_argument(
        "--force", action="store_true", help="Refaz a análise de todos os clipes, ignorando o cache"
    )
    catalog_parser.add_argument(
        "--projects-dir",
        default="output/projects",
        help="Diretório base dos projetos (padrão: output/projects)",
    )
    catalog_parser.add_argument(
        "--no-live", action="store_true", help="Sem linha de andamento ao vivo; só as linhas finais de cada etapa"
    )

    # Subcomando: briefing
    briefing_parser = subparsers.add_parser(
        "briefing", help="Produtor: lê o pedido e o material e escreve o briefing da edição (não edita nada)"
    )
    briefing_parser.add_argument("target", help="Vídeo ou pasta de clipes (a pasta é catalogada se preciso)")
    briefing_parser.add_argument("--pedido", default="", help='Pedido livre, ex: "vlog de 8 minutos, ritmo rápido"')
    briefing_parser.add_argument(
        "--receita", default=None, metavar="NOME",
        help=f"Força a receita; a IA só preenche o resto (disponíveis: {', '.join(recipe_names())})",
    )
    briefing_parser.add_argument(
        "--director-provider", "--director", dest="director", choices=["none", "mock", *_llm_providers()],
        default="gemini",
        help="Provedor de IA do Produtor (as mesmas opções do Diretor; none = só a regra sem IA; padrão: gemini)",
    )
    briefing_parser.add_argument("--director-model", default=None, help="Modelo (padrão: o do provedor)")
    briefing_parser.add_argument(
        "--director-fallback", default=None, metavar="LISTA",
        help="Modelos de reserva \"provedor:modelo,...\"; \"none\" desliga (padrão: os do provedor)",
    )
    briefing_parser.add_argument(
        "--director-thinking", choices=["off", "low", "high"], default="low", help="Raciocínio (padrão: low)"
    )
    briefing_parser.add_argument(
        "--director-timeout", type=float, default=None, help=f"Tempo limite em segundos da chamada ({TIMEOUT_HELP})"
    )
    briefing_parser.add_argument(
        "--api-retries", type=_non_negative_int, default=DEFAULT_API_RETRIES, metavar="N",
        help=f"Novas tentativas da API em falhas temporárias (padrão: {DEFAULT_API_RETRIES})",
    )
    briefing_parser.add_argument(
        "--projects-dir", default="output/projects", help="Diretório base dos projetos (padrão: output/projects)"
    )
    briefing_parser.add_argument(
        "--no-live", action="store_true", help="Sem linha de andamento ao vivo (se o catálogo precisar rodar)"
    )

    # Subcomando: visual
    visual_parser = subparsers.add_parser(
        "visual", help="Tomadas, miniaturas, medidas e descrição por IA dos clipes de um projeto"
    )
    visual_parser.add_argument(
        "folder", help="Pasta de clipes (a mesma do catalog) ou pasta do projeto; o catálogo é gerado se faltar"
    )
    visual_parser.add_argument(
        "--papeis", default=None, metavar="LISTA",
        help="Papéis a analisar, separados por vírgula, ex: principal,broll (padrão: todos)",
    )
    visual_parser.add_argument(
        "--so-estimar", action="store_true",
        help="Gera tomadas e medidas, mostra nº de tomadas, imagens, chamadas e tokens estimados, e para "
        "(sem chamar a IA)",
    )
    visual_parser.add_argument(
        "--visual-provider", choices=["none", "mock", *_llm_providers()], default="gemini",
        help="Provedor da descrição por IA; none = só tomadas e medidas (padrão: gemini)",
    )
    visual_parser.add_argument("--visual-model", default=None, help="Modelo (padrão: o do provedor)")
    visual_parser.add_argument(
        "--visual-fallback", default=None, metavar="LISTA",
        help="Modelos de reserva \"provedor:modelo,...\"; \"none\" desliga (padrão: os do provedor, como no Diretor)",
    )
    visual_parser.add_argument(
        "--visual-thinking", choices=["off", "low", "high"], default="low",
        help="Raciocínio da descrição (padrão: low)",
    )
    visual_parser.add_argument(
        "--visual-timeout", type=float, default=None,
        help=f"Tempo limite em segundos de cada chamada (padrão: 180; ollama: {DEFAULT_OLLAMA_TIMEOUT_SEC:.0f})",
    )
    visual_parser.add_argument(
        "--detalhe-imagem", choices=["low", "auto", "high"], default="auto",
        help="Resolução com que a IA lê as miniaturas: low = ~4x menos tokens no Gemini 3 (padrão: auto)",
    )
    visual_parser.add_argument(
        "--visual-paralelo", type=_positive_int, default=None, metavar="N",
        help=f"Lotes enviados à IA ao mesmo tempo; 1 = um por vez; cai pela metade se o provedor limitar "
        f"(padrão: {DEFAULT_AI_PARALLEL}; ollama: 1, uma GPU)",
    )
    visual_parser.add_argument(
        "--api-retries", type=_non_negative_int, default=DEFAULT_API_RETRIES, metavar="N",
        help=f"Novas tentativas da API em falhas temporárias (padrão: {DEFAULT_API_RETRIES})",
    )
    visual_parser.add_argument(
        "--force", action="store_true", help="Refaz tomadas e medidas de todos os clipes, ignorando o cache"
    )
    visual_parser.add_argument(
        "--projects-dir", default="output/projects", help="Diretório base dos projetos (padrão: output/projects)"
    )
    visual_parser.add_argument(
        "--no-live", action="store_true", help="Sem linha de andamento ao vivo; só as linhas finais de cada etapa"
    )

    # Subcomando: roteiro
    roteiro_parser = subparsers.add_parser(
        "roteiro",
        help="Roteiro unificado (fala + imagem + ritmo) de um arquivo ou pasta de clipes, com IDs para o Editor de IA",
    )
    roteiro_parser.add_argument(
        "target", help="Vídeo (projeto de 1 clipe) ou pasta de clipes (a mesma do catalog); o catálogo é gerado se faltar"
    )
    roteiro_parser.add_argument(
        "--intervalo", type=float, default=2.0, metavar="SEG",
        help="Uma foto a cada SEG segundos em todos os clipes, além das extras em início de frase e troca de cena "
        "(padrão: 2)",
    )
    roteiro_parser.add_argument(
        "--visao-provider", choices=["none", "mock", *_llm_providers()], default=None,
        help="IA que descreve as fotos; none = sem descrição (padrão: ollama, Gemma 3 4B local, quando instalado)",
    )
    roteiro_parser.add_argument(
        "--visao-model", default=None, help="Modelo da IA de visão (padrão: gemma3:4b no ollama; o do provedor nos outros)"
    )
    roteiro_parser.add_argument(
        "--visao-thinking", choices=["off", "low", "high"], default="off",
        help="Raciocínio da IA de visão, se o modelo tiver (padrão: off)",
    )
    roteiro_parser.add_argument(
        "--visao-timeout", type=float, default=120.0, help="Tempo limite em segundos de cada foto (padrão: 120)"
    )
    roteiro_parser.add_argument(
        "--visao-paralelo", type=_positive_int, default=1, metavar="N",
        help="Clipes descritos ao mesmo tempo (as fotos de um clipe vão em ordem); padrão 1 (GPU local)",
    )
    roteiro_parser.add_argument(
        "--api-retries", type=_non_negative_int, default=DEFAULT_API_RETRIES, metavar="N",
        help=f"Novas tentativas da API em falhas temporárias (padrão: {DEFAULT_API_RETRIES})",
    )
    roteiro_parser.add_argument(
        "--so-estimar", action="store_true",
        help="Faz catálogo, silêncios, transcrição e fotos; mostra fotos, chamadas e tempo estimado da IA de visão "
        "e para (sem chamá-la)",
    )
    roteiro_parser.add_argument(
        "--threshold-db", type=_threshold_arg, default="auto",
        help="Limiar de silêncio em dB, ou `auto` (padrão: auto; o mesmo do edit, para usar o mesmo cache)",
    )
    roteiro_parser.add_argument(
        "--min-silence-sec", type=float, default=0.5, help="Duração mínima de um silêncio (padrão: 0.5)"
    )
    roteiro_parser.add_argument("--language", default=None, help="Idioma da transcrição (ex: pt); padrão: automático")
    roteiro_parser.add_argument(
        "--whisper-model", default="auto", help="Modelo do faster-whisper (padrão: auto, como no edit)"
    )
    roteiro_parser.add_argument(
        "--whisper-device", choices=["auto", "cpu", "cuda"], default="auto", help="Dispositivo do faster-whisper"
    )
    roteiro_parser.add_argument("--whisper-compute-type", default="auto", help="Precisão do faster-whisper")
    roteiro_parser.add_argument(
        "--force", action="store_true",
        help="Refaz silêncios, volume, cenas e fotos (a transcrição e as descrições em cache continuam valendo)",
    )
    roteiro_parser.add_argument(
        "--projects-dir", default="output/projects", help="Diretório base dos projetos (padrão: output/projects)"
    )
    roteiro_parser.add_argument(
        "--no-live", action="store_true", help="Sem linha de andamento ao vivo; só as linhas finais de cada etapa"
    )

    # Subcomando: benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmarks reproduzíveis")
    benchmark_sub = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    bench_tr = benchmark_sub.add_parser(
        "transcription", help="Velocidade e WER de configurações de transcrição"
    )
    bench_tr.add_argument("--audio", help="Áudio a transcrever (um item)")
    bench_tr.add_argument(
        "--reference",
        help="Transcrição de referência (.txt; linhas com # são ignoradas). Sem ela, só mede velocidade",
    )
    bench_tr.add_argument("--language", help="Idioma do áudio (ex: en); padrão: detecção automática")
    bench_tr.add_argument(
        "--manifest",
        help="JSON com vários itens (name, audio, reference opcional, language opcional); "
        "alternativa a --audio/--reference",
    )
    bench_tr.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Configurações modelo:dispositivo:compute_type, separadas por espaço ou vírgula "
        "(ex: small:cpu:int8 small:cuda:float16)",
    )
    bench_tr.add_argument("--name", help="Nome da execução (padrão: nome do áudio ou do manifesto)")
    bench_tr.add_argument(
        "--results-dir",
        default="benchmark/results",
        help="Pasta base dos resultados; cria <data>-<nome>/ dentro dela (padrão: benchmark/results)",
    )

    bench_rescore = benchmark_sub.add_parser(
        "rescore",
        help="Recalcula as métricas de uma execução a partir das transcrições salvas (sem transcrever)",
    )
    bench_rescore.add_argument("run_dir", help="Pasta da execução (benchmark/results/<data>-<nome>)")
    bench_rescore.add_argument(
        "--reference", help="Referência a usar no lugar da gravada no results.json (execuções de um item)"
    )

    # Subcomando: versions
    versions_parser = subparsers.add_parser("versions", help="Lista as versões de um projeto")
    versions_parser.add_argument("project", help="Pasta do projeto")

    # Subcomando: ollama-check
    check_parser = subparsers.add_parser(
        "ollama-check",
        help="Diagnóstico do Ollama local: servidor no ar, modelos baixados e uma chamada de texto e uma com imagem",
    )
    check_parser.add_argument(
        "--modelo-texto", default=DEFAULT_CHECK_TEXT_MODEL,
        help=f"Modelo da chamada de texto (padrão: {DEFAULT_CHECK_TEXT_MODEL}, o do Diretor)",
    )
    check_parser.add_argument(
        "--modelo-imagem", default=DEFAULT_CHECK_IMAGE_MODEL,
        help=f"Modelo da chamada com imagem (padrão: {DEFAULT_CHECK_IMAGE_MODEL}); \"none\" pula",
    )
    check_parser.add_argument(
        "--timeout", type=float, default=DEFAULT_OLLAMA_TIMEOUT_SEC,
        help=f"Tempo limite em segundos de cada chamada (padrão: {DEFAULT_OLLAMA_TIMEOUT_SEC:.0f})",
    )

    for sub in (edit_parser, revise_parser, briefing_parser, visual_parser, roteiro_parser, check_parser):
        _add_ollama_args(sub)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_ollama_args(args)

    if args.command == "render":
        sys.exit(handle_plan_render(args))
    elif args.command == "quick-cut":
        sys.exit(handle_quick_cut(args))
    elif args.command == "analyze":
        sys.exit(handle_analyze(args))
    elif args.command == "report-ai":
        sys.exit(handle_report_ai(args))
    elif args.command == "edit":
        sys.exit(handle_edit(args))
    elif args.command == "revise":
        sys.exit(handle_revise(args))
    elif args.command == "versions":
        sys.exit(handle_versions(args))
    elif args.command == "catalog":
        sys.exit(handle_catalog(args))
    elif args.command == "briefing":
        sys.exit(handle_briefing(args))
    elif args.command == "visual":
        sys.exit(handle_visual(args))
    elif args.command == "ollama-check":
        sys.exit(handle_ollama_check(args))
    elif args.command == "roteiro":
        sys.exit(handle_roteiro(args))
    elif args.command == "benchmark":
        if args.benchmark_command == "rescore":
            sys.exit(handle_benchmark_rescore(args))
        sys.exit(handle_benchmark_transcription(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
