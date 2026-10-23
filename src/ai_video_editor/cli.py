"""Interface de linha de comando (CLI) para o motor do AI Video Editor."""

import argparse
import json
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
from ai_video_editor.llm.registry import provider_names
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
    with status:
        return prepare_briefing(
            video_path, args.pedido, Path(args.projects_dir), lambda: _announce_producer(_build_producer(args)),
            forced_recipe=args.receita, on_progress=printer, on_stage_progress=printer.progress,
        )


def _announce_producer(producer):
    if producer is not None and producer.model_chain:
        print(f"[produtor] {producer.model_chain[0]}", flush=True)
    return producer


def _briefing_recipe_problem(recipe: Recipe, is_folder: bool, video_path: Path, projects_dir: Path) -> Optional[str]:
    """Por que a receita escolhida pelo briefing não serve para este material (None = serve)."""
    if recipe.multi_clip != is_folder:
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
    if recipe.multi_clip != is_folder:
        kind = "uma pasta de clipes" if recipe.multi_clip else "um vídeo só"
        print(f"Erro: a receita '{recipe.name}' edita {kind}; recebido {video_path}.", file=sys.stderr)
        return 1
    if args.so_plano and "coverage" not in recipe.steps:
        print(
            f"Erro: --so-plano mostra o plano da cobertura, e a receita '{recipe.name}' não tem cobertura "
            "(use uma pasta com `visual` feito, --transcribe e a receita vlog_cobertura).",
            file=sys.stderr,
        )
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
            director_timeout_sec=args.director_timeout,
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
            thinking=args.coverage_thinking, timeout_sec=args.coverage_timeout,
        )
        if coverage.provider not in ("none", "mock"):
            coverage.chain()  # valida provedor e reservas
    except ValueError as e:
        print(f"Erro: parâmetros inválidos: {e}", file=sys.stderr)
        return 1

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
                    use_briefing=settings.uses_briefing, plan_only=args.so_plano,
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
    if isinstance(result, SpineResult) and result.plan_only:
        _print_coverage_plan(result)
        print(f"\nPlano da cobertura (sem render): {result.plan_path}")
        print(f"Projeto: {result.project_dir}")
        print(f"Decisão: {result.project_dir / 'coverage_decision.json'}")
        print(f"Duração prevista: {result.final_duration_sec:.2f}s")
        _print_run_summary(result.timings)
        return 0
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
            _print_coverage_plan(result)
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


def _print_coverage_plan(result: SpineResult) -> None:
    """Resumo da cobertura e uma linha por bloco de assunto."""
    run = result.coverage
    if run is None:
        return
    breath_sec = sum(b.duration_sec for b in run.placed_breaths)
    print(
        f"Cobertura: {len(run.placed_coverages)} coberturas ({run.covered_sec:.1f}s de {run.spine_duration_sec:.1f}s), "
        f"{len(run.placed_breaths)} respiros (+{breath_sec:.1f}s), imagem ruim: {run.bad_label} — origem {run.origin_label}"
    )
    if run.video_summary:
        print(f"  Do que o vídeo trata: {run.video_summary}")
    for line in run.plan:
        print(f"  {line}")


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
        timeout_sec=args.reviser_timeout or params.director_timeout_sec,
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
        timeout_sec=args.director_timeout,
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
        timeout_sec=args.visual_timeout,
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
        ai_parallel=args.visual_paralelo,
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
        default=120.0,
        help="Tempo limite em segundos da chamada ao Diretor (padrão: 120)",
    )
    edit_parser.add_argument(
        "--coverage-provider", "--coverage", dest="coverage", choices=["none", "mock", *_llm_providers()], default=None,
        help="Diretor de cobertura (receita vlog_cobertura): none = sem IA (um bloco por clipe e só o tapa das "
        "telas uniformes, pelo código); mock = simulado (padrão: o do Diretor)",
    )
    edit_parser.add_argument(
        "--so-plano", action="store_true",
        help="Pasta com cobertura: roda até a cobertura e mostra/salva o plano por bloco de assunto "
        "(coverage_plan.md), sem renderizar",
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
        "--director-timeout", type=float, default=120.0, help="Tempo limite em segundos da chamada (padrão: 120)"
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
        "--visual-timeout", type=float, default=180.0,
        help="Tempo limite em segundos de cada chamada (padrão: 180)",
    )
    visual_parser.add_argument(
        "--detalhe-imagem", choices=["low", "auto", "high"], default="auto",
        help="Resolução com que a IA lê as miniaturas: low = ~4x menos tokens no Gemini 3 (padrão: auto)",
    )
    visual_parser.add_argument(
        "--visual-paralelo", type=_positive_int, default=DEFAULT_AI_PARALLEL, metavar="N",
        help=f"Lotes enviados à IA ao mesmo tempo; 1 = um por vez; cai pela metade se o provedor limitar "
        f"(padrão: {DEFAULT_AI_PARALLEL})",
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
    elif args.command == "benchmark":
        if args.benchmark_command == "rescore":
            sys.exit(handle_benchmark_rescore(args))
        sys.exit(handle_benchmark_transcription(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
