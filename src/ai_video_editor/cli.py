"""Interface de linha de comando (CLI) para o motor do AI Video Editor."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, get_args

from ai_video_editor.analyzer.schema import FrameExtractionConfig
from ai_video_editor.analyzer.service import VideoIngestService
from ai_video_editor.core.schema import ClipSegment, EditPlan, LoudnessPreset
from ai_video_editor.director.schema import UserBrief
from ai_video_editor.intelligence.retry import DEFAULT_API_RETRIES, RetryPolicy
from ai_video_editor.intelligence.schema import AIAnalysisConfig
from ai_video_editor.llm.chain import ModelChain, resolve_chain
from ai_video_editor.llm.registry import provider_names
from ai_video_editor.intelligence.service import MultimodalAnalysisService
from ai_video_editor.live_status import LiveStatus
from ai_video_editor.pipeline import (
    STAGES,
    EditPipeline,
    EditPipelineParams,
    ProjectManifest,
    ProjectReviser,
)
from ai_video_editor.pipeline.project import MANIFEST_FILE
from ai_video_editor.pipeline.report import format_timestamp
from ai_video_editor.pipeline.schema import RunTimings, StageProgress
from ai_video_editor.pipeline.timing import STAGE_LABELS, format_duration, summarize_run
from ai_video_editor.pipeline.versions import ensure_versions, parse_version_ref
from ai_video_editor.planning import SilenceCutParams
from ai_video_editor.renderer.executor import RenderExecutor
from ai_video_editor.revision import RevisionError
from ai_video_editor.subtitles import SubtitleParams
from ai_video_editor.transcription.schema import TranscriptionConfig

LOUDNESS_CHOICES = list(get_args(LoudnessPreset))


def handle_plan_render(args: argparse.Namespace) -> int:
    """Executa um EditPlan definido em arquivo JSON."""
    plan_path = Path(args.plan_file)
    if not plan_path.is_file():
        print(f"Erro: arquivo de plano não encontrado: {plan_path}", file=sys.stderr)
        return 1

    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plan = EditPlan.model_validate(data)
    print(f"Plano carregado: '{plan.title}' com {len(plan.segments)} segmentos.")
    print(f"Destino: {plan.target_resolution[0]}x{plan.target_resolution[1]} @ {plan.target_fps}fps")

    executor = RenderExecutor()
    result = executor.execute(plan)

    print("Renderização concluída com sucesso!")
    print(f"Arquivo gerado: {result.output_path}")
    print(f"Duração: {result.duration_sec:.2f}s | Resolução: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Tamanho: {result.size_bytes} bytes")
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
    """Mostra o andamento de cada etapa do comando `edit` (linha ao vivo em um terminal)."""

    def __init__(self, status: LiveStatus):
        self.status = status

    def __call__(self, stage: str, event: str, detail: str) -> None:
        step = f"[{STAGES.index(stage) + 1}/{len(STAGES)}] {STAGE_LABELS[stage]}"
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


def handle_edit(args: argparse.Namespace) -> int:
    """Corta as pausas de um vídeo falado, com pasta de projeto e cache por etapa."""
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Erro: vídeo não encontrado: {video_path}", file=sys.stderr)
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
            brief=UserBrief(
                intensity=args.intensity,
                target_duration_sec=args.target_duration,
                remove_fillers=not args.keep_fillers,
                instructions=args.instructions,
            ),
            subtitles=not args.no_subtitles,
            subtitle=SubtitleParams(max_chars_per_line=args.subtitle_max_chars),
            transcript_min_pause_sec=args.transcript_min_pause,
            loudness=args.loudness,
        )
    except ValueError as e:
        print(f"Erro: parâmetros inválidos: {e}", file=sys.stderr)
        return 1

    status = LiveStatus(live=False if args.no_live else None)
    printer = EditProgressPrinter(status)
    pipeline = EditPipeline(
        projects_root=Path(args.projects_dir), on_progress=printer, on_stage_progress=printer.progress
    )
    try:
        with status:
            result = pipeline.run(video_path, params, force=args.force)
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
    if result.version is not None:
        print(f"Versão: v{result.version} (revise com: revise \"{result.project_dir}\" \"pedido\")")
    print(
        f"Duração: {result.original_duration_sec:.2f}s -> {result.final_duration_sec:.2f}s "
        f"({pct:.1f}% removido)"
    )
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


def handle_benchmark_transcription(args: argparse.Namespace) -> int:
    """Mede velocidade e WER de configurações de transcrição em um ou mais áudios."""
    from datetime import date

    from ai_video_editor.benchmark import (
        BenchmarkConfig,
        BenchmarkItem,
        BenchmarkManifest,
        format_markdown,
        run_benchmark,
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
    run = run_benchmark(name, configs, items, on_progress=print)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    markdown = format_markdown(run)
    (out_dir / "results.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\n{markdown}")
    print(f"Resultados em: {out_dir}")
    return 0 if any(config_run.status == "ok" for config_run in run.configs) else 1


def handle_versions(args: argparse.Namespace) -> int:
    """Lista as versões de um projeto (migra projetos antigos para v1)."""
    project_dir = Path(args.project)
    manifest_path = project_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        print(f"Erro: {project_dir} não é uma pasta de projeto (falta {MANIFEST_FILE}).", file=sys.stderr)
        return 1
    try:
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


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("deve ser 0 ou maior")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Constrói o analisador de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        prog="ai-video-editor",
        description="Motor de edição, ingestão técnica e análise multimodal com IA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: render
    render_parser = subparsers.add_parser("render", help="Renderiza um EditPlan a partir de JSON")
    render_parser.add_argument("plan_file", help="Caminho do arquivo de plano JSON")

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
        "edit", help="Remove as pausas de um vídeo falado (pasta de projeto com cache)"
    )
    edit_parser.add_argument("video", help="Caminho do vídeo a editar")
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
        "--whisper-model", default="small", help="Modelo do faster-whisper (padrão: small)"
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
        "--intensity",
        choices=["light", "medium", "aggressive"],
        default="light",
        help="Quanto o Diretor deve cortar (padrão: light)",
    )
    edit_parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Duração final desejada em segundos (opcional)",
    )
    edit_parser.add_argument(
        "--instructions", default=None, help="Pedido livre ao Diretor (ex: 'corte as piadas')"
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

    for live_parser in (edit_parser, revise_parser):
        live_parser.add_argument(
            "--no-live",
            action="store_true",
            help="Sem linha de andamento ao vivo; só as linhas finais de cada etapa "
            "(automático quando a saída não é um terminal)",
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
    elif args.command == "benchmark":
        sys.exit(handle_benchmark_transcription(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
