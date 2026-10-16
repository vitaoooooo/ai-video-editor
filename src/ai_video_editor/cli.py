"""Interface de linha de comando (CLI) para o motor do AI Video Editor."""

import argparse
import json
import sys
from pathlib import Path

from ai_video_editor.analyzer.schema import FrameExtractionConfig
from ai_video_editor.analyzer.service import VideoIngestService
from ai_video_editor.core.schema import ClipSegment, EditPlan
from ai_video_editor.intelligence.schema import AIAnalysisConfig
from ai_video_editor.intelligence.service import MultimodalAnalysisService
from ai_video_editor.renderer.executor import RenderExecutor


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
        reports = MultimodalAnalysisService.from_report_file(report_file, config=ai_config)
    except Exception as e:
        print(f"Erro durante a execução da análise multimodal: {e}", file=sys.stderr)
        return 1

    print(f"\nConcluído! {len(reports)} vídeo(s) analisado(s) com sucesso:")
    for rep in reports:
        print(f"\n* Relatório Markdown: {rep.markdown_path}")
        print(f"* Dados Estruturados (JSON): {rep.json_path}")
        print(f"* Resumo: {rep.analysis_response.summary[:120]}...")

    return 0


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
        default="gemini-2.5-flash",
        help="Identificador do modelo (padrão: gemini-2.5-flash)",
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
