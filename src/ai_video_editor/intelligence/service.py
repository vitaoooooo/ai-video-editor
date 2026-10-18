"""Serviço orquestrador para análise multimodal com IA e geração de relatórios."""

import json
import time
from pathlib import Path
from typing import Optional, Sequence

from ai_video_editor.analyzer.schema import (
    BatchAnalysisResult,
    ExtractedFrame,
    VideoAnalysisResult,
)
from ai_video_editor.intelligence.image_proxy import create_image_proxies
from ai_video_editor.intelligence.exceptions import AIProviderError
from ai_video_editor.intelligence.provider import MultimodalAIProvider
from ai_video_editor.intelligence.providers.gemini import GeminiMultimodalProvider
from ai_video_editor.intelligence.providers.mock import MockMultimodalProvider
from ai_video_editor.intelligence.reporter import MarkdownReportGenerator
from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    AIRunMetrics,
    CompleteAIReport,
    VisualAnalysisResponse,
)


def select_representative_frames(
    frames: list[ExtractedFrame],
    max_count: int,
) -> list[ExtractedFrame]:
    """Sub-amostra frames de forma uniforme caso a quantidade exceda o teto configurado.

    Garante que vídeos longos com muitos frames não estourem os limites
    de imagens por requisição do provedor de IA.
    """
    if len(frames) <= max_count:
        return frames

    if max_count <= 1:
        return [frames[len(frames) // 2]]

    step = (len(frames) - 1) / float(max_count - 1)
    indices = [int(round(i * step)) for i in range(max_count)]
    # Garante índices únicos preservando a ordem cronológica
    selected_indices = sorted(list(dict.fromkeys(indices)))
    return [frames[i] for i in selected_indices]


class MultimodalAnalysisService:
    """Orquestra pré-processamento de imagens, inferência de IA e geração do relatório Markdown."""

    def __init__(
        self,
        provider: Optional[MultimodalAIProvider] = None,
        reporter: Optional[MarkdownReportGenerator] = None,
    ):
        self.provider = provider
        self.reporter = reporter or MarkdownReportGenerator()

    def _resolve_provider(self, config: AIAnalysisConfig) -> MultimodalAIProvider:
        """Instancia o provedor apropriado com base na configuração."""
        if self.provider is not None:
            return self.provider

        provider_name = config.provider.lower().strip()
        if provider_name == "mock":
            return MockMultimodalProvider()
        elif provider_name == "gemini":
            return GeminiMultimodalProvider()
        else:
            raise ValueError(f"Provedor de IA desconhecido: '{config.provider}'. Suportados: 'gemini', 'mock'.")

    def analyze_and_report(
        self,
        analysis_result: VideoAnalysisResult,
        config: Optional[AIAnalysisConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> CompleteAIReport:
        """Executa a análise de IA e produz os arquivos ai_analysis.json e ai_report.md.

        Args:
            analysis_result: Resultado técnico e frames da Fase 2.
            config: Parâmetros do modelo e restrições de custo.
            output_dir: Diretório para os novos artefatos (padrão: output_dir da Fase 2).

        Returns:
            CompleteAIReport com a resposta e os caminhos dos arquivos gravados.
        """
        cfg = config or AIAnalysisConfig()
        dest_dir = (output_dir or analysis_result.output_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        active_provider = self._resolve_provider(cfg)

        # 1. Limita quantidade de frames para respeitar cotas e orçamento
        selected_frames = select_representative_frames(
            analysis_result.frames,
            max_count=cfg.max_images_per_request,
        )

        # 2. Cria proxies leves (ex: 1024px, JPEG comprimido) sem alterar originais 4K
        proxy_dir = dest_dir / "proxies"
        proxy_frames = create_image_proxies(
            frames=selected_frames,
            proxy_dir=proxy_dir,
            max_dimension=cfg.max_image_dimension,
            quality=cfg.image_quality,
        )

        # 3. Invoca o provedor multimodal
        metrics_path = dest_dir / "ai_run_metrics.json"
        start = time.perf_counter()
        try:
            ai_response = active_provider.analyze_frames(
                metadata=analysis_result.metadata,
                frames=proxy_frames,
                config=cfg,
            )
        except AIProviderError as exc:
            failed = AIRunMetrics(
                provider=active_provider.provider_name,
                model_id=cfg.model_id,
                frames_sent=len(proxy_frames),
                latency_sec=time.perf_counter() - start,
                input_tokens=active_provider.last_input_tokens,
                output_tokens=active_provider.last_output_tokens,
                success=False,
                error_type=type(exc).__name__,
            )
            metrics_path.write_text(failed.model_dump_json(indent=2), encoding="utf-8")
            raise

        metrics = AIRunMetrics(
            provider=active_provider.provider_name,
            model_id=cfg.model_id,
            frames_sent=len(proxy_frames),
            latency_sec=time.perf_counter() - start,
            input_tokens=active_provider.last_input_tokens,
            output_tokens=active_provider.last_output_tokens,
            success=True,
        )
        metrics_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

        # 4. Salva o resultado estruturado em JSON separado
        json_path = dest_dir / "ai_analysis.json"
        json_path.write_text(ai_response.model_dump_json(indent=2), encoding="utf-8")

        # 5. Compila e grava o relatório Markdown
        md_path = dest_dir / "ai_report.md"
        self.reporter.generate(
            analysis_result=analysis_result,
            ai_response=ai_response,
            output_file=md_path,
            metrics=metrics,
        )

        return CompleteAIReport(
            analysis_response=ai_response,
            markdown_path=md_path.resolve(),
            json_path=json_path.resolve(),
            metrics=metrics,
        )

    @classmethod
    def from_report_file(
        cls,
        report_json_path: Path,
        config: Optional[AIAnalysisConfig] = None,
        provider: Optional[MultimodalAIProvider] = None,
    ) -> list[CompleteAIReport]:
        """Carrega analysis_report.json e executa análise de IA para todos os vídeos processados com sucesso."""
        path = report_json_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de relatório técnico não encontrado: {path}")

        raw_data = json.loads(path.read_text(encoding="utf-8"))
        batch = BatchAnalysisResult.model_validate(raw_data)

        service = cls(provider=provider)
        reports: list[CompleteAIReport] = []

        for item in batch.successful:
            rep = service.analyze_and_report(item, config=config)
            reports.append(rep)

        return reports
