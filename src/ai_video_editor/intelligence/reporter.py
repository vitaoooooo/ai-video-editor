"""Gerador de relatórios estruturados em Markdown a partir de análises técnicas e de IA."""

import os
from pathlib import Path
from typing import Optional

from ai_video_editor.analyzer.schema import VideoAnalysisResult
from ai_video_editor.intelligence.schema import AIRunMetrics, VisualAnalysisResponse


def _format_timestamp(seconds: float) -> str:
    """Formata segundos no formato legível MM:SS.mmm."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{mins:02d}:{secs:02d}.{millis:03d}"


class MarkdownReportGenerator:
    """Compila análises técnicas e multimodais em um documento Markdown rastreável."""

    def generate(
        self,
        analysis_result: VideoAnalysisResult,
        ai_response: VisualAnalysisResponse,
        output_file: Path,
        metrics: Optional[AIRunMetrics] = None,
    ) -> Path:
        """Gera o arquivo Markdown e salva no disco.

        Args:
            analysis_result: Dados técnicos e frames da Fase 2.
            ai_response: Interpretação estruturada da IA da Fase 3.
            output_file: Destino final do arquivo .md.
            metrics: Métricas da execução da IA (seção omitida se None).

        Returns:
            Path resolvido do relatório gerado.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        report_dir = output_file.parent.resolve()

        meta = analysis_result.metadata
        v = meta.video_stream
        a = meta.audio_stream

        # Mapeia caminhos relativos de imagens para garantir que o Markdown renderize em qualquer leitor
        frame_lookup = {f.frame_index: f for f in analysis_result.frames}

        lines: list[str] = [
            f"# Relatório de Análise Visual e Técnica — `{meta.source_path.name}`",
            "",
            "> **Aviso Metodológico:** A análise baseia-se em amostragem discreta de frames. "
            "Eventos breves ocorridos entre os instantes amostrados não são visíveis nesta avaliação. "
            "As descrições distinguem estritamente fatos visuais observados de inferências hipotéticas.",
            "",
            "---",
            "",
            "## 1. Metadados Técnicos do Vídeo",
            "",
            "| Propriedade | Valor |",
            "| :--- | :--- |",
            f"| **Arquivo Fonte** | `{meta.source_path.name}` |",
            f"| **Duração Total** | `{meta.duration_sec:.2f}s` ({_format_timestamp(meta.duration_sec)}) |",
            f"| **Resolução** | `{v.width}x{v.height}` |",
            f"| **Taxa de Quadros (FPS)** | `{v.fps:.2f} fps` |",
            f"| **Codec de Vídeo** | `{v.codec_name}` (pix_fmt: `{v.pixel_format or 'desconhecido'}`) |",
            f"| **Codec de Áudio** | `{a.codec_name if a else 'Sem áudio'}` |",
            f"| **Tamanho do Arquivo** | `{meta.size_bytes:,} bytes` |",
            f"| **Total de Frames Analisados** | `{len(analysis_result.frames)}` |",
            f"| **Estratégia de Amostragem** | `{analysis_result.sampling_strategy_used}` |",
            "",
            "---",
            "",
            "## 2. Síntese Geral da IA",
            "",
            f"* **Provedor:** `{ai_response.provider_used}`",
            f"* **Modelo:** `{ai_response.model_used}`",
            "",
            f"**Resumo do Conteúdo:**",
            f"> {ai_response.summary}",
            "",
            "---",
            "",
            "## 3. Linha do Tempo Visual Detalhada",
            "",
        ]

        for obs in ai_response.chronological_timeline:
            ts_str = _format_timestamp(obs.target_timestamp_sec)
            lines.append(f"### Frame {obs.frame_index} — `{ts_str}` ({obs.target_timestamp_sec:.2f}s)")
            lines.append("")

            # Localiza imagem para inserir link ou miniatura relativa
            if frame_obj := frame_lookup.get(obs.frame_index):
                try:
                    rel_img_path = os.path.relpath(frame_obj.image_path.resolve(), report_dir).replace("\\", "/")
                    lines.append(f"![Frame {obs.frame_index}]({rel_img_path})")
                    lines.append("")
                except ValueError:
                    # Em caso de unidades de disco distintas no Windows, usa caminho absoluto como fallback
                    lines.append(f"![Frame {obs.frame_index}]({frame_obj.image_path.as_uri()})")
                    lines.append("")

            lines.append(f"* **Conteúdo Observável:** {obs.observable_content}")
            if obs.inferences:
                lines.append(f"* **Inferências/Contexto:** *{obs.inferences}*")
            if obs.visual_quality_notes:
                lines.append(f"* **Qualidade Técnica:** {obs.visual_quality_notes}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## 4. Hipóteses de Mudanças de Cena e Eventos",
            "",
        ])

        if ai_response.scene_transitions:
            lines.append("| Intervalo | Tipo | Evidência Observada | Confirmação Exata? |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for tr in ai_response.scene_transitions:
                interval_str = f"`{tr.from_timestamp_sec:.2f}s` → `{tr.to_timestamp_sec:.2f}s`"
                exact_str = "Sim" if tr.is_confirmed_exact else "Não (amostra esparsa)"
                lines.append(f"| {interval_str} | `{tr.transition_type}` | {tr.evidence_description} | {exact_str} |")
            lines.append("")
        else:
            lines.append("*Nenhuma transição de cena abrupta detectada entre os frames amostrados.*")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## 5. Limitações Metodológicas e Pontos Cegos",
            "",
        ])

        for limitation in ai_response.limitations_and_blindspots:
            lines.append(f"* {limitation}")
        lines.append("")

        if metrics is not None:
            tokens_in = "n/d" if metrics.input_tokens is None else str(metrics.input_tokens)
            tokens_out = "n/d" if metrics.output_tokens is None else str(metrics.output_tokens)
            lines.extend([
                "---",
                "",
                "## 6. Métricas da execução",
                "",
                f"* **Provedor / Modelo:** `{metrics.provider}` / `{metrics.model_id}`",
                f"* **Frames enviados:** {metrics.frames_sent}",
                f"* **Latência:** {metrics.latency_sec:.2f}s",
                f"* **Tentativas de API:** {metrics.api_attempts}",
                f"* **Tokens (entrada / saída):** {tokens_in} / {tokens_out}",
                "",
            ])

        # Grava o relatório final no disco
        content = "\n".join(lines)
        output_file.write_text(content, encoding="utf-8")
        return output_file.resolve()
