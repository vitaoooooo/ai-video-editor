"""Testes unitários para o gerador de relatórios Markdown."""

from pathlib import Path

from ai_video_editor.analyzer.schema import (
    AudioStreamInfo,
    ExtractedFrame,
    VideoAnalysisResult,
    VideoMetadata,
    VideoStreamInfo,
)
from ai_video_editor.intelligence.reporter import MarkdownReportGenerator, _format_timestamp
from ai_video_editor.intelligence.schema import (
    SceneTransitionHypothesis,
    VisualAnalysisResponse,
    VisualObservation,
)


def test_format_timestamp():
    """Valida a formatação de segundos para MM:SS.mmm."""
    assert _format_timestamp(0.0) == "00:00.000"
    assert _format_timestamp(1.5) == "00:01.500"
    assert _format_timestamp(65.123) == "01:05.123"
    assert _format_timestamp(3600.0) == "60:00.000"


def test_markdown_report_generator_content(tmp_path: Path):
    """Gera um relatório Markdown e valida a presença de todas as seções obrigatórias."""
    # Cria frames e imagens dummy
    img_dir = tmp_path / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)
    frame_path_0 = img_dir / "frame_0000.jpg"
    frame_path_0.write_bytes(b"dummy_image_data")

    meta = VideoMetadata(
        source_path=tmp_path / "clipe_teste.mp4",
        duration_sec=14.5,
        size_bytes=1048576,
        video_stream=VideoStreamInfo(
            codec_name="h264",
            width=1920,
            height=1080,
            fps=30.0,
            pixel_format="yuv420p",
        ),
        audio_stream=AudioStreamInfo(
            codec_name="aac",
            sample_rate=48000,
            channels=2,
        ),
    )

    analysis_res = VideoAnalysisResult(
        metadata=meta,
        sampling_strategy_used="interval: every 2.00s",
        planned_timestamps_sec=[0.0],
        frames=[
            ExtractedFrame(
                frame_index=0,
                target_timestamp_sec=0.0,
                image_path=frame_path_0,
                width=1920,
                height=1080,
                size_bytes=50000,
            )
        ],
        output_dir=img_dir,
    )

    ai_resp = VisualAnalysisResponse(
        provider_used="gemini",
        model_used="gemini-3.8-flash",
        summary="Apresentador em estúdio de gravação iluminado.",
        chronological_timeline=[
            VisualObservation(
                frame_index=0,
                target_timestamp_sec=0.0,
                observable_content="Indivíduo posicionado no centro do enquadramento vestindo camisa azul.",
                inferences="Possível gravação de curso ou tutorial educacional.",
                visual_quality_notes="Iluminação uniforme em três pontos.",
            )
        ],
        scene_transitions=[
            SceneTransitionHypothesis(
                from_timestamp_sec=0.0,
                to_timestamp_sec=2.0,
                transition_type="cut",
                evidence_description="Corte rápido para tela de captura de software.",
                is_confirmed_exact=False,
            )
        ],
        limitations_and_blindspots=[
            "Movimentos rápidos de mãos entre os segundos 0 e 2 podem ter sido perdidos devido à amostragem esparsa."
        ],
    )

    report_dest = tmp_path / "relatorios" / "ai_report.md"
    generator = MarkdownReportGenerator()
    generated_file = generator.generate(analysis_res, ai_resp, report_dest)

    assert generated_file.is_file()
    content = generated_file.read_text(encoding="utf-8")

    # 1. Valida cabeçalho e metadados
    assert "clipe_teste.mp4" in content
    assert "1920x1080" in content
    assert "h264" in content
    assert "aac" in content

    # 2. Valida síntese e modelo
    assert "gemini-3.8-flash" in content
    assert "Apresentador em estúdio" in content

    # 3. Valida linha do tempo e separação de fatos vs inferências
    assert "Indivíduo posicionado no centro" in content
    assert "Possível gravação de curso" in content
    assert "frame_0000.jpg" in content

    # 4. Valida aviso explícito de amostragem esparsa na tabela de transições
    assert "Não (amostra esparsa)" in content
    assert "Corte rápido para tela de captura" in content

    # 5. Valida limitações metodológicas
    assert "amostragem esparsa" in content
