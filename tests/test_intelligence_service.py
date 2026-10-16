"""Testes de integração para o serviço orquestrador de análise multimodal."""

import json
from pathlib import Path
from PIL import Image

from ai_video_editor.analyzer.schema import (
    BatchAnalysisResult,
    ExtractedFrame,
    VideoAnalysisResult,
    VideoMetadata,
    VideoStreamInfo,
)
from ai_video_editor.intelligence.providers.mock import MockMultimodalProvider
from ai_video_editor.intelligence.schema import AIAnalysisConfig
from ai_video_editor.intelligence.service import (
    MultimodalAnalysisService,
    select_representative_frames,
)


def _create_frame_image(path: Path) -> Path:
    """Cria uma imagem de frame sintética em disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 360), color=(100, 150, 200))
    img.save(path, format="JPEG")
    return path


def test_select_representative_frames():
    """Valida a sub-amostragem uniforme de frames quando a quantidade excede o limite."""
    # Cria lista de 10 frames com índices de 0 a 9
    frames = [
        ExtractedFrame(
            frame_index=i,
            target_timestamp_sec=float(i * 2),
            image_path=Path(f"f{i}.jpg"),
            width=640,
            height=360,
            size_bytes=1000,
        )
        for i in range(10)
    ]

    # Solicita teto de 4 frames
    selected = select_representative_frames(frames, max_count=4)
    assert len(selected) == 4
    # Deve conter início e fim da sequência
    assert selected[0].frame_index == 0
    assert selected[-1].frame_index == 9

    # Solicita teto maior que o total disponível -> retorna todos
    all_selected = select_representative_frames(frames, max_count=20)
    assert len(all_selected) == 10


def test_analyze_and_report_end_to_end(tmp_path: Path):
    """Executa o ciclo completo da IA multimodal usando Mock e verifica os arquivos gerados."""
    frames_dir = tmp_path / "video_frames"
    frame_files = [_create_frame_image(frames_dir / f"frame_{i:04d}.jpg") for i in range(3)]

    meta = VideoMetadata(
        source_path=tmp_path / "clipe.mp4",
        duration_sec=6.0,
        size_bytes=800000,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )

    analysis_res = VideoAnalysisResult(
        metadata=meta,
        sampling_strategy_used="interval: every 2.00s",
        planned_timestamps_sec=[0.0, 2.0, 4.0],
        frames=[
            ExtractedFrame(
                frame_index=i,
                target_timestamp_sec=float(i * 2),
                image_path=frame_files[i],
                width=640,
                height=360,
                size_bytes=frame_files[i].stat().st_size,
            )
            for i in range(3)
        ],
        output_dir=frames_dir,
    )

    # Executa o serviço configurado com o MockProvider
    service = MultimodalAnalysisService(provider=MockMultimodalProvider())
    cfg = AIAnalysisConfig(provider="mock", model_id="gemini-2.5-flash", max_images_per_request=10)

    report_out = service.analyze_and_report(analysis_res, config=cfg)

    # 1. Valida existência dos artefatos
    assert report_out.markdown_path.is_file()
    assert report_out.json_path.is_file()

    # 2. Valida criação do diretório de proxies
    proxy_dir = frames_dir / "proxies"
    assert proxy_dir.is_dir()
    proxy_images = list(proxy_dir.glob("*.jpg"))
    assert len(proxy_images) == 3

    # 3. Valida conteúdo do JSON estruturado da IA
    json_data = json.loads(report_out.json_path.read_text(encoding="utf-8"))
    assert json_data["provider_used"] == "mock"
    assert len(json_data["chronological_timeline"]) == 3

    # 4. Valida conteúdo do relatório Markdown
    md_content = report_out.markdown_path.read_text(encoding="utf-8")
    assert "# Relatório de Análise Visual e Técnica" in md_content
    assert "gemini-2.5-flash" in md_content


def test_from_report_file(tmp_path: Path):
    """Valida o carregamento a partir de analysis_report.json produzido pela Fase 2."""
    session_dir = tmp_path / "session_test"
    video_dir = session_dir / "vid1_hash123"
    frame_file = _create_frame_image(video_dir / "frame_0000.jpg")

    meta = VideoMetadata(
        source_path=tmp_path / "demo.mp4",
        duration_sec=3.0,
        size_bytes=300000,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )

    analysis_item = VideoAnalysisResult(
        metadata=meta,
        sampling_strategy_used="fixed_count: 1",
        planned_timestamps_sec=[0.0],
        frames=[
            ExtractedFrame(
                frame_index=0,
                target_timestamp_sec=0.0,
                image_path=frame_file,
                width=640,
                height=360,
                size_bytes=frame_file.stat().st_size,
            )
        ],
        output_dir=video_dir,
    )

    batch_manifest = BatchAnalysisResult(
        session_id="session_test",
        session_dir=session_dir,
        total_requested=1,
        successful=[analysis_item],
        failed={},
    )

    report_json_path = session_dir / "analysis_report.json"
    session_dir.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(batch_manifest.model_dump_json(indent=2), encoding="utf-8")

    cfg = AIAnalysisConfig(provider="mock")
    reports = MultimodalAnalysisService.from_report_file(
        report_json_path,
        config=cfg,
        provider=MockMultimodalProvider(),
    )

    assert len(reports) == 1
    assert reports[0].markdown_path.is_file()
    assert reports[0].json_path.is_file()
