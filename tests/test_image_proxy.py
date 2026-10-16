"""Testes unitários para redimensionamento e criação de imagens proxy."""

from pathlib import Path
from PIL import Image

from ai_video_editor.analyzer.schema import ExtractedFrame
from ai_video_editor.intelligence.image_proxy import create_image_proxies


def _create_synthetic_image(path: Path, width: int, height: int, color=(200, 50, 50)) -> Path:
    """Gera uma imagem sintética colorida para testes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path, format="JPEG", quality=95)
    return path


def test_create_image_proxies_4k_downscaling(tmp_path: Path):
    """Verifica se imagem 4K (3840x2160) é redimensionada para 1024x576 preservando o original."""
    src_4k = _create_synthetic_image(tmp_path / "raw" / "frame_4k.jpg", 3840, 2160)
    original_size = src_4k.stat().st_size

    input_frame = ExtractedFrame(
        frame_index=0,
        target_timestamp_sec=3.5,
        image_path=src_4k,
        width=3840,
        height=2160,
        size_bytes=original_size,
    )

    proxy_dir = tmp_path / "proxies"
    proxies = create_image_proxies([input_frame], proxy_dir=proxy_dir, max_dimension=1024, quality=80)

    assert len(proxies) == 1
    p = proxies[0]

    # Verifica se os metadados temporais foram rigorosamente preservados
    assert p.frame_index == 0
    assert p.target_timestamp_sec == 3.5

    # Verifica se a resolução foi reduzida mantendo a proporção 16:9
    assert p.width == 1024
    assert p.height == 576
    assert p.image_path.is_file()

    # O arquivo proxy deve existir separadamente do original
    assert p.image_path != src_4k

    # A imagem original em disco deve permanecer intacta com resolução 4K
    with Image.open(src_4k) as orig_img:
        assert orig_img.size == (3840, 2160)

    # O proxy deve ser substancialmente menor em bytes
    assert p.size_bytes < original_size


def test_create_image_proxies_small_image_no_upscaling(tmp_path: Path):
    """Imagens que já possuem dimensões menores que o teto não devem sofrer upscale."""
    src_small = _create_synthetic_image(tmp_path / "raw" / "frame_small.jpg", 640, 360)
    input_frame = ExtractedFrame(
        frame_index=1,
        target_timestamp_sec=1.0,
        image_path=src_small,
        width=640,
        height=360,
        size_bytes=src_small.stat().st_size,
    )

    proxy_dir = tmp_path / "proxies"
    proxies = create_image_proxies([input_frame], proxy_dir=proxy_dir, max_dimension=1024)

    assert len(proxies) == 1
    p = proxies[0]
    assert p.width == 640
    assert p.height == 360
