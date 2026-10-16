"""Módulo para criação de imagens proxy otimizadas para envio a modelos multimodais."""

from pathlib import Path
from PIL import Image

from ai_video_editor.analyzer.schema import ExtractedFrame


def create_image_proxies(
    frames: list[ExtractedFrame],
    proxy_dir: Path,
    max_dimension: int = 1024,
    quality: int = 80,
) -> list[ExtractedFrame]:
    """Gera proxies redimensionados e comprimidos a partir dos frames originais.

    Preserva intactos os frames originais em disco e cria versões leves
    adequadas para envio a APIs de visão, economizando tokens e largura de banda.

    Args:
        frames: Lista de frames extraídos na Fase 2.
        proxy_dir: Diretório de destino exclusivo para armazenar os proxies.
        max_dimension: Limite máximo em pixels para a maior dimensão (largura ou altura).
        quality: Qualidade de compressão JPEG (20 a 100).

    Returns:
        Nova lista de ExtractedFrame apontando para as imagens proxy geradas.
    """
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_frames: list[ExtractedFrame] = []

    for frame in frames:
        src_path = frame.image_path.resolve()
        if not src_path.is_file():
            continue

        # Nome determinístico para o proxy dentro da pasta dedicada
        proxy_name = f"proxy_{frame.frame_index:04d}_{int(frame.target_timestamp_sec * 1000):06d}ms.jpg"
        dest_path = proxy_dir / proxy_name

        with Image.open(src_path) as img:
            # Converte formatos como RGBA/PNG para RGB para permitir salvar como JPEG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width, height = img.size

            # Calcula proporção se qualquer dimensão exceder o limite configurado
            if max(width, height) > max_dimension:
                scale = max_dimension / float(max(width, height))
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized_img = img
                new_width, new_height = width, height

            # Salva o arquivo proxy compactado
            resized_img.save(dest_path, format="JPEG", quality=quality, optimize=True)

        size_bytes = dest_path.stat().st_size

        proxy_frames.append(
            ExtractedFrame(
                frame_index=frame.frame_index,
                target_timestamp_sec=frame.target_timestamp_sec,
                image_path=dest_path.resolve(),
                width=new_width,
                height=new_height,
                size_bytes=size_bytes,
            )
        )

    return proxy_frames
