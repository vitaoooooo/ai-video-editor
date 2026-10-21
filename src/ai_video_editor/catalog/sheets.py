"""Folha de contato de um papel: a miniatura do meio de cada clipe, com o ID e a duração escritos."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ai_video_editor.catalog.schema import CatalogClip

TILE_WIDTH = 240
TILE_HEIGHT = 135
LABEL_HEIGHT = 22
COLUMNS = 6
MARGIN = 6
BACKGROUND = (24, 24, 24)
LABEL_COLOR = (235, 235, 235)


def _font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fonte bitmap fixa
        return ImageFont.load_default()


def _clock(seconds: float) -> str:
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _middle_thumbnail(clip: CatalogClip) -> str:
    return clip.thumbnails[len(clip.thumbnails) // 2]


def build_contact_sheet(clips: list[CatalogClip], catalog_dir: Path, output: Path) -> Path:
    """Grade com uma miniatura por clipe (`clips` já em ordem), rotulada `P001 · 0:42`."""
    columns = min(COLUMNS, len(clips))
    rows = (len(clips) + columns - 1) // columns
    cell_w, cell_h = TILE_WIDTH + MARGIN, TILE_HEIGHT + LABEL_HEIGHT + MARGIN
    sheet = Image.new("RGB", (columns * cell_w + MARGIN, rows * cell_h + MARGIN), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = _font()
    for index, clip in enumerate(clips):
        x = MARGIN + (index % columns) * cell_w
        y = MARGIN + (index // columns) * cell_h
        with Image.open(catalog_dir / _middle_thumbnail(clip)) as thumb:
            thumb = thumb.convert("RGB")
            thumb.thumbnail((TILE_WIDTH, TILE_HEIGHT))
            sheet.paste(thumb, (x + (TILE_WIDTH - thumb.width) // 2, y + (TILE_HEIGHT - thumb.height) // 2))
        label = clip.id or ""
        if clip.media is not None:
            label += f" · {_clock(clip.media.duration_sec)}"
        draw.text((x + 2, y + TILE_HEIGHT + 3), label, fill=LABEL_COLOR, font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=85)
    return output
