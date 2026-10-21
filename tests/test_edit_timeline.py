"""Testes da linha do tempo dos cortes (`edit_timeline.svg`), sem render."""

import xml.etree.ElementTree as ET

from ai_video_editor.pipeline.edit_timeline import CATEGORIES, render_edit_timeline, timeline_from_diff_file
from ai_video_editor.pipeline.schema import DiffEntry, EditDiff

_NS = {"svg": "http://www.w3.org/2000/svg"}


def _sentence(sid: int, start: float, end: float, status: str = "kept", **kwargs) -> DiffEntry:
    extra = {"origin": "director", "category": "filler", "reason": "Enrolação & <ruído>."} if status != "kept" else {}
    return DiffEntry(
        type="sentence", original_start_sec=start, original_end_sec=end, sentence_id=sid,
        text=f"Frase {sid} com \"aspas\"", status=status, **extra, **kwargs,
    )


def _pause(start: float, end: float) -> DiffEntry:
    return DiffEntry(type="pause", original_start_sec=start, original_end_sec=end, removed_sec=end - start, removed_pct=100)


def _diff(duration: float, entries: list[DiffEntry], final: float) -> EditDiff:
    return EditDiff(
        video_name="aula & teste.mp4", original_duration_sec=duration, final_duration_sec=final,
        removed_pause_sec=sum(e.duration_sec for e in entries if e.type == "pause"), entries=entries,
    )


def _items(root: ET.Element) -> list[ET.Element]:
    return [g for g in root.iter(f"{{{_NS['svg']}}}g") if "item" in g.get("class", "").split()]


def test_svg_is_valid_xml_with_one_item_per_diff_entry_and_a_legend():
    diff = _diff(
        20.0,
        [
            _sentence(0, 0.1, 3.0),
            _pause(3.2, 4.6),
            _sentence(1, 4.6, 7.0, "applied", removed_sec=2.4, removed_pct=100),
            _sentence(2, 9.0, 12.0, "partial", removed_sec=1.0, removed_pct=33.3),
            _sentence(3, 13.0, 15.0, "not_applied"),
            _sentence(4, 16.0, 19.5),
        ],
        final=15.0,
    )
    root = ET.fromstring(render_edit_timeline(diff))
    assert root.tag == f"{{{_NS['svg']}}}svg"

    items = _items(root)
    assert len(items) == len(diff.entries)
    kinds = [g.get("class").split()[1] for g in items]
    assert sorted(kinds) == sorted(["kept", "pause", "removed", "partial", "partial", "kept"])

    legend = root.find("svg:g[@class='legend']", _NS)
    labels = [t.text for t in legend.findall("svg:text", _NS)]
    assert labels == [label for _, _, label in CATEGORIES]

    texts = [t.text for t in root.findall("svg:text", _NS)]
    assert texts[0] == "aula & teste.mp4: 0:20 → 0:15 (25,0% removido)"
    assert root.find("svg:rect[@class='background']", _NS) is not None
    # Marcas de tempo a cada 5s numa linha de 20s
    assert {"0:00", "0:05", "0:10", "0:15", "0:20"} <= set(texts)


def test_long_video_wraps_into_rows_and_items_crossing_a_row_get_one_rect_per_row(tmp_path):
    entries = [_sentence(0, 0.0, 590.0), _sentence(1, 590.0, 610.0, "applied", removed_sec=20, removed_pct=100)]
    entries += [_pause(1200.0, 1201.0), _sentence(2, 1300.0, 1500.0)]
    diff = _diff(1500.0, entries, final=1479.0)
    root = ET.fromstring(render_edit_timeline(diff))

    tracks = root.findall("svg:rect[@class='track']", _NS)
    assert len(tracks) == 3  # 10 min por linha
    items = _items(root)
    assert len(items) == len(entries)
    crossing = next(g for g in items if g.get("data-sentence-id") == "1")
    assert len(crossing.findall("svg:rect", _NS)) == 2
    # Pausa de 1s ainda aparece com largura mínima
    pause = next(g for g in items if "pause" in g.get("class"))
    assert float(pause.find("svg:rect", _NS).get("width")) >= 1.2

    diff_path = tmp_path / "edit_diff.json"
    diff_path.write_text(diff.model_dump_json(), encoding="utf-8")
    svg = timeline_from_diff_file(diff_path)
    assert svg == tmp_path / "edit_timeline.svg"
    assert len(_items(ET.parse(svg).getroot())) == len(entries)
