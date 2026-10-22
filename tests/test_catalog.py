"""Testes do comando `catalog`: papéis, IDs, datas, ajuste de relógio, cache por clipe, fala e saídas."""

import io
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_video_editor.catalog import CatalogBuilder, MockSpeechDetector, ProjectCatalog
from ai_video_editor.catalog import builder as builder_module
from ai_video_editor.catalog.builder import (
    assign_roles,
    date_conflicts,
    natural_key,
    order_role,
    project_warnings,
    scan_folder,
)
from ai_video_editor.catalog.dates import date_from_filename, date_from_mtime, date_from_tags
from ai_video_editor.catalog.roles import (
    format_offset,
    match_folder_role,
    parse_clock_adjustment,
    parse_offset,
    parse_role_override,
    role_for_folder,
)
from ai_video_editor.catalog.schema import (
    LOG_MAX_LUMA_SPREAD,
    LOG_MAX_SATURATION,
    CatalogClip,
    ClipMedia,
    ColorSignals,
    RecordingDate,
)
from ai_video_editor.catalog.speech import merge_segments
from ai_video_editor.cli import CatalogProgressPrinter, build_parser, handle_catalog
from ai_video_editor.live_status import LiveStatus, describe_progress
from ai_video_editor.pipeline.schema import StageProgress
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from tests.conftest import create_corrupted_media_file, create_synthetic_audio_only

# --- papéis, ajustes e datas (sem FFmpeg) ---


@pytest.mark.parametrize(
    "folder, role",
    [
        ("principal", "principal"), ("Principal", "principal"), ("MAIN", "principal"), ("A-Roll", "principal"),
        ("Prïncipal", "principal"), ("drone", "drone"), ("Drones", "drone"), ("POV", "pov"), ("GoPro", "pov"),
        ("gopros", "pov"), ("broll", "broll"), ("B-Roll", "broll"), ("b_roll", "broll"), ("Extras", "outro"),
        ("Câmera 2", "outro"),
    ],
)
def test_role_for_folder_tolerates_case_accents_and_synonyms(folder, role):
    assert role_for_folder(folder) == role


@pytest.mark.parametrize(
    "folder, role",
    [
        # pastas do material real
        ("Osmo Pocket - PRINCIPAL", "principal"), ("Fx30 - BROLL", "broll"), ("Go3S - POV", "pov"),
        ("Imagens de Drone", "drone"),
        ("Fx30 - B Roll", "broll"), ("A cam Sony", "principal"), ("Câmera Drones DJI", "drone"),
    ],
)
def test_role_for_folder_containing_a_synonym_as_a_word(folder, role):
    assert match_folder_role(folder) == (role, "contém", [role])


@pytest.mark.parametrize("folder", ["Dronezinho", "Povo", "Mainstream", "Outros"])
def test_role_contains_needs_whole_words(folder):
    role, source, _ = match_folder_role(folder)
    assert (role, source) == ("outro", "padrão")


def test_assign_roles_contains_ambiguous_and_override():
    assignments, warnings = assign_roles(
        ["Drone", "Osmo Pocket - PRINCIPAL", "Drone e POV", "Fx30 - BROLL"], {"fx30 - broll": "outro"}
    )
    roles = {a.folder: (a.role, a.source) for a in assignments}
    assert roles == {
        "Drone": ("drone", "padrão"),
        "Osmo Pocket - PRINCIPAL": ("principal", "contém"),
        "Drone e POV": ("outro", "padrão"),
        "Fx30 - BROLL": ("outro", "--papel"),
    }
    assert warnings == [
        'Pasta "Drone e POV" cita mais de um papel (drone, pov); ficou como outro. Use --papel "Drone e POV=papel".'
    ]


def test_role_override_and_clock_adjustment_parsing():
    assert parse_role_override("Minha Pasta=B-Roll") == ("Minha Pasta", "broll")
    assert parse_role_override("a=b=drone") == ("a=b", "drone")
    with pytest.raises(ValueError, match="papel desconhecido 'cinema'"):
        parse_role_override("x=cinema")
    with pytest.raises(ValueError, match="pasta=papel"):
        parse_role_override("drone")

    assert parse_offset("+3h") == 10800
    assert parse_offset("-1h30m") == -5400
    assert parse_offset("45s") == 45
    assert parse_offset("+1h2m3s") == 3723
    assert parse_offset("-0.5h") == -1800
    for bad in ("", "3", "+h", "3x", "1h-2m"):
        with pytest.raises(ValueError, match="ajuste inválido"):
            parse_offset(bad)
    assert parse_clock_adjustment("Drones=+3h") == ("drone", 10800)
    with pytest.raises(ValueError, match="papel=ajuste"):
        parse_clock_adjustment("+3h")
    assert format_offset(10800) == "+3h" and format_offset(-5430) == "-1h30m30s"


def test_assign_roles_root_overrides_and_unknown_override():
    assignments, warnings = assign_roles(["", "Drone", "Celular"], {"celular": "broll", "Inexistente": "pov"})
    roles = {a.folder: (a.role, a.source) for a in assignments}
    assert roles == {"": ("principal", "raiz"), "Drone": ("drone", "padrão"), "Celular": ("broll", "--papel")}
    assert warnings == ["--papel Inexistente=pov: nenhuma subpasta com esse nome."]


def test_dates_from_tags_prefer_quicktime_and_skip_zeroed_clocks():
    tags = {"creation_time": "2024-01-15T12:30:45.000000Z", "com.apple.quicktime.creationdate": "2024-01-15T09:30:45-0300"}
    date = date_from_tags(tags, {})
    assert date.source == "quicktime" and date.reliable
    assert date.value == datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)

    zeroed = date_from_tags({"creation_time": "1970-01-01T00:00:00Z"}, {"creation_time": "2023-05-02 10:00:00"})
    assert zeroed.source == "stream" and zeroed.value == datetime(2023, 5, 2, 10, tzinfo=timezone.utc)
    assert date_from_tags({"creation_time": "1904-01-01T00:00:00Z"}, {}) is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DJI_20240115123045_0001_D.MP4", (2024, 1, 15, 12, 30, 45)),
        ("VID_20240115_123045.mp4", (2024, 1, 15, 12, 30, 45)),
        ("PXL_20240115_123045123.mp4", (2024, 1, 15, 12, 30, 45)),
        ("2024-01-15 12-30-45.mkv", (2024, 1, 15, 12, 30, 45)),
        ("IMG_20231231_235959.MOV", (2023, 12, 31, 23, 59, 59)),
    ],
)
def test_date_from_filename_patterns_are_local_time(name, expected):
    date = date_from_filename(name)
    assert date.source == "filename" and date.reliable
    assert date.value == datetime(*expected).astimezone().astimezone(timezone.utc)


def test_date_from_filename_without_date_and_mtime_is_unreliable():
    assert date_from_filename("GX010123.MP4") is None
    assert date_from_filename("clip_20241399_250000.mp4") is None
    mtime = date_from_mtime(1_705_321_845 * 10**9)
    assert mtime.source == "mtime" and not mtime.reliable
    assert mtime.value == datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)


def test_merge_speech_segments_across_chunks():
    merged = merge_segments([(59.8, 60.0), (60.0, 61.5), (0.0, 1.0), (1.2, 2.0), (5.0, 6.0)])
    assert [(s.start_sec, s.end_sec) for s in merged] == [(0.0, 2.0), (5.0, 6.0), (59.8, 61.5)]


def test_progress_in_clips():
    assert describe_progress(StageProgress(done=12, total=40, unit="clips")) == "12/40 clipes (30%)"


def _clip(role, cid, when, source="container", fps=30.0, audio=True, resolution=(1920, 1080), codec="h264"):
    value = datetime(2024, 1, 15, tzinfo=timezone.utc) + timedelta(hours=when)
    return CatalogClip(
        id=cid, role=role, folder=role, relative_path=f"{role}/{cid}.mp4", size_bytes=1,
        media=ClipMedia(duration_sec=10, width=resolution[0], height=resolution[1], fps=fps, codec=codec, has_audio=audio),
        recorded_at=RecordingDate(value=value, source=source, raw="x", reliable=source != "mtime"),
        adjusted_at=value,
    )


def test_project_warnings_clock_mtime_mixed_formats_and_silent():
    clips = [
        _clip("principal", "P001", 10), _clip("principal", "P002", 11, resolution=(3840, 2160)),
        _clip("drone", "D001", 13.5, audio=False, fps=25.0),  # relógio ~3h adiantado
        _clip("pov", "V001", 10.5),  # dentro do intervalo
        _clip("broll", "B001", 99, source="mtime"),  # mtime não conta para relógio suspeito
    ]
    warnings = project_warnings(clips, {})
    text = "\n".join(warnings)
    assert "Relógio suspeito no papel drone" in text and "--ajuste-relogio drone=-3h" in text
    assert "papel pov" not in text and "papel broll" not in text
    assert "Datas pouco confiáveis (broll): B001" in text
    assert "Formatos misturados (principal): resoluções 1920x1080, 3840x2160." in warnings
    assert "Taxas de quadros diferentes no projeto: 25, 30 fps." in warnings
    assert "Clipes sem áudio: D001." in warnings


def _timed(name, start, duration, folder="Drone", role="drone"):
    """Clipe com data `start` e duração; o ID vem da ordem de `order_role`."""
    return CatalogClip(
        role=role, folder=folder, relative_path=f"{folder}/{name}", size_bytes=1,
        media=ClipMedia(duration_sec=duration, width=1920, height=1080, fps=30, codec="h264", has_audio=True),
        recorded_at=RecordingDate(value=start, source="container", raw="x"), adjusted_at=start,
    )


def _with_ids(clips, prefix="D"):
    ordered = order_role(clips)
    for n, clip in enumerate(ordered, start=1):
        clip.id = f"{prefix}{n:03d}"
    return ordered


def test_natural_key_orders_numbers_by_value():
    names = ["OsmoPocket_FullHD (27).mp4", "OsmoPocket_FullHD (4).mp4", "OsmoPocket_FullHD (15).mp4", "osmopocket_fullhd (5).MP4"]
    assert sorted(names, key=natural_key) == [
        "OsmoPocket_FullHD (4).mp4", "osmopocket_fullhd (5).MP4", "OsmoPocket_FullHD (15).mp4", "OsmoPocket_FullHD (27).mp4",
    ]


def test_export_dates_overlap_mark_role_unreliable_and_order_by_name():
    day = datetime(2025, 3, 1, tzinfo=timezone.utc)
    # Material real: Drone (1) tem 2:02 e "começa" 13:39:56; Drone (2) "começa" 13:40:45.
    drone1 = _timed("Drone (1).mp4", day.replace(hour=13, minute=39, second=56), 122)
    drone2 = _timed("Drone (2).mp4", day.replace(hour=13, minute=40, second=45), 40)
    assert [(a.relative_path, b.relative_path) for a, b in date_conflicts([drone2, drone1])] == [
        ("Drone/Drone (1).mp4", "Drone/Drone (2).mp4")
    ]

    # Exportados em lote: o (27) foi exportado antes do (4); a ordem passa a ser a do nome.
    clips = [
        _timed("OsmoPocket_FullHD (27).mp4", day, 60, folder="Osmo", role="principal"),
        _timed("OsmoPocket_FullHD (4).mp4", day + timedelta(seconds=20), 60, folder="Osmo", role="principal"),
        _timed("OsmoPocket_FullHD (15).mp4", day + timedelta(seconds=45), 60, folder="Osmo", role="principal"),
    ]
    ordered = _with_ids(clips, "P")
    assert [Path(c.relative_path).name for c in ordered] == [
        "OsmoPocket_FullHD (4).mp4", "OsmoPocket_FullHD (15).mp4", "OsmoPocket_FullHD (27).mp4",
    ]
    assert all(c.order_source == "nome" and not c.recorded_at.reliable for c in ordered)
    export = [w for w in project_warnings(ordered, {}) if w.startswith("Datas não confiáveis (principal)")]
    assert len(export) == 1
    assert "P001 começa antes de P003 terminar" in export[0]
    assert "As datas parecem de exportação, não de gravação" in export[0]
    assert "ordenado pelo número no nome do arquivo" in export[0]


def test_dates_without_overlap_keep_date_order_with_natural_tiebreak():
    day = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    ordered = _with_ids([_timed("clip (2).mp4", day + timedelta(minutes=5), 60), _timed("clip (10).mp4", day, 60)])
    assert [c.relative_path for c in ordered] == ["Drone/clip (10).mp4", "Drone/clip (2).mp4"]
    assert all(c.order_source == "data" and c.recorded_at.reliable for c in ordered)
    # mesma data e duração curta (sem sobreposição): desempate natural, não alfabético
    tied = _with_ids([_timed("clip (10).mp4", day, 0.5), _timed("clip (9).mp4", day, 0.5)])
    assert [c.relative_path for c in tied] == ["Drone/clip (9).mp4", "Drone/clip (10).mp4"]
    assert not any(w.startswith("Datas não confiáveis") for w in project_warnings(ordered, {}))


def test_two_cameras_in_the_same_role_may_record_at_the_same_time():
    day = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    clips = [
        _timed("a.mp4", day, 120, folder="Osmo", role="principal"),
        _timed("b.mp4", day + timedelta(seconds=30), 120, folder="Celular", role="principal"),
    ]
    assert date_conflicts(clips) == []
    assert [c.order_source for c in order_role(clips)] == ["data", "data"]


@pytest.mark.parametrize(
    "spread, saturation, suspect",
    [
        (0.25, 0.03, True),
        (0.25, 0.10, False),  # só contraste baixo: cena clara, não é log
        (0.60, 0.02, False),  # só saturação baixa: cena pouco colorida
        (0.40, 0.07, False),  # marcado pelos limites antigos (0,45 e 0,08)
        (LOG_MAX_LUMA_SPREAD, LOG_MAX_SATURATION - 0.01, False),
    ],
)
def test_log_suspect_needs_low_contrast_and_low_saturation(spread, saturation, suspect):
    color = ColorSignals(frames=3, luma_spread=spread, saturation=saturation)
    assert color.log_suspect is suspect
    dumped = color.model_dump()
    assert (dumped["luma_spread"], dumped["saturation"], dumped["log_suspect"]) == (spread, saturation, suspect)


def test_color_signals_from_old_cache_ignore_old_flags():
    old = '{"frames": 3, "luma_spread": 0.4, "saturation": 0.07, "low_contrast": true, "low_saturation": true}'
    assert ColorSignals.model_validate_json(old).log_suspect is False


def test_cli_catalog_rejects_bad_options(tmp_path, capsys):
    parser = build_parser()
    assert handle_catalog(parser.parse_args(["catalog", str(tmp_path / "nao-existe")])) == 1
    assert "pasta não encontrada" in capsys.readouterr().err
    assert handle_catalog(parser.parse_args(["catalog", str(tmp_path), "--papel", "x=cinema"])) == 1
    assert "papel desconhecido" in capsys.readouterr().err
    assert handle_catalog(parser.parse_args(["catalog", str(tmp_path), "--ajuste-relogio", "drone=3"])) == 1
    assert "ajuste inválido" in capsys.readouterr().err


def test_catalog_progress_printer_numbering():
    out = io.StringIO()
    printer = CatalogProgressPrinter(LiveStatus(stream=out, live=False))
    printer("media", "cached", "")
    printer("speech", "notice", "faster-whisper não instalado")
    assert out.getvalue().splitlines() == [
        "[2/4] Metadados e miniaturas: em cache",
        "[3/4] Detecção de fala: faster-whisper não instalado",
    ]


# --- pasta sintética (FFmpeg) ---

FIXED_MTIME = datetime(2024, 1, 16, 8, 0, tzinfo=timezone.utc).timestamp()


def _make_clip(
    path: Path,
    size: str = "320x240",
    fps: int = 30,
    duration: float = 2.0,
    audio: bool = True,
    creation_time: str | None = None,
    source: str = "testsrc",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    video = f"{source}=duration={duration}:size={size}:rate={fps}" if source != "flat" else (
        f"color=c=0x707878:duration={duration}:size={size}:rate={fps}"
    )
    args = [str(get_ffmpeg_path()), "-v", "error", "-y", "-f", "lavfi", "-i", video]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac", "-shortest"]
    if creation_time:
        args += ["-metadata", f"creation_time={creation_time}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, capture_output=True, check=True)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return path


@pytest.fixture(scope="session")
def clips_folder(tmp_path_factory) -> Path:
    """Vlog sintético (somente leitura): papéis por subpasta, datas por tag/nome/mtime, arquivos inválidos."""
    root = tmp_path_factory.mktemp("catalog") / "Viagem Serra"
    # principal: o nome "b" foi gravado antes do "a"; a raiz também é principal
    _make_clip(root / "Principal" / "a.mp4", creation_time="2024-01-15T12:30:00Z")
    _make_clip(root / "Principal" / "b.mp4", size="640x360", creation_time="2024-01-15T12:00:00Z")
    _make_clip(root / "raiz_20240116_120000.mp4", audio=False)
    _make_clip(root / "Drones" / "DJI_20240115123000_0001_D.MP4", size="480x270", fps=25, audio=False)
    _make_clip(root / "GoPro" / "GX010001.MP4", fps=60, source="flat")  # sem data: mtime; imagem "lavada"
    _make_clip(root / "B-Roll" / "VID_20240115_121000.mp4", audio=False, duration=1.5)
    create_corrupted_media_file(root / "B-Roll" / "quebrado.mp4")
    create_synthetic_audio_only(root / "B-Roll" / "so_audio.mp4", duration_sec=1.0)
    _make_clip(root / "Extras" / "making_of.mov", duration=1.0)
    (root / "notas.txt").write_text("não é vídeo", encoding="utf-8")
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return root


def _listing(root: Path) -> list[tuple[str, int, int]]:
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def _by_path(catalog: ProjectCatalog) -> dict:
    return {c.relative_path: c for c in catalog.clips}


@pytest.mark.slow
def test_catalog_end_to_end(tmp_path, clips_folder):
    before = _listing(clips_folder)
    detector = MockSpeechDetector(segments=[(0.2, 1.2)])
    catalog = CatalogBuilder(projects_root=tmp_path / "projects", speech_detector=detector).build(clips_folder)
    assert _listing(clips_folder) == before  # nada muda na pasta de origem

    project = catalog.project_dir
    assert project.parent == (tmp_path / "projects").resolve()
    assert project.name.startswith("viagem-serra-")
    clips = _by_path(catalog)

    # papéis, IDs e ordem (data de gravação; desempate pelo nome)
    ids = {path: c.id for path, c in clips.items()}
    assert ids == {
        "Principal/b.mp4": "P001",
        "Principal/a.mp4": "P002",
        "raiz_20240116_120000.mp4": "P003",
        "Drones/DJI_20240115123000_0001_D.MP4": "D001",
        "GoPro/GX010001.MP4": "V001",
        "B-Roll/VID_20240115_121000.mp4": "B001",
        "Extras/making_of.mov": "X001",
        "B-Roll/quebrado.mp4": None,
        "B-Roll/so_audio.mp4": None,
    }
    assert catalog.ignored_files == 1
    assert {a.folder: a.role for a in catalog.roles} == {
        "": "principal", "B-Roll": "broll", "Drones": "drone", "Extras": "outro", "GoPro": "pov", "Principal": "principal",
    }

    # origem das datas
    assert clips["Principal/a.mp4"].recorded_at.source == "container"
    assert clips["Principal/a.mp4"].recorded_at.value == datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    assert clips["Drones/DJI_20240115123000_0001_D.MP4"].recorded_at.source == "filename"
    gopro = clips["GoPro/GX010001.MP4"]
    assert gopro.recorded_at.source == "mtime" and not gopro.recorded_at.reliable

    # dados técnicos
    b = clips["Principal/b.mp4"].media
    assert (b.width, b.height, b.fps, b.codec, b.bit_depth, b.pix_fmt) == (640, 360, 30.0, "h264", 8, "yuv420p")
    assert b.has_audio and b.audio_codec == "aac" and b.duration_sec == pytest.approx(2.0, abs=0.1) and b.bitrate
    assert clips["Drones/DJI_20240115123000_0001_D.MP4"].media.fps == 25.0
    assert not clips["Drones/DJI_20240115123000_0001_D.MP4"].media.has_audio

    # fala: mock nos clipes com áudio; sem áudio não chama o detector
    assert clips["Principal/b.mp4"].speech.status == "analisada"
    assert clips["Principal/b.mp4"].speech.ratio == pytest.approx(0.5, abs=0.03)
    assert clips["Drones/DJI_20240115123000_0001_D.MP4"].speech.status == "sem_audio"
    assert len(detector.calls) == 4  # a, b, GoPro, making_of

    # sinais de cor
    assert gopro.color.log_suspect
    assert not clips["Principal/a.mp4"].color.log_suspect

    # arquivos com problema viram aviso
    assert clips["B-Roll/quebrado.mp4"].error == "arquivo ilegível ou corrompido"
    assert clips["B-Roll/so_audio.mp4"].error == "sem trilha de vídeo"
    text = "\n".join(catalog.warnings)
    assert "Arquivo com problema (fora do catálogo): B-Roll/quebrado.mp4" in text
    assert "Datas pouco confiáveis (pov): V001" in text
    assert "Formatos misturados (principal): resoluções 320x240, 640x360." in catalog.warnings
    assert "Clipes sem áudio: P003, D001, B001." in catalog.warnings
    assert "Suspeita de perfil de cor log (baixo contraste e saturação): V001" in text

    # saídas
    catalog_dir = project / "catalog"
    loaded = ProjectCatalog.load(catalog_dir / "catalog.json")
    assert loaded.schema_version == 2 and len(loaded.clips) == 9
    assert clips["Principal/b.mp4"].thumbnails == ["thumbs/P001_10.jpg", "thumbs/P001_50.jpg", "thumbs/P001_90.jpg"]
    assert len(list((catalog_dir / "thumbs").glob("*.jpg"))) == 7 * 3
    assert set(catalog.contact_sheets) == {"principal", "drone", "pov", "broll", "outro"}
    for name in catalog.contact_sheets.values():
        assert (catalog_dir / name).is_file()
    md = (catalog_dir / "catalog.md").read_text(encoding="utf-8")
    assert md.startswith("# Catálogo — Viagem Serra")
    assert "| principal | (raiz), Principal | 3 |" in md
    assert "| P001 | `Principal/b.mp4` | 0:02 | 640x360 | 30 | h264 | 8 (yuv420p) | aac | 50% |" in md
    assert "(modificação ⚠)" in md and "— suspeita: perfil log" in md
    color = clips["Principal/a.mp4"].color
    assert f"| luma {color.luma_spread:.2f} · sat {color.saturation:.3f} |" in md
    assert "suspeita: baixa" not in md and "suspeita: baixo" not in md
    assert "## Arquivos com problema" in md and "`B-Roll/so_audio.mp4` | sem trilha de vídeo" in md


@pytest.mark.slow
def test_catalog_cache_skips_reprocessing(tmp_path, clips_folder, monkeypatch):
    calls = {"probe": 0, "thumbs": 0}
    original_probe, original_thumbs = builder_module.probe_clip, builder_module.thumbnails_and_color

    def probe(*args):
        calls["probe"] += 1
        return original_probe(*args)

    def thumbs(*args):
        calls["thumbs"] += 1
        return original_thumbs(*args)

    monkeypatch.setattr(builder_module, "probe_clip", probe)
    monkeypatch.setattr(builder_module, "thumbnails_and_color", thumbs)
    events = []
    detector = MockSpeechDetector()
    builder = CatalogBuilder(
        projects_root=tmp_path / "projects", speech_detector=detector,
        on_progress=lambda stage, event, detail: events.append((stage, event)),
    )
    first = builder.build(clips_folder)
    assert calls == {"probe": 9, "thumbs": 7} and len(detector.calls) == 4
    assert not any(c.cached for c in first.clips)

    events.clear()
    second = builder.build(clips_folder)
    assert calls == {"probe": 9, "thumbs": 7} and len(detector.calls) == 4
    assert all(c.cached for c in second.clips)
    assert ("media", "cached") in events and ("speech", "cached") in events
    assert [c.id for c in second.clips] == [c.id for c in first.clips]
    assert second.project_dir == first.project_dir

    builder.build(clips_folder, force=True)
    assert calls == {"probe": 18, "thumbs": 14} and len(detector.calls) == 8


@pytest.mark.slow
def test_catalog_role_override_and_clock_adjustment(tmp_path, clips_folder):
    catalog = CatalogBuilder(projects_root=tmp_path / "projects", speech_detector=MockSpeechDetector()).build(
        clips_folder, role_overrides={"extras": "broll"}, clock_adjustments={"principal": -3600.0}
    )
    clips = _by_path(catalog)
    # making_of (mtime 16/01) vai para broll, depois do VID de 15/01
    assert clips["Extras/making_of.mov"].id == "B002" and clips["Extras/making_of.mov"].role == "broll"
    a = clips["Principal/a.mp4"]
    assert a.clock_offset_sec == -3600.0
    assert a.adjusted_at == a.recorded_at.value - timedelta(hours=1)
    assert catalog.clock_adjustments == {"principal": -3600.0}
    md = (catalog.project_dir / "catalog" / "catalog.md").read_text(encoding="utf-8")
    assert "Papéis definidos por `--papel`: `Extras` → broll." in md
    assert "Ajuste de relógio aplicado antes de ordenar: principal -1h." in md
    assert "contact_sheet_outro.jpg" not in catalog.contact_sheets.values()


@pytest.mark.slow
def test_catalog_without_vad_marks_speech_unknown_then_recovers(tmp_path, clips_folder):
    root = tmp_path / "projects"
    unavailable = MockSpeechDetector(available=False)
    catalog = CatalogBuilder(projects_root=root, speech_detector=unavailable).build(clips_folder)
    clips = _by_path(catalog)
    assert clips["Principal/a.mp4"].speech.status == "desconhecido"
    assert clips["Drones/DJI_20240115123000_0001_D.MP4"].speech.status == "sem_audio"
    assert any("Detecção de fala indisponível" in w for w in catalog.warnings)
    md = (catalog.project_dir / "catalog" / "catalog.md").read_text(encoding="utf-8")
    assert "| desconhecido |" in md

    # Com um detector disponível, só a fala é refeita (mídia continua do cache).
    detector = MockSpeechDetector()
    again = CatalogBuilder(projects_root=root, speech_detector=detector).build(clips_folder)
    assert len(detector.calls) == 4
    assert _by_path(again)["Principal/a.mp4"].speech.status == "analisada"


@pytest.mark.slow
def test_real_vad_when_installed(tmp_path, clips_folder):
    pytest.importorskip("faster_whisper")
    from ai_video_editor.catalog.speech import FasterWhisperVAD

    result = FasterWhisperVAD().detect(clips_folder / "Principal" / "a.mp4", 2.0)
    assert result.status == "analisada" and 0.0 <= result.ratio <= 1.0


@pytest.mark.slow
def test_cli_catalog(tmp_path, clips_folder, capsys, monkeypatch):
    monkeypatch.setattr(builder_module, "FasterWhisperVAD", lambda: MockSpeechDetector())
    args = build_parser().parse_args(
        ["catalog", str(clips_folder), "--projects-dir", str(tmp_path / "projects"), "--no-live",
         "--papel", "Extras=broll", "--ajuste-relogio", "drone=+1h"]
    )
    assert handle_catalog(args) == 0
    out = capsys.readouterr().out
    assert "[1/4] Varredura: ok — 9 vídeos em 5 pasta(s), 1 arquivo(s) que não são vídeo ignorado(s)" in out
    assert "[3/4] Detecção de fala: ok — 4 clipes analisados, 4 com fala" in out
    assert "Clipes: 7 (principal 3, drone 1, pov 1, broll 2)" in out
    assert "Folha de contato (broll):" in out and "Tempo total:" in out


def test_scan_ignores_hidden_and_non_video(tmp_path):
    (tmp_path / "Drone").mkdir()
    for name in ("Drone/a.MP4", "Drone/._a.MP4", ".hidden/b.mp4", "c.mov", "d.LRV", "e.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    sources, ignored = scan_folder(tmp_path)
    assert [(s.relative, s.folder) for s in sources] == [("c.mov", ""), ("Drone/a.MP4", "Drone")]
    assert ignored == 2
