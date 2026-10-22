"""`catalog.md`: resumo por papel, avisos e tabela por clipe."""

from datetime import datetime
from typing import Optional

from ai_video_editor.catalog.roles import format_offset
from ai_video_editor.catalog.schema import ROLES, CatalogClip, ProjectCatalog
from ai_video_editor.catalog.speech import SPEECH_MIN_RATIO
from ai_video_editor.pipeline.timing import format_duration

_DATE_SOURCE_LABELS = {
    "quicktime": "tag QuickTime",
    "container": "tag do container",
    "stream": "tag da trilha",
    "filename": "nome do arquivo",
    "mtime": "modificação ⚠",
}

_SPEECH_LABELS = {"desconhecido": "desconhecido", "sem_audio": "sem áudio"}


def _clock(seconds: float) -> str:
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _local(value: Optional[datetime]) -> str:
    """Data no fuso desta máquina, sem segundos fracionários."""
    if value is None:
        return "—"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GB"
    return f"{size_bytes / 1024**2:.0f} MB"


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _speech(clip: CatalogClip) -> str:
    speech = clip.speech
    if speech is None:
        return "—"
    if speech.status != "analisada":
        return _SPEECH_LABELS[speech.status]
    return _pct(speech.ratio or 0.0)


def _signals(clip: CatalogClip) -> str:
    """Valores medidos (para calibrar os limites) e a suspeita de log, se houver."""
    color = clip.color
    if color is None:
        return "—"
    values = f"luma {color.luma_spread:.2f} · sat {color.saturation:.3f}"
    return f"{values} — suspeita: perfil log" if color.log_suspect else values


def _speech_share(clips: list[CatalogClip]) -> str:
    analyzed = [c for c in clips if c.speech and c.speech.status == "analisada"]
    if not analyzed:
        unknown = sum(1 for c in clips if c.speech and c.speech.status == "desconhecido")
        return "desconhecida" if unknown else "sem áudio"
    total = sum(c.media.duration_sec for c in analyzed)
    speech = sum(c.speech.speech_sec or 0.0 for c in analyzed)
    with_speech = sum(1 for c in analyzed if (c.speech.ratio or 0) >= SPEECH_MIN_RATIO)
    share = _pct(speech / total) if total > 0 else "0%"
    return f"{share} do tempo; {with_speech}/{len(clips)} clipes com fala"


def build_catalog_markdown(catalog: ProjectCatalog) -> str:
    valid = [c for c in catalog.clips if c.error is None]
    total_duration = sum(c.media.duration_sec for c in valid)
    lines = [
        f"# Catálogo — {catalog.source_dir.name}",
        "",
        "| Campo | Valor |",
        "| --- | --- |",
        f"| Pasta | `{catalog.source_dir}` |",
        f"| Projeto | `{catalog.project_dir}` |",
        f"| Clipes | {len(valid)} ({_clock(total_duration)}, {_size(sum(c.size_bytes for c in valid))}) |",
        f"| Arquivos com problema | {len(catalog.clips) - len(valid)} |",
        f"| Arquivos ignorados (não são vídeo) | {catalog.ignored_files} |",
        f"| Gerado em | {_local(catalog.created_at)} (tempo: {format_duration(catalog.total_sec)}) |",
        "",
        "## Papéis",
        "",
        "| Papel | Pastas | Clipes | Duração | Resoluções | fps | Codecs | Datas | Fala |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for role in ROLES:
        members = catalog.clips_of(role)
        if not members:
            continue
        folders = sorted({a.folder or "(raiz)" for a in catalog.roles if a.role == role})
        dates = [c.adjusted_at for c in members if c.adjusted_at]
        span = "—"
        if dates:
            span = _local(min(dates))
            if max(dates) != min(dates):
                span += f" → {_local(max(dates))}"
        if any(c.order_source == "nome" for c in members):
            span = f"não confiáveis ⚠ ({span})"
        lines.append(
            f"| {role} | {', '.join(folders)} | {len(members)} | {_clock(sum(c.media.duration_sec for c in members))} | "
            f"{', '.join(sorted({c.media.resolution for c in members}))} | "
            f"{', '.join(sorted({f'{c.media.fps:g}' for c in members}, key=float))} | "
            f"{', '.join(sorted({c.media.codec for c in members}))} | {span} | {_speech_share(members)} |"
        )
    contains = [a for a in catalog.roles if a.source == "contém"]
    if contains:
        lines += ["", "Papéis pela palavra no nome da pasta: " + ", ".join(f"`{a.folder}` → {a.role}" for a in contains) + "."]
    overrides = [a for a in catalog.roles if a.source == "--papel"]
    if overrides:
        lines += ["", "Papéis definidos por `--papel`: " + ", ".join(f"`{a.folder}` → {a.role}" for a in overrides) + "."]
    if catalog.clock_adjustments:
        adjustments = ", ".join(f"{role} {format_offset(sec)}" for role, sec in catalog.clock_adjustments.items())
        lines += ["", f"Ajuste de relógio aplicado antes de ordenar: {adjustments}."]

    lines += ["", "## Avisos", ""]
    lines += [f"- {w}" for w in catalog.warnings] or ["Nenhum."]

    for role in ROLES:
        members = catalog.clips_of(role)
        if not members:
            continue
        sheet = catalog.contact_sheets.get(role)
        lines += ["", f"## {role}", ""]
        if sheet:
            lines += [f"Folha de contato: `{sheet}`", ""]
        lines += [
            "| ID | Arquivo | Duração | Resolução | fps | Codec | Bits | Áudio | Fala | Gravação (origem) | Sinais |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in members:
            m = c.media
            origin = _DATE_SOURCE_LABELS[c.recorded_at.source] if c.recorded_at else "—"
            if c.clock_offset_sec:
                origin += f", {format_offset(c.clock_offset_sec)}"
            if c.order_source == "nome":
                origin += ", não confiável; ordem pelo nome"
            rotation = f" ↻{m.rotation}°" if m.rotation else ""
            lines.append(
                f"| {c.id} | `{c.relative_path}` | {_clock(m.duration_sec)} | {m.resolution}{rotation} | {m.fps:g} | "
                f"{m.codec} | {m.bit_depth or '—'} ({m.pix_fmt or '?'}) | {m.audio_codec or 'não'} | {_speech(c)} | "
                f"{_local(c.adjusted_at)} ({origin}) | {_signals(c)} |"
            )

    broken = [c for c in catalog.clips if c.error is not None]
    if broken:
        lines += ["", "## Arquivos com problema", "", "| Arquivo | Problema |", "| --- | --- |"]
        lines += [f"| `{c.relative_path}` | {c.error} |" for c in broken]
    return "\n".join(lines) + "\n"
