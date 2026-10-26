"""`roteiro.md`: o roteiro unificado legível (e o texto que o Editor de IA vai ler).

Índice no topo (clipes, papéis, duração, momentos, frases e o tamanho estimado em tokens); depois, por
clipe, o cabeçalho e os momentos (tempo, descrição, problemas) com as frases que começam em cada um
(ID, marcas de ritmo e texto). Tempos só como contexto (mm:ss); tudo o que a IA pode citar tem ID.
"""

from collections import Counter

from ai_video_editor.roteiro.schema import Moment, ProjectRoteiro, RoteiroClip, RoteiroSentence

# Pausa a partir da qual a marca ⏸ aparece na frase.
PAUSE_MARK_SEC = 1.0
# Caracteres por token na estimativa (regra de bolso para português/inglês).
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimativa grosseira: caracteres / 4."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def clock(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _num(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def rhythm_marks(sentence: RoteiroSentence) -> str:
    """Só o que foge do normal: volume, velocidade e pausa longa antes da frase.

    A pausa "depois" é a "antes" da frase seguinte (fica só no JSON, para não repetir).
    """
    rhythm = sentence.rhythm
    marks: list[str] = []
    if rhythm.volume not in (None, "normal"):
        marks.append(rhythm.volume)
    if rhythm.speed not in (None, "normal"):
        marks.append(rhythm.speed)
    if rhythm.pause_before_sec >= PAUSE_MARK_SEC:
        marks.append(f"⏸{_num(rhythm.pause_before_sec)}s antes")
    return f" ({', '.join(marks)})" if marks else ""


def _clip_header(clip: RoteiroClip, moments: int, sentences: int) -> str:
    parts = [clip.id, clip.role, clock(clip.duration_sec)]
    if clip.has_speech:
        parts.append("com fala")
    elif clip.speech_status == "sem_audio" or not clip.has_audio:
        parts.append("sem áudio")
    else:
        parts.append("sem fala")
    if clip.recorded_at is not None:
        parts.append(f"gravado {clip.recorded_at.strftime('%Y-%m-%d %H:%M')}")
    return f"## {' · '.join(parts)}\n\n{moments} momento(s), {sentences} frase(s) · `{clip.relative_path}`\n"


def _moment_block(moment: Moment, sentences: list[RoteiroSentence]) -> str:
    title = moment.description or "(sem descrição)"
    issues = f" [{', '.join(moment.issues)}]" if moment.issues else ""
    lines = [f"### {moment.id} · {clock(moment.start_sec)}–{clock(moment.end_sec)} · {title}{issues}"]
    details = []
    if moment.place:
        details.append(f"lugar: {moment.place}")
    if moment.people:
        details.append(f"pessoas: {moment.people}" + (" (falando para a câmera)" if moment.talking_to_camera else ""))
    if moment.tags:
        details.append(f"tags: {', '.join(moment.tags)}")
    if moment.scene_change:
        details.append("troca de cena")
    if details:
        lines.append(" · ".join(details))
    for sentence in sentences:
        lines.append(f"- {sentence.id}{rhythm_marks(sentence)} — {sentence.text}")
    return "\n".join(lines) + "\n"


def clip_section(roteiro: ProjectRoteiro, clip: RoteiroClip) -> str:
    moments = [m for m in roteiro.moments if m.clip_id == clip.id]
    sentences = [s for s in roteiro.sentences if s.clip_id == clip.id]
    starts: dict[str, list[RoteiroSentence]] = {m.id: [] for m in moments}
    for sentence in sentences:  # cada frase aparece uma vez: no momento em que começa
        if sentence.moment_ids:
            starts.setdefault(sentence.moment_ids[0], []).append(sentence)
    parts = [_clip_header(clip, len(moments), len(sentences))]
    parts += [_moment_block(m, starts.get(m.id, [])) for m in moments]
    return "\n".join(parts)


def _index(roteiro: ProjectRoteiro, total_tokens: int) -> str:
    roles = Counter(c.role for c in roteiro.clips)
    duration = sum(c.duration_sec for c in roteiro.clips)
    role_text = ", ".join(f"{role} {count}" for role, count in roles.items())
    lines = [
        f"# Roteiro — {roteiro.source.name}",
        "",
        f"{len(roteiro.clips)} clipe(s) ({role_text}) · {clock(duration)} de material · "
        f"{len(roteiro.moments)} momentos · {len(roteiro.sentences)} frases · "
        f"{len(roteiro.photos)} fotos",
        f"Tamanho deste roteiro: ~{_thousands(total_tokens)} tokens (estimativa: caracteres/{CHARS_PER_TOKEN})",
        "",
        "| Clipe | Papel | Duração | Fala | Momentos | Frases | Tokens (est.) |",
        "|---|---|---|---|---|---|---|",
    ]
    for clip in roteiro.clips:
        moments = sum(1 for m in roteiro.moments if m.clip_id == clip.id)
        sentences = sum(1 for s in roteiro.sentences if s.clip_id == clip.id)
        speech = "sim" if clip.has_speech else "não"
        lines.append(
            f"| {clip.id} | {clip.role} | {clock(clip.duration_sec)} | {speech} | {moments} | {sentences} | "
            f"~{_thousands(clip.estimated_tokens)} |"
        )
    lines += [
        "",
        "Como ler: `## clipe` (ordem do catálogo) → `### momento` (trecho com a mesma imagem: tempo, descrição, "
        "[problemas]) → frases que começam nele (`S…`). Ritmo só quando foge do normal: volume (sussurro, baixo, "
        f"alto, em relação ao próprio clipe), velocidade (lenta, rápida) e ⏸ pausa antes da frase de {_num(PAUSE_MARK_SEC)}s ou mais. "
        "Tempos são só contexto: cite os IDs.",
    ]
    return "\n".join(lines) + "\n\n"


def build_roteiro_markdown(roteiro: ProjectRoteiro) -> str:
    """Monta o roteiro.md e preenche as estimativas de tokens (total e por clipe) em `roteiro`."""
    sections = []
    for clip in roteiro.clips:
        text = clip_section(roteiro, clip)
        clip.estimated_tokens = estimate_tokens(text)
        sections.append(text)
    body = "\n".join(sections)
    warnings = ""
    if roteiro.warnings:
        warnings = "\n## Avisos\n\n" + "\n".join(f"- {w}" for w in roteiro.warnings) + "\n"
    # O índice entra na conta: a 1ª passada mede sem ele, a 2ª com o total já conhecido.
    total = estimate_tokens(body + warnings)
    for _ in range(2):
        total = estimate_tokens(_index(roteiro, total) + body + warnings)
    roteiro.estimated_tokens = total
    return _index(roteiro, total) + body + warnings
