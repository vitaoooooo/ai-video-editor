"""Testes do status de aplicação das remoções e da transcrição comparativa (sem render)."""

from pathlib import Path

import pytest

from ai_video_editor.audio.schema import AudioAnalysisParams, AudioAnalysisResult, SilenceInterval
from ai_video_editor.director import DirectorDecision, DirectorRun, Sentence, SentenceRemoval, UserBrief
from ai_video_editor.pipeline.cutting import plan_keep_ranges
from ai_video_editor.pipeline.edit_diff import build_edit_diff, render_edit_transcript, write_edit_diff
from ai_video_editor.pipeline.report import _director_section, build_revision_section
from ai_video_editor.pipeline.schema import EditDiff
from ai_video_editor.planning import SilenceCutParams, TimeRange
from ai_video_editor.subtitles import build_subtitle_blocks
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment
from tests.test_edit_pipeline import _words

CUT = SilenceCutParams()  # padding 0.15 antes / 0.20 depois
FPS = 30


def _audio(duration: float, silences: list[tuple[float, float]]) -> AudioAnalysisResult:
    """Análise sintética: a fala é o complemento dos silêncios."""
    speech, cursor = [], 0.0
    for a, b in silences:
        if a > cursor:
            speech.append(SilenceInterval.from_bounds(cursor, a))
        cursor = b
    if cursor < duration:
        speech.append(SilenceInterval.from_bounds(cursor, duration))
    return AudioAnalysisResult(
        source_path=Path("v.mp4"),
        duration_sec=duration,
        params=AudioAnalysisParams(threshold_db=-35.0, min_silence_sec=0.3),
        silences=[SilenceInterval.from_bounds(a, b) for a, b in silences],
        speech=speech,
    )


def _sentences(spans: list[tuple[float, float, str]]) -> list[Sentence]:
    return [
        Sentence(id=i, start_sec=a, end_sec=b, text=t, segment_ids=[i]) for i, (a, b, t) in enumerate(spans)
    ]


def _removal(sid: int, category: str = "filler", reason: str = "Enrolação.") -> SentenceRemoval:
    return SentenceRemoval(sentence_id=sid, category=category, reason=reason)


def _diff(audio, sentences, removals, final_duration=None, **kwargs) -> EditDiff:
    keep, results = plan_keep_ranges(audio, sentences, set(removals), CUT)
    final = final_duration if final_duration is not None else sum(r.duration for r in keep)
    return build_edit_diff(
        "aula.mp4", sentences, removals, keep, results, FPS, audio.duration_sec, final, **kwargs
    )


# --- A. status de aplicação ---


def test_short_sentence_between_kept_ones_is_not_applied_and_not_struck():
    # Caso "anymore": 0,28s colada nas vizinhas; a folga delas anula o corte inteiro.
    audio = _audio(16.0, [(0.0, 0.5), (15.0, 16.0)])
    sentences = _sentences([(0.5, 10.0, "First part."), (10.0, 10.28, "anymore"), (10.28, 15.0, "Last part.")])
    keep, results = plan_keep_ranges(audio, sentences, {1}, CUT)
    assert [r.outcome for r in results] == ["protected"]
    assert all(r.applied is None for r in results)

    diff = _diff(audio, sentences, {1: _removal(1)})
    entry = diff.entry_for(1)
    assert entry.status == "not_applied" and entry.removed_sec == 0.0
    assert entry.application_note == "protegida pela folga das frases vizinhas"
    assert entry.edited_start_sec is not None  # continua no vídeo
    assert entry.application_label() == "não aplicada: protegida pela folga das frases vizinhas"

    md = render_edit_transcript(diff)
    line = next(l for l in md.splitlines() if "anymore" in l)
    assert line.startswith("⚠️ ") and "~~" not in line
    assert "**remoção não aplicada**, continua no vídeo: protegida pela folga das frases vizinhas" in line
    assert line.endswith("vizinhas. Remoção pedida pelo Diretor · filler: Enrolação.")
    assert diff.removal_counts() == "1 remoção pedida, 0 aplicadas, 0 parciais, 1 não aplicada"


# Fala até 10,02s com um silêncio dentro do início da frase 1 (o Whisper começou a frase cedo):
# o ajuste ao silêncio deixaria 0,22s da frase no vídeo, um pedaço de palavra.
_FRAGMENT_SILENCES = [(0.0, 0.4), (10.02, 10.6), (12.05, 12.45), (16.0, 17.0)]


def test_small_leftover_is_removed_by_extending_the_cut_to_the_sentence_limit():
    audio = _audio(17.0, _FRAGMENT_SILENCES)
    sentences = _sentences([(0.5, 9.5, "Antes."), (10.0, 12.0, "Frase que sai."), (12.5, 16.0, "Depois.")])
    keep, results = plan_keep_ranges(audio, sentences, {1}, CUT)
    assert results[0].outcome == "extended"
    assert (results[0].applied.start_sec, results[0].applied.end_sec) == (10.0, 12.3)

    entry = _diff(audio, sentences, {1: _removal(1)}).entry_for(1)
    assert entry.status == "applied" and entry.removed_pct == 100.0


def test_small_leftover_that_would_invade_a_kept_sentence_undoes_the_cut():
    # A frase mantida anterior (tempos do Whisper) termina depois do início da removida.
    audio = _audio(17.0, _FRAGMENT_SILENCES)
    sentences = _sentences([(0.5, 10.1, "Antes."), (10.0, 12.0, "Frase que sai."), (12.5, 16.0, "Depois.")])
    keep, results = plan_keep_ranges(audio, sentences, {1}, CUT)
    assert results[0].outcome == "undone" and results[0].applied is None
    assert keep == [TimeRange(start_sec=0.0, end_sec=16.2)]  # nada da frase saiu

    diff = _diff(audio, sentences, {1: _removal(1)})
    entry = diff.entry_for(1)
    assert entry.status == "not_applied"
    assert "não deixar meia palavra" in entry.application_note
    line = next(l for l in render_edit_transcript(diff).splitlines() if "Frase que sai." in l)
    assert line.startswith("⚠️ ") and "~~" not in line


def test_partial_removal_reports_seconds_and_percent():
    # Caso "By": o silêncio mais próximo do fim está dentro da frase e sobram 0,4s dela.
    audio = _audio(20.0, [(0.0, 0.4), (10.6, 11.0), (14.0, 20.0)])
    sentences = _sentences([(0.5, 9.5, "Antes."), (10.0, 11.4, "By"), (11.6, 14.0, "Depois.")])
    diff = _diff(audio, sentences, {1: _removal(1)})
    entry = diff.entry_for(1)
    assert entry.status == "partial"
    assert 0.0 < entry.removed_sec < 1.4 * 0.9
    label = entry.application_label()
    assert label.startswith(f"parcial ({entry.removed_sec:.2f}".replace(".", ","))
    assert f"{entry.removed_pct:.0f}%)" in label
    line = next(l for l in render_edit_transcript(diff).splitlines() if " By " in l)
    assert line.startswith("◐ ") and "**remoção parcial**: saíram" in line and "~~" not in line


def test_director_table_and_revision_section_show_the_application_status():
    audio = _audio(16.0, [(0.0, 0.5), (15.0, 16.0)])
    sentences = _sentences([(0.5, 10.0, "First part."), (10.0, 10.28, "anymore"), (10.28, 15.0, "Last part.")])
    removals = {1: _removal(1)}
    diff = _diff(audio, sentences, removals)
    run = DirectorRun(
        provider="mock",
        model_id="mock",
        brief=UserBrief(),
        sentences=sentences,
        decision=DirectorDecision(removals=list(removals.values()), summary="ok"),
    )
    table = "\n".join(_director_section(run, diff))
    assert "| Motivo | Aplicação |" in table
    assert "| anymore | Enrolação. | não aplicada: protegida pela folga das frases vizinhas |" in table

    section = build_revision_section(
        "tira o anymore", 1, 2, "ok", "normal", "normal",
        [(1, "anymore", "mantida", "removida", "não aplicada: protegida", "Enrolação.")],
    )
    assert "| Frase | Texto | Antes | Depois | Aplicação | Motivo |" in section
    assert "| 1 | anymore | mantida | removida | não aplicada: protegida | Enrolação. |" in section


# --- B. transcrição comparativa ---


def _lesson():
    """Falas reais 0–3, 4.5–7.5, 9–12, 13.5–16.5; a frase 1 é removida."""
    texts = [
        "Hoje vamos falar de cortes.",
        "Frase repetida que sai do video.",
        "Primeiro abrimos o editor.",
        "Obrigado por assistir.",
    ]
    segments = []
    for i, (start, text) in enumerate(zip([0.1, 4.6, 9.1, 13.6], texts)):
        words = _words(text, start)
        segments.append(
            TranscriptSegment(id=i, start_sec=start, end_sec=words[-1].end_sec, text=text, words=words)
        )
    transcript = Transcript(
        source_path=Path("aula.mp4"), language="pt", duration_sec=16.5, model_used="mock", segments=segments
    )
    sentences = _sentences([(s.start_sec, s.end_sec, s.text) for s in segments])
    audio = _audio(16.5, [(3.0, 4.5), (7.5, 9.0), (12.0, 13.5)])
    return transcript, sentences, audio


def test_applied_removal_is_struck_with_reason_and_long_pauses_get_a_marker(tmp_path):
    transcript, sentences, audio = _lesson()
    diff = _diff(audio, sentences, {1: _removal(1, "repetition", "Repete.")}, min_pause_sec=1.0)
    entry = diff.entry_for(1)
    assert entry.status == "applied" and entry.origin == "director" and entry.edited_start_sec is None

    md = render_edit_transcript(diff)
    assert "~~Frase repetida que sai do video.~~ `[00:04.600–00:06.950]` — removida pelo Diretor · repetition: Repete." in md
    # Cortes 3.2–6.95 (pausa + frase), 7.7–8.85 e 12.2–13.35: o que sai fora da frase vira marcador
    pauses = [e for e in diff.entries if e.type == "pause"]
    assert [(p.original_start_sec, p.original_end_sec) for p in pauses] == [(3.2, 4.6), (7.7, 8.85), (12.2, 13.35)]
    assert "⏸ pausa de 1,4s removida `[00:03.200]`" in md
    assert "⏸ pausa de 1,1s removida `[00:12.200]`" in md
    assert diff.removed_pause_sec == pytest.approx(1.4 + 1.15 + 1.15, abs=0.001)
    assert "| Frases mantidas | 3 |" in md
    assert "| Frases marcadas para remoção | 1 (1 aplicada, 0 parciais, 0 não aplicadas) |" in md
    assert "| Pausas removidas | 3,7s (no texto, só as de 1,0s ou mais) |" in md

    # Pausas menores que o limite só entram no total
    fewer = _diff(audio, sentences, {1: _removal(1)}, min_pause_sec=1.2)
    assert [p.original_start_sec for p in fewer.entries if p.type == "pause"] == [3.2]
    assert fewer.removed_pause_sec == diff.removed_pause_sec

    # Ordem cronológica e JSON validado pelo modelo
    starts = [e.original_start_sec for e in diff.entries]
    assert starts == sorted(starts)
    transcript_path, diff_path, timeline_path = write_edit_diff(diff, tmp_path)
    assert timeline_path.name == "edit_timeline.svg" and timeline_path.is_file()
    assert EditDiff.model_validate_json(diff_path.read_text(encoding="utf-8")) == diff
    assert transcript_path.read_text(encoding="utf-8") == md


def test_edited_times_match_the_subtitles():
    transcript, sentences, audio = _lesson()
    keep, results = plan_keep_ranges(audio, sentences, {1}, CUT)
    diff = build_edit_diff("aula.mp4", sentences, {1: _removal(1)}, keep, results, FPS, 16.5, 9.7)
    blocks = build_subtitle_blocks(transcript, keep, FPS)
    for sid in (0, 2, 3):
        entry = diff.entry_for(sid)
        first_word = entry.text.split()[0]
        block = next(b for b in blocks if b.text.startswith(first_word))
        assert entry.edited_start_sec == pytest.approx(block.start_sec, abs=0.001)
    md = render_edit_transcript(diff)
    assert "`[00:09.100 → 00:04.183]` Primeiro abrimos o editor." in md
    assert "`[00:00.100 → 00:00.100]` Hoje vamos falar de cortes." in md


def test_revision_marks_restored_and_newly_removed_sentences():
    transcript, sentences, audio = _lesson()
    diff = _diff(
        audio,
        sentences,
        {3: _removal(3, "other", "Despedida.")},
        origins={3: "revision"},
        version=2,
        base_version=1,
        request="mantém a repetida e tira o final",
        restored={1: "O usuário quer essa frase."},
        removed_now={3},
    )
    assert diff.entry_for(1).revision_change == "restored" and diff.entry_for(1).status == "kept"
    assert diff.entry_for(3).revision_change == "removed" and diff.entry_for(3).origin == "revision"

    md = render_edit_transcript(diff)
    assert "| Versão | v2 (revisão da v1) |" in md
    assert "| Pedido | mantém a repetida e tira o final |" in md
    restored = next(l for l in md.splitlines() if "Frase repetida" in l)
    assert restored.startswith("✅ `[00:04.600 → ") and "*restaurada nesta revisão: O usuário quer essa frase.*" in restored
    removed = next(l for l in md.splitlines() if "Obrigado" in l)
    assert removed.startswith("🆕 ~~Obrigado por assistir.~~")
    assert "removida nesta revisão (pedido: “mantém a repetida e tira o final”) · other: Despedida." in removed
