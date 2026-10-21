"""Teste de referência do perfil `fala`: o `edit` completo comparado com um snapshot versionado.

Roda o pipeline inteiro (probe → silêncios → transcrição → Diretor → plano → render → legendas →
relatório) num vídeo sintético com fala simulada (tom), transcrição mock e Diretor mock deterministas,
e compara o resultado com `tests/data/reference_fala/`:

- `snapshot.json`: etapas e resultado de cada uma, frases removidas, trechos do EditPlan (tempos
  arredondados a 3 casas) e duração final;
- `legendas.srt`: o .srt gerado.

Serve de rede de segurança para refatorações: o snapshot NÃO deve mudar sem uma mudança intencional
de comportamento. Para regenerá-lo (e revisar o diff no git antes de commitar):

    UPDATE_REFERENCE=1 python -m pytest -q tests/test_reference_fala.py

Tolerâncias: tempos de corte e legendas com ±2 ms (a detecção de silêncio depende do decodificador
AAC da versão do FFmpeg); duração final do render com ±50 ms (depende do codificador).
"""

import json
import os
from pathlib import Path

import pytest

from ai_video_editor.core.schema import EditPlan
from ai_video_editor.director import DirectorRun
from ai_video_editor.pipeline import EditPipeline, EditPipelineParams
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment, Word
from tests.conftest import create_tone_silence_video

REFERENCE_DIR = Path(__file__).parent / "data" / "reference_fala"
SNAPSHOT_FILE = REFERENCE_DIR / "snapshot.json"
SUBTITLES_FILE = REFERENCE_DIR / "legendas.srt"

TIME_TOLERANCE_SEC = 0.002
DURATION_TOLERANCE_SEC = 0.05

# Cinco falas de 3s separadas por 1.5s de silêncio: 0–3, 4.5–7.5, 9–12, 13.5–16.5, 18–21.
# A frase 1 repete a 0 (o Diretor mock remove repetições).
_SENTENCES = [
    (0.1, "Hoje vamos falar de cortes."),
    (4.6, "Hoje vamos falar de cortes!"),
    (9.1, "Primeiro abrimos o editor e importamos o vídeo da aula."),
    (13.6, "Depois cortamos as pausas longas, sem pressa."),
    (18.1, "Obrigado por assistir."),
]


def _transcript(video: Path) -> Transcript:
    segments = []
    for i, (start, text) in enumerate(_SENTENCES):
        tokens = text.split()
        step = 2.8 / len(tokens)
        words = [
            Word(text=t, start_sec=round(start + k * step, 3), end_sec=round(start + (k + 1) * step - 0.05, 3))
            for k, t in enumerate(tokens)
        ]
        segments.append(
            TranscriptSegment(id=i, start_sec=start, end_sec=words[-1].end_sec, text=text, words=words)
        )
    return Transcript(source_path=video, language="pt", duration_sec=21.0, model_used="mock", segments=segments)


def _srt_entries(text: str) -> list[tuple[float, float, str]]:
    """(início, fim, texto) de cada bloco do SRT."""

    def to_sec(stamp: str) -> float:
        h, m, rest = stamp.strip().split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    entries = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().splitlines()
        start, end = lines[1].split(" --> ")
        entries.append((to_sec(start), to_sec(end), "\n".join(lines[2:])))
    return entries


def _snapshot(result, project_dir: Path) -> dict:
    plan = EditPlan.model_validate_json((project_dir / "edit_plan.json").read_text(encoding="utf-8"))
    run = DirectorRun.model_validate_json((project_dir / "director_decision.json").read_text(encoding="utf-8"))
    return {
        "outcomes": dict(result.outcomes),
        "removed_sentences": [
            {"id": s.id, "text": s.text, "category": r.category} for s, r in run.removed_sentences
        ],
        "plan_segments": [[round(s.start_sec, 3), round(s.end_sec, 3)] for s in plan.segments],
        "target_fps": plan.target_fps,
        "resolution": [plan.target_resolution[0], plan.target_resolution[1]],
        "original_duration_sec": round(result.original_duration_sec, 3),
        "final_duration_sec": round(result.final_duration_sec, 3),
    }


@pytest.fixture(scope="module")
def reference_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("reference") / "aula-referencia.mp4"
    return create_tone_silence_video(path, [("tone", 3.0), ("silence", 1.5)] * 4 + [("tone", 3.0)])


@pytest.mark.slow
def test_reference_fala_matches_snapshot(tmp_path, reference_video):
    pipeline = EditPipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=MockTranscriptionProvider(custom_transcript=_transcript(reference_video)),
    )
    result = pipeline.run(reference_video, EditPipelineParams(transcribe=True, director="mock"))
    actual = _snapshot(result, result.project_dir)
    srt_text = result.subtitles_path.read_text(encoding="utf-8")

    if os.environ.get("UPDATE_REFERENCE") == "1":
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_FILE.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        SUBTITLES_FILE.write_text(srt_text, encoding="utf-8")
        pytest.skip("snapshot de referência regenerado (UPDATE_REFERENCE=1)")

    assert SNAPSHOT_FILE.is_file(), "snapshot ausente; gere com UPDATE_REFERENCE=1"
    expected = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))

    assert actual["outcomes"] == expected["outcomes"]
    assert actual["removed_sentences"] == expected["removed_sentences"]
    assert actual["target_fps"] == expected["target_fps"]
    assert actual["resolution"] == expected["resolution"]
    assert len(actual["plan_segments"]) == len(expected["plan_segments"])
    for got, want in zip(actual["plan_segments"], expected["plan_segments"]):
        assert got == pytest.approx(want, abs=TIME_TOLERANCE_SEC)
    assert actual["original_duration_sec"] == pytest.approx(expected["original_duration_sec"], abs=DURATION_TOLERANCE_SEC)
    assert actual["final_duration_sec"] == pytest.approx(expected["final_duration_sec"], abs=DURATION_TOLERANCE_SEC)

    got_srt = _srt_entries(srt_text)
    want_srt = _srt_entries(SUBTITLES_FILE.read_text(encoding="utf-8"))
    assert [text for *_, text in got_srt] == [text for *_, text in want_srt]
    for (gs, ge, _), (ws, we, _) in zip(got_srt, want_srt):
        assert (gs, ge) == pytest.approx((ws, we), abs=TIME_TOLERANCE_SEC)

    # Segunda execução: tudo do cache, mesmo resultado.
    again = pipeline.run(reference_video, EditPipelineParams(transcribe=True, director="mock"))
    assert {stage: outcome for stage, outcome in again.outcomes.items() if outcome != "skipped"} == {
        stage: "cached" for stage, outcome in expected["outcomes"].items() if outcome != "skipped"
    }
    cached = _snapshot(again, again.project_dir)
    assert {k: v for k, v in cached.items() if k != "outcomes"} == {
        k: v for k, v in actual.items() if k != "outcomes"
    }
