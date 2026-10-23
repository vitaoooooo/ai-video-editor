"""Testes da espinha falada (`edit <pasta>`, receita `vlog`): escolha dos clipes, frases com IDs globais,
Diretor com `retake`, Timeline, legendas no tempo final, cache por clipe, versões e render."""

import shutil
from pathlib import Path
from typing import Optional

import pytest

from ai_video_editor.catalog.schema import CatalogClip, ClipMedia, ProjectCatalog, SpeechDetection
from ai_video_editor.catalog.speech import SpeechDetector, merge_segments, speech_result
from ai_video_editor.cli import build_parser, handle_edit, handle_revise
from ai_video_editor.core.timeline import Timeline
from ai_video_editor.director import (
    DirectorClip,
    DirectorContext,
    DirectorDecision,
    DirectorRun,
    SentenceRemoval,
    UserBrief,
    build_clip_sentences,
    validate_decision,
)
from ai_video_editor.director.prompt import build_director_prompt
from ai_video_editor.director.providers.mock import MockDirector
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.pipeline import EditPipelineParams, ProjectReviser, SpinePipeline
from ai_video_editor.pipeline.clips import MIN_SPEECH_SEC, select_spine_clips
from ai_video_editor.pipeline.spine_schema import SpineDiff, SpineManifest, SpinePlan
from ai_video_editor.planning import TimeRange
from ai_video_editor.planning.timeline_plan import ClipCut, build_spine_timeline, clip_spans, spine_fps
from ai_video_editor.recipes import PIECES, get_recipe
from ai_video_editor.renderer.encoders import EncoderSelection
from ai_video_editor.renderer.timeline_executor import TimelineRenderResult
from ai_video_editor.revision.exceptions import ProjectStateError
from ai_video_editor.revision.providers.mock import MockReviser
from ai_video_editor.subtitles import ClipSubtitleSource, build_clip_subtitle_blocks
from ai_video_editor.transcription.exceptions import TranscriptionError
from ai_video_editor.transcription.normalize import SegmentAdjustments
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment, TranscriptionConfig, Word
from tests.conftest import create_tone_silence_video

# Falas simuladas por clipe (início, texto); cada uma dura ~1,8 s dentro de um trecho de tom.
SCRIPTS = {
    "clip_a": [(0.1, "Chegamos na praia."), (3.1, "Chegamos na praia."), (6.1, "O mar está lindo hoje.")],
    "clip_b": [(0.1, "Agora vamos almoçar."), (3.1, "Depois voltamos ao hotel.")],
}
# Tom = fala; silêncio = pausa. clip_a a 30 fps, clip_b a 25 fps, clip_c sem fala, broll de cobertura.
PATTERNS = {
    "principal/clip_a.mp4": ([("tone", 2.0), ("silence", 1.0)] * 2 + [("tone", 2.0)], 30),
    "principal/clip_b.mp4": ([("tone", 2.0), ("silence", 1.0), ("tone", 2.0)], 25),
    "principal/clip_c.mp4": ([("silence", 3.0)], 30),
    "broll/praia.mp4": ([("silence", 2.0)], 24),
}
SPEECH = {"clip_a": [(0.0, 8.0)], "clip_b": [(0.0, 5.0)], "clip_c": [], "praia": []}


def _segments(clip: str, start_id: int = 0) -> list[TranscriptSegment]:
    segments = []
    for i, (start, text) in enumerate(SCRIPTS[clip]):
        tokens = text.split()
        step = 1.8 / len(tokens)
        words = [
            Word(text=t, start_sec=round(start + k * step, 3), end_sec=round(start + (k + 1) * step - 0.05, 3))
            for k, t in enumerate(tokens)
        ]
        segments.append(TranscriptSegment(id=start_id + i, start_sec=start, end_sec=words[-1].end_sec, text=text, words=words))
    return segments


class FolderTranscriber(TranscriptionProvider):
    """Transcrição simulada por nome de arquivo; registra quais clipes foram transcritos."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-pasta"

    def transcribe(self, media_path: Path, config: TranscriptionConfig, on_progress=None) -> Transcript:
        self.calls.append(Path(media_path).name)
        segments = _segments(Path(media_path).stem)
        if on_progress is not None:
            on_progress(segments[-1].end_sec, segments[-1].end_sec)
        return Transcript(
            source_path=media_path, language="pt", duration_sec=segments[-1].end_sec, model_used="mock", segments=segments
        )


class NamedSpeechDetector(SpeechDetector):
    """VAD simulado do catálogo: trechos de fala pelo nome do arquivo."""

    @property
    def name(self) -> str:
        return "teste"

    def detect(self, path: Path, duration_sec: float) -> SpeechDetection:
        return speech_result(merge_segments(SPEECH.get(path.stem, [])), duration_sec, self.name)


class FakeTimelineExecutor:
    """Render falso: grava um arquivo vazio e devolve a duração da Timeline (sem FFmpeg)."""

    def __init__(self) -> None:
        self.timelines: list[Timeline] = []

    def resolve_encoder(self) -> EncoderSelection:
        return EncoderSelection()

    def execute(self, timeline: Timeline, on_progress=None) -> TimelineRenderResult:
        self.timelines.append(timeline)
        timeline.output_path.parent.mkdir(parents=True, exist_ok=True)
        timeline.output_path.write_bytes(b"x")
        return TimelineRenderResult(
            output_path=timeline.output_path, duration_sec=timeline.duration_sec, frame_count=timeline.total_frames,
            resolution=timeline.resolution, fps=timeline.fps, size_bytes=1, command_executed=[],
        )


@pytest.fixture(scope="session")
def vlog_folder(tmp_path_factory) -> Path:
    """Pasta sintética somente leitura: 3 principais (30 fps, 25 fps, sem fala) + 1 broll."""
    root = tmp_path_factory.mktemp("vlog") / "viagem"
    for n, (relative, (pattern, fps)) in enumerate(PATTERNS.items()):
        path = create_tone_silence_video(root / relative, pattern, width=160, height=120, fps=fps)
        stamp = 1_700_000_000 + n * 60  # ordem de gravação = ordem dos arquivos (pelo mtime)
        import os

        os.utime(path, (stamp, stamp))
    return root


def _copy_folder(folder: Path, tmp_path: Path) -> Path:
    target = tmp_path / folder.name
    shutil.copytree(folder, target, copy_function=shutil.copy2)
    return target


def _pipeline(tmp_path: Path, transcriber=None, executor=None, director=None) -> SpinePipeline:
    return SpinePipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=transcriber or FolderTranscriber(),
        render_executor=executor or FakeTimelineExecutor(),
        edit_director=director,
        speech_detector=NamedSpeechDetector(),
    )


PARAMS = EditPipelineParams(transcribe=True, director="mock", loudness="off")


# --- receita, frases, prompt e validação (sem FFmpeg) ---


def test_vlog_recipe_is_registered_with_new_pieces():
    vlog = get_recipe("vlog")
    assert vlog.steps == ["catalog", "silences", "transcription", "director", "timeline", "render", "subtitles", "report"]
    assert {"catalog", "timeline"} <= set(PIECES)
    assert "clipes" in vlog.description and "b-roll" in vlog.description
    assert vlog.dependencies("render") == ["timeline"]
    assert vlog.multi_clip and not get_recipe("fala").multi_clip


def _clip_transcript(clip: str, punctuated: bool = True) -> Transcript:
    segments = _segments(clip)
    if not punctuated:
        segments = [s.model_copy(update={"text": s.text.rstrip(".!?")}) for s in segments]
    return Transcript(source_path=Path(f"{clip}.mp4"), language="pt", duration_sec=8.0, model_used="mock", segments=segments)


def test_clip_sentences_have_global_ids_and_never_cross_clips():
    # Sem pontuação, a última fala do clipe A emendaria na primeira do B se fossem um vídeo só.
    sentences = build_clip_sentences(
        [("P001", _clip_transcript("clip_a", punctuated=False)), ("P002", _clip_transcript("clip_b", punctuated=False))]
    )
    assert [s.id for s in sentences] == list(range(len(sentences)))
    assert [s.clip_id for s in sentences] == ["P001", "P001", "P001", "P002", "P002"]
    assert all(len({x.clip_id for x in [s]}) == 1 for s in sentences)
    assert sentences[3].text == "Agora vamos almoçar" and sentences[3].start_sec == pytest.approx(0.1)


def _clip_context() -> DirectorContext:
    return DirectorContext(
        video_name="viagem", duration_sec=13.0, language="pt",
        clips=[DirectorClip(id="P001", name="clip_a.mp4", duration_sec=8.0), DirectorClip(id="P002", name="clip_b.mp4", duration_sec=34.4)],
    )


def test_prompt_shows_clips_as_sections_with_retake():
    sentences = build_clip_sentences([("P001", _clip_transcript("clip_a")), ("P002", _clip_transcript("clip_b"))])
    prompt = build_director_prompt(sentences, UserBrief(instructions="foque na praia"), _clip_context())
    assert "Clipe P001 (0:08) — clip_a.mp4" in prompt and "Clipe P002 (0:34) — clip_b.mp4" in prompt
    assert prompt.index("Clipe P001") < prompt.index("[0]") < prompt.index("Clipe P002") < prompt.index("[3]")
    assert "retake (tomada repetida" in prompt and "foque na praia" in prompt
    single = build_director_prompt(sentences, UserBrief(), DirectorContext(video_name="v.mp4", duration_sec=8.0))
    assert "retake" not in single and "Clipe P001" not in single


def test_removing_a_whole_clip_needs_retake_or_off_topic():
    sentences = build_clip_sentences([("P001", _clip_transcript("clip_a")), ("P002", _clip_transcript("clip_b"))])
    whole_b = lambda category: DirectorDecision(  # noqa: E731
        removals=[SentenceRemoval(sentence_id=i, category=category, reason="x") for i in (3, 4)], summary="s"
    )
    brief = UserBrief(target_duration_sec=4.0)  # libera remover bastante tempo
    with pytest.raises(AIResponseValidationError) as exc:
        validate_decision(whole_b("repetition"), sentences, brief)
    assert "clipe P002 ficaria sem nenhuma frase" in exc.value.details
    validate_decision(whole_b("retake"), sentences, brief)
    validate_decision(whole_b("off_topic"), sentences, brief)
    # Sem clip_id (vídeo só) a regra não se aplica.
    single = [s.model_copy(update={"clip_id": None}) for s in sentences]
    validate_decision(whole_b("repetition"), single, brief)


def test_mock_director_removes_earlier_takes_as_retake():
    sentences = build_clip_sentences([("P001", _clip_transcript("clip_a")), ("P002", _clip_transcript("clip_b"))])
    decision = MockDirector().decide(sentences, UserBrief(), _clip_context())
    assert [(r.sentence_id, r.category) for r in decision.removals] == [(0, "retake")]
    assert "frase 1" in decision.removals[0].reason


# --- escolha dos clipes ---


def _catalog_clip(clip_id: Optional[str], role: str, ratio: Optional[float], duration: float = 10.0, **kw) -> CatalogClip:
    status = kw.pop("status", "analisada")
    has_audio = kw.pop("has_audio", True)
    return CatalogClip(
        id=clip_id, role=role, folder=role, relative_path=f"{role}/{clip_id or 'x'}.mp4", size_bytes=1,
        media=ClipMedia(duration_sec=duration, width=1920, height=1080, fps=29.97, codec="h264", has_audio=has_audio),
        speech=SpeechDetection(
            status=status, ratio=ratio, speech_sec=None if ratio is None else ratio * duration, detector="t"
        ),
        **kw,
    )


def test_select_spine_clips_by_role_and_speech(tmp_path):
    catalog = ProjectCatalog(
        source_dir=tmp_path, project_dir=tmp_path / "p", listing_hash="h",
        clips=[
            _catalog_clip("P002", "principal", 0.80),
            _catalog_clip("P001", "principal", 0.50),
            _catalog_clip("P003", "principal", 0.05),  # pouca fala: cobertura
            _catalog_clip("P004", "principal", 0.10, duration=60.0),  # 10% mas 6 s de fala: entra
            _catalog_clip("P005", "principal", None, status="desconhecido"),  # VAD indisponível: entra
            _catalog_clip("P006", "principal", None, status="sem_audio", has_audio=False),
            _catalog_clip("D001", "drone", 0.0),
            _catalog_clip(None, "principal", None, error="arquivo ilegível"),
        ],
    )
    selection = select_spine_clips(catalog)
    assert [c.id for c in selection.clips] == ["P001", "P002", "P004", "P005"]
    assert 60.0 * 0.10 >= MIN_SPEECH_SEC
    unused = {u.id: u for u in selection.unused}
    assert unused["P003"].coverage and "pouca fala" in unused["P003"].reason
    assert unused["P006"].coverage and unused["P006"].reason == "sem áudio"
    assert unused["D001"].coverage and unused["D001"].reason == "papel drone"
    assert not unused[None].coverage and "ilegível" in unused[None].reason
    assert any("P005" in w for w in selection.warnings)


# --- Timeline e legendas (puras) ---


def test_spine_timeline_keeps_clip_order_and_drops_sub_frame_ranges(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    clips = [
        ClipCut(clip_id="P001", source_path=a, keep=[TimeRange(start_sec=0.0, end_sec=1.0), TimeRange(start_sec=2.0, end_sec=2.01)]),
        ClipCut(clip_id="P002", source_path=b, keep=[TimeRange(start_sec=0.5, end_sec=1.5)]),
    ]
    timeline = build_spine_timeline(clips, tmp_path / "out.mp4", fps=30)
    assert [(i.clip_id, i.label) for i in timeline.video_main] == [("P001", "P001 #1"), ("P002", "P002 #1")]
    assert timeline.resolution == (1920, 1080) and timeline.total_frames == 60
    assert clip_spans(timeline) == {"P001": (0, 30), "P002": (30, 60)}
    assert spine_fps([29.97, 29.97, 23.976]) == 30 and spine_fps([25.0, 30.0]) == 25


def test_clip_subtitles_are_shifted_to_the_final_time():
    a = _clip_transcript("clip_a")
    b = _clip_transcript("clip_b")
    blocks = build_clip_subtitle_blocks(
        [
            ClipSubtitleSource(transcript=a, kept_ranges=[TimeRange(start_sec=5.9, end_sec=8.0)], offset_sec=0.0),
            ClipSubtitleSource(transcript=b, kept_ranges=[TimeRange(start_sec=0.0, end_sec=2.2)], offset_sec=2.1),
        ],
        fps=30,
    )
    assert [b.text for b in blocks] == ["O mar está lindo hoje.", "Agora vamos almoçar."]
    assert blocks[0].start_sec == pytest.approx(0.2, abs=0.02)
    assert blocks[1].start_sec == pytest.approx(2.2, abs=0.02)  # 2,1 (início do clipe) + 0,1
    assert [b.index for b in blocks] == [1, 2] and blocks[0].end_sec <= blocks[1].start_sec


# --- pipeline (FFmpeg: catálogo e silêncios; render falso) ---


@pytest.mark.slow
def test_edit_folder_builds_spine_with_retake_and_reports(vlog_folder, tmp_path):
    transcriber, executor = FolderTranscriber(), FakeTimelineExecutor()
    result = _pipeline(tmp_path, transcriber, executor).run(vlog_folder, PARAMS)
    project = result.project_dir

    # Espinha: os dois principais com fala, em ordem; o sem fala e o broll ficam para a cobertura.
    assert result.clips_used == ["P001", "P002"] and result.clips_unused == 2
    assert set(result.outcomes) == set(get_recipe("vlog").steps)
    assert all(outcome == "ran" for outcome in result.outcomes.values())
    assert sorted(transcriber.calls) == ["clip_a.mp4", "clip_b.mp4"]

    # Diretor: IDs globais, cada frase de um clipe, primeira tomada removida como retake.
    run = DirectorRun.model_validate_json((project / "director_decision.json").read_text(encoding="utf-8"))
    assert [s.id for s in run.sentences] == [0, 1, 2, 3, 4]
    assert [s.clip_id for s in run.sentences] == ["P001", "P001", "P001", "P002", "P002"]
    assert [(s.id, r.category) for s, r in run.removed_sentences] == [(0, "retake")]

    # Timeline válida, salva no projeto e na versão; clipes em sequência, 1920x1080 @ 30 (fps do primeiro).
    timeline = Timeline.model_validate_json((project / "timeline.json").read_text(encoding="utf-8"))
    assert timeline == executor.timelines[0]
    assert (timeline.resolution, timeline.fps) == ((1920, 1080), 30)
    clip_order = [i.clip_id for i in timeline.video_main]
    assert clip_order == sorted(clip_order) and set(clip_order) == {"P001", "P002"}
    assert all(i.source_path.name == ("clip_a.mp4" if i.clip_id == "P001" else "clip_b.mp4") for i in timeline.video_main)
    first_a = min(i.source_in for i in timeline.video_main if i.clip_id == "P001")
    assert first_a > 2.0  # a primeira tomada (0–2 s) saiu
    assert result.final_duration_sec == pytest.approx(timeline.duration_sec, abs=0.001)

    # Legendas no tempo final: a fala do clipe B começa onde o clipe B começa no vídeo.
    spans = clip_spans(timeline)
    srt = result.subtitles_path.read_text(encoding="utf-8")
    assert "Chegamos na praia." in srt and "Agora vamos almoçar." in srt
    block_b = next(b for b in srt.strip().split("\n\n") if "Agora vamos almoçar." in b)
    start_b = block_b.splitlines()[1].split(" --> ")[0]
    h, m, rest = start_b.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))
    assert spans["P002"][0] / 30 <= seconds < spans["P002"][0] / 30 + 0.3

    # Relatórios por clipe, clipes não usados e linha do tempo por clipe.
    report = (project / "edit_report.md").read_text(encoding="utf-8")
    assert "## Resumo por clipe" in report and "| P001 |" in report and "| P002 |" in report
    assert "## Clipes não usados" in report and "disponível para cobertura (rodada 3)" in report
    assert "retake" in report and "aplicada" in report and "1920x1080 @ 30fps" in report
    transcript = (project / "edit_transcript.md").read_text(encoding="utf-8")
    assert transcript.index("## Clipe P001") < transcript.index("## Clipe P002") < transcript.index("## Clipes não usados")
    assert "~~Chegamos na praia.~~" in transcript
    diff = SpineDiff.model_validate_json((project / "edit_diff.json").read_text(encoding="utf-8"))
    assert [c.clip_id for c in diff.clips] == ["P001", "P002"]
    assert diff.merged().entry_for(0).status == "applied"
    svg = (project / "edit_timeline.svg").read_text(encoding="utf-8")
    assert 'data-clip-id="P001"' in svg and 'data-clip-id="P002"' in svg

    # Versão v1 com a timeline; revise recusado com mensagem clara.
    manifest = SpineManifest.load(project / "project.json")
    assert manifest.current_version == 1 and (project / "versions" / "v1" / "timeline.json").is_file()
    with pytest.raises(ProjectStateError, match="rodada 4"):
        ProjectReviser(MockReviser()).revise(project, "corta menos")
    assert handle_revise(build_parser().parse_args(["revise", str(project), "corta menos"])) == 1

    # Segunda execução: tudo do cache, nenhum clipe transcrito de novo, mesma versão.
    again = _pipeline(tmp_path, transcriber, executor).run(vlog_folder, PARAMS)
    assert set(again.outcomes.values()) == {"cached"} and len(transcriber.calls) == 2
    assert again.version == 1


@pytest.mark.slow
def test_swapping_one_clip_only_reprocesses_that_clip(vlog_folder, tmp_path):
    folder = _copy_folder(vlog_folder, tmp_path)
    transcriber = FolderTranscriber()
    first = _pipeline(tmp_path, transcriber).run(folder, PARAMS)
    assert sorted(transcriber.calls) == ["clip_a.mp4", "clip_b.mp4"]

    # Outro arquivo no lugar do clip_b (mesmo nome, conteúdo diferente).
    create_tone_silence_video(folder / "principal" / "clip_b.mp4", [("tone", 2.0), ("silence", 1.5), ("tone", 2.0)], 160, 120, 25)
    transcriber.calls.clear()
    second = _pipeline(tmp_path, transcriber).run(folder, PARAMS)
    assert second.project_dir != first.project_dir  # a listagem mudou: novo projeto
    assert transcriber.calls == ["clip_b.mp4"]
    manifest = SpineManifest.load(second.manifest_path)
    assert manifest.stages["silences"].metrics["clips_cached"] == 1
    assert manifest.stages["transcription"].metrics["clips_cached"] == 1


@pytest.mark.slow
def test_whole_clip_removal_is_rejected_then_falls_back(vlog_folder, tmp_path):
    """Remover todas as frases de um clipe sem retake/off_topic: nova tentativa e, se insistir, só silêncio."""
    bad = DirectorDecision(
        removals=[SentenceRemoval(sentence_id=i, category="filler", reason="x") for i in (3, 4)], summary="s"
    )
    director = MockDirector(responses=[bad, bad])
    result = _pipeline(tmp_path, director=director).run(vlog_folder, PARAMS)
    assert len(director.calls) == 2 and "P002" in (director.calls[1].previous_error or "")
    assert any("Diretor ignorado" in w for w in result.warnings)
    assert result.clips_used == ["P001", "P002"]


def test_cli_rejects_recipe_for_the_wrong_input(vlog_folder, tmp_path, capsys):
    video = vlog_folder / "principal" / "clip_a.mp4"
    assert handle_edit(build_parser().parse_args(["edit", str(video), "--receita", "vlog"])) == 1
    assert "edita uma pasta de clipes" in capsys.readouterr().err
    assert handle_edit(build_parser().parse_args(["edit", str(vlog_folder), "--receita", "fala"])) == 1
    assert "edita um vídeo só" in capsys.readouterr().err
    args = build_parser().parse_args(["edit", str(vlog_folder), "--pedido", "foque na praia"])
    assert args.pedido == "foque na praia" and args.instructions is None


# --- render de verdade ---


@pytest.mark.slow
def test_render_spine_has_expected_duration(vlog_folder, tmp_path):
    result = SpinePipeline(
        projects_root=tmp_path / "projects",
        transcription_provider=FolderTranscriber(),
        speech_detector=NamedSpeechDetector(),
    ).run(vlog_folder, PARAMS)
    timeline = Timeline.model_validate_json(result.timeline_path.read_text(encoding="utf-8"))
    plan = SpinePlan.model_validate_json((result.project_dir / "spine_plan.json").read_text(encoding="utf-8"))
    kept = sum(r.end_sec - r.start_sec for c in plan.clips for r in c.keep)
    items = len(timeline.video_main)
    # Cada item é quantizado em frames (±1/2 frame) e o render confere ±1 frame no total.
    assert result.final_duration_sec == pytest.approx(kept, abs=items / timeline.fps)
    assert result.final_duration_sec == pytest.approx(timeline.duration_sec, abs=1 / timeline.fps)
    assert result.output_path.is_file() and result.output_path.stat().st_size > 1000
    assert (result.project_dir / "versions" / "v1" / result.output_path.name).is_file()


def test_old_catalog_format_is_rebuilt_and_current_one_reused(vlog_folder, tmp_path):
    """O edit <pasta> segue a regra do Produtor: só reaproveita catálogo com a listagem e o schema_version atuais."""
    from ai_video_editor.catalog.schema import CATALOG_SCHEMA_VERSION
    from ai_video_editor.pipeline.clips import catalog_location, catalog_path, load_or_build_catalog

    root = tmp_path / "projects"
    project_dir, digest = catalog_location(vlog_folder, root)
    current = ProjectCatalog(source_dir=vlog_folder, project_dir=project_dir, listing_hash=digest)
    built = ProjectCatalog(source_dir=vlog_folder, project_dir=project_dir, listing_hash="novo")
    calls = []

    class Builder:
        def build(self, source_dir):
            calls.append(source_dir)
            return built

    old = current.model_copy(update={"schema_version": CATALOG_SCHEMA_VERSION - 1})
    old.save(catalog_path(project_dir))
    assert load_or_build_catalog(vlog_folder, root, Builder) == (built, True) and len(calls) == 1

    current.save(catalog_path(project_dir))
    catalog, generated = load_or_build_catalog(vlog_folder, root, Builder)
    assert not generated and catalog.listing_hash == digest and len(calls) == 1


class AdjustingTranscriber(FolderTranscriber):
    """Como o FolderTranscriber, mas a saída do clip_b precisou de normalização; `fail` quebra um clipe."""

    def __init__(self, fail: Optional[str] = None) -> None:
        super().__init__()
        self.fail = fail

    def transcribe(self, media_path: Path, config: TranscriptionConfig, on_progress=None) -> Transcript:
        if Path(media_path).stem == self.fail:
            raise ValueError("segmentos devem estar em ordem cronológica")
        transcript = super().transcribe(media_path, config, on_progress)
        adjusted = Path(media_path).stem == "clip_b"
        self._last_adjustments = SegmentAdjustments(segments_adjusted=1, words_dropped=2) if adjusted else SegmentAdjustments()
        return transcript


@pytest.mark.slow
def test_transcription_adjustments_are_warned_and_survive_the_cache(vlog_folder, tmp_path):
    transcriber = AdjustingTranscriber()
    result = _pipeline(tmp_path, transcriber).run(vlog_folder, PARAMS)
    note = "Transcrição (P002): saída do Whisper fora de ordem corrigida — 1 segmento com início ajustado, 2 palavras descartadas."
    assert note in result.warnings and not any("(P001)" in w for w in result.warnings)
    assert f"> ⚠️ **Atenção:** {note}" in (result.project_dir / "edit_report.md").read_text(encoding="utf-8")

    # Nova execução com outro parâmetro: a transcrição vem do cache e o aviso continua.
    transcriber.calls.clear()
    again = _pipeline(tmp_path, transcriber).run(vlog_folder, PARAMS.model_copy(update={"min_silence_sec": 0.6}))
    assert transcriber.calls == [] and note in again.warnings


@pytest.mark.slow
def test_clip_transcription_error_names_the_clip(vlog_folder, tmp_path):
    with pytest.raises(TranscriptionError, match=r"^P002 \(clip_b\.mp4\): segmentos devem estar em ordem cronológica"):
        _pipeline(tmp_path, AdjustingTranscriber(fail="clip_b")).run(vlog_folder, PARAMS)
