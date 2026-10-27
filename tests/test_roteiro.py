"""Testes do roteiro unificado (`roteiro <arquivo-ou-pasta>`): fotos, descarte, ritmo, descrição por IA
(mock), momentos, caches e o roteiro.json/roteiro.md. Sem rede; FFmpeg pesado com `slow`."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest

from ai_video_editor.audio.levels import AUTO_RULE_VERSION, measure_levels
from ai_video_editor.audio.schema import SilenceInterval
from ai_video_editor.catalog.schema import SpeechDetection
from ai_video_editor.catalog.speech import SpeechDetector, merge_segments, speech_result
from ai_video_editor.cli import _build_photo_describer, build_parser, handle_roteiro
from ai_video_editor.director.sentences import build_clip_sentences
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.ollama_check import DEFAULT_CHECK_IMAGE_MODEL
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.schema import ImageInput, ModelSpec
from ai_video_editor.pipeline import EditPipelineParams, SpinePipeline
from ai_video_editor.pipeline.clips import (
    clip_cache_file,
    silences_fingerprint,
    transcription_fingerprint,
)
from ai_video_editor.pipeline.project import file_hash, fingerprint, project_dir_for
from ai_video_editor.renderer.ffmpeg_locator import get_ffmpeg_path
from ai_video_editor.roteiro import (
    PhotoDescriber,
    PhotoDescription,
    ProjectRoteiro,
    RoteiroBuilder,
    RoteiroParams,
    mock_vision_responder,
)
from ai_video_editor.roteiro.describer import VISION_PROMPT_VERSION, PhotoRequest, build_vision_prompt
from ai_video_editor.roteiro.moments import (
    MAX_MOMENT_SEC,
    PROMPT_TAGS,
    build_moments,
    finalize_issues,
    is_prompt_tag,
    link_sentences,
)
from ai_video_editor.roteiro.photos import (
    DUPLICATE_MAX_DIFF,
    MAX_KEPT_GAP_SEC,
    ExtractedPhoto,
    plan_photos,
    select_photos,
)
from ai_video_editor.roteiro.report import build_roteiro_markdown, estimate_tokens
from ai_video_editor.roteiro.rhythm import measure_rhythm
from ai_video_editor.roteiro.schema import Moment, RoteiroClip, RoteiroPhoto, RoteiroSentence
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig, TranscriptSegment, Word
from ai_video_editor.visual.schema import ShotMeasures

# --- material sintético ---

# Falas simuladas por arquivo (início, texto); cada palavra ocupa 0,3 s.
SCRIPTS = {
    "fala": [(0.1, "Chegamos na praia."), (3.6, "O mar está lindo hoje.")],
    "pov": [(0.2, "Olha essa vista daqui de cima.")],
    "cenas": [(5.0, "Agora a parte azul do vídeo.")],
}
SPEECH = {"fala": [(0.0, 2.0), (3.5, 6.0)], "pov": [(0.0, 4.0)], "cenas": [(5.0, 7.0)], "voo": []}


def _segments(stem: str, word_sec: float = 0.3) -> list[TranscriptSegment]:
    segments = []
    for i, (start, text) in enumerate(SCRIPTS[stem]):
        tokens = text.split()
        words = [
            Word(text=t, start_sec=round(start + k * word_sec, 3), end_sec=round(start + (k + 1) * word_sec - 0.02, 3))
            for k, t in enumerate(tokens)
        ]
        segments.append(TranscriptSegment(id=i, start_sec=start, end_sec=words[-1].end_sec, text=text, words=words))
    return segments


class FileTranscriber(TranscriptionProvider):
    """Transcrição simulada pelo nome do arquivo; registra quais arquivos foram transcritos."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-roteiro"

    def transcribe(self, media_path: Path, config: TranscriptionConfig, on_progress=None) -> Transcript:
        self.calls.append(Path(media_path).name)
        segments = _segments(Path(media_path).stem)
        return Transcript(
            source_path=media_path, language="pt", duration_sec=segments[-1].end_sec, model_used="mock",
            segments=segments,
        )


class NamedSpeechDetector(SpeechDetector):
    @property
    def name(self) -> str:
        return "teste-roteiro"

    def detect(self, path: Path, duration_sec: float) -> SpeechDetection:
        return speech_result(merge_segments(SPEECH.get(path.stem, [])), duration_sec, self.name)


def make_video(path: Path, colors: list[tuple[str, float]], audio: Optional[list[tuple[str, float]]] = None) -> Path:
    """MP4 com cenas de cor sólida (trocas de cena nítidas, trechos parados) e áudio de tom/silêncio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [str(get_ffmpeg_path()), "-y", "-v", "error"]
    for color, dur in colors:
        args += ["-f", "lavfi", "-i", f"color=c={color}:s=160x120:r=25:d={dur}"]
    for kind, dur in audio or []:
        source = f"sine=frequency=440:sample_rate=44100:duration={dur}" if kind == "tone" else f"anullsrc=r=44100:cl=mono:d={dur}"
        args += ["-f", "lavfi", "-i", source]
    n, m = len(colors), len(audio or [])
    graph = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    if m:
        graph += ";" + "".join(f"[{n + i}:a]" for i in range(m)) + f"concat=n={m}:v=0:a=1[a]"
    args += ["-filter_complex", graph, "-map", "[v]"]
    if m:
        args += ["-map", "[a]", "-c:a", "aac", "-ar", "44100"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, capture_output=True, check=True)
    return path


@pytest.fixture(scope="session")
def roles_folder(tmp_path_factory) -> Path:
    """Pasta somente leitura com 3 papéis: principal com fala, drone sem áudio, POV com fala."""
    root = tmp_path_factory.mktemp("roteiro") / "passeio"
    make_video(root / "principal" / "fala.mp4", [("red", 3), ("blue", 3)],
               [("tone", 2.0), ("silence", 1.5), ("tone", 2.5)])
    make_video(root / "drone" / "voo.mp4", [("green", 4)])
    make_video(root / "pov" / "pov.mp4", [("yellow", 4)], [("tone", 4.0)])
    for n, path in enumerate(sorted(root.rglob("*.mp4"))):
        stamp = 1_700_000_000 + n * 60
        os.utime(path, (stamp, stamp))
    return root


@pytest.fixture(scope="session")
def scenes_video(tmp_path_factory) -> Path:
    """12 s: vermelho até 3 s, azul parado de 3 a 9 s, verde depois; fala de 5 a 7 s."""
    return make_video(
        tmp_path_factory.mktemp("cenas") / "cenas.mp4", [("red", 3), ("blue", 6), ("green", 3)],
        [("silence", 5.0), ("tone", 2.0), ("silence", 5.0)],
    )


def mock_describer(responder=mock_vision_responder) -> tuple[PhotoDescriber, MockLLMClient]:
    client = MockLLMClient(responses=[responder])
    chain = ModelChain([ModelSpec(provider="mock", model_id="mock")], thinking="off", clients={"mock": client})
    return PhotoDescriber(chain), client


def builder(tmp_path: Path, describer=None, transcriber=None, **kwargs) -> RoteiroBuilder:
    return RoteiroBuilder(
        projects_root=tmp_path / "projects",
        describer=describer,
        transcription_provider=transcriber or FileTranscriber(),
        speech_detector=NamedSpeechDetector(),
        **kwargs,
    )


# --- fotos: planejamento e descarte (sem FFmpeg) ---


def test_plan_photos_grid_plus_sentence_and_scene_extras():
    photos = plan_photos(12.0, 2.0, sentence_starts=[5.0, 6.2], scene_cuts=[3.0, 9.0])
    assert [(p.time_sec, p.reasons) for p in photos] == [
        (0.0, ["intervalo"]), (2.0, ["intervalo"]), (3.0, ["cena"]), (4.0, ["intervalo"]), (5.0, ["frase"]),
        (6.0, ["intervalo", "frase"]),  # 6,2 s está a menos de 0,5 s da foto da grade: vira só um motivo
        (8.0, ["intervalo"]), (9.0, ["cena"]), (10.0, ["intervalo"]),
    ]
    assert [p.grid for p in photos] == [True, True, False, True, False, True, True, False, True]
    assert [p.time_sec for p in plan_photos(3.0, 2.0)] == [0.0, 2.0]


def _photo(t: float) -> ExtractedPhoto:
    return ExtractedPhoto(t, ["intervalo"], Path(f"t{t}.jpg"))


def test_select_photos_discards_near_duplicates_of_last_kept():
    flat, other = bytes([100] * 256), bytes([200] * 256)
    near = bytes([100 + int(DUPLICATE_MAX_DIFF) - 1] * 256)
    photos = [_photo(0), _photo(2), _photo(4), _photo(6)]
    selection = select_photos(photos, [flat, near, other, flat])
    assert [p.time_sec for p, _ in selection.kept] == [0, 4, 6]
    assert [(p.time_sec, index) for p, index, _ in selection.discarded] == [(2, 0)]
    assert selection.kept[0][1] is None and selection.kept[1][1] == 100.0


def test_select_photos_keeps_one_photo_every_max_gap_even_if_still():
    times = [k * 2.0 for k in range(12)]  # 0..22 s, imagem parada
    selection = select_photos([_photo(t) for t in times], [bytes(256)] * len(times))
    kept = [p.time_sec for p, _ in selection.kept]
    assert kept == [0.0, MAX_KEPT_GAP_SEC, 2 * MAX_KEPT_GAP_SEC]


# --- ritmo (sem FFmpeg) ---


def _sentence_transcript(specs: list[tuple[float, float, int]]) -> Transcript:
    """(início, fim, palavras) por frase, cada uma um segmento com palavras espalhadas."""
    segments = []
    for i, (start, end, n) in enumerate(specs):
        step = (end - start) / n
        words = [Word(text=f"p{k}", start_sec=round(start + k * step, 3), end_sec=round(start + (k + 1) * step - 0.01, 3))
                 for k in range(n)]
        segments.append(TranscriptSegment(id=i, start_sec=start, end_sec=end, text=" ".join(w.text for w in words) + ".",
                                          words=words))
    return Transcript(source_path=Path("x.mp4"), language="pt", duration_sec=specs[-1][1], model_used="t", segments=segments)


def test_rhythm_volume_speed_and_pauses_from_levels():
    transcript = _sentence_transcript([(0.0, 2.0, 5), (3.0, 5.0, 8), (5.5, 7.5, 3), (9.0, 11.0, 5), (11.2, 13.2, 5)])
    sentences = build_clip_sentences([("P001", transcript)])
    window = 0.05
    levels = [-90.0] * int(14 / window)
    for (start, end), level in zip([(0, 2), (3, 5), (5.5, 7.5), (9, 11), (11.2, 13.2)], [-20, -20, -20, -12, -34]):
        for i in range(int(start / window), int(end / window)):
            levels[i] = level
    silences = [SilenceInterval.from_bounds(7.6, 9.4)]  # o Whisper "comeu" parte da pausa
    rhythms = measure_rhythm(sentences, transcript, 14.0, levels, silences)
    assert [r.volume for r in rhythms] == ["normal", "normal", "normal", "alto", "sussurro"]
    assert [r.relative_db for r in rhythms] == [0.0, 0.0, 0.0, 8.0, -14.0]
    assert [r.speed for r in rhythms] == ["normal", "rápida", "lenta", "normal", "normal"]
    assert rhythms[1].words_per_sec == 4.0 and rhythms[2].words_per_sec == 1.5
    assert rhythms[0].pause_before_sec == 0.0 and rhythms[1].pause_before_sec == 1.0
    assert rhythms[3].pause_before_sec == 1.8  # o silêncio detectado (1,8 s) vence o intervalo entre frases (1,5 s)
    assert rhythms[4].pause_after_sec == pytest.approx(0.8)  # até o fim do clipe
    without = measure_rhythm(sentences, transcript, 14.0)
    assert all(r.volume is None and r.level_db is None for r in without)


# --- momentos (sem FFmpeg) ---


def _described(clip: str, n: int, t: float, place: str, tags: list[str]) -> RoteiroPhoto:
    return RoteiroPhoto(
        id=f"{clip}.F{n:03d}", clip_id=clip, time_sec=t, reasons=["intervalo"], file="x.jpg", image_hash="h",
        status="descrita", description=PhotoDescription(what=f"foto {n}", place=place, tags=tags),
    )


def test_moments_group_similar_photos_and_break_on_scene_or_change():
    photos = [
        _described("P001", 1, 0.0, "praia", ["mar", "areia", "sol"]),
        _described("P001", 2, 2.0, "Praia", ["mar", "areia"]),  # mesmo lugar (caixa à parte), 2 tags em comum
        _described("P001", 3, 4.0, "praia", ["mar", "barco"]),  # só 1 tag em comum: novo momento
        _described("P001", 4, 6.0, "praia", ["mar", "barco", "céu"]),
        _described("P001", 5, 8.0, "praia", ["mar", "barco"]),  # troca de cena antes: novo momento
    ]
    moments = build_moments("P001", photos, 10.0, scene_cuts=[7.5])
    assert [(m.id, m.start_sec, m.end_sec, m.photo_ids) for m in moments] == [
        ("P001.M01", 0.0, 4.0, ["P001.F001", "P001.F002"]),
        ("P001.M02", 4.0, 8.0, ["P001.F003", "P001.F004"]),
        ("P001.M03", 8.0, 10.0, ["P001.F005"]),
    ]
    assert moments[0].tags[:2] == ["mar", "areia"] and moments[2].scene_change
    assert moments[0].description == "foto 1"


def test_moments_respect_max_duration_and_link_sentences():
    photos = [_described("P001", n + 1, n * 2.0, "rua", ["rua", "carro"]) for n in range(20)]  # 40 s iguais
    moments = build_moments("P001", photos, 40.0, scene_cuts=[])
    assert len(moments) >= 2 and all(m.duration <= MAX_MOMENT_SEC for m in moments)
    assert moments[0].start_sec == 0.0 and moments[-1].end_sec == 40.0
    assert all(a.end_sec == b.start_sec for a, b in zip(moments, moments[1:]))
    boundary = moments[0].end_sec
    sentences = [
        RoteiroSentence(id="S000", index=0, clip_id="P001", start_sec=1.0, end_sec=3.0, text="a"),
        RoteiroSentence(id="S001", index=1, clip_id="P001", start_sec=boundary - 1, end_sec=boundary + 1, text="b"),
    ]
    link_sentences(sentences, moments)
    assert sentences[0].moment_ids == ["P001.M01"]
    assert sentences[1].moment_ids == ["P001.M01", "P001.M02"]
    assert moments[1].sentence_ids == ["S001"]
    empty = build_moments("D001", [], 7.0, [])
    assert [(m.id, m.start_sec, m.end_sec) for m in empty] == [("D001.M01", 0.0, 7.0)]


def _moment_issues(issues: list[str], measures: ShotMeasures | None) -> list[str]:
    descriptions = {
        f"P001.F{n:03d}": PhotoDescription(what="x", tags=["mar"], issue=issue) for n, issue in enumerate(issues, 1)
    }
    moment = Moment(id="P001.M01", clip_id="P001", start_sec=0.0, end_sec=4.0, photo_ids=list(descriptions),
                    measures=measures)
    finalize_issues([moment], descriptions)
    return moment.issues


def test_light_issues_come_only_from_the_measures():
    measures = ShotMeasures(samples=4, brightness=0.5, motion=0.0, luma_spread=0.5)
    # "clara"/"escura" da IA são ignorados; os outros problemas da IA continuam.
    assert _moment_issues(["clara", "escura", "borrada", "tremida", "obstruída", "tela uniforme"], measures) == [
        "borrada", "tremida", "obstruída", "tela uniforme",
    ]
    assert _moment_issues(["clara", "clara"], None) == []
    dark = measures.model_copy(update={"brightness": 0.1, "dark": True})
    assert _moment_issues(["clara", "nenhum"], dark) == ["escura"]
    bright = measures.model_copy(update={"brightness": 0.95, "bright": True, "blurry": True})
    assert _moment_issues(["escura"], bright) == ["borrão alto", "clara"]


def test_never_dark_and_bright_together():
    both = ShotMeasures(samples=4, brightness=0.7, motion=0.0, luma_spread=0.9, dark=True, bright=True)
    assert _moment_issues(["escura"], both) == ["clara"]
    assert _moment_issues(["clara"], both.model_copy(update={"brightness": 0.3})) == ["escura"]


def test_prompt_tags_are_filtered_when_building_moments():
    assert all(is_prompt_tag(t) for t in [
        "papel principal", "Papel", "principal", "drone", "pov", "broll", "b-roll", "contexto", "sem fala",
        "com fala", "Silêncio", "foto", "imagem", "P001", "p001.f002", "D003.M04", "X001.T03",
    ])
    assert not any(is_prompt_tag(t) for t in ["praia", "mar", "fala alta", "a380", "pessoa"])
    assert "papel principal" in PROMPT_TAGS
    noise = ["papel principal", "sem fala", "contexto", "P001.F001"]
    photos = [
        _described("P001", 1, 0.0, "", ["mar", "areia", *noise]),
        _described("P001", 2, 2.0, "", ["mar", "areia", "Silêncio", "papel"]),
        _described("P001", 3, 4.0, "", ["rua", "carro", *noise]),  # só ruído em comum com as anteriores
    ]
    moments = build_moments("P001", photos, 6.0, scene_cuts=[])
    assert [m.photo_ids for m in moments] == [["P001.F001", "P001.F002"], ["P001.F003"]]
    assert moments[0].tags == ["mar", "areia"] and moments[1].tags == ["rua", "carro"]
    # A descrição (a do cache) não muda: o filtro é só na hora de montar o momento.
    assert photos[0].description.tags == ["mar", "areia", *[t.lower() for t in noise]]


def test_vision_defaults_to_gemma_and_keeps_the_cache_fingerprint():
    describer = _build_photo_describer(build_parser().parse_args(["roteiro", "x.mp4"]), lambda *a: None)
    assert DEFAULT_CHECK_IMAGE_MODEL == "gemma3:4b" and describer.model_chain == ["ollama:gemma3:4b"]
    assert build_parser().parse_args(["ollama-check"]).modelo_imagem == "gemma3:4b"
    # Mesmo fingerprint das 644 descrições já gravadas com `--visao-model gemma3:4b` (não pode mudar).
    assert VISION_PROMPT_VERSION == 1 and describer.fingerprint == "6859a494d42aad5e"
    qwen = _build_photo_describer(
        build_parser().parse_args(["roteiro", "x.mp4", "--visao-model", "qwen2.5vl:3b"]), lambda *a: None
    )
    assert qwen.model_chain == ["ollama:qwen2.5vl:3b"]


# --- descrição por IA (mock, sem FFmpeg) ---


def test_photo_description_normalizes_ai_answers():
    d = PhotoDescription.model_validate({
        "what": " Mulher na praia ", "tags": ["Mar", "mar", " areia ", "a", "b", "c", "d", "e"], "issue": "Obstruida",
    })
    assert d.what == "Mulher na praia" and d.tags == ["mar", "areia", "a", "b", "c", "d"] and d.issue == "obstruída"
    assert PhotoDescription.model_validate({"what": "x", "tags": "rua, carro", "issue": "desfocada"}).issue == "borrada"
    with pytest.raises(ValueError):
        PhotoDescription.model_validate({"what": "x", "tags": ["a"], "issue": "fantasma"})


def test_vision_prompt_has_speech_and_previous_description(tmp_path):
    image = tmp_path / "f.jpg"
    from PIL import Image

    Image.new("RGB", (32, 32), (200, 20, 20)).save(image)
    previous = PhotoDescription(what="Homem no barco", place="lago", tags=["barco", "lago"])
    request = PhotoRequest("P001.F002", "P001", "pov", image, "olha o peixe", previous)
    prompt = build_vision_prompt(request)
    assert '"olha o peixe"' in prompt and "Homem no barco | lugar: lago | tags: barco, lago" in prompt
    assert "papel pov" in prompt
    silent = build_vision_prompt(PhotoRequest("D001.F001", "D001", "drone", image, ""))
    assert "Fala neste momento do vídeo: sem fala" in silent and "primeira foto do clipe" in silent

    describer, client = mock_describer()
    outcome = describer.describe(request)
    assert outcome.description.tags == ["vermelho", "simulada"] and outcome.api_calls == 1
    assert client.image_calls[0][0].label == "P001.F002"


def test_invalid_vision_answer_is_retried_with_the_error_then_gives_up(tmp_path):
    image = tmp_path / "f.jpg"
    from PIL import Image

    Image.new("RGB", (32, 32)).save(image)
    answers = iter([{"what": "x", "tags": []}, {"what": "Rua", "tags": ["rua"]}])
    describer, client = mock_describer(lambda prompt, images, schema: next(answers))
    outcome = describer.describe(PhotoRequest("P001.F001", "P001", "principal", image, ""))
    assert outcome.description.what == "Rua" and outcome.corrections == 1
    assert "recusada" in client.calls[1][0] and "tags" in client.calls[1][0]

    bad, client = mock_describer(lambda prompt, images, schema: {"what": ""})
    failed = bad.describe(PhotoRequest("P001.F001", "P001", "principal", image, ""))
    assert failed.description is None and "3 tentativas" in failed.error and len(client.calls) == 3


# --- caches compartilhados com o `edit` (sem FFmpeg) ---


def test_shared_clip_fingerprints_are_the_ones_of_edit():
    assert silences_fingerprint("abc", "auto", 0.5) == fingerprint("silences", "abc", ("auto", AUTO_RULE_VERSION), 0.5)
    assert silences_fingerprint("abc", -35.0, 0.5) == fingerprint("silences", "abc", ("fixed", -35.0), 0.5)
    provider = FileTranscriber()
    config = TranscriptionConfig(language="pt")
    effective = provider.effective_settings(config)
    expected = config.model_copy(
        update={"model_size": provider.effective_model(config), "device": effective[0], "compute_type": effective[1]}
    )
    assert transcription_fingerprint("abc", provider, config) == fingerprint("transcription", "abc", "mock-roteiro", expected)


def test_markdown_has_ids_rhythm_marks_and_token_estimate():
    from ai_video_editor.roteiro.schema import Moment, SentenceRhythm

    roteiro = ProjectRoteiro(
        project_dir=Path("p"), source=Path("viagem"), kind="pasta", interval_sec=2.0,
        clips=[RoteiroClip(id="P001", role="principal", relative_path="principal/a.mp4", order=1, duration_sec=65,
                           has_speech=True)],
        moments=[Moment(id="P001.M01", clip_id="P001", start_sec=0, end_sec=65, description="Mulher na praia",
                        tags=["praia"], issues=["tremida"])],
        sentences=[RoteiroSentence(
            id="S000", index=0, clip_id="P001", start_sec=1, end_sec=3, text="Oi gente.", moment_ids=["P001.M01"],
            rhythm=SentenceRhythm(volume="alto", speed="rápida", pause_before_sec=1.5),
        )],
    )
    text = build_roteiro_markdown(roteiro)
    assert "## P001 · principal · 1:05 · com fala" in text
    assert "### P001.M01 · 0:00–1:05 · Mulher na praia [tremida]" in text
    assert "- S000 (alto, rápida, ⏸1,5s antes) — Oi gente." in text
    assert roteiro.estimated_tokens == estimate_tokens(text) and roteiro.clips[0].estimated_tokens > 0
    assert f"~{roteiro.estimated_tokens} tokens (estimativa: caracteres/4)" in text


def test_cli_roteiro_options():
    args = build_parser().parse_args(["roteiro", "pasta", "--visao-provider", "mock", "--intervalo", "3", "--so-estimar"])
    assert (args.visao_provider, args.intervalo, args.so_estimar, args.visao_paralelo) == ("mock", 3.0, True, 1)
    defaults = build_parser().parse_args(["roteiro", "x.mp4"])
    assert defaults.visao_provider is None and defaults.intervalo == 2.0 and defaults.threshold_db == "auto"


# --- execução real (FFmpeg) ---


@pytest.mark.slow
def test_folder_with_three_roles_keeps_every_clip(roles_folder, tmp_path):
    describer, client = mock_describer()
    transcriber = FileTranscriber()
    result = builder(tmp_path, describer, transcriber).build(roles_folder)
    roteiro = result.roteiro

    # Todos os clipes entram (nenhum descartado por papel), na ordem do catálogo; o sem fala também.
    assert [(c.id, c.role, c.has_speech) for c in roteiro.clips] == [
        ("P001", "principal", True), ("D001", "drone", False), ("V001", "pov", True),
    ]
    assert sorted(transcriber.calls) == ["fala.mp4", "pov.mp4"]  # POV com fala também é transcrito
    assert {m.clip_id for m in roteiro.moments} == {"P001", "D001", "V001"}
    drone = [p for p in roteiro.photos if p.clip_id == "D001"]
    assert drone and all(p.status == "descrita" and p.speech == "" for p in drone)
    assert [s.clip_id for s in roteiro.sentences] == ["P001", "P001", "V001"]
    assert [s.id for s in roteiro.sentences] == ["S000", "S001", "S002"]
    assert all(s.moment_ids for s in roteiro.sentences)

    # A fala do momento e a descrição anterior chegam no prompt da visão.
    prompts = {p.split("Foto ", 1)[1].split(" ", 1)[0]: p for p, _ in client.calls}
    assert '"Chegamos na praia."' in prompts["P001.F001"]
    assert "primeira foto do clipe" in prompts["P001.F001"]
    assert "Imagem simulada predominantemente vermelho." in prompts["P001.F002"] or "vermelho" in prompts["P001.F002"]
    assert "Fala neste momento do vídeo: sem fala" in prompts["D001.F001"]

    # roteiro.json válido e roteiro.md com IDs; fotos copiadas.
    loaded = ProjectRoteiro.load(result.json_path)
    assert loaded.schema_version == 1 and len(loaded.moments) == len(roteiro.moments)
    text = result.md_path.read_text(encoding="utf-8")
    for needle in ("P001.M01", "D001.M01", "V001.M01", "S000", "S002", "sem áudio", "Chegamos na praia."):
        assert needle in text
    assert all((result.json_path.parent / p.file).is_file() for p in roteiro.photos)
    assert roteiro.estimated_tokens > 0 and all(c.estimated_tokens > 0 for c in roteiro.clips)

    # Segunda execução: nada de IA nem de transcrição.
    calls = len(client.calls)
    again = builder(tmp_path, describer, transcriber).build(roles_folder).roteiro
    assert len(client.calls) == calls and sorted(transcriber.calls) == ["fala.mp4", "pov.mp4"]
    assert again.vision.cached == len(again.photos) and again.vision.described == 0


@pytest.mark.slow
def test_photos_extras_discard_moments_and_retry_of_missing_description(scenes_video, tmp_path):
    def flaky(prompt, images, schema):
        if "Foto P001.F002 " in prompt:
            return {"what": ""}  # sempre inválida nesta execução
        return mock_vision_responder(prompt, images, schema)

    describer, client = mock_describer(flaky)
    roteiro = builder(tmp_path, describer).build(scenes_video).roteiro

    clip = roteiro.clips[0]
    assert roteiro.kind == "arquivo" and [c.id for c in roteiro.clips] == ["P001"]
    assert roteiro.project_dir == project_dir_for(scenes_video, file_hash(scenes_video), tmp_path / "projects").resolve()
    assert clip.scene_changes == 2
    # Grade a cada 2 s + cenas em 3 s e 9 s + frase depois de pausa em 5 s; o azul parado sai repetido.
    photos = [(p.time_sec, p.reasons) for p in roteiro.photos]
    assert [r for _, r in photos] == [["intervalo"], ["cena"], ["cena"]]
    assert [t for t, _ in photos][1] == pytest.approx(3.0, abs=0.05)
    assert clip.photos_before == 9 and clip.photos_kept == 3
    discarded = {round(d.time_sec): (d.reasons, d.similar_to) for d in clip.discarded}
    assert discarded[5] == (["frase"], "P001.F002") and discarded[2] == (["intervalo"], "P001.F001")

    # Momentos pelas trocas de cena; a frase fica no momento azul.
    assert [(m.id, round(m.start_sec), round(m.end_sec)) for m in roteiro.moments] == [
        ("P001.M01", 0, 3), ("P001.M02", 3, 9), ("P001.M03", 9, 12),
    ]
    assert roteiro.sentences[0].moment_ids == ["P001.M02"]

    # A foto sem descrição fica com aviso e é a única pedida de novo na próxima execução.
    missing = [p for p in roteiro.photos if p.status == "sem_descricao"]
    assert [p.id for p in missing] == ["P001.F002"] and roteiro.vision.failed == 1
    good, client2 = mock_describer()
    again = builder(tmp_path, good).build(scenes_video).roteiro
    assert len(client2.calls) == 1 and "Foto P001.F002 " in client2.calls[0][0]
    assert all(p.status == "descrita" for p in again.photos)


@pytest.mark.slow
def test_cached_descriptions_are_reused_and_cleaned_in_the_moments(scenes_video, tmp_path):
    def noisy(prompt, images, schema):
        good = mock_vision_responder(prompt, images, schema)
        return good.model_copy(update={"tags": [*good.tags, "papel principal", "sem fala"], "issue": "clara"})

    describer, _ = mock_describer(noisy)
    builder(tmp_path, describer).build(scenes_video)
    again, client = mock_describer()
    roteiro = builder(tmp_path, again).build(scenes_video).roteiro
    assert client.calls == [] and roteiro.vision.cached == len(roteiro.photos)
    assert all("papel principal" in p.description.tags for p in roteiro.photos)  # cache intacto
    assert all("clara" not in m.issues and not {"papel principal", "sem fala"} & set(m.tags) for m in roteiro.moments)
    assert all(not {"escura", "clara"} <= set(m.issues) for m in roteiro.moments)


@pytest.mark.slow
def test_ctrl_c_during_vision_keeps_described_photos(scenes_video, tmp_path):
    def slow(prompt, images, schema):
        time.sleep(0.2)
        return mock_vision_responder(prompt, images, schema)

    describer, client = mock_describer(slow)
    seen: list[int] = []

    def on_stage_progress(stage, progress):
        if stage == "vision" and progress.done:
            seen.append(int(progress.done))
            if progress.done >= 1:
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        builder(tmp_path, describer, on_stage_progress=on_stage_progress).build(scenes_video)
    assert seen == [1]
    resumed, client2 = mock_describer()
    roteiro = builder(tmp_path, resumed).build(scenes_video).roteiro
    assert roteiro.vision.cached >= 1 and roteiro.vision.described == len(roteiro.photos) - roteiro.vision.cached
    assert len(client2.calls) == roteiro.vision.described


@pytest.mark.slow
def test_reuses_edit_caches_for_folder_and_single_file(roles_folder, scenes_video, tmp_path, monkeypatch):
    # Pasta: a espinha do `edit <pasta>` transcreve o principal; o roteiro só transcreve o POV.
    transcriber = FileTranscriber()
    from tests.test_spine import FakeTimelineExecutor

    SpinePipeline(
        projects_root=tmp_path / "projects", transcription_provider=transcriber,
        render_executor=FakeTimelineExecutor(), speech_detector=NamedSpeechDetector(),
    ).run(roles_folder, EditPipelineParams(transcribe=True, loudness="off"))
    assert transcriber.calls == ["fala.mp4"]
    import ai_video_editor.pipeline.clips as clips_module

    detected: list[str] = []
    original = clips_module.detect_silences
    monkeypatch.setattr(clips_module, "detect_silences", lambda path, *a: detected.append(Path(path).name) or original(path, *a))
    builder(tmp_path, None, transcriber).build(roles_folder)
    assert transcriber.calls == ["fala.mp4", "pov.mp4"] and detected == ["pov.mp4"]

    # Arquivo único: a transcrição do projeto do `edit <vídeo>` (mesmo fingerprint) é aproveitada.
    single = FileTranscriber()
    project = project_dir_for(scenes_video, file_hash(scenes_video), tmp_path / "projects").resolve()
    project.mkdir(parents=True)
    digest = file_hash(scenes_video)
    fp = transcription_fingerprint(digest, single, TranscriptionConfig())
    Transcript(source_path=scenes_video, language="pt", duration_sec=7, model_used="edit",
               segments=_segments("cenas")).save(project / "transcript.json")
    (project / "project.json").write_text(
        json.dumps({"stages": {"transcription": {"status": "done", "fingerprint": fp}}}), encoding="utf-8"
    )
    roteiro = builder(tmp_path, None, single).build(scenes_video).roteiro
    assert single.calls == [] and roteiro.sentences[0].text == "Agora a parte azul do vídeo."
    assert clip_cache_file(tmp_path / "projects", digest, "transcript", fp).is_file()
    assert all(p.status == "pendente" for p in roteiro.photos)  # sem IA de visão


@pytest.mark.slow
def test_rhythm_from_synthetic_audio(tmp_path):
    """Tom em volumes diferentes: frase alta e baixa pelo volume medido; rápida e lenta pelas palavras."""
    wav = tmp_path / "ritmo.wav"
    pieces = [(0.5, 2.0), (0.0, 1.0), (0.5, 2.0), (0.0, 1.0), (1.0, 2.0), (0.0, 1.0), (0.2, 2.0), (0.0, 1.0), (0.5, 2.0)]
    args = [str(get_ffmpeg_path()), "-y", "-v", "error"]
    for amplitude, dur in pieces:
        src = f"sine=frequency=300:sample_rate=16000:duration={dur}" if amplitude else f"anullsrc=r=16000:cl=mono:d={dur}"
        args += ["-f", "lavfi", "-i", src]
    graph = ";".join(f"[{i}:a]volume={a or 1}[a{i}]" for i, (a, _) in enumerate(pieces))
    graph += ";" + "".join(f"[a{i}]" for i in range(len(pieces))) + f"concat=n={len(pieces)}:v=0:a=1[out]"
    subprocess.run([*args, "-filter_complex", graph, "-map", "[out]", str(wav)], capture_output=True, check=True)

    # Frases sobre os tons: 0–2 (normal), 3–5 (normal, rápida), 6–8 (alta), 9–11 (baixa), 12–14 (normal, lenta).
    transcript = _sentence_transcript([(0, 2, 5), (3, 5, 8), (6, 8, 5), (9, 11, 5), (12, 14, 3)])
    sentences = build_clip_sentences([("P001", transcript)])
    rhythms = measure_rhythm(sentences, transcript, 14.0, measure_levels(wav))
    assert [r.volume for r in rhythms] == ["normal", "normal", "alto", "baixo", "normal"]
    assert rhythms[2].relative_db == pytest.approx(6.0, abs=0.6) and rhythms[3].relative_db == pytest.approx(-8.0, abs=0.6)
    assert [r.speed for r in rhythms] == ["normal", "rápida", "normal", "normal", "lenta"]
    assert [r.pause_before_sec for r in rhythms] == [0.0, 1.0, 1.0, 1.0, 1.0]


@pytest.mark.slow
def test_cli_roteiro_estimate_and_run(roles_folder, tmp_path, capsys, monkeypatch):
    import ai_video_editor.roteiro.builder as builder_module

    monkeypatch.setattr(builder_module.RoteiroBuilder, "transcription_provider", property(lambda self: FileTranscriber()))
    monkeypatch.setattr(builder_module.RoteiroBuilder, "speech_detector", property(lambda self: NamedSpeechDetector()))
    monkeypatch.setattr(builder_module.RoteiroBuilder, "_catalog_builder", lambda self: _catalog_builder(tmp_path))
    base = ["roteiro", str(roles_folder), "--projects-dir", str(tmp_path / "projects"), "--no-live", "--visao-provider", "mock"]
    assert handle_roteiro(build_parser().parse_args([*base, "--so-estimar"])) == 0
    out = capsys.readouterr().out
    assert "a IA de visão não foi chamada" in out and "A descrever:" in out and "Tempo estimado" in out
    assert not list((tmp_path / "projects").glob("*/roteiro/roteiro.json"))
    assert handle_roteiro(build_parser().parse_args(base)) == 0
    out = capsys.readouterr().out
    assert "Roteiro concluído!" in out and "Momentos:" in out and "tokens (estimativa" in out and "Tempos:" in out


def _catalog_builder(tmp_path: Path):
    from ai_video_editor.catalog import CatalogBuilder

    return CatalogBuilder(projects_root=tmp_path / "projects", speech_detector=NamedSpeechDetector())


@pytest.mark.slow
def test_scan_reuses_the_visual_pass(scenes_video, tmp_path, monkeypatch):
    """Com o `visual` já feito, o roteiro não decodifica o clipe de novo para achar cenas e medidas."""
    import shutil

    from ai_video_editor.catalog import CatalogBuilder
    from ai_video_editor.visual import VisualBuilder

    folder = tmp_path / "material"
    folder.mkdir()
    shutil.copy2(scenes_video, folder / "cenas.mp4")
    projects = tmp_path / "projects"
    VisualBuilder(
        projects_root=projects, catalog_builder=CatalogBuilder(projects_root=projects, speech_detector=NamedSpeechDetector())
    ).build(folder)
    import ai_video_editor.roteiro.builder as builder_module

    def no_scan(path):
        raise AssertionError("scan_clip não devia rodar")

    monkeypatch.setattr(builder_module, "scan_clip", no_scan)
    roteiro = builder(tmp_path).build(folder).roteiro
    assert roteiro.clips[0].scene_changes == 2 and len(roteiro.moments) == 3
