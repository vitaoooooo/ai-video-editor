"""Comando `roteiro`: o roteiro unificado (fala + imagem + ritmo) de qualquer material, com IDs.

Etapas: catálogo (lido ou gerado; um arquivo = projeto de 1 clipe) → silêncios e volume (por clipe) →
transcrição (todo clipe com fala, de qualquer papel) → frases e ritmo → fotos (FFmpeg: grade fixa +
extras em início de frase depois de pausa e em troca de cena; quase iguais descartadas) → descrição de
cada foto (IA de visão) → momentos → roteiro (`roteiro/roteiro.json` e `roteiro/roteiro.md`).

O programa só mede, descreve, confere e monta; nenhuma regra depende de papel ou de tipo de vídeo: o
papel e "tem fala" são informação para a IA. Clipes sem fala entram com momentos e descrições.

Caches (fora do projeto, chave = hash do arquivo + parâmetros), em `<projects>/_cache/edit_clips/<hash>/`:
silêncios e transcrição (os MESMOS arquivos e fingerprints do `edit <pasta>`; num arquivo único, também
os do projeto do `edit <vídeo>` são aproveitados), picos por janela de 50 ms (`levels-*.json`), trocas
de cena e medidas (`scan-*.json`, amostras brutas do `scan_clip`) e as fotos extraídas (`photos-*/`).
Descrições por foto em `<projects>/_cache/roteiro_fotos/` (ver `describer.py`).
"""

import json
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from pydantic import BaseModel, Field

from ai_video_editor.audio import AudioAnalysisResult
from ai_video_editor.audio.levels import WINDOW_SEC, measure_levels
from ai_video_editor.catalog.schema import CatalogClip, ProjectCatalog, StageTime
from ai_video_editor.director.sentences import build_clip_sentences
from ai_video_editor.intelligence.exceptions import AIAuthenticationError
from ai_video_editor.pipeline.clips import (
    catalog_location,
    clip_cache_file,
    load_or_build_catalog,
    load_or_detect_silences,
    silences_fingerprint,
    transcribe_to_cache,
    transcription_fingerprint,
)
from ai_video_editor.pipeline.project import (
    AUDIO_ANALYSIS_FILE,
    DEFAULT_PROJECTS_ROOT,
    MANIFEST_FILE,
    TRANSCRIPT_FILE,
    file_hash,
    fingerprint,
    project_dir_for,
)
from ai_video_editor.pipeline.schema import StageProgress
from ai_video_editor.pipeline.timing import format_duration
from ai_video_editor.roteiro.describer import (
    IMAGE_TOKENS_ESTIMATE,
    OUTPUT_TOKENS_ESTIMATE,
    PROMPT_TOKENS_ESTIMATE,
    SECONDS_PER_PHOTO_ESTIMATE,
    PhotoDescriber,
    PhotoRequest,
    cache_file,
    image_hash,
    load_cached,
    save_cached,
    short_speech,
)
from ai_video_editor.roteiro.moments import build_moments, finalize_issues, link_sentences, moment_rules
from ai_video_editor.roteiro.photos import (
    PAUSE_PHOTO_SEC,
    PHOTO_FORMAT_VERSION,
    PHOTO_INTERVAL_SEC,
    PHOTO_MAX_SIDE,
    PHOTO_QUALITY,
    ExtractedPhoto,
    PhotoSelection,
    extract_photos,
    photo_rules,
    plan_photos,
    scene_changes,
    select_photos,
)
from ai_video_editor.roteiro.report import build_roteiro_markdown
from ai_video_editor.roteiro.rhythm import measure_rhythm, rhythm_rules
from ai_video_editor.roteiro.schema import (
    DiscardedPhoto,
    ProjectRoteiro,
    RoteiroClip,
    RoteiroEstimate,
    RoteiroPhoto,
    RoteiroSentence,
    VisionRun,
)
from ai_video_editor.transcription.provider import TranscriptionProvider
from ai_video_editor.transcription.schema import Transcript, TranscriptionConfig
from ai_video_editor.visual.shots import ANALYSIS_WIDTH, MEASURE_FPS, SCENE_THRESHOLD, ClipScan, mark_shaky, scan_clip

ROTEIRO_DIR = "roteiro"
ROTEIRO_JSON = "roteiro.json"
ROTEIRO_MD = "roteiro.md"
PHOTOS_DIR = "fotos"
# Arquivo único: o resumo do clipe (ffprobe + fala) fica no projeto, para não rodar o VAD de novo.
SINGLE_CLIP_FILE = "clip.json"
PHOTOS_MANIFEST = "photos.json"

ROTEIRO_STAGES = ("catalog", "silences", "transcription", "rhythm", "photos", "vision", "moments", "report")
ROTEIRO_STAGE_LABELS = {
    "catalog": "Catálogo",
    "silences": "Silêncios e volume",
    "transcription": "Transcrição",
    "rhythm": "Frases e ritmo",
    "photos": "Fotos",
    "vision": "Descrição das fotos",
    "moments": "Momentos",
    "report": "Roteiro",
}

# Um clipe é transcrito com pelo menos isso de fala detectada pelo catálogo (ou fala desconhecida).
MIN_SPEECH_TO_TRANSCRIBE_SEC = 1.0
# Fotos seguidas com erro da API (não de validação) antes de parar de enviar as restantes.
MAX_CONSECUTIVE_FAILURES = 3
# Formatos dos caches próprios do roteiro (sobem quando o conteúdo muda).
LEVELS_FORMAT = "levels-v1"
SCAN_FORMAT = "scan-v1"

# (etapa, evento, detalhe); eventos: "start", "done", "skipped", "notice".
RoteiroProgress = Callable[[str, str, str], None]
RoteiroStageProgress = Callable[[str, StageProgress], None]


class RoteiroParams(BaseModel):
    """Parâmetros do `roteiro`. Silêncios e transcrição com os mesmos padrões do `edit` (mesmo cache)."""

    interval_sec: float = Field(default=PHOTO_INTERVAL_SEC, gt=0.0)
    threshold_db: Union[float, Literal["auto"]] = "auto"
    min_silence_sec: float = Field(default=0.5, gt=0.0)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)


class RoteiroResult(BaseModel):
    project_dir: Path
    roteiro: Optional[ProjectRoteiro] = None
    estimate: Optional[RoteiroEstimate] = None
    json_path: Optional[Path] = None
    md_path: Optional[Path] = None
    stage_times: list[StageTime] = Field(default_factory=list)


@dataclass
class _Clip:
    """Um clipe ao longo das etapas."""

    info: CatalogClip
    path: Path
    hash: str
    order: int
    duration: float
    transcribe: bool
    audio: Optional[AudioAnalysisResult] = None
    levels: Optional[list[float]] = None
    transcript: Optional[Transcript] = None
    scan: Optional[ClipScan] = None
    cuts: list[float] = field(default_factory=list)
    extracted: list[ExtractedPhoto] = field(default_factory=list)
    selection: Optional[PhotoSelection] = None
    discarded: list[DiscardedPhoto] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.info.id or ""


@dataclass
class _State:
    target: Path
    kind: Literal["arquivo", "pasta"]
    project_dir: Path
    params: RoteiroParams
    clips: list[_Clip] = field(default_factory=list)
    sentences: list[RoteiroSentence] = field(default_factory=list)
    photos: list[RoteiroPhoto] = field(default_factory=list)
    photo_files: dict[str, Path] = field(default_factory=dict)  # ID → imagem no cache
    warnings: list[str] = field(default_factory=list)
    stage_times: list[StageTime] = field(default_factory=list)


def needs_transcription(clip: CatalogClip) -> bool:
    """Todo clipe com fala detectada, de qualquer papel; fala desconhecida (sem VAD) também."""
    if clip.media is None or not clip.media.has_audio:
        return False
    speech = clip.speech
    if speech is None or speech.status == "desconhecido":
        return True
    if speech.status == "sem_audio":
        return False
    return (speech.speech_sec or 0.0) >= MIN_SPEECH_TO_TRANSCRIBE_SEC


def reliable_date(clip: CatalogClip):
    """Data/hora de gravação só quando confiável (nunca a de modificação do arquivo)."""
    recorded = clip.recorded_at
    if recorded is None or not recorded.reliable or recorded.source == "mtime":
        return None
    return clip.adjusted_at or recorded.value


def photo_speech(transcript: Optional[Transcript], start: float, end: float) -> str:
    """Palavras do Whisper que começam em [início, fim); vazio = sem fala."""
    if transcript is None:
        return ""
    words: list[str] = []
    for segment in transcript.segments:
        if segment.end_sec <= start or segment.start_sec >= end:
            continue
        if segment.words:
            words += [w.text.strip() for w in segment.words if start <= w.start_sec < end and w.text.strip()]
        elif start <= segment.start_sec < end:
            words += segment.text.split()
    return short_speech(words)


class RoteiroBuilder:
    """Monta o roteiro unificado de um arquivo ou de uma pasta de clipes."""

    def __init__(
        self,
        projects_root: Path = DEFAULT_PROJECTS_ROOT,
        describer: Optional[PhotoDescriber] = None,
        transcription_provider: Optional[TranscriptionProvider] = None,
        speech_detector=None,
        on_progress: Optional[RoteiroProgress] = None,
        on_stage_progress: Optional[RoteiroStageProgress] = None,
        workers: Optional[int] = None,
        vision_parallel: int = 1,
    ):
        self.projects_root = Path(projects_root)
        self.describer = describer
        self._transcription_provider = transcription_provider
        self._speech_detector = speech_detector
        self.on_progress = on_progress
        self.on_stage_progress = on_stage_progress
        self.workers = workers or max(1, min(4, os.cpu_count() or 2))
        self.vision_parallel = max(1, vision_parallel)

    @property
    def transcription_provider(self) -> TranscriptionProvider:
        if self._transcription_provider is None:
            from ai_video_editor.transcription.providers.faster_whisper import FasterWhisperProvider

            self._transcription_provider = FasterWhisperProvider(
                on_warning=lambda message: self._emit("transcription", "notice", message)
            )
        return self._transcription_provider

    @property
    def speech_detector(self):
        if self._speech_detector is None:
            from ai_video_editor.catalog.speech import FasterWhisperVAD

            self._speech_detector = FasterWhisperVAD()
        return self._speech_detector

    # --- eventos ---

    def _emit(self, stage: str, event: str, detail: str = "") -> None:
        if self.on_progress is not None:
            self.on_progress(stage, event, detail)

    def _progress(self, stage: str, progress: StageProgress) -> None:
        if self.on_stage_progress is not None:
            self.on_stage_progress(stage, progress)

    def _stage(self, s: _State, stage: str, work: Callable[[], str]) -> None:
        self._emit(stage, "start")
        started = time.perf_counter()
        detail = work()
        elapsed = time.perf_counter() - started
        s.stage_times.append(StageTime(name=stage, elapsed_sec=round(elapsed, 3)))
        self._emit(stage, "done", f"{detail} ({format_duration(elapsed)})")

    # --- execução ---

    def build(
        self,
        target: Path,
        params: Optional[RoteiroParams] = None,
        estimate_only: bool = False,
        force: bool = False,
    ) -> RoteiroResult:
        """Executa as etapas (com cache) e grava o roteiro; com `estimate_only`, para antes da IA de visão.

        `force` refaz as medidas do programa (silêncios, volume, cenas e fotos); a transcrição e as
        descrições em cache continuam valendo (são caras e não mudam com as regras do programa).
        """
        started = time.perf_counter()
        params = params or RoteiroParams()
        target = Path(target).resolve()
        if not target.exists():
            raise FileNotFoundError(f"arquivo ou pasta não encontrado: {target}")
        s = self._open(target, params)

        self._stage(s, "catalog", lambda: self._catalog(s))
        self._stage(s, "silences", lambda: self._silences(s, force))
        transcribe = [c for c in s.clips if c.transcribe and c.audio is not None]
        if transcribe:
            self._stage(s, "transcription", lambda: self._transcription(s, transcribe))
        else:
            self._emit("transcription", "skipped", "nenhum clipe com fala")
        self._stage(s, "rhythm", lambda: self._rhythm(s))
        self._stage(s, "photos", lambda: self._photos(s, force))

        if estimate_only:
            estimate = self._estimate(s)
            return RoteiroResult(project_dir=s.project_dir, estimate=estimate, stage_times=s.stage_times)

        vision: Optional[VisionRun] = None
        if self.describer is None:
            self._emit("vision", "skipped", "sem IA de visão (--visao-provider none)")
        else:
            holder: list[VisionRun] = []
            self._stage(s, "vision", lambda: self._vision(s, holder))
            vision = holder[0]

        roteiro = ProjectRoteiro(
            project_dir=s.project_dir, source=s.target, kind=s.kind, interval_sec=params.interval_sec,
            sentences=s.sentences, photos=s.photos, vision=vision,
        )
        self._stage(s, "moments", lambda: self._moments(s, roteiro))
        roteiro_dir = s.project_dir / ROTEIRO_DIR
        report_started = time.perf_counter()
        self._emit("report", "start")
        roteiro.rules = self._rules(params)
        roteiro.warnings = [*s.warnings, *(f"{c.id}: {w}" for c in s.clips for w in c.warnings)]
        roteiro.stage_times = s.stage_times
        self._write_photos(s, roteiro_dir)
        markdown = build_roteiro_markdown(roteiro)
        s.stage_times.append(StageTime(name="report", elapsed_sec=round(time.perf_counter() - report_started, 3)))
        roteiro.total_sec = round(time.perf_counter() - started, 3)
        json_path, md_path = roteiro_dir / ROTEIRO_JSON, roteiro_dir / ROTEIRO_MD
        roteiro.save(json_path)
        md_path.write_text(markdown, encoding="utf-8")
        self._emit("report", "done", f"{md_path} (~{roteiro.estimated_tokens} tokens estimados)")
        return RoteiroResult(
            project_dir=s.project_dir, roteiro=roteiro, json_path=json_path, md_path=md_path,
            stage_times=s.stage_times,
        )

    def _open(self, target: Path, params: RoteiroParams) -> _State:
        if target.is_dir():
            project_dir, _ = catalog_location(target, self.projects_root)
            return _State(target=target, kind="pasta", project_dir=project_dir, params=params)
        project_dir = project_dir_for(target, file_hash(target), self.projects_root).resolve()
        return _State(target=target, kind="arquivo", project_dir=project_dir, params=params)

    # --- 1. catálogo ---

    def _catalog_builder(self):
        from ai_video_editor.catalog import CATALOG_STAGE_LABELS, CatalogBuilder

        def on_progress(stage: str, event: str, detail: str) -> None:
            if event == "start":
                self._progress("catalog", StageProgress(note=CATALOG_STAGE_LABELS.get(stage, stage)))
            elif event == "notice":
                self._emit("catalog", "notice", detail)

        return CatalogBuilder(
            projects_root=self.projects_root,
            speech_detector=self._speech_detector,
            on_progress=on_progress,
            on_stage_progress=lambda stage, progress: self._progress(
                "catalog", progress.model_copy(update={"note": CATALOG_STAGE_LABELS.get(stage, stage)})
            ),
        )

    def _catalog(self, s: _State) -> str:
        if s.kind == "pasta":
            catalog, built = load_or_build_catalog(s.target, self.projects_root, self._catalog_builder)
            s.project_dir = Path(catalog.project_dir).resolve()
            infos = self._catalog_clips(s, catalog)
            how = "gerado agora" if built else "lido"
        else:
            infos, how = [self._single_clip(s)], "arquivo único"
        for order, info in enumerate(infos, start=1):
            path = (s.target if s.kind == "arquivo" else s.target / info.relative_path).resolve()
            s.clips.append(_Clip(
                info=info, path=path, hash=file_hash(path), order=order, duration=info.media.duration_sec,
                transcribe=needs_transcription(info),
            ))
        roles: dict[str, int] = {}
        for clip in s.clips:
            roles[clip.info.role] = roles.get(clip.info.role, 0) + 1
        per_role = ", ".join(f"{role} {n}" for role, n in roles.items())
        speaking = sum(1 for c in s.clips if c.transcribe)
        return f"{len(s.clips)} clipe(s) ({per_role}), {speaking} com fala a transcrever; catálogo {how}"

    def _catalog_clips(self, s: _State, catalog: ProjectCatalog) -> list[CatalogClip]:
        clips = []
        for clip in catalog.clips:
            if clip.id is None or clip.error is not None or clip.media is None:
                s.warnings.append(f"Fora do roteiro (arquivo com problema): {clip.relative_path} — {clip.error}")
                continue
            clips.append(clip)
        if not clips:
            raise ValueError(f"nenhum clipe de vídeo legível em {s.target}")
        return clips

    def _single_clip(self, s: _State) -> CatalogClip:
        """Arquivo único = projeto de 1 clipe (P001). O resumo (ffprobe + fala) fica no projeto."""
        from ai_video_editor.catalog.builder import describe_clip

        detector = self.speech_detector
        digest = file_hash(s.target)
        path = s.project_dir / ROTEIRO_DIR / SINGLE_CLIP_FILE
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("hash") == digest and data.get("detector") == detector.name:
                    return CatalogClip.model_validate(data["clip"])
            except (ValueError, KeyError):
                pass
        clip = describe_clip(s.target, detector)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"hash": digest, "detector": detector.name, "clip": clip.model_dump(mode="json")}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return clip

    def _adopt_edit_cache(self, s: _State, stage: str, fp: str, artifact: str, target: Path) -> None:
        """Arquivo único: aproveita o resultado do `edit <vídeo>` do mesmo projeto (mesmo fingerprint)."""
        if s.kind != "arquivo" or target.is_file():
            return
        manifest = s.project_dir / MANIFEST_FILE
        source = s.project_dir / artifact
        if not manifest.is_file() or not source.is_file():
            return
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))["stages"][stage]
        except (ValueError, KeyError, TypeError):
            return
        if record.get("status") == "done" and record.get("fingerprint") == fp:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    # --- 2. silêncios e volume ---

    def _silences(self, s: _State, force: bool) -> str:
        clips = [c for c in s.clips if c.transcribe]
        if not clips:
            return "nenhum clipe com fala"
        params = s.params
        total = len(clips)
        cached = 0
        self._progress("silences", StageProgress(done=0, total=total, unit="clips"))

        def work(clip: _Clip) -> tuple[_Clip, bool, Optional[str]]:
            fp = silences_fingerprint(clip.hash, params.threshold_db, params.min_silence_sec)
            path = clip_cache_file(self.projects_root, clip.hash, "silences", fp)
            self._adopt_edit_cache(s, "silences", fp, AUDIO_ANALYSIS_FILE, path)
            try:
                clip.audio, from_cache = load_or_detect_silences(
                    clip.path, path, params.threshold_db, params.min_silence_sec, force
                )
                clip.levels, levels_cached = self._levels(clip, force)
            except Exception as e:  # noqa: BLE001 — um clipe com problema não derruba os outros
                return clip, False, str(e)
            return clip, from_cache and levels_cached, None

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for done, (clip, from_cache, error) in enumerate(pool.map(work, clips), start=1):
                cached += from_cache
                if error is not None:
                    clip.transcribe = False
                    clip.warnings.append(f"áudio não analisado ({error}); clipe tratado como sem fala")
                    self._emit("silences", "notice", f"{clip.id}: {error}")
                self._progress("silences", StageProgress(done=done, total=total, unit="clips", note=clip.id))
        silence = sum(x.duration for c in clips if c.audio for x in c.audio.silences)
        return f"{total} clipe(s) ({cached} do cache), {silence:.1f}s de silêncio"

    def _levels(self, clip: _Clip, force: bool) -> tuple[list[float], bool]:
        fp = fingerprint("levels", clip.hash, WINDOW_SEC, LEVELS_FORMAT)
        path = clip_cache_file(self.projects_root, clip.hash, "levels", fp)
        if path.is_file() and not force:
            try:
                return [float(v) for v in json.loads(path.read_text(encoding="utf-8"))], True
            except ValueError:
                pass
        levels = measure_levels(clip.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(levels), encoding="utf-8")
        return levels, False

    # --- 3. transcrição ---

    def _transcription(self, s: _State, clips: list[_Clip]) -> str:
        provider = self.transcription_provider
        config = s.params.transcription
        total_sec = sum(c.duration for c in clips)
        done_sec, cached, segments = 0.0, 0, 0
        for k, clip in enumerate(clips):
            note = f"{clip.id} ({k + 1}/{len(clips)})"
            self._progress("transcription", StageProgress(done=done_sec, total=total_sec, note=note))
            fp = transcription_fingerprint(clip.hash, provider, config)
            path = clip_cache_file(self.projects_root, clip.hash, "transcript", fp)
            self._adopt_edit_cache(s, "transcription", fp, TRANSCRIPT_FILE, path)
            if path.is_file():
                clip.transcript = Transcript.load(path)
                cached += 1
            else:
                offset = done_sec
                clip.transcript = transcribe_to_cache(
                    provider, clip.id, clip.path, path, config,
                    on_progress=lambda done, _total, offset=offset, note=note: self._progress(
                        "transcription", StageProgress(done=offset + done, total=total_sec, note=note)
                    ),
                )
            done_sec += clip.duration
            segments += len(clip.transcript.segments)
        return f"{len(clips)} clipe(s) ({cached} do cache), {segments} segmentos"

    # --- 4. frases e ritmo ---

    def _rhythm(self, s: _State) -> str:
        transcribed = [c for c in s.clips if c.transcript is not None]
        sentences = build_clip_sentences([(c.id, c.transcript) for c in transcribed])
        by_clip: dict[str, list] = {}
        for sentence in sentences:
            by_clip.setdefault(sentence.clip_id, []).append(sentence)
        for clip in transcribed:
            mine = by_clip.get(clip.id, [])
            rhythms = measure_rhythm(
                mine, clip.transcript, clip.duration, clip.levels, clip.audio.silences if clip.audio else None
            )
            for sentence, rhythm in zip(mine, rhythms):
                s.sentences.append(RoteiroSentence(
                    id=f"S{sentence.id:03d}", index=sentence.id, clip_id=clip.id, start_sec=sentence.start_sec,
                    end_sec=sentence.end_sec, text=sentence.text, rhythm=rhythm,
                ))
        marked = sum(
            1 for x in s.sentences
            if x.rhythm.volume not in (None, "normal") or x.rhythm.speed not in (None, "normal")
        )
        return f"{len(s.sentences)} frase(s) em {len(by_clip)} clipe(s), {marked} fora do ritmo normal"

    # --- 5. fotos ---

    def _photos(self, s: _State, force: bool) -> str:
        total = len(s.clips)
        self._progress("photos", StageProgress(done=0, total=total, unit="clips"))
        starts: dict[str, list[float]] = {}
        for sentence in s.sentences:
            if sentence.rhythm.pause_before_sec >= PAUSE_PHOTO_SEC:
                starts.setdefault(sentence.clip_id, []).append(sentence.start_sec)

        def work(clip: _Clip) -> tuple[_Clip, Optional[str]]:
            try:
                clip.scan = self._scan(clip, force)
                clip.cuts = scene_changes(clip.scan.cuts, clip.duration)
                planned = plan_photos(clip.duration, s.params.interval_sec, starts.get(clip.id, []), clip.cuts)
                clip.extracted = self._extract(clip, planned, s.params.interval_sec, force)
                clip.selection = select_photos(clip.extracted)
            except Exception as e:  # noqa: BLE001
                return clip, str(e)
            return clip, None

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for done, (clip, error) in enumerate(pool.map(work, s.clips), start=1):
                if error is not None:
                    clip.warnings.append(f"fotos não extraídas ({error})")
                    self._emit("photos", "notice", f"{clip.id}: {error}")
                self._progress("photos", StageProgress(done=done, total=total, unit="clips", note=clip.id))
        for clip in s.clips:
            self._clip_photos(s, clip)
        before = sum(len(c.extracted) for c in s.clips)
        return f"{before} fotos, {len(s.photos)} mantidas ({before - len(s.photos)} quase iguais descartadas)"

    def _scan(self, clip: _Clip, force: bool) -> ClipScan:
        """Trocas de cena e medidas por amostra (`scan_clip` do `visual`), com cache bruto por arquivo.

        Sem cache próprio, aproveita o do comando `visual` (a mesma passada do FFmpeg, uma decodificação
        a menos): as amostras são as mesmas e as trocas de cena são os inícios das tomadas `cena`.
        """
        fp = fingerprint("scan", clip.hash, SCENE_THRESHOLD, ANALYSIS_WIDTH, MEASURE_FPS, SCAN_FORMAT)
        path = clip_cache_file(self.projects_root, clip.hash, "scan", fp)
        if path.is_file() and not force:
            try:
                return ClipScan.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        scan = None if force else self._visual_scan(clip)
        if scan is None:
            scan = scan_clip(clip.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(scan.model_dump_json(), encoding="utf-8")
        return scan

    def _visual_scan(self, clip: _Clip) -> Optional[ClipScan]:
        """Passada do `visual` já feita para este arquivo (`_cache/clips/visual-*/shots.json`), se houver."""
        from ai_video_editor.catalog.builder import CLIP_CACHE_DIR
        from ai_video_editor.visual.builder import VisualBuilder, clip_cache_key

        stat = clip.path.stat()
        key = clip_cache_key(clip.path, stat.st_size, stat.st_mtime_ns)
        cached = VisualBuilder._load_shots(self.projects_root.resolve() / CLIP_CACHE_DIR / key)
        if cached is None or not cached.samples:
            return None
        return ClipScan(cuts=[shot.start_sec for shot in cached.shots if shot.boundary == "cena"], samples=cached.samples)

    def _extract(self, clip: _Clip, planned, interval: float, force: bool) -> list[ExtractedPhoto]:
        """Fotos do cache (mesmos tempos planejados) ou extraídas agora."""
        fp = fingerprint(
            "photos", clip.hash, interval, [(p.time_sec, p.reasons, p.grid) for p in planned],
            PHOTO_MAX_SIDE, PHOTO_QUALITY, PHOTO_FORMAT_VERSION,
        )
        folder = clip_cache_file(self.projects_root, clip.hash, "photos", fp).with_suffix("")
        manifest = folder / PHOTOS_MANIFEST
        if manifest.is_file() and not force:
            try:
                items = json.loads(manifest.read_text(encoding="utf-8"))
                photos = [ExtractedPhoto(i["time_sec"], i["reasons"], folder / i["file"]) for i in items]
                if photos and all(p.file.is_file() for p in photos):
                    return photos
            except (ValueError, KeyError, TypeError):
                pass
        if folder.exists():
            shutil.rmtree(folder)
        photos = extract_photos(clip.path, planned, folder, interval)
        items = [{"time_sec": p.time_sec, "reasons": p.reasons, "file": p.file.name} for p in photos]
        manifest.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return photos

    def _clip_photos(self, s: _State, clip: _Clip) -> None:
        """Fotos mantidas do clipe com IDs (`P007.F023`) e a fala até a próxima; descartadas só contadas."""
        if clip.selection is None:
            return
        kept = clip.selection.kept
        ids: list[str] = []
        for number, (photo, diff) in enumerate(kept, start=1):
            photo_id = f"{clip.id}.F{number:03d}"
            ids.append(photo_id)
            end = kept[number][0].time_sec if number < len(kept) else clip.duration
            s.photo_files[photo_id] = photo.file
            s.photos.append(RoteiroPhoto(
                id=photo_id, clip_id=clip.id, time_sec=photo.time_sec, reasons=photo.reasons,
                file=f"{PHOTOS_DIR}/{photo_id}.jpg", image_hash=image_hash(photo.file), diff=diff,
                speech=photo_speech(clip.transcript, photo.time_sec, end),
            ))
        clip.discarded = [
            DiscardedPhoto(time_sec=photo.time_sec, reasons=photo.reasons, similar_to=ids[index], diff=diff)
            for photo, index, diff in clip.selection.discarded
        ]

    # --- 6. descrição das fotos ---

    def _estimate(self, s: _State) -> RoteiroEstimate:
        cached = 0
        if self.describer is not None:
            fp = self.describer.fingerprint
            cached = sum(
                1 for p in s.photos if load_cached(cache_file(self.projects_root, fp, p.image_hash)) is not None
            )
        todo = len(s.photos) - cached
        return RoteiroEstimate(
            photos_before=sum(len(c.extracted) for c in s.clips),
            photos_kept=len(s.photos),
            cached=cached,
            to_describe=todo,
            calls=todo,
            input_tokens=todo * (IMAGE_TOKENS_ESTIMATE + PROMPT_TOKENS_ESTIMATE),
            output_tokens=todo * OUTPUT_TOKENS_ESTIMATE,
            seconds=round(todo * SECONDS_PER_PHOTO_ESTIMATE / self.vision_parallel, 1),
        )

    def _vision(self, s: _State, holder: list[VisionRun]) -> str:
        describer = self.describer
        fp = describer.fingerprint
        run = VisionRun(
            provider=describer.provider_name, model_chain=describer.model_chain, parallel=self.vision_parallel,
            photos=len(s.photos),
        )
        holder.append(run)
        for photo in s.photos:
            cached = load_cached(cache_file(self.projects_root, fp, photo.image_hash))
            if cached is not None:
                photo.description, photo.status = cached, "descrita"
        run.cached = sum(1 for p in s.photos if p.status == "descrita")
        todo = len(s.photos) - run.cached
        if todo == 0:
            self._emit("vision", "notice", "todas as fotos já descritas (cache)")
            return f"{len(s.photos)} fotos, todas do cache"
        roles = {c.id: c.info.role for c in s.clips}
        by_clip: dict[str, list[RoteiroPhoto]] = {}
        for photo in s.photos:
            by_clip.setdefault(photo.clip_id, []).append(photo)
        jobs: "queue.Queue[str]" = queue.Queue()
        for clip_id, photos in by_clip.items():
            if any(p.status != "descrita" for p in photos):
                jobs.put(clip_id)
        results: "queue.Queue[tuple]" = queue.Queue()
        stop = threading.Event()
        failures = {"consecutive": 0}
        lock = threading.Lock()

        def work() -> None:
            # Um clipe por vez, em ordem: a foto seguinte recebe a descrição da anterior.
            while not stop.is_set():
                try:
                    clip_id = jobs.get_nowait()
                except queue.Empty:
                    return
                previous = None
                for photo in by_clip[clip_id]:
                    if stop.is_set():
                        return
                    if photo.description is not None:
                        previous = photo.description
                        continue
                    with lock:
                        gave_up = failures["consecutive"] >= MAX_CONSECUTIVE_FAILURES
                    if gave_up:
                        photo.status, photo.note = "sem_descricao", "não enviada: a IA falhou nas fotos anteriores"
                        results.put(("photo", photo, None, 0.0))
                        continue
                    request = PhotoRequest(
                        photo_id=photo.id, clip_id=clip_id, role=roles.get(clip_id, ""),
                        image=s.photo_files[photo.id], speech=photo.speech, previous=previous,
                    )
                    began = time.perf_counter()
                    try:
                        outcome = describer.describe(request)
                    except AIAuthenticationError as e:
                        results.put(("fatal", e))
                        return
                    elapsed = time.perf_counter() - began
                    if outcome.description is not None:
                        save_cached(cache_file(self.projects_root, fp, photo.image_hash), outcome.description)
                        photo.description, photo.status = outcome.description, "descrita"
                        previous = outcome.description
                        with lock:
                            failures["consecutive"] = 0
                    else:
                        photo.status, photo.note = "sem_descricao", outcome.error
                        if outcome.error.startswith("erro da IA"):
                            with lock:
                                failures["consecutive"] += 1
                    results.put(("photo", photo, outcome, elapsed))

        # Threads daemon: um Ctrl+C sai na hora (a foto em andamento é abandonada; as prontas já estão no cache).
        threads = [threading.Thread(target=work, daemon=True) for _ in range(min(self.vision_parallel, jobs.qsize()))]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        done, described_sec = 0, 0.0
        try:
            while done < todo:
                try:
                    item = results.get(timeout=0.2)
                except queue.Empty:
                    if not any(t.is_alive() for t in threads) and results.empty():
                        break
                    continue
                if item[0] == "fatal":
                    raise item[1]
                _, photo, outcome, elapsed = item
                done += 1
                if outcome is not None:
                    run.api_calls += outcome.api_calls
                    run.corrections += outcome.corrections
                    if outcome.input_tokens is not None:
                        run.input_tokens = (run.input_tokens or 0) + outcome.input_tokens
                    if outcome.output_tokens is not None:
                        run.output_tokens = (run.output_tokens or 0) + outcome.output_tokens
                    if outcome.model:
                        run.effective_model = outcome.model
                if photo.status == "descrita":
                    run.described += 1
                    described_sec += elapsed
                else:
                    run.failed += 1
                    self._emit("vision", "notice", f"{photo.id} sem descrição: {photo.note}")
                self._progress("vision", StageProgress(
                    done=done, total=todo, unit="photos", note=describer.chain.current.label
                ))
        except BaseException:
            stop.set()
            raise
        run.wall_sec = round(time.perf_counter() - started, 3)
        if run.described:
            run.sec_per_photo = round(described_sec / run.described, 2)
        detail = f"{run.described} foto(s) descrita(s), {run.cached} do cache"
        if run.failed:
            detail += f", {run.failed} sem descrição (pedidas de novo na próxima execução)"
        if run.sec_per_photo is not None:
            detail += f"; {run.sec_per_photo:.1f}s por foto".replace(".", ",")
        return detail

    # --- 7. momentos ---

    def _moments(self, s: _State, roteiro: ProjectRoteiro) -> str:
        by_clip: dict[str, list[RoteiroPhoto]] = {}
        for photo in roteiro.photos:
            by_clip.setdefault(photo.clip_id, []).append(photo)
        for clip in s.clips:
            samples = clip.scan.samples if clip.scan is not None else None
            roteiro.moments += build_moments(clip.id, by_clip.get(clip.id, []), clip.duration, clip.cuts, samples)
        measures = [m.measures for m in roteiro.moments if m.measures is not None]
        roteiro.motion_threshold = mark_shaky(measures) if measures else None
        finalize_issues(roteiro.moments, {p.id: p.description for p in roteiro.photos if p.description is not None})
        link_sentences(roteiro.sentences, roteiro.moments)
        roteiro.clips = [self._roteiro_clip(s, clip, by_clip.get(clip.id, [])) for clip in s.clips]
        return f"{len(roteiro.moments)} momento(s) em {len(s.clips)} clipe(s)"

    def _roteiro_clip(self, s: _State, clip: _Clip, photos: list[RoteiroPhoto]) -> RoteiroClip:
        info = clip.info
        return RoteiroClip(
            id=clip.id, role=info.role, relative_path=info.relative_path, order=clip.order,
            duration_sec=clip.duration, recorded_at=reliable_date(info),
            has_audio=bool(info.media and info.media.has_audio),
            speech_status=info.speech.status if info.speech else None,
            has_speech=any(x.clip_id == clip.id for x in s.sentences),
            transcribed=clip.transcript is not None,
            language=clip.transcript.language if clip.transcript is not None else None,
            silence_total_sec=round(sum(x.duration for x in clip.audio.silences), 3) if clip.audio else None,
            scene_changes=len(clip.cuts),
            photos_before=len(clip.extracted), photos_kept=len(photos),
            discarded=clip.discarded,
            warnings=clip.warnings,
        )

    # --- 8. roteiro ---

    def _write_photos(self, s: _State, roteiro_dir: Path) -> None:
        """Copia as fotos mantidas do cache para `roteiro/fotos/<ID>.jpg` (os IDs mudam se o material mudar)."""
        folder = roteiro_dir / PHOTOS_DIR
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for photo in s.photos:
            shutil.copyfile(s.photo_files[photo.id], roteiro_dir / photo.file)

    @staticmethod
    def _rules(params: RoteiroParams) -> dict[str, float]:
        rules = {
            "interval_sec": params.interval_sec,
            "min_speech_to_transcribe_sec": MIN_SPEECH_TO_TRANSCRIBE_SEC,
            **photo_rules(),
            **rhythm_rules(),
            **moment_rules(),
        }
        return {name: float(value) for name, value in rules.items()}
