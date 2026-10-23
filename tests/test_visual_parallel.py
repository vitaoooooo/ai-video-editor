"""Lotes da IA em paralelo no `visual` (sem rede e sem FFmpeg: tomadas montadas e IA simulada com atraso)."""

import threading
import time
from pathlib import Path

import pytest

from ai_video_editor import cli
from ai_video_editor.intelligence.exceptions import AIProviderError, AIRateLimitError
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm import ModelChain, ModelSpec
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.visual import VisualBuilder, VisualDescriber, mock_visual_responder
from ai_video_editor.visual import builder as builder_module
from ai_video_editor.visual.builder import (
    VISUAL_CACHE_VERSION,
    CachedAnalyses,
    CachedShot,
    ClipShotsCache,
    ai_fingerprint,
)
from ai_video_editor.visual.describer import prompt_sections
from ai_video_editor.visual.report import batch_timing_line, build_visual_markdown
from ai_video_editor.visual.schema import (
    ClipVisual,
    ProjectVisual,
    Shot,
    ShotFrame,
    ShotMeasures,
    VisualBatchResponse,
)

DELAY_SEC = 0.2


def _shot(shot_id: str, index: int) -> Shot:
    return Shot(
        id=shot_id, index=index, start_sec=0.0, end_sec=4.0, boundary="inicio",
        frames=[ShotFrame(id=f"{shot_id}.F{n}", position=n, time_sec=n, file=f"frames/{shot_id}.F{n}.jpg")
                for n in (1, 2, 3)],
        measures=ShotMeasures(samples=4, brightness=0.5, motion=0.01, luma_spread=0.6), has_speech=False,
    )


def _project(root: Path, clips: dict[str, int]):
    """Clipes montados à mão ({ID: nº de tomadas}), com miniaturas falsas no cache."""
    cache_root = root / "cache"
    visuals, caches = [], {}
    for clip_id, count in clips.items():
        key = f"visual-{clip_id}"
        shots, items = [], []
        for index in range(1, count + 1):
            files = [f"T{index:02d}_F{n}.jpg" for n in (1, 2, 3)]
            for name in files:
                (cache_root / key).mkdir(parents=True, exist_ok=True)
                (cache_root / key / name).write_bytes(b"\xff\xd8jpeg")
            items.append(CachedShot(index=index, start_sec=0.0, end_sec=4.0, boundary="inicio",
                                    frame_times=[0.4, 2.0, 3.6], frame_files=files))
            shots.append(_shot(f"{clip_id}.T{index:02d}", index))
        caches[clip_id] = (key, ClipShotsCache(
            format_version=VISUAL_CACHE_VERSION, path=root / f"{clip_id}.mp4", size_bytes=1, mtime_ns=1,
            params={}, duration_sec=4.0 * count, shots=items,
        ))
        visuals.append(ClipVisual(clip_id=clip_id, role="broll", relative_path=f"{clip_id}.mp4",
                                  duration_sec=4.0 * count, shots=shots))
    return visuals, caches, cache_root


def _delayed(responder=mock_visual_responder, delay: float = DELAY_SEC):
    def respond(prompt, images, schema):
        time.sleep(delay)
        return responder(prompt, images, schema)
    return respond


def _builder(tmp_path: Path, responder, parallel: int, events: list | None = None) -> VisualBuilder:
    chain = ModelChain(
        [ModelSpec(provider="mock", model_id="mock")],
        retry=RetryPolicy(max_retries=0, sleep=lambda _: None),
        clients={"mock": MockLLMClient(responses=[responder])},
    )
    on_progress = (lambda stage, event, detail: events.append((event, detail))) if events is not None else None
    return VisualBuilder(projects_root=tmp_path, describer=VisualDescriber(chain), ai_parallel=parallel,
                         on_progress=on_progress)


def _run(tmp_path: Path, responder, parallel: int, clips: dict[str, int], events: list | None = None):
    builder = _builder(tmp_path, responder, parallel, events)
    visuals, caches, cache_root = _project(tmp_path, clips)
    fingerprint = ai_fingerprint(builder.describer)
    started = time.perf_counter()
    run = builder._describe(visuals, [], caches, cache_root, fingerprint)
    return run, visuals, caches, cache_root, fingerprint, time.perf_counter() - started


def _cached(cache_root: Path, key: str, fingerprint: str) -> CachedAnalyses:
    return CachedAnalyses.model_validate_json((cache_root / key / f"ai-{fingerprint}.json").read_text("utf-8"))


def test_parallel_is_faster_and_identical_to_sequential(tmp_path):
    clips = {"B001": 24, "B002": 24}  # 48 tomadas = 6 lotes de 8
    seq = _run(tmp_path / "seq", _delayed(), 1, clips)
    par = _run(tmp_path / "par", _delayed(), 3, clips)

    seq_run, seq_visuals, _, seq_root, fingerprint, seq_sec = seq
    par_run, par_visuals, _, par_root, _, par_sec = par
    assert seq_sec >= 6 * DELAY_SEC
    assert par_sec < seq_sec * 0.6
    assert [v.model_dump() for v in par_visuals] == [v.model_dump() for v in seq_visuals]
    for clip_id in clips:
        key = f"visual-{clip_id}"
        assert _cached(par_root, key, fingerprint) == _cached(seq_root, key, fingerprint)
    # Métricas por lote, na ordem dos lotes e sem misturar as chamadas das threads.
    assert [s.number for s in par_run.batch_stats] == [1, 2, 3, 4, 5, 6]
    assert par_run.batches == 6 and par_run.api_calls == 6 and par_run.images_sent == 144
    assert all(s.api_calls == 1 and s.duration_sec >= DELAY_SEC for s in par_run.batch_stats)
    assert par_run.parallel == par_run.parallel_final == 3
    assert par_run.effective_model == seq_run.effective_model == "mock:mock"
    timing = {"batch_stats", "wall_sec", "parallel", "parallel_final"}
    assert par_run.model_dump(exclude=timing) == seq_run.model_dump(exclude=timing)


def test_batch_timing_in_report(tmp_path):
    run, visuals, *_ = _run(tmp_path, _delayed(delay=0.01), 2, {"B001": 16})
    line = batch_timing_line(run)
    assert line.startswith("Lotes: 2 em ") and "(paralelo 2)" in line and "média" in line and "máx" in line
    visual = ProjectVisual(project_dir=tmp_path, source_dir=tmp_path, roles=["broll"], clips=visuals, ai=run)
    assert line in build_visual_markdown(visual)
    loaded = ProjectVisual.model_validate_json(visual.model_dump_json())
    assert loaded.ai.batch_stats[0].shots == 8 and loaded.ai.batch_stats[0].images == 24


def test_cache_is_saved_after_each_batch_and_survives_interrupt(tmp_path):
    seen: list[int] = []
    holder: dict = {}

    def respond(prompt, images, schema):
        path = holder["root"] / "visual-B001" / f"ai-{holder['fingerprint']}.json"
        seen.append(len(CachedAnalyses.model_validate_json(path.read_text("utf-8")).analyses)
                    if path.is_file() else 0)
        if len(seen) == 3:
            raise KeyboardInterrupt  # Ctrl+C no meio do 3º lote
        return mock_visual_responder(prompt, images, schema)

    builder = _builder(tmp_path, respond, parallel=1)
    visuals, caches, cache_root = _project(tmp_path, {"B001": 32})
    holder.update(root=cache_root, fingerprint=ai_fingerprint(builder.describer))

    with pytest.raises(KeyboardInterrupt):
        builder._describe(visuals, [], caches, cache_root, holder["fingerprint"])

    assert seen == [0, 8, 16]
    kept = _cached(cache_root, "visual-B001", holder["fingerprint"])
    assert sorted(kept.analyses) == list(range(1, 17))
    assert not list((cache_root / "visual-B001").glob("*.tmp"))


def test_rate_limit_halves_parallelism_once(tmp_path):
    in_flight = 0
    lock = threading.Lock()
    concurrency: dict[str, int] = {}

    def respond(prompt, images, schema):
        nonlocal in_flight
        first = prompt_sections(prompt)[0][0]
        with lock:
            in_flight += 1
            concurrency[first] = in_flight
        try:
            if first == "B001.T01":
                error = AIRateLimitError("mock", details="429 RESOURCE_EXHAUSTED")
                error.status_code = 429
                raise error
            time.sleep(0.1)
            return mock_visual_responder(prompt, images, schema)
        finally:
            with lock:
                in_flight -= 1

    events: list = []
    run, visuals, *_ = _run(tmp_path, respond, 4, {"B001": 56}, events)  # 7 lotes

    assert run.parallel == 4 and run.parallel_final == 2
    notices = [d for e, d in events if e == "notice" and "limite do provedor" in d]
    assert notices == ["limite do provedor no lote 1: 2 lote(s) ao mesmo tempo daqui em diante"]
    # Depois da redução, nunca mais de 2 lotes ao mesmo tempo (lotes 5 a 7).
    assert max(concurrency[f"B001.T{n:02d}"] for n in (33, 41, 49)) <= 2
    shots = visuals[0].shots
    assert all(s.status == "sem_analise" for s in shots[:8])
    assert all(s.status == "analisada" for s in shots[8:])
    assert run.batch_stats[0].rate_limited and run.batch_stats[0].failed_shots == 8


def test_two_batches_of_the_same_clip_never_write_the_cache_together(tmp_path, monkeypatch):
    original = builder_module.VisualBuilder._save_analyses
    active: dict[Path, int] = {}
    overlap: list[Path] = []
    lock = threading.Lock()

    def slow_save(visual, cache_dir, fingerprint):
        with lock:
            active[cache_dir] = active.get(cache_dir, 0) + 1
            if active[cache_dir] > 1:
                overlap.append(cache_dir)
        time.sleep(0.1)
        original(visual, cache_dir, fingerprint)
        with lock:
            active[cache_dir] -= 1

    monkeypatch.setattr(builder_module.VisualBuilder, "_save_analyses", staticmethod(slow_save))
    run, _, _, cache_root, fingerprint, _ = _run(tmp_path, _delayed(delay=0.05), 3, {"B001": 24})

    assert run.batches == 3 and not overlap
    assert sorted(_cached(cache_root, "visual-B001", fingerprint).analyses) == list(range(1, 25))


def test_api_failures_still_stop_after_two_batches_in_a_row(tmp_path):
    def respond(prompt, images, schema):
        raise AIProviderError("Serviço fora", provider="mock")

    run, visuals, *_ = _run(tmp_path, respond, 1, {"B001": 32})
    notes = [s.note for s in visuals[0].shots]
    assert all("erro da IA" in n for n in notes[:16])
    assert all("não enviada" in n for n in notes[16:])
    assert run.api_calls == 2


class _PerModelMock(MockLLMClient):
    """Responde por modelo: `a` falha com cota esgotada depois que as duas threads chegam."""

    def __init__(self):
        super().__init__()
        self.barrier = threading.Barrier(2)

    def _request(self, prompt, schema, config, level, images=()):
        if config.model_id == "a":
            self.barrier.wait(timeout=5)
            error = AIProviderError("cota", provider="mock")
            error.quota_exhausted = True
            raise error
        return VisualBatchResponse(shots=[])


def test_chain_switches_once_when_parallel_calls_fail_together():
    events: list[str] = []
    chain = ModelChain(
        [ModelSpec(provider="mock", model_id="a"), ModelSpec(provider="mock", model_id="b")],
        retry=RetryPolicy(max_retries=0, sleep=lambda _: None), on_event=events.append,
        clients={"mock": _PerModelMock()},
    )
    results: list = []

    def call():
        chain.generate("p", VisualBatchResponse)
        results.append((chain.last_metrics.model_id, chain.last_metrics.api_calls, chain.last_switches))

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(chain.switches) == 1 and chain.current.model_id == "b"
    assert len([e for e in events if "trocando" in e]) == 1
    assert sorted(results) == [("b", 2, 1), ("b", 2, 1)]


class _LazySdk:
    """Cliente do SDK falso: fecha quando um cliente mais novo o substitui (como o `__del__` do SDK)."""

    def __init__(self):
        self.closed = False


class _LazyClientMock(MockLLMClient):
    """Provedor sem cliente injetado: o cliente do SDK é criado na 1ª chamada, devagar."""

    def __init__(self, responder):
        super().__init__(responses=[responder])
        self._client = None
        self.created: list[_LazySdk] = []

    def _create_client(self):
        time.sleep(0.05)
        for old in self.created:
            old.closed = True
        sdk = _LazySdk()
        self.created.append(sdk)
        return sdk

    def _request(self, prompt, schema, config, level, images=()):
        sdk = self.get_client()
        time.sleep(0.01)
        if sdk.closed:
            raise AIProviderError("Cannot send a request, as the client has been closed.", provider="mock")
        return super()._request(prompt, schema, config, level, images)


def test_parallel_batches_with_lazy_sdk_client(tmp_path):
    """Regressão: com o cliente criado sob demanda, as threads não podem criar um cliente cada."""
    provider = _LazyClientMock(mock_visual_responder)
    chain = ModelChain([ModelSpec(provider="mock", model_id="mock")],
                       retry=RetryPolicy(max_retries=0, sleep=lambda _: None), clients={"mock": provider})
    builder = VisualBuilder(projects_root=tmp_path, describer=VisualDescriber(chain), ai_parallel=3)
    visuals, caches, cache_root = _project(tmp_path, {"B001": 24, "B002": 24})
    run = builder._describe(visuals, [], caches, cache_root, ai_fingerprint(builder.describer))

    assert len(provider.created) == 1
    assert run.batches == 6 and run.api_calls == 6
    assert all(shot.analysis is not None for visual in visuals for shot in visual.shots)


def test_cli_visual_parallel_option():
    args = cli.build_parser().parse_args(["visual", "pasta"])
    assert cli.visual_parallel(args) == 3
    assert cli.build_parser().parse_args(["visual", "pasta", "--visual-paralelo", "1"]).visual_paralelo == 1
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["visual", "pasta", "--visual-paralelo", "0"])
