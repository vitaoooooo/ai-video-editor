"""Testes da nova tentativa automática em falhas temporárias do Gemini (sem rede, sem dormir)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata, VideoStreamInfo
from ai_video_editor.cli import build_parser
from ai_video_editor.director import DirectorContext, DirectorDecision, UserBrief, run_director
from ai_video_editor.director.providers.gemini import GeminiDirector, generate_structured
from ai_video_editor.director.schema import Sentence
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.providers.gemini import GeminiMultimodalProvider, map_gemini_error
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.intelligence.schema import AIAnalysisConfig

CONTEXT = DirectorContext(video_name="v.mp4", duration_sec=10.0, language="pt")
DECISION_JSON = json.dumps({"removals": [], "summary": "nada a remover"})


class ApiError(Exception):
    """Imita `google.genai.errors.APIError`: `code`, texto e, às vezes, resposta HTTP."""

    def __init__(self, code: int, text: str, headers: dict | None = None):
        super().__init__(f"{code} {text}")
        self.code = code
        if headers is not None:
            self.response = SimpleNamespace(headers=headers)


class FakeModels:
    """Falha com cada exceção da lista e depois responde com sucesso."""

    def __init__(self, failures: list[Exception], text: str = DECISION_JSON):
        self.failures = list(failures)
        self.text = text
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(text=self.text, usage_metadata=None)


def _policy(**kwargs) -> tuple[RetryPolicy, list[float], list[str]]:
    sleeps: list[float] = []
    messages: list[str] = []
    policy = RetryPolicy(
        sleep=sleeps.append,
        rand=lambda: 0.0,
        on_retry=lambda event: messages.append(event.message),
        **kwargs,
    )
    return policy, sleeps, messages


def _sentences() -> list[Sentence]:
    return [Sentence(id=0, start_sec=0.0, end_sec=2.0, text="Olá pessoal.", segment_ids=[0])]


def _director(models: FakeModels, policy: RetryPolicy) -> GeminiDirector:
    return GeminiDirector(client=SimpleNamespace(models=models), thinking="off", retry=policy)


def test_retries_503_twice_then_succeeds():
    policy, sleeps, messages = _policy()
    models = FakeModels([ApiError(503, "UNAVAILABLE"), ApiError(503, "UNAVAILABLE")])
    director = _director(models, policy)

    decision = director.decide(_sentences(), UserBrief(), CONTEXT)

    assert decision.summary == "nada a remover"
    assert models.calls == 3
    assert director.last_api_attempts == 3
    assert sleeps == [2.0, 4.0]
    assert messages == [
        "Gemini indisponível (503), nova tentativa em 2s (2/5)",
        "Gemini indisponível (503), nova tentativa em 4s (3/5)",
    ]


def test_backoff_grows_exponentially_and_caps():
    policy, sleeps, _ = _policy(max_retries=4)
    models = FakeModels([ApiError(503, "UNAVAILABLE")] * 4)
    generate_structured(
        SimpleNamespace(models=models), "gemini-3.8-flash", "off", "p", DirectorDecision, 0.2, 30, policy
    )
    assert sleeps == [2.0, 4.0, 8.0, 16.0]


def test_jitter_adds_to_the_wait():
    policy = RetryPolicy(rand=lambda: 1.0, jitter=0.25)
    assert policy.delay_for(0) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "error",
    [
        ApiError(400, "INVALID_ARGUMENT: bad request"),
        ApiError(404, "NOT_FOUND: models/xyz is not found"),
        ApiError(403, "PERMISSION_DENIED: API key not valid"),
        ApiError(401, "UNAUTHENTICATED"),
    ],
)
def test_permanent_errors_are_never_retried(error):
    policy, sleeps, messages = _policy()
    models = FakeModels([error])
    director = _director(models, policy)

    with pytest.raises(AIProviderError) as exc_info:
        director.decide(_sentences(), UserBrief(), CONTEXT)

    assert models.calls == 1
    assert sleeps == [] and messages == []
    assert not exc_info.value.transient
    assert director.last_api_attempts == 1


def test_auth_error_type_is_preserved():
    policy, _, _ = _policy()
    director = _director(FakeModels([ApiError(403, "PERMISSION_DENIED")]), policy)
    with pytest.raises(AIAuthenticationError):
        director.decide(_sentences(), UserBrief(), CONTEXT)


def test_exhausted_retries_raise_with_original_detail():
    policy, sleeps, _ = _policy(max_retries=2)
    models = FakeModels([ApiError(503, "UNAVAILABLE: The model is overloaded. Try again later.")] * 5)
    director = _director(models, policy)

    with pytest.raises(AIUnavailableError) as exc_info:
        director.decide(_sentences(), UserBrief(), CONTEXT)

    message = str(exc_info.value)
    assert models.calls == 3
    assert len(sleeps) == 2
    assert "temporariamente indisponível" in message
    assert "The model is overloaded. Try again later." in message
    assert "3 tentativas" in message
    assert exc_info.value.api_attempts == 3
    assert director.last_api_attempts == 3


def test_504_deadline_exceeded_is_retried_and_detail_kept():
    policy, sleeps, messages = _policy(max_retries=1)
    models = FakeModels([ApiError(504, "DEADLINE_EXCEEDED: Deadline expired before operation could complete")] * 2)
    with pytest.raises(AIUnavailableError) as exc_info:
        _director(models, policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert models.calls == 2
    assert "DEADLINE_EXCEEDED" in str(exc_info.value)
    assert messages == ["Gemini demorou demais para responder (504), nova tentativa em 2s (2/2)"]


def test_retry_after_header_is_respected():
    policy, sleeps, messages = _policy()
    models = FakeModels([ApiError(429, "RESOURCE_EXHAUSTED: rate limit", headers={"retry-after": "30"})])
    _director(models, policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert sleeps == [30.0]
    assert messages == ["Gemini com limite de requisições (429), nova tentativa em 30s (2/5)"]


def test_retry_delay_in_error_body_is_respected():
    policy, sleeps, _ = _policy()
    error = ApiError(429, "RESOURCE_EXHAUSTED: quota per minute. Please retry in 12.5s.")
    _director(FakeModels([error]), policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert sleeps == [12.5]


def test_short_retry_after_does_not_shorten_the_backoff():
    policy, sleeps, _ = _policy()
    error = ApiError(429, "RESOURCE_EXHAUSTED", headers={"Retry-After": "1"})
    _director(FakeModels([error]), policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert sleeps == [2.0]


def test_daily_quota_is_not_retried():
    policy, sleeps, _ = _policy()
    error = ApiError(429, "RESOURCE_EXHAUSTED: Quota exceeded for GenerateRequestsPerDayPerProjectPerModel")
    models = FakeModels([error])
    with pytest.raises(AIRateLimitError):
        _director(models, policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert models.calls == 1 and sleeps == []


def test_connection_failure_is_retried():
    policy, sleeps, messages = _policy()
    models = FakeModels([ConnectionError("Server disconnected without sending a response.")])
    _director(models, policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert models.calls == 2
    assert messages == ["falha de conexão com o Gemini, nova tentativa em 2s (2/5)"]


def test_zero_retries_turns_retry_off():
    policy, sleeps, _ = _policy(max_retries=0)
    models = FakeModels([ApiError(503, "UNAVAILABLE")])
    with pytest.raises(AIUnavailableError) as exc_info:
        _director(models, policy).decide(_sentences(), UserBrief(), CONTEXT)
    assert models.calls == 1 and sleeps == []
    assert "tentativas" not in str(exc_info.value)


def test_thinking_level_refusal_still_escalates_without_counting_as_retry():
    refusal = ApiError(400, "INVALID_ARGUMENT: Thinking level MINIMAL is not supported by this model")
    policy, sleeps, _ = _policy()
    models = FakeModels([refusal])
    director = _director(models, policy)
    director.decide(_sentences(), UserBrief(), CONTEXT)
    assert models.calls == 2
    assert sleeps == []
    assert "não suportado" in director.effective_thinking_label


def test_error_detail_never_leaks_api_key():
    error = ApiError(503, "UNAVAILABLE for url https://x/v1:generate?key=AIzaSyA1234567890abcdefghijklmnop")
    mapped = map_gemini_error(error, 30)
    assert "AIzaSy" not in str(mapped)
    assert "key=" not in str(mapped).lower()


def test_run_director_records_api_attempts_in_metrics():
    policy, _, _ = _policy()
    models = FakeModels([ApiError(503, "UNAVAILABLE")])
    run = run_director(_director(models, policy), _sentences(), UserBrief(), CONTEXT)
    assert run.decision is not None
    assert [a.api_attempts for a in run.attempts] == [2]


def _frame_fixture(tmp_path: Path):
    metadata = VideoMetadata(
        source_path=tmp_path / "vid.mp4",
        duration_sec=2.0,
        size_bytes=500,
        video_stream=VideoStreamInfo(codec_name="h264", width=640, height=360, fps=30.0),
    )
    frame = ExtractedFrame(
        frame_index=0, target_timestamp_sec=0.0, image_path=tmp_path / "f.jpg",
        width=640, height=360, size_bytes=10,
    )
    return metadata, frame


def test_visual_provider_retries_503_and_reports_attempts(tmp_path: Path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(Image, "open", lambda path: object())
    payload = json.dumps({
        "provider_used": "gemini", "model_used": "gemini-3.8-flash",
        "summary": "ok", "chronological_timeline": [], "scene_transitions": [],
        "limitations_and_blindspots": [],
    })
    models = FakeModels([ApiError(503, "UNAVAILABLE"), ApiError(504, "DEADLINE_EXCEEDED")], text=payload)
    policy, sleeps, messages = _policy()
    provider = GeminiMultimodalProvider(api_key="x", retry=policy)
    monkeypatch.setattr(provider, "_get_client", lambda: SimpleNamespace(models=models))
    metadata, frame = _frame_fixture(tmp_path)

    provider.analyze_frames(metadata, [frame], AIAnalysisConfig())

    assert models.calls == 3
    assert provider.last_api_attempts == 3
    assert sleeps == [2.0, 4.0]
    assert len(messages) == 2


def test_visual_provider_does_not_retry_400(tmp_path: Path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(Image, "open", lambda path: object())
    models = FakeModels([ApiError(400, "INVALID_ARGUMENT")])
    policy, sleeps, _ = _policy()
    provider = GeminiMultimodalProvider(api_key="x", retry=policy)
    monkeypatch.setattr(provider, "_get_client", lambda: SimpleNamespace(models=models))
    metadata, frame = _frame_fixture(tmp_path)

    with pytest.raises(AIProviderError) as exc_info:
        provider.analyze_frames(metadata, [frame], AIAnalysisConfig())

    assert models.calls == 1 and sleeps == []
    assert "INVALID_ARGUMENT" in str(exc_info.value)


@pytest.mark.parametrize("command", [["edit", "v.mp4"], ["revise", "proj", "pedido"], ["report-ai", "--report", "r.json"]])
def test_cli_api_retries_flag(command):
    parser = build_parser()
    assert parser.parse_args(command).api_retries == 4
    assert parser.parse_args(command + ["--api-retries", "0"]).api_retries == 0
    assert parser.parse_args(command + ["--api-retries", "7"]).api_retries == 7
    with pytest.raises(SystemExit):
        parser.parse_args(command + ["--api-retries", "-1"])
