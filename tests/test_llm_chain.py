"""Testes da cadeia de modelos com troca automática (clientes falsos, sem rede e sem dormir)."""

import json
from types import SimpleNamespace

import pytest

from ai_video_editor.cli import _build_reviser, build_parser
from ai_video_editor.director import DirectorContext, DirectorDecision, Sentence, UserBrief, run_director
from ai_video_editor.director.providers.llm import LLMDirector
from ai_video_editor.intelligence.exceptions import AIAuthenticationError, AIProviderError, AIUnavailableError
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm import ModelChain, ModelSpec, register_provider, resolve_chain
from ai_video_editor.llm.providers.gemini import GeminiLLMClient
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.providers.openai import OpenAILLMClient
from ai_video_editor.llm.registry import create_client, provider_names
from ai_video_editor.pipeline import EditPipelineParams
from ai_video_editor.pipeline.report import _director_section, build_revision_section
from ai_video_editor.revision import RevisionContext, SentenceState, run_reviser
from ai_video_editor.revision.providers.llm import LLMReviser

CONTEXT = DirectorContext(video_name="v.mp4", duration_sec=10.0, language="pt")
DECISION = {"removals": [], "summary": "nada a remover"}


class ApiError(Exception):
    """Imita `google.genai.errors.APIError`."""

    def __init__(self, code: int, text: str):
        super().__init__(f"{code} {text}")
        self.code = code


class FakeGeminiModels:
    """`generate_content` por modelo: uma lista de falhas por modelo; depois responde."""

    def __init__(self, failures: dict[str, list[Exception]], always_fail: dict[str, Exception] | None = None):
        self.failures = {k: list(v) for k, v in failures.items()}
        self.always_fail = always_fail or {}
        self.calls: list[str] = []

    def generate_content(self, **kwargs):
        model = kwargs["model"]
        self.calls.append(model)
        if model in self.always_fail:
            raise self.always_fail[model]
        if self.failures.get(model):
            raise self.failures[model].pop(0)
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20)
        return SimpleNamespace(text=json.dumps(DECISION), usage_metadata=usage)


class FakeOpenAIResponses:
    def __init__(self):
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=kwargs["text_format"].model_validate(DECISION), output=[], status="completed",
            usage=SimpleNamespace(input_tokens=50, output_tokens=7),
        )


def _policy(max_retries: int = 4) -> RetryPolicy:
    return RetryPolicy(max_retries=max_retries, sleep=lambda _: None, rand=lambda: 0.0)


def _chain(specs: str, gemini_models=None, openai=None, events=None, **clients) -> ModelChain:
    provided = dict(clients)
    if gemini_models is not None:
        provided["gemini"] = GeminiLLMClient(client=SimpleNamespace(models=gemini_models))
    if openai is not None:
        provided["openai"] = OpenAILLMClient(client=SimpleNamespace(responses=openai))
    first, _, rest = specs.partition(",")
    provider, model = first.split(":")
    return ModelChain(
        resolve_chain(provider, model, rest or "none"),
        thinking="low", retry=_policy(), on_event=(events.append if events is not None else None),
        clients=provided,
    )


def _sentences() -> list[Sentence]:
    return [Sentence(id=0, start_sec=0.0, end_sec=2.0, text="Olá pessoal.", segment_ids=[0])]


def test_persistent_503_switches_to_next_model_with_warning():
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": ApiError(503, "UNAVAILABLE: high demand")})
    events: list[str] = []
    chain = _chain("gemini:gemini-3.8-flash,gemini:gemini-3.6-flash", gemini_models=models, events=events)

    result = chain.generate("p", DirectorDecision)

    assert result.value.summary == "nada a remover"
    assert models.calls == ["gemini-3.8-flash"] * 5 + ["gemini-3.6-flash"]
    assert events == ["gemini-3.8-flash indisponível (503) após 5 tentativas → trocando para gemini-3.6-flash"]
    assert [s.model_dump() for s in chain.switches] == [{
        "from_model": "gemini:gemini-3.8-flash", "to_model": "gemini:gemini-3.6-flash",
        "reason": "indisponível (503) após 5 tentativas", "status_code": 503, "api_attempts": 5,
    }]
    assert chain.last_metrics.model_id == "gemini-3.6-flash"
    assert chain.last_metrics.api_calls == 6
    assert (chain.last_metrics.input_tokens, chain.last_metrics.output_tokens) == (100, 20)


def test_switch_is_sticky_for_the_next_call():
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": ApiError(503, "UNAVAILABLE")})
    chain = _chain("gemini:gemini-3.8-flash,gemini:gemini-3.6-flash", gemini_models=models)
    chain.generate("p", DirectorDecision)
    models.calls.clear()

    chain.generate("p", DirectorDecision)

    assert models.calls == ["gemini-3.6-flash"]
    assert len(chain.switches) == 1


@pytest.mark.parametrize(
    "error, expected",
    [
        (ApiError(403, "PERMISSION_DENIED: API key not valid"), AIAuthenticationError),
        (ApiError(400, "INVALID_ARGUMENT: bad request"), AIProviderError),
        (ApiError(404, "NOT_FOUND: model"), AIProviderError),
    ],
)
def test_auth_and_invalid_request_fail_without_switching(error, expected):
    models = FakeGeminiModels({"gemini-3.8-flash": [error]})
    events: list[str] = []
    chain = _chain("gemini:gemini-3.8-flash,gemini:gemini-3.6-flash", gemini_models=models, events=events)

    with pytest.raises(expected):
        chain.generate("p", DirectorDecision)

    assert models.calls == ["gemini-3.8-flash"]
    assert events == [] and chain.switches == []


def test_daily_quota_switches_without_retrying():
    quota = ApiError(429, "RESOURCE_EXHAUSTED: Quota exceeded for GenerateRequestsPerDayPerProjectPerModel")
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": quota})
    events: list[str] = []
    chain = _chain("gemini:gemini-3.8-flash,gemini:gemini-3.6-flash", gemini_models=models, events=events)

    chain.generate("p", DirectorDecision)

    assert models.calls == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert events == ["gemini-3.8-flash com a cota esgotada (429) → trocando para gemini-3.6-flash"]


def test_last_model_failure_is_raised():
    error = ApiError(503, "UNAVAILABLE")
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": error, "gemini-3.6-flash": error})
    chain = _chain("gemini:gemini-3.8-flash,gemini:gemini-3.6-flash", gemini_models=models)

    with pytest.raises(AIUnavailableError, match="5 tentativas"):
        chain.generate("p", DirectorDecision)

    assert len(models.calls) == 10
    assert chain.last_metrics.api_calls == 10


def test_switches_between_providers_gemini_to_openai():
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": ApiError(503, "UNAVAILABLE")})
    responses = FakeOpenAIResponses()
    events: list[str] = []
    chain = _chain("gemini:gemini-3.8-flash,openai:gpt-6-luna", gemini_models=models, openai=responses, events=events)

    result = chain.generate("p", DirectorDecision)

    assert result.value.summary == "nada a remover"
    assert events == ["gemini-3.8-flash indisponível (503) após 5 tentativas → trocando para openai:gpt-6-luna"]
    assert responses.calls[0]["model"] == "gpt-6-luna"
    assert result.metrics.provider == "openai" and result.metrics.model_id == "gpt-6-luna"
    assert result.metrics.thinking == "low (reasoning.effort=low)"


def test_provider_without_key_is_skipped_with_one_warning(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # sem .env
    models = FakeGeminiModels({})
    events: list[str] = []
    chain = ModelChain(
        [ModelSpec(provider="openai", model_id="gpt-6-luna"), ModelSpec(provider="openai", model_id="gpt-6.1-sol"),
         ModelSpec(provider="gemini", model_id="gemini-3.8-flash")],
        retry=_policy(), on_event=events.append,
        clients={"gemini": GeminiLLMClient(client=SimpleNamespace(models=models))},
    )

    chain.generate("p", DirectorDecision)
    chain.generate("p", DirectorDecision)

    assert events == ["openai sem chave configurada (OPENAI_API_KEY) → pulando openai:gpt-6-luna"]
    assert models.calls == ["gemini-3.8-flash", "gemini-3.8-flash"]
    assert chain.switches == []
    assert chain.last_metrics.provider == "gemini"


def test_no_key_anywhere_raises_authentication_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    chain = ModelChain([ModelSpec(provider="gemini", model_id="gemini-3.8-flash")], retry=_policy())
    with pytest.raises(AIAuthenticationError) as exc_info:
        chain.generate("p", DirectorDecision)
    assert "GEMINI_API_KEY" in exc_info.value.details


def test_key_in_dotenv_counts_as_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('OPENAI_API_KEY="sk-test"\n', encoding="utf-8")
    assert OpenAILLMClient().has_credentials()


def test_director_records_requested_and_effective_model_and_report_shows_it():
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": ApiError(503, "UNAVAILABLE")})
    director = LLMDirector(_chain("gemini:gemini-3.8-flash,gemini:gemini-3.6-flash", gemini_models=models))

    run = run_director(director, _sentences(), UserBrief(), CONTEXT)

    # pedido (fingerprint) continua sendo o primeiro elo
    assert (run.provider, run.model_id) == ("gemini", "gemini-3.8-flash")
    assert director.model_chain == ["gemini:gemini-3.8-flash", "gemini:gemini-3.6-flash"]
    assert run.model_chain == director.model_chain
    assert (run.effective_provider, run.effective_model_id) == ("gemini", "gemini-3.6-flash")
    assert run.attempts[0].model_id == "gemini-3.6-flash"
    assert run.attempts[0].api_attempts == 6
    assert len(run.model_switches) == 1

    reloaded = type(run).model_validate_json(run.model_dump_json())
    report = "\n".join(_director_section(reloaded))
    assert "| Diretor | gemini (gemini-3.8-flash) |" in report
    assert "| Cadeia de modelos | gemini:gemini-3.8-flash → gemini:gemini-3.6-flash |" in report
    assert "| Modelo usado | gemini:gemini-3.6-flash |" in report
    assert (
        "| Troca de modelo | gemini:gemini-3.8-flash → gemini:gemini-3.6-flash: "
        "indisponível (503) após 5 tentativas |" in report
    )


def test_reviser_uses_the_chain_and_report_shows_the_switch():
    models = FakeGeminiModels({}, always_fail={"gemini-3.8-flash": ApiError(503, "UNAVAILABLE")})
    reviser = LLMReviser(_chain("gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite", gemini_models=models))
    states = [SentenceState(sentence=s) for s in _sentences()]
    context = RevisionContext(video_name="a.mp4", duration_sec=10, base_version=1, request="x")

    run = run_reviser(reviser, states, UserBrief(), context)

    assert run.effective_model_id == "gemini-3.5-flash-lite" and run.model_id == "gemini-3.8-flash"
    section = build_revision_section("x", 1, 2, "ok", "normal", "normal", [], reviser=run)
    assert "| Revisor | gemini (gemini-3.8-flash) |" in section
    assert "| Modelo usado | gemini:gemini-3.5-flash-lite |" in section
    assert "gemini:gemini-3.8-flash → gemini:gemini-3.5-flash-lite" in section


# --- resolução da cadeia, registro e CLI ---


def test_default_chain_and_fallback_options():
    assert [s.label for s in EditPipelineParams(director="gemini").director_chain()] == [
        "gemini:gemini-3.8-flash", "gemini:gemini-3.6-flash", "gemini:gemini-3.5-flash-lite",
    ]
    assert [s.label for s in resolve_chain("gemini", "gemini-3.6-flash")] == [
        "gemini:gemini-3.6-flash", "gemini:gemini-3.5-flash-lite",
    ]
    assert [s.label for s in resolve_chain("gemini", None, "none")] == ["gemini:gemini-3.8-flash"]
    assert [s.label for s in resolve_chain("gemini", None, "openai:gpt-6-luna, gemini-3.6-flash")] == [
        "gemini:gemini-3.8-flash", "openai:gpt-6-luna", "gemini:gemini-3.6-flash",
    ]
    assert [s.label for s in resolve_chain("openai")] == ["openai:gpt-6-luna"]
    with pytest.raises(ValueError):
        EditPipelineParams(director="gemini", director_fallback="nao-existe:x")


def test_registry_accepts_new_providers():
    register_provider("fake-test", MockLLMClient)
    try:
        assert "fake-test" in provider_names()
        assert isinstance(create_client("fake-test"), MockLLMClient)
        assert [s.label for s in resolve_chain("gemini", None, "fake-test:m1")][-1] == "fake-test:m1"
    finally:
        from ai_video_editor.llm import registry

        registry._FACTORIES.pop("fake-test", None)


def test_cli_director_and_reviser_chain_flags():
    parser = build_parser()
    args = parser.parse_args(["edit", "v.mp4", "--director-provider", "openai", "--director-fallback", "none"])
    assert (args.director, args.director_model, args.director_fallback) == ("openai", None, "none")
    assert parser.parse_args(["edit", "v.mp4", "--director", "gemini"]).director == "gemini"
    args = parser.parse_args(["revise", "p", "x", "--reviser-provider", "openai", "--reviser-fallback", "gemini:gemini-3.6-flash"])
    assert (args.reviser, args.reviser_fallback) == ("openai", "gemini:gemini-3.6-flash")
    assert parser.parse_args(["revise", "p", "x", "--reviser", "mock"]).reviser == "mock"


def test_cli_reviser_inherits_the_project_director_chain():
    parser = build_parser()
    params = EditPipelineParams(director="gemini", director_model="gemini-3.6-flash", director_fallback="openai:gpt-6-luna")

    reviser = _build_reviser(parser.parse_args(["revise", "p", "x"]), params)
    assert reviser.model_chain == ["gemini:gemini-3.6-flash", "openai:gpt-6-luna"]

    reviser = _build_reviser(parser.parse_args(["revise", "p", "x", "--reviser-provider", "openai"]), params)
    assert reviser.model_chain == ["openai:gpt-6-luna"]

    reviser = _build_reviser(parser.parse_args(["revise", "p", "x", "--reviser-fallback", "none"]), params)
    assert reviser.model_chain == ["gemini:gemini-3.6-flash"]


def test_switch_notice_is_printed_on_the_director_step(capsys):
    from ai_video_editor.cli import _print_progress

    _print_progress("director", "notice", "gemini-3.8-flash indisponível (503) após 5 tentativas → trocando para gemini-3.6-flash")
    assert capsys.readouterr().out.strip() == (
        "[4/8] Diretor: gemini-3.8-flash indisponível (503) após 5 tentativas → trocando para gemini-3.6-flash"
    )
