"""Testes do provedor OpenAI com cliente falso (sem rede)."""

from types import SimpleNamespace

import pytest

from ai_video_editor.director import DirectorDecision
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm import LLMCallConfig
from ai_video_editor.llm.providers.openai import OpenAILLMClient, map_openai_error, supports_reasoning

PAYLOAD = {"removals": [{"sentence_id": 2, "category": "filler", "reason": "Enrolação."}], "summary": "Uma removida."}


class FakeResponses:
    """`responses.parse`: falha com cada exceção da lista e depois responde."""

    def __init__(self, failures=(), response=None):
        self.failures = list(failures)
        self.response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        if self.response is not None:
            return self.response
        return SimpleNamespace(
            output_parsed=kwargs["text_format"].model_validate(PAYLOAD),
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text")])],
            status="completed",
            usage=SimpleNamespace(input_tokens=321, output_tokens=45),
        )


def _sdk_error(cls_name: str, status: int | None = None, message: str = "erro", code=None, type_=None):
    """Exceção com o mesmo nome e atributos das classes do SDK `openai`."""
    cls = type(cls_name, (Exception,), {})
    error = cls(message)
    error.status_code = status
    error.code = code
    error.type = type_
    error.response = SimpleNamespace(headers={})
    return error


def _client(responses: FakeResponses) -> OpenAILLMClient:
    return OpenAILLMClient(client=SimpleNamespace(responses=responses))


def _policy(max_retries=4):
    sleeps: list[float] = []
    return RetryPolicy(max_retries=max_retries, sleep=sleeps.append, rand=lambda: 0.0), sleeps


def test_parses_structured_output_and_usage():
    responses = FakeResponses()
    config = LLMCallConfig(model_id="gpt-6-luna", thinking="low", timeout_sec=30)

    result = _client(responses).generate("prompt", DirectorDecision, config, _policy()[0])

    assert result.value.removals[0].sentence_id == 2
    assert (result.metrics.input_tokens, result.metrics.output_tokens) == (321, 45)
    assert result.metrics.provider == "openai" and result.metrics.api_calls == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-6-luna"
    assert call["input"] == "prompt"
    assert call["text_format"] is DirectorDecision
    assert call["reasoning"] == {"effort": "low"}
    assert call["timeout"] == 30
    assert "temperature" not in call  # modelo de raciocínio com esforço > none


def test_thinking_off_maps_to_effort_none_with_temperature():
    responses = FakeResponses()
    client = _client(responses)
    config = LLMCallConfig(model_id="gpt-6-luna", thinking="off", temperature=0.2)
    client.generate("p", DirectorDecision, config, _policy()[0])
    assert responses.calls[0]["reasoning"] == {"effort": "none"}
    assert responses.calls[0]["temperature"] == 0.2
    assert client.thinking_label("gpt-6-luna", "off") == "off (reasoning.effort=none)"


def test_model_without_reasoning_ignores_thinking_and_says_so():
    responses = FakeResponses()
    client = _client(responses)
    result = client.generate("p", DirectorDecision, LLMCallConfig(model_id="gpt-4.1-mini", thinking="high"), _policy()[0])
    assert "reasoning" not in responses.calls[0]
    assert responses.calls[0]["temperature"] == 0.2
    assert "ignorado" in result.metrics.thinking
    assert supports_reasoning("o4-mini") and supports_reasoning("gpt-5-mini") and not supports_reasoning("gpt-4o")


def test_refused_effort_escalates_like_gemini():
    refusal = _sdk_error(
        "BadRequestError", 400,
        "Error code: 400 - Unsupported value: 'reasoning.effort' does not support 'none' with this model.",
    )
    responses = FakeResponses([refusal])
    policy, sleeps = _policy()
    result = _client(responses).generate("p", DirectorDecision, LLMCallConfig(model_id="gpt-6.1-sol", thinking="off"), policy)
    assert [c["reasoning"]["effort"] for c in responses.calls] == ["none", "low"]
    assert result.metrics.thinking == "off → low (none não suportado)"
    assert sleeps == []


def test_refusal_item_is_a_validation_error():
    refused = SimpleNamespace(
        output_parsed=None, status="completed", usage=None,
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="não posso")])],
    )
    client = _client(FakeResponses(response=refused))
    with pytest.raises(AIResponseValidationError, match="openai"):
        client.generate("p", DirectorDecision, LLMCallConfig(model_id="gpt-6-luna"), _policy()[0])


def test_incomplete_response_is_a_validation_error():
    incomplete = SimpleNamespace(
        output_parsed=None, status="incomplete", output=[], usage=SimpleNamespace(input_tokens=5, output_tokens=9),
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    client = _client(FakeResponses(response=incomplete))
    with pytest.raises(AIResponseValidationError) as exc_info:
        client.generate("p", DirectorDecision, LLMCallConfig(model_id="gpt-6-luna"), _policy()[0])
    assert "max_output_tokens" in exc_info.value.details
    assert client.last_metrics.output_tokens == 9  # tokens registrados mesmo com resposta inválida


@pytest.mark.parametrize(
    "error, expected, transient, quota",
    [
        (_sdk_error("RateLimitError", 429, "Rate limit reached for requests"), AIRateLimitError, True, False),
        (_sdk_error("RateLimitError", 429, "You exceeded your quota", code="insufficient_quota"), AIRateLimitError, False, True),
        (_sdk_error("RateLimitError", 429, "no credits", code="credit_balance_exhausted"), AIRateLimitError, False, True),
        (_sdk_error("InternalServerError", 503, "Model temporarily overloaded"), AIUnavailableError, True, False),
        (_sdk_error("InternalServerError", 500, "Server error"), AIUnavailableError, True, False),
        (_sdk_error("APITimeoutError", None, "Request timed out."), AITimeoutError, True, False),
        (_sdk_error("APIConnectionError", None, "Connection error."), AIUnavailableError, True, False),
        (_sdk_error("AuthenticationError", 401, "Incorrect API key provided"), AIAuthenticationError, False, False),
        (_sdk_error("NotFoundError", 404, "The model `gpt-x` does not exist"), AIProviderError, False, False),
        (_sdk_error("BadRequestError", 400, "Invalid schema"), AIProviderError, False, False),
    ],
)
def test_error_mapping(error, expected, transient, quota):
    mapped = map_openai_error(error, 30)
    assert type(mapped) is expected
    assert mapped.transient is transient
    assert mapped.quota_exhausted is quota
    assert mapped.provider == "openai"


def test_transient_errors_are_retried_and_permanent_are_not():
    policy, sleeps = _policy()
    responses = FakeResponses([_sdk_error("InternalServerError", 503, "overloaded")] * 2)
    result = _client(responses).generate("p", DirectorDecision, LLMCallConfig(model_id="gpt-6-luna"), policy)
    assert result.metrics.api_calls == 3 and sleeps == [2.0, 4.0]

    policy, sleeps = _policy()
    responses = FakeResponses([_sdk_error("AuthenticationError", 401, "bad key")])
    with pytest.raises(AIAuthenticationError):
        _client(responses).generate("p", DirectorDecision, LLMCallConfig(model_id="gpt-6-luna"), policy)
    assert len(responses.calls) == 1 and sleeps == []


def test_error_detail_never_leaks_api_key():
    error = _sdk_error("AuthenticationError", 401, "Incorrect API key provided: sk-proj-abcdefghijklmnopqrstuvwxyz123456")
    assert "sk-proj" not in str(map_openai_error(error, 30))


def test_real_sdk_exceptions_are_mapped():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    def status_error(cls, status, body):
        return cls("erro", response=httpx.Response(status, request=request), body=body)

    quota = map_openai_error(status_error(openai.RateLimitError, 429, {"code": "insufficient_quota", "type": "insufficient_quota"}), 30)
    assert isinstance(quota, AIRateLimitError) and quota.quota_exhausted and not quota.transient
    rate = map_openai_error(status_error(openai.RateLimitError, 429, {"message": "Rate limit reached"}), 30)
    assert rate.transient and rate.status_code == 429
    assert isinstance(map_openai_error(status_error(openai.AuthenticationError, 401, None), 30), AIAuthenticationError)
    assert map_openai_error(status_error(openai.InternalServerError, 503, None), 30).transient
    assert isinstance(map_openai_error(openai.APITimeoutError(request=request), 30), AITimeoutError)
    assert map_openai_error(openai.APIConnectionError(request=request), 30).transient
