"""Testes do provedor Ollama com transporte HTTP falso (sem rede e sem Ollama)."""

import base64
import io
import json
import urllib.error

import pytest
from pydantic import BaseModel

from ai_video_editor import cli
from ai_video_editor.director import DirectorDecision
from ai_video_editor.intelligence.exceptions import (
    AIContextLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm import ImageInput, LLMCallConfig, ModelChain, ModelSpec
from ai_video_editor.llm.chain import resolve_chain
from ai_video_editor.llm.ollama_check import run_ollama_check
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.providers.ollama import (
    DEFAULT_NUM_CTX,
    OllamaHTTPError,
    OllamaLLMClient,
    OllamaTransport,
    normalize_host,
)
from ai_video_editor.llm.registry import create_client, default_timeout, provider_names
from ai_video_editor.pipeline import EditPipelineParams

PAYLOAD = {"removals": [{"sentence_id": 2, "category": "filler", "reason": "Enrolação."}], "summary": "Uma removida."}
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


class Answer(BaseModel):
    summary: str


def chat_response(content, prompt_tokens=120, output_tokens=30, **extra) -> dict:
    text = content if isinstance(content, str) else json.dumps(content)
    return {
        "model": "qwen2.5:7b", "message": {"role": "assistant", "content": text}, "done": True,
        "done_reason": "stop", "prompt_eval_count": prompt_tokens, "eval_count": output_tokens, **extra,
    }


class FakeTransport:
    """`request`: devolve (ou lança) os itens da lista em ordem; o último se repete."""

    def __init__(self, *items):
        self.items = list(items) or [chat_response(PAYLOAD)]
        self.calls: list[tuple[str, str, dict | None, float]] = []

    def request(self, method, path, body=None, timeout=30.0):
        self.calls.append((method, path, body, timeout))
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(method, path, body)
        return item


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for env in ("OLLAMA_HOST", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE"):
        monkeypatch.delenv(env, raising=False)


def _client(transport: FakeTransport, **kwargs) -> OllamaLLMClient:
    return OllamaLLMClient(client=transport, **kwargs)


def _policy(max_retries=2):
    sleeps: list[float] = []
    return RetryPolicy(max_retries=max_retries, sleep=sleeps.append, rand=lambda: 0.0), sleeps


def _config(**kwargs) -> LLMCallConfig:
    return LLMCallConfig(model_id=kwargs.pop("model_id", "qwen2.5:7b"), **kwargs)


@pytest.fixture
def images(tmp_path):
    result = []
    for label in ("P001.T01.F1", "P001.T01.F2"):
        path = tmp_path / f"{label}.jpg"
        path.write_bytes(JPEG_BYTES + label.encode())
        result.append(ImageInput(path=path, label=label))
    return result


# --- payload ---


def test_text_call_sends_schema_in_format_and_explicit_num_ctx():
    transport = FakeTransport()
    client = _client(transport)

    result = client.generate("prompt", DirectorDecision, _config(temperature=0.3, timeout_sec=45), _policy()[0])

    assert result.value.removals[0].sentence_id == 2
    assert (result.metrics.input_tokens, result.metrics.output_tokens) == (120, 30)
    assert result.metrics.provider == "ollama" and result.metrics.api_calls == 1
    method, path, body, timeout = transport.calls[0]
    assert (method, path, timeout) == ("POST", "/api/chat", 45)
    assert body["model"] == "qwen2.5:7b"
    assert body["messages"] == [{"role": "user", "content": "prompt"}]
    assert body["format"] == DirectorDecision.model_json_schema()
    assert body["stream"] is False
    assert body["options"] == {"num_ctx": DEFAULT_NUM_CTX, "temperature": 0.3}
    assert "keep_alive" not in body and "think" not in body
    assert "num_ctx=8192" in result.metrics.notes


def test_thinking_is_ignored_and_recorded():
    client = _client(FakeTransport())
    assert client.thinking_setting("qwen2.5:7b", "high") is None
    assert "ignorado" in client.thinking_label("qwen2.5:7b", "high")
    result = client.generate("p", DirectorDecision, _config(thinking="high"), _policy()[0])
    assert "ignorado" in result.metrics.thinking


def test_images_go_base64_each_after_its_caption(images):
    transport = FakeTransport(chat_response({"summary": "ok"}))
    images[0] = images[0].model_copy(update={"detail": "low"})

    result = _client(transport).generate("descreva", Answer, _config(model_id="qwen2.5vl:3b"), _policy()[0], images)

    messages = transport.calls[0][2]["messages"]
    assert messages[0] == {"role": "user", "content": "descreva"}
    assert [m["content"] for m in messages[1:]] == ["Imagem P001.T01.F1:", "Imagem P001.T01.F2:"]
    for message, image in zip(messages[1:], images):
        assert message["role"] == "user"
        assert message["images"] == [base64.b64encode(image.read_bytes()).decode("ascii")]
    assert result.metrics.images == 2
    assert any("detalhe da imagem low ignorado" in note for note in result.metrics.notes)


def test_text_only_call_has_no_images_key():
    transport = FakeTransport()
    _client(transport).generate("p", DirectorDecision, _config(), _policy()[0])
    assert all("images" not in m for m in transport.calls[0][2]["messages"])


def test_num_ctx_host_and_keep_alive_from_env_or_arguments(monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "16384")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "3600")
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    transport = FakeTransport()
    client = _client(transport)
    client.generate("p", DirectorDecision, _config(), _policy()[0])
    body = transport.calls[0][2]
    assert body["options"]["num_ctx"] == 16384
    assert body["keep_alive"] == 3600
    assert client.host == "http://127.0.0.1:11434"

    client = _client(FakeTransport(), num_ctx=4096, keep_alive="10m", host="pc-da-gpu:9999")
    assert (client.num_ctx, client.keep_alive, client.host) == (4096, "10m", "http://pc-da-gpu:9999")
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "muito")
    with pytest.raises(ValueError, match="OLLAMA_CONTEXT_LENGTH"):
        OllamaLLMClient()


def test_normalize_host():
    assert normalize_host("http://localhost:11434/") == "http://localhost:11434"
    assert normalize_host("localhost") == "http://localhost:11434"
    assert normalize_host("https://ollama.local") == "https://ollama.local:443"
    assert normalize_host("") == "http://localhost:11434"


def test_no_key_needed():
    client = OllamaLLMClient()
    assert client.api_key_env == "" and client.has_credentials()


# --- contexto ---


def test_input_touching_num_ctx_is_a_clear_permanent_error():
    transport = FakeTransport(chat_response(PAYLOAD, prompt_tokens=4096, output_tokens=20))
    policy, sleeps = _policy()

    with pytest.raises(AIContextLimitError) as info:
        _client(transport, num_ctx=4096).generate("p", DirectorDecision, _config(), policy)

    error = info.value
    assert not error.transient and not error.quota_exhausted
    assert "4096 tokens" in str(error) and "--ollama-ctx" in str(error) and "reduza a entrada" in str(error)
    assert len(transport.calls) == 1 and sleeps == []
    # A cadeia não troca de modelo por isso.
    chain = ModelChain(
        [ModelSpec(provider="ollama", model_id="qwen2.5:7b"), ModelSpec(provider="mock", model_id="mock")],
        retry=_policy()[0], clients={"ollama": _client(transport, num_ctx=4096), "mock": MockLLMClient()},
    )
    with pytest.raises(AIContextLimitError):
        chain.generate("p", DirectorDecision)


def test_output_filling_the_context_is_also_detected():
    transport = FakeTransport(chat_response(PAYLOAD, prompt_tokens=3900, output_tokens=190))
    with pytest.raises(AIContextLimitError, match="resposta"):
        _client(transport, num_ctx=4096).generate("p", DirectorDecision, _config(), _policy()[0])


def test_input_below_the_limit_passes():
    transport = FakeTransport(chat_response(PAYLOAD, prompt_tokens=4000, output_tokens=40))
    result = _client(transport, num_ctx=4096).generate("p", DirectorDecision, _config(), _policy()[0])
    assert result.value.summary == "Uma removida."


# --- erros ---


def test_connection_refused_says_ollama_is_not_running_and_does_not_switch():
    transport = FakeTransport(ConnectionRefusedError(10061, "No connection could be made"))
    policy, sleeps = _policy()
    chain = ModelChain(
        [ModelSpec(provider="ollama", model_id="qwen2.5:7b"), ModelSpec(provider="mock", model_id="mock")],
        retry=policy, clients={"ollama": _client(transport), "mock": MockLLMClient()},
    )
    with pytest.raises(AIUnavailableError) as info:
        chain.generate("p", DirectorDecision)
    assert "Ollama não está rodando" in str(info.value) and "ollama serve" in str(info.value)
    assert not info.value.transient
    assert len(transport.calls) == 1 and sleeps == [] and chain.switches == []


def test_missing_model_says_to_pull_it():
    error = OllamaHTTPError(404, 'model "qwen2.5:7b" not found, try pulling it first', model="qwen2.5:7b")
    with pytest.raises(Exception) as info:
        _client(FakeTransport(error)).generate("p", DirectorDecision, _config(), _policy()[0])
    assert "rode `ollama pull qwen2.5:7b`" in str(info.value)
    assert not info.value.transient


def test_timeout_is_transient_and_retried():
    transport = FakeTransport(TimeoutError("timed out"), chat_response(PAYLOAD))
    policy, sleeps = _policy()
    result = _client(transport).generate("p", DirectorDecision, _config(timeout_sec=600), policy)
    assert result.metrics.api_calls == 2 and len(sleeps) == 1

    with pytest.raises(AITimeoutError) as info:
        _client(FakeTransport(TimeoutError("timed out"))).generate("p", DirectorDecision, _config(), _policy(0)[0])
    assert info.value.transient


def test_runner_crash_and_busy_server_are_transient_other_500_is_not():
    client = OllamaLLMClient()
    crash = OllamaHTTPError(500, "llama-server process has terminated: exit status 0xc0000409: CUDA error")
    assert client.map_error(crash, 600).transient
    assert client.map_error(OllamaHTTPError(503, "server busy"), 600).transient
    oom = client.map_error(OllamaHTTPError(500, "model requires more system memory (7.1 GiB)"), 600)
    assert not oom.transient and "memory" in str(oom)


def test_invalid_json_is_a_validation_error():
    with pytest.raises(AIResponseValidationError) as info:
        _client(FakeTransport(chat_response("isso não é JSON"))).generate("p", DirectorDecision, _config(), _policy()[0])
    assert "JSON inválido" in info.value.details
    with pytest.raises(AIResponseValidationError):
        _client(FakeTransport(chat_response({"outra": "coisa"}))).generate("p", DirectorDecision, _config(), _policy()[0])


def test_urllib_transport_maps_refused_connection_and_http_errors(monkeypatch):
    def refused(*_, **__):
        raise urllib.error.URLError(ConnectionRefusedError(10061, "recusou"))

    monkeypatch.setattr("urllib.request.urlopen", refused)
    with pytest.raises(ConnectionRefusedError):
        OllamaTransport("http://localhost:11434").request("GET", "/api/tags")

    def not_found(request, timeout):
        body = io.BytesIO(json.dumps({"error": 'model "x:1b" not found, try pulling it first'}).encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, body)

    monkeypatch.setattr("urllib.request.urlopen", not_found)
    with pytest.raises(OllamaHTTPError) as info:
        OllamaTransport("http://localhost:11434").request("POST", "/api/chat", {"model": "x:1b"})
    assert (info.value.status_code, info.value.model) == (404, "x:1b")
    assert "try pulling" in info.value.message


# --- registro, cadeia e CLI ---


def test_registered_with_tagged_model_ids_in_chains():
    assert "ollama" in provider_names()
    assert isinstance(create_client("ollama"), OllamaLLMClient)
    assert [s.label for s in resolve_chain("ollama")] == ["ollama:qwen2.5:7b"]
    assert [s.label for s in resolve_chain("ollama", None, "qwen2.5:3b, gemini:gemini-3.6-flash")] == [
        "ollama:qwen2.5:7b", "ollama:qwen2.5:3b", "gemini:gemini-3.6-flash",
    ]
    assert [s.label for s in resolve_chain("gemini", None, "ollama:qwen2.5:7b")] == [
        "gemini:gemini-3.8-flash", "ollama:qwen2.5:7b",
    ]
    # Fora do Ollama, provedor desconhecido continua sendo erro.
    with pytest.raises(ValueError):
        EditPipelineParams(director="gemini", director_fallback="nao-existe:x")


def test_cli_accepts_ollama_everywhere():
    parser = cli.build_parser()
    args = parser.parse_args(["edit", "v.mp4", "--director-provider", "ollama", "--coverage-provider", "ollama"])
    assert (args.director, args.coverage) == ("ollama", "ollama")
    assert parser.parse_args(["revise", "p", "x", "--reviser-provider", "ollama"]).reviser == "ollama"
    assert parser.parse_args(["briefing", "pasta", "--director", "ollama"]).director == "ollama"
    assert parser.parse_args(["visual", "pasta", "--visual-provider", "ollama"]).visual_provider == "ollama"
    args = parser.parse_args(["edit", "v.mp4", "--director", "gemini", "--director-fallback", "ollama:qwen2.5:7b"])
    assert [s.label for s in EditPipelineParams(director=args.director, director_fallback=args.director_fallback)
            .director_chain()][-1] == "ollama:qwen2.5:7b"


def test_cli_timeouts_default_to_the_provider():
    parser = cli.build_parser()
    assert default_timeout("ollama", 120.0) == 600.0 and default_timeout("gemini", 120.0) == 120.0
    assert cli._director_timeout(parser.parse_args(["edit", "v.mp4", "--director", "ollama"])) == 600.0
    assert cli._director_timeout(parser.parse_args(["edit", "v.mp4"])) == 120.0
    assert cli._director_timeout(parser.parse_args(["edit", "v.mp4", "--director", "ollama", "--director-timeout", "90"])) == 90
    args = parser.parse_args(["edit", "v.mp4", "--coverage", "ollama"])
    assert cli._coverage_timeout(args) == 600.0
    assert cli._coverage_timeout(parser.parse_args(["edit", "v.mp4", "--director", "ollama"])) is None  # herda 600

    params = EditPipelineParams(director="gemini")
    reviser = cli._build_reviser(parser.parse_args(["revise", "p", "x", "--reviser", "ollama"]), params)
    assert reviser.chain.timeout_sec == 600.0
    reviser = cli._build_reviser(parser.parse_args(["revise", "p", "x"]), params)
    assert reviser.chain.timeout_sec == 120.0


def test_cli_ollama_options_set_the_environment(monkeypatch):
    args = cli.build_parser().parse_args(
        ["visual", "pasta", "--ollama-host", "http://outro:11434", "--ollama-ctx", "12288", "--ollama-keep-alive", "30m"]
    )
    cli.apply_ollama_args(args)
    client = OllamaLLMClient()
    assert (client.host, client.num_ctx, client.keep_alive) == ("http://outro:11434", 12288, "30m")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["edit", "v.mp4", "--ollama-ctx", "100"])


def test_visual_parallel_is_one_with_ollama_and_warns_if_more(capsys):
    parser = cli.build_parser()
    assert cli.visual_parallel(parser.parse_args(["visual", "pasta", "--visual-provider", "ollama"])) == 1
    assert capsys.readouterr().err == ""
    assert cli.visual_parallel(
        parser.parse_args(["visual", "pasta", "--visual-provider", "ollama", "--visual-paralelo", "3"])
    ) == 3
    assert "mesma GPU" in capsys.readouterr().err
    assert cli.visual_parallel(parser.parse_args(["visual", "pasta"])) == 3


def test_visual_describer_uses_the_ollama_timeout():
    args = cli.build_parser().parse_args(["visual", "pasta", "--visual-provider", "ollama"])
    describer = cli._build_visual_describer(args, lambda *a: None)
    assert describer.chain.timeout_sec == 600.0
    assert describer.chain.labels == ["ollama:qwen2.5:7b"]


# --- ollama-check ---


def _check_server(missing=()):
    models = [m for m in ("qwen2.5:7b", "qwen2.5vl:3b") if m not in missing]

    def respond(method, path, body):
        if path == "/api/version":
            return {"version": "0.40.10"}
        if path == "/api/tags":
            return {"models": [{"name": m} for m in models]}
        if body.get("messages", [{}])[-1].get("images"):
            return chat_response({"cor_esquerda": "vermelho", "cor_direita": "azul", "descricao": "duas cores"},
                                 prompt_tokens=900, load_duration=3_000_000_000)
        return chat_response({"ok": True, "soma": 5}, prompt_tokens=60)

    return FakeTransport(respond)


def test_ollama_check_reports_both_calls_with_time():
    transport = _check_server()
    lines: list[str] = []
    report = run_ollama_check(on_line=lines.append, client=_client(transport))

    assert report.ok, lines
    assert report.version == "0.40.10"
    text, image = [s for s in report.steps if s.name in ("texto", "imagem")]
    assert text.latency_sec is not None and text.input_tokens == 60
    assert image.input_tokens == 900 and "certas" in image.detail
    chat_bodies = [body for _, path, body, _ in transport.calls if path == "/api/chat"]
    assert [b["model"] for b in chat_bodies] == ["qwen2.5:7b", "qwen2.5vl:3b"]
    assert chat_bodies[1]["messages"][1]["content"] == "Imagem TESTE.F1:"
    assert any("Chamada de imagem (qwen2.5vl:3b): JSON válido" in line and "carga do modelo 3,0 s" in line
               for line in lines)


def test_ollama_check_server_down_and_missing_model():
    lines: list[str] = []
    report = run_ollama_check(on_line=lines.append, client=_client(FakeTransport(ConnectionRefusedError("recusou"))))
    assert not report.ok and "Ollama não está rodando" in lines[-1]

    lines = []
    report = run_ollama_check(on_line=lines.append, client=_client(_check_server(missing=("qwen2.5vl:3b",))))
    assert not report.ok
    assert any("rode `ollama pull qwen2.5vl:3b`" in line for line in lines)
    assert [s.name for s in report.steps if s.ok] == ["servidor", "modelo qwen2.5:7b", "texto"]
