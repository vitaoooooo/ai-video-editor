"""Testes do provedor Ollama com transporte HTTP falso (sem rede e sem Ollama)."""

import base64
import io
import json
import threading
import urllib.error

import pytest
from pydantic import BaseModel

from ai_video_editor import cli
from ai_video_editor.director import DirectorDecision
from ai_video_editor.intelligence.exceptions import (
    AIContextLimitError,
    AIProviderError,
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
    compact_schema,
    extract_json_object,
    normalize_host,
    reset_format_modes,
)
from ai_video_editor.llm.registry import create_client, default_timeout, provider_names
from ai_video_editor.pipeline import EditPipelineParams
from ai_video_editor.roteiro.describer import PhotoDescriber, PhotoRequest
from ai_video_editor.roteiro.schema import PhotoDescription

PAYLOAD = {"removals": [{"sentence_id": 2, "category": "filler", "reason": "Enrolação."}], "summary": "Uma removida."}
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"
# Resposta zerada do Ollama 0.40 quando o sampler da gramática do JSON schema quebra (medida no material real).
ZERO_RESPONSE = {"model": "", "created_at": "0001-01-01T00:00:00Z", "message": {"role": "", "content": ""}, "done": False}


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
    for env in ("OLLAMA_HOST", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE", "OLLAMA_FORMATO"):
        monkeypatch.delenv(env, raising=False)
    reset_format_modes()
    yield
    reset_format_modes()


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


def test_output_limit_goes_as_num_predict_only_when_asked():
    transport = FakeTransport()
    client = _client(transport)
    client.generate("prompt", DirectorDecision, _config(max_output_tokens=340), _policy()[0])
    assert transport.calls[0][2]["options"] == {"num_ctx": DEFAULT_NUM_CTX, "temperature": 0.2, "num_predict": 340}


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


def _check_server(missing=(), broken_schema=False):
    models = [m for m in ("qwen2.5:7b", "gemma3:4b") if m not in missing]

    def respond(method, path, body):
        if path == "/api/version":
            return {"version": "0.40.10"}
        if path == "/api/tags":
            return {"models": [{"name": m} for m in models]}
        if body.get("messages", [{}])[-1].get("images"):
            if broken_schema and "format" in body:
                return ZERO_RESPONSE
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
    text, schema_step, text_step, image = [
        s for s in report.steps if s.name in ("texto", "imagem (schema)", "imagem (texto)", "imagem")
    ]
    assert text.latency_sec is not None and text.input_tokens == 60
    assert schema_step.input_tokens == 900 and "certas" in schema_step.detail and text_step.ok
    assert image.ok and image.detail == "schema, texto"
    chat_bodies = [body for _, path, body, _ in transport.calls if path == "/api/chat"]
    assert [b["model"] for b in chat_bodies] == ["qwen2.5:7b", "gemma3:4b", "gemma3:4b"]
    assert chat_bodies[1]["messages"][1]["content"] == "Imagem TESTE.F1:"
    assert "format" in chat_bodies[1] and "format" not in chat_bodies[2]
    assert any("Chamada com imagem (gemma3:4b), JSON schema em format: JSON válido" in line
               and "carga do modelo 3,0 s" in line for line in lines)
    assert any("Formato com imagem (gemma3:4b): os dois funcionaram" in line for line in lines)


def test_ollama_check_says_only_text_format_worked():
    lines: list[str] = []
    client = _client(_check_server(broken_schema=True))
    report = run_ollama_check(on_line=lines.append, client=client)

    assert report.ok, lines
    steps = {s.name: s for s in report.steps}
    assert not steps["imagem (schema)"].ok and "interrompeu a geração" in steps["imagem (schema)"].detail
    assert steps["imagem (texto)"].ok and steps["imagem"].detail == "texto"
    assert any("só o JSON pelo texto funcionou" in line for line in lines)
    assert client.format_mode == "auto"  # o diagnóstico devolve o formato configurado


def test_ollama_check_server_down_and_missing_model():
    lines: list[str] = []
    report = run_ollama_check(on_line=lines.append, client=_client(FakeTransport(ConnectionRefusedError("recusou"))))
    assert not report.ok and "Ollama não está rodando" in lines[-1]

    lines = []
    report = run_ollama_check(on_line=lines.append, client=_client(_check_server(missing=("gemma3:4b",))))
    assert not report.ok
    assert any("rode `ollama pull gemma3:4b`" in line for line in lines)
    assert [s.name for s in report.steps if s.ok] == ["servidor", "modelo qwen2.5:7b", "texto"]


# --- formato: resposta zerada do Ollama e JSON pelo texto do pedido ---


def _photo(**extra) -> dict:
    return {"what": "Pessoa na praia.", "place": "praia", "people": 1, "tags": ["praia", "mar"], **extra}


def _schema_breaks(answer: str):
    """Servidor que devolve a resposta zerada a todo pedido com `format` e `answer` aos outros."""

    def respond(method, path, body):
        return ZERO_RESPONSE if "format" in body else chat_response(answer)

    return respond


def test_zero_response_is_an_ollama_failure_not_invalid_json(monkeypatch):
    monkeypatch.setenv("OLLAMA_FORMATO", "schema")
    transport = FakeTransport(ZERO_RESPONSE)
    with pytest.raises(AIProviderError) as info:
        _client(transport).generate("p", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0])

    error = info.value
    assert not isinstance(error, AIResponseValidationError) and error.transient is False
    assert "O Ollama interrompeu a geração" in str(error) and "formato JSON forçado" in str(error)
    assert "server.log" in str(error)
    # OLLAMA_FORMATO=schema: sem o fallback, uma chamada só, ainda com `format`.
    assert len(transport.calls) == 1 and "format" in transport.calls[0][2]


def test_finished_empty_answer_is_still_response_without_content():
    empty = {"model": "qwen2.5vl:3b", "message": {"role": "assistant", "content": ""}, "done": True}
    transport = FakeTransport(empty)
    with pytest.raises(AIResponseValidationError) as info:
        _client(transport).generate("p", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0])
    assert info.value.details == "resposta sem conteúdo" and len(transport.calls) == 1


def test_zero_response_retries_at_once_without_format_and_model_stays_in_text():
    answer = "Claro! Aqui está:\n```json\n" + json.dumps(_photo(people="2", talking_to_camera="false")) + "\n```\nPronto."
    transport = FakeTransport(_schema_breaks(answer))
    warnings: list[str] = []
    client = _client(transport, on_warning=warnings.append)
    config = _config(model_id="qwen2.5vl:3b")

    result = client.generate("Descreva a foto.", PhotoDescription, config, _policy()[0])

    assert result.value.people == 2 and result.value.talking_to_camera is False and result.value.place == "praia"
    first, second = (body for _, _, body, _ in transport.calls)
    assert "format" in first and "format" not in second
    prompt = second["messages"][0]["content"]
    assert prompt.startswith("Descreva a foto.\n\nResponda só com um objeto JSON com estes campos")
    assert prompt.endswith(json.dumps(compact_schema(PhotoDescription), ensure_ascii=False, separators=(",", ":")))
    assert first["messages"][0]["content"] == "Descreva a foto."
    expected = "qwen2.5vl:3b: o Ollama falhou com o JSON schema; usando JSON pelo texto do pedido"
    assert warnings == [expected] and result.metrics.warnings == [expected]
    assert expected in result.metrics.notes
    assert result.metrics.input_tokens == 120  # métricas da resposta que valeu

    # A próxima chamada do mesmo modelo já vai sem `format`, sem aviso de novo.
    result = client.generate("Outra foto.", PhotoDescription, config, _policy()[0])
    assert len(transport.calls) == 3 and "format" not in transport.calls[2][2]
    assert result.metrics.warnings == [] and warnings == [expected]
    assert "formato: JSON pelo texto do pedido (sem format)" in result.metrics.notes
    # Outro modelo continua no JSON schema.
    other = FakeTransport()
    _client(other).generate("p", DirectorDecision, _config(), _policy()[0])
    assert "format" in other.calls[0][2]


def test_zero_response_also_without_format_is_the_ollama_error():
    transport = FakeTransport(ZERO_RESPONSE)
    with pytest.raises(AIProviderError) as info:
        _client(transport).generate("p", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0])
    assert "interrompeu" in str(info.value) and "sem o formato JSON forçado" in str(info.value)
    assert len(transport.calls) == 2


def test_forced_text_format_starts_without_format(monkeypatch, images):
    monkeypatch.setenv("OLLAMA_FORMATO", "texto")
    transport = FakeTransport(chat_response(json.dumps(_photo())))
    client = _client(transport)
    client.generate("Descreva.", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0], images[:1])

    body = transport.calls[0][2]
    assert len(transport.calls) == 1 and "format" not in body
    assert "Responda só com um objeto JSON" in body["messages"][0]["content"]
    # A imagem continua na sua mensagem, depois do prompt.
    assert body["messages"][1]["content"] == "Imagem P001.T01.F1:" and body["messages"][1]["images"]
    assert body["options"] == {"num_ctx": DEFAULT_NUM_CTX, "temperature": 0.2}


def test_text_format_without_json_is_a_validation_error():
    transport = FakeTransport(chat_response("Vejo uma praia com uma pessoa."))
    client = _client(transport, format_mode="texto")
    with pytest.raises(AIResponseValidationError) as info:
        client.generate("p", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0])
    assert "nenhum objeto JSON" in info.value.details
    with pytest.raises(AIResponseValidationError):  # JSON sem os obrigatórios
        _client(FakeTransport(chat_response('{"place": "praia"}')), format_mode="texto").generate(
            "p", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0]
        )


def test_text_format_still_checks_the_context():
    transport = FakeTransport(chat_response(json.dumps(_photo()), prompt_tokens=4090))
    with pytest.raises(AIContextLimitError):
        _client(transport, num_ctx=4096, format_mode="texto").generate(
            "p", PhotoDescription, _config(model_id="qwen2.5vl:3b"), _policy()[0]
        )


def test_extract_json_object():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Resposta: {"a": {"b": [1, 2]}} e mais texto {"c": 3}') == {"a": {"b": [1, 2]}}
    assert extract_json_object('chaves soltas { aqui } e depois {"a": "{x}"}') == {"a": "{x}"}
    with pytest.raises(ValueError):
        extract_json_object("[1, 2] sem objeto")


def test_compact_schema_keeps_only_types_enums_and_required():
    compact = compact_schema(PhotoDescription)
    assert set(compact) == {"type", "properties", "required"}
    assert compact["required"] == ["what", "tags"]
    props = compact["properties"]
    assert props["what"] == {"type": "string"} and props["people"] == {"type": "integer"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert "tela uniforme" in props["issue"]["enum"]
    assert "description" not in json.dumps(compact) and "title" not in json.dumps(compact)
    nested = compact_schema(DirectorDecision)  # $ref de modelo aninhado resolvido
    assert "$ref" not in json.dumps(nested)
    assert "properties" in nested["properties"]["removals"]["items"]


def test_format_option_validation(monkeypatch):
    monkeypatch.setenv("OLLAMA_FORMATO", "xml")
    with pytest.raises(ValueError, match="OLLAMA_FORMATO"):
        OllamaLLMClient()
    assert OllamaLLMClient(format_mode="TEXTO").format_mode == "texto"


def test_cli_ollama_format_option(monkeypatch):
    parser = cli.build_parser()
    args = parser.parse_args(["roteiro", "pasta", "--ollama-formato", "texto", "--ollama-ctx", "4096"])
    cli.apply_ollama_args(args)
    client = OllamaLLMClient()
    assert (client.format_mode, client.num_ctx) == ("texto", 4096)
    for command in (["edit", "v.mp4"], ["visual", "p"], ["ollama-check"]):
        assert parser.parse_args([*command, "--ollama-formato", "schema"]).ollama_formato == "schema"
    with pytest.raises(SystemExit):
        parser.parse_args(["visual", "p", "--ollama-formato", "xml"])


def test_switch_warning_once_with_many_threads():
    workers = 6
    barrier = threading.Barrier(workers, timeout=10)
    answer = json.dumps(_photo())

    def respond(method, path, body):
        if "format" in body:
            barrier.wait()  # todas as threads recebem a resposta zerada ao mesmo tempo
            return ZERO_RESPONSE
        return chat_response(answer)

    transport = FakeTransport(respond)
    warnings: list[str] = []
    client = _client(transport, on_warning=warnings.append)
    config = _config(model_id="qwen2.5vl:3b")
    results, errors = [], []

    def work():
        try:
            results.append(client.generate("p", PhotoDescription, config, _policy()[0]))
        except Exception as e:  # pragma: no cover - aparece no assert
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == [] and len(results) == workers
    assert len(warnings) == 1
    assert sum(len(r.metrics.warnings) for r in results) == 1
    assert all(r.value.place == "praia" for r in results)


def test_photo_describer_does_not_spend_a_correction_on_the_zero_response(tmp_path):
    image = tmp_path / "P001.F01.jpg"
    image.write_bytes(JPEG_BYTES)
    transport = FakeTransport(_schema_breaks(json.dumps(_photo())))
    chain = ModelChain(
        [ModelSpec(provider="ollama", model_id="qwen2.5vl:3b")], retry=_policy()[0],
        clients={"ollama": _client(transport)},
    )
    retries: list[str] = []
    describer = PhotoDescriber(chain, on_retry=retries.append)
    fingerprint = describer.fingerprint

    outcome = describer.describe(PhotoRequest(photo_id="P001.F01", clip_id="P001", role="principal",
                                              image=image, speech=""))

    assert outcome.description is not None and outcome.description.place == "praia"
    assert outcome.corrections == 0 and retries == []
    assert outcome.warnings == ["qwen2.5vl:3b: o Ollama falhou com o JSON schema; usando JSON pelo texto do pedido"]
    # O formato não entra na chave do cache: as descrições já gravadas continuam valendo.
    assert describer.fingerprint == fingerprint
    outcome = describer.describe(PhotoRequest(photo_id="P001.F02", clip_id="P001", role="principal",
                                              image=image, speech=""))
    assert outcome.warnings == [] and len(transport.calls) == 3
