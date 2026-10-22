"""Imagens na camada `llm`: payload do Gemini e da OpenAI com clientes falsos, mock e cadeia (sem rede)."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ai_video_editor.intelligence.exceptions import AIProviderError
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm import ImageInput, LLMCallConfig, ModelChain, ModelSpec
from ai_video_editor.llm.providers.gemini import GeminiLLMClient
from ai_video_editor.llm.providers.mock import MockLLMClient
from ai_video_editor.llm.providers.openai import OpenAILLMClient
from ai_video_editor.llm.stage import ChainBackedStage

JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


class Answer(BaseModel):
    summary: str


@pytest.fixture
def images(tmp_path):
    result = []
    for label in ("P001.T01.F1", "P001.T01.F2"):
        path = tmp_path / f"{label}.jpg"
        path.write_bytes(JPEG_BYTES + label.encode())
        result.append(ImageInput(path=path, label=label))
    return result


def _policy():
    return RetryPolicy(max_retries=2, sleep=lambda _: None, rand=lambda: 0.0)


def _gemini(models) -> GeminiLLMClient:
    return GeminiLLMClient(client=SimpleNamespace(models=models))


def _gemini_models(failures=()):
    models = MagicMock()
    response = MagicMock(text='{"summary": "ok"}')
    response.usage_metadata = SimpleNamespace(prompt_token_count=900, candidates_token_count=20)
    models.generate_content.side_effect = [*failures, response]
    return models


class FakeResponses:
    def __init__(self, failures=()):
        self.failures = list(failures)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(
            output_parsed=kwargs["text_format"].model_validate({"summary": "ok"}),
            output=[],
            status="completed",
            usage=SimpleNamespace(input_tokens=700, output_tokens=10),
        )


# --- Gemini ---


def test_gemini_sends_prompt_then_labeled_inline_images(images):
    models = _gemini_models()
    config = LLMCallConfig(model_id="gemini-3.8-flash")

    result = _gemini(models).generate("Descreva.", Answer, config, _policy(), images)

    assert result.value.summary == "ok"
    assert result.metrics.images == 2
    contents = models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 1 and contents[0].role == "user"
    parts = contents[0].parts
    assert parts[0].text == "Descreva."
    assert parts[1].text == "Imagem P001.T01.F1:"
    assert parts[2].inline_data.mime_type == "image/jpeg"
    assert parts[2].inline_data.data == JPEG_BYTES + b"P001.T01.F1"
    assert parts[3].text == "Imagem P001.T01.F2:"
    assert parts[4].inline_data.data == JPEG_BYTES + b"P001.T01.F2"
    # detalhe "auto": o parâmetro não é enviado
    assert models.generate_content.call_args.kwargs["config"].media_resolution is None


def test_gemini_low_detail_sets_media_resolution(images):
    models = _gemini_models()
    low = [image.model_copy(update={"detail": "low"}) for image in images]

    _gemini(models).generate("x", Answer, LLMCallConfig(model_id="gemini-3.8-flash"), _policy(), low)

    assert "LOW" in str(models.generate_content.call_args.kwargs["config"].media_resolution)


def test_gemini_text_only_call_is_unchanged():
    models = _gemini_models()

    result = _gemini(models).generate("Só texto.", Answer, LLMCallConfig(model_id="gemini-3.8-flash"), _policy())

    kwargs = models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "Só texto."
    assert kwargs["config"].media_resolution is None
    assert result.metrics.images == 0


def test_gemini_retries_with_images_and_counts_calls(images):
    models = _gemini_models(failures=[Exception("HTTP 503: Service Unavailable")])

    result = _gemini(models).generate("x", Answer, LLMCallConfig(model_id="gemini-3.8-flash"), _policy(), images)

    assert models.generate_content.call_count == 2
    assert result.metrics.api_calls == 2 and result.metrics.images == 2
    # a nova tentativa reenvia as mesmas imagens
    assert len(models.generate_content.call_args.kwargs["contents"][0].parts) == 5


def test_missing_image_fails_before_calling(tmp_path):
    models = _gemini_models()
    missing = ImageInput(path=tmp_path / "nao-existe.jpg", label="X")

    with pytest.raises(FileNotFoundError):
        _gemini(models).generate("x", Answer, LLMCallConfig(model_id="gemini-3.8-flash"), _policy(), [missing])
    models.generate_content.assert_not_called()


# --- OpenAI ---


def test_openai_sends_input_text_and_input_image(images):
    responses = FakeResponses()
    client = OpenAILLMClient(client=SimpleNamespace(responses=responses))

    result = client.generate("Descreva.", Answer, LLMCallConfig(model_id="gpt-6-luna"), _policy(), images)

    assert result.metrics.images == 2
    message = responses.calls[0]["input"]
    assert message[0]["role"] == "user"
    content = message[0]["content"]
    assert content[0] == {"type": "input_text", "text": "Descreva."}
    assert content[1] == {"type": "input_text", "text": "Imagem P001.T01.F1:"}
    assert content[2]["type"] == "input_image"
    assert content[2]["detail"] == "auto"
    prefix = "data:image/jpeg;base64,"
    assert content[2]["image_url"].startswith(prefix)
    assert base64.b64decode(content[2]["image_url"][len(prefix):]) == JPEG_BYTES + b"P001.T01.F1"
    assert content[4]["type"] == "input_image"


def test_openai_text_only_call_is_unchanged():
    responses = FakeResponses()
    client = OpenAILLMClient(client=SimpleNamespace(responses=responses))

    client.generate("Só texto.", Answer, LLMCallConfig(model_id="gpt-6-luna"), _policy())

    assert responses.calls[0]["input"] == "Só texto."


# --- mock, cadeia e etapa ---


def test_mock_records_images_and_calls_responder(images):
    seen = {}

    def responder(prompt, imgs, schema):
        seen["labels"] = [image.label for image in imgs]
        return schema(summary=f"{len(imgs)} imagens")

    mock = MockLLMClient(responses=[responder])
    result = mock.generate("x", Answer, LLMCallConfig(model_id="mock"), images=images)

    assert result.value.summary == "2 imagens"
    assert seen["labels"] == ["P001.T01.F1", "P001.T01.F2"]
    assert mock.image_calls == [images]
    assert mock.calls[0][0] == "x"


def test_chain_switches_model_keeping_images(images):
    first = MockLLMClient(responses=[_transient("fora")], name="a")
    second = MockLLMClient(responses=[{"summary": "ok"}], name="b")
    chain = ModelChain(
        [ModelSpec(provider="a", model_id="m1"), ModelSpec(provider="b", model_id="m2")],
        retry=RetryPolicy(max_retries=0, sleep=lambda _: None),
        clients={"a": first, "b": second},
    )

    result = chain.generate("x", Answer, images)

    assert result.value.summary == "ok"
    assert first.image_calls == [images] and second.image_calls == [images]
    assert chain.last_metrics.images == 2 and chain.switches


def _transient(message: str) -> AIProviderError:
    error = AIProviderError(message, provider="a")
    error.transient = True
    return error


def test_stage_without_images_calls_chain_like_before():
    calls = []

    class SpyChain(ModelChain):
        def generate(self, *args):
            calls.append(args)
            return super().generate(*args)

    mock = MockLLMClient(responses=[{"summary": "ok"}])
    stage = ChainBackedStage(SpyChain([ModelSpec(provider="mock", model_id="mock")], clients={"mock": mock}))

    assert stage._generate("x", Answer).summary == "ok"
    assert calls == [("x", Answer)]  # mesma assinatura de antes, sem argumento de imagens
