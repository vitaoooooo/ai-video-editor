"""Diagnóstico do Ollama local (`ollama-check`): servidor, modelos baixados e chamadas mínimas.

As chamadas usam o mesmo `OllamaLLMClient` do Diretor e do `visual` (mesmo `num_ctx` e mapeamento
de erros), sem novas tentativas, para o tempo medido ser o de uma chamada. A de texto usa o formato
configurado (`--ollama-formato`, padrão `auto`); a com imagem é feita nos dois formatos (`schema` e
`texto`), e o diagnóstico diz qual funcionou: basta um.
"""

import tempfile
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ai_video_editor.intelligence.exceptions import AIProviderError
from ai_video_editor.intelligence.retry import RetryPolicy
from ai_video_editor.llm.providers.ollama import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT_SEC,
    TEXT_FORMAT_NOTE,
    OllamaLLMClient,
)
from ai_video_editor.llm.schema import ImageInput, LLMCallConfig, LLMCallMetrics

DEFAULT_CHECK_TEXT_MODEL = DEFAULT_OLLAMA_MODEL
# gemma3:4b: no material real (RTX 2060) descreveu 644/644 fotos; o qwen2.5vl:3b devolveu lixo na GPU.
DEFAULT_CHECK_IMAGE_MODEL = "gemma3:4b"
CHECK_IMAGE_LABEL = "TESTE.F1"
# Formatos testados na chamada com imagem (nome no diagnóstico).
IMAGE_FORMATS = (("schema", "JSON schema em format"), ("texto", "JSON pelo texto do pedido"))

TEXT_PROMPT = (
    "Teste de diagnóstico. Responda só com o JSON pedido: `ok` = true e `soma` = o resultado de 2 + 3."
)
IMAGE_PROMPT = (
    f"Teste de diagnóstico. A imagem {CHECK_IMAGE_LABEL} tem duas metades de cores diferentes. "
    "Responda só com o JSON pedido: a cor da metade esquerda, a cor da metade direita (em português) "
    "e uma descrição curta da imagem."
)


class TextProbe(BaseModel):
    ok: bool
    soma: int


class ImageProbe(BaseModel):
    cor_esquerda: str
    cor_direita: str
    descricao: str


class CheckStep(BaseModel):
    """Um item do diagnóstico."""

    name: str
    ok: bool
    # Informativo: a falha não reprova o diagnóstico (outro passo resume, como o formato da imagem).
    optional: bool = False
    detail: str = ""
    latency_sec: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class OllamaCheckReport(BaseModel):
    host: str
    num_ctx: int
    version: Optional[str] = None
    installed: list[str] = Field(default_factory=list)
    steps: list[CheckStep] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(step.ok or step.optional for step in self.steps)


def full_model_name(model: str) -> str:
    """`qwen2.5` → `qwen2.5:latest` (a tag que o Ollama assume)."""
    return model if ":" in model else f"{model}:latest"


def write_check_image(path: Path) -> Path:
    """Miniatura sintética 320x180: metade esquerda vermelha, direita azul."""
    from PIL import Image

    image = Image.new("RGB", (320, 180), (220, 30, 30))
    image.paste((30, 60, 220), (160, 0, 320, 180))
    image.save(path, "JPEG", quality=90)
    return path


def _seconds(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " s"


def _call_line(label: str, metrics: Optional[LLMCallMetrics], extra: str = "") -> str:
    parts = []
    if metrics is not None:
        parts.append(_seconds(metrics.latency_sec))
        if metrics.input_tokens is not None:
            parts.append(f"tokens {metrics.input_tokens} de entrada / {metrics.output_tokens} de saída")
        parts.extend(note for note in metrics.notes if not note.startswith("num_ctx") and note != TEXT_FORMAT_NOTE)
    if extra:
        parts.append(extra)
    return f"{label}: JSON válido" + (f" em {'; '.join(parts)}" if parts else "")


def run_ollama_check(
    text_model: str = DEFAULT_CHECK_TEXT_MODEL,
    image_model: Optional[str] = DEFAULT_CHECK_IMAGE_MODEL,
    timeout_sec: float = DEFAULT_OLLAMA_TIMEOUT_SEC,
    on_line: Callable[[str], None] = print,
    client: Optional[OllamaLLMClient] = None,
) -> OllamaCheckReport:
    """Roda o diagnóstico e escreve uma linha por item em `on_line`.

    Para no primeiro passo se o servidor não responde; modelo não baixado pula a chamada dele
    (e o diagnóstico falha, com o `ollama pull` a rodar).
    """
    client = client or OllamaLLMClient()
    report = OllamaCheckReport(host=client.host, num_ctx=client.num_ctx)
    on_line(f"Ollama em {client.host} (num_ctx={client.num_ctx}, timeout {timeout_sec:.0f} s)")

    try:
        report.version = client.version()
        report.installed = client.installed_models()
    except Exception as e:
        error = client.map_error(e, 10.0)
        report.steps.append(CheckStep(name="servidor", ok=False, detail=str(error)))
        on_line(f"Servidor: FALHOU — {error}")
        return report
    report.steps.append(CheckStep(name="servidor", ok=True, detail=f"versão {report.version}"))
    on_line(f"Servidor: no ar (versão {report.version})")
    on_line(f"Modelos baixados: {', '.join(report.installed) or '(nenhum)'}")

    installed = {full_model_name(name) for name in report.installed}
    wanted = [("texto", text_model)] + ([("imagem", image_model)] if image_model else [])
    available: dict[str, bool] = {}
    for _, model in wanted:
        if model in available:
            continue
        available[model] = full_model_name(model) in installed
        if available[model]:
            report.steps.append(CheckStep(name=f"modelo {model}", ok=True, detail="baixado"))
        else:
            report.steps.append(CheckStep(name=f"modelo {model}", ok=False, detail=f"rode `ollama pull {model}`"))
            on_line(f"Modelo {model}: FALTA — rode `ollama pull {model}`")

    retry = RetryPolicy(max_retries=0)
    with tempfile.TemporaryDirectory() as tmp:
        for kind, model in wanted:
            if not available[model]:
                continue
            config = LLMCallConfig(model_id=model, thinking="off", temperature=0.0, timeout_sec=timeout_sec)
            if kind == "texto":
                _probe(report, client, on_line, "texto", f"Chamada de texto ({model})",
                       lambda: _text_call(client, config, retry))
                continue
            image = ImageInput(path=write_check_image(Path(tmp) / "teste.jpg"), label=CHECK_IMAGE_LABEL, detail="auto")
            configured = client.format_mode
            worked = []
            try:
                for mode, name in IMAGE_FORMATS:
                    client.format_mode = mode
                    if _probe(report, client, on_line, f"imagem ({mode})", f"Chamada com imagem ({model}), {name}",
                              lambda: _image_call(client, config, retry, image), optional=True):
                        worked.append(mode)
            finally:
                client.format_mode = configured
            report.steps.append(CheckStep(name="imagem", ok=bool(worked), detail=", ".join(worked) or "nenhum"))
            on_line(f"Formato com imagem ({model}): {_format_verdict(worked)}")

    on_line("Tudo certo." if report.ok else "Diagnóstico com falhas (veja acima).")
    return report


def _text_call(client: OllamaLLMClient, config: LLMCallConfig, retry: RetryPolicy):
    result = client.generate(TEXT_PROMPT, TextProbe, config, retry)
    right = result.value.ok and result.value.soma == 5
    return result, "resposta certa" if right else f"resposta errada ({result.value.model_dump()})"


def _image_call(client: OllamaLLMClient, config: LLMCallConfig, retry: RetryPolicy, image: ImageInput):
    result = client.generate(IMAGE_PROMPT, ImageProbe, config, retry, [image])
    value = result.value
    right = "vermelh" in value.cor_esquerda.lower() and "azul" in value.cor_direita.lower()
    colors = f"cores {'certas' if right else 'erradas'} ({value.cor_esquerda} / {value.cor_direita})"
    return result, f"{colors}; \"{value.descricao}\""


def _probe(
    report: OllamaCheckReport,
    client: OllamaLLMClient,
    on_line: Callable[[str], None],
    name: str,
    label: str,
    call: Callable,
    optional: bool = False,
) -> bool:
    """Faz uma chamada do diagnóstico, registra o passo e escreve a linha; True se deu JSON válido."""
    try:
        result, extra = call()
    except AIProviderError as error:
        metrics = client.last_metrics
        report.steps.append(CheckStep(
            name=name, ok=False, optional=optional, detail=str(error),
            latency_sec=metrics.latency_sec if metrics else None,
        ))
        on_line(f"{label}: FALHOU — {error}")
        return False
    metrics = result.metrics
    report.steps.append(CheckStep(
        name=name, ok=True, optional=optional, detail=extra, latency_sec=metrics.latency_sec,
        input_tokens=metrics.input_tokens, output_tokens=metrics.output_tokens,
    ))
    on_line(_call_line(label, metrics, extra))
    return True


def _format_verdict(worked: list[str]) -> str:
    if worked == ["schema", "texto"]:
        return "os dois funcionaram (auto usa o JSON schema e passa para o texto se o Ollama falhar)"
    if worked == ["texto"]:
        return ("só o JSON pelo texto funcionou; o padrão auto passa para ele na primeira falha "
                "(ou use --ollama-formato texto)")
    if worked == ["schema"]:
        return "só o JSON schema funcionou (use --ollama-formato schema)"
    return "nenhum funcionou (veja acima)"
