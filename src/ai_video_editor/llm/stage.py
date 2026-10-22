"""Base das etapas de IA (Diretor, revisor) que chamam uma `ModelChain`."""

from typing import Optional, Sequence

from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.per_thread import PerThread
from ai_video_editor.llm.schema import ImageInput, ModelSwitch, T


class ChainBackedStage:
    """Preenche os atributos comuns das etapas a partir da cadeia de modelos.

    `provider_name`/`model_id`/`thinking_label` são os PEDIDOS (primeiro elo; entram no
    fingerprint); `effective_*` e `model_switches` dizem o que foi usado de fato.
    Os `last_*` são da thread que chamou (lotes em paralelo não se misturam).
    """

    last_input_tokens = PerThread(None)
    last_output_tokens = PerThread(None)
    last_api_attempts = PerThread(1)
    last_model_switches = PerThread(0)

    def __init__(self, chain: ModelChain):
        self.chain = chain
        self.retry = chain.retry
        self.model_chain: list[str] = chain.labels
        self.thinking_label: Optional[str] = chain.thinking_label()
        self.effective_thinking_label: Optional[str] = self.thinking_label
        self.effective_provider: Optional[str] = None
        self.effective_model_id: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return self.chain.requested.provider

    @property
    def model_id(self) -> str:
        return self.chain.requested.model_id

    @property
    def model_switches(self) -> list[ModelSwitch]:
        return self.chain.switches

    def _generate(self, prompt: str, schema: type[T], images: Optional[Sequence[ImageInput]] = None) -> T:
        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_model_switches = 0
        self.effective_thinking_label = self.thinking_label
        try:
            if images:
                return self.chain.generate(prompt, schema, images).value
            return self.chain.generate(prompt, schema).value
        finally:
            self.last_model_switches = self.chain.last_switches
            metrics = self.chain.last_metrics
            if metrics is not None:
                self.last_input_tokens = metrics.input_tokens
                self.last_output_tokens = metrics.output_tokens
                self.last_api_attempts = max(1, metrics.api_calls)
                self.effective_thinking_label = metrics.thinking
                self.effective_provider = metrics.provider
                self.effective_model_id = metrics.model_id
