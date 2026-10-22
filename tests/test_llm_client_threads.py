"""Criação do cliente do SDK sob demanda com várias threads (sem rede)."""

import threading
import time

from ai_video_editor.llm import ModelChain, ModelSpec
from ai_video_editor.llm import chain as chain_module
from ai_video_editor.llm.providers.mock import MockLLMClient

THREADS = 8


class _SlowCreateMock(MockLLMClient):
    """Cliente criado sob demanda (nada injetado), devagar, contando quantos foram criados."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
        self.created: list[object] = []

    def _create_client(self):
        time.sleep(0.05)  # janela para as outras threads verem `_client` vazio
        client = object()
        self.created.append(client)
        return client


def _run_together(target) -> list:
    barrier = threading.Barrier(THREADS)
    results: list = []

    def call():
        barrier.wait(timeout=5)
        results.append(target())

    threads = [threading.Thread(target=call) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_get_client_creates_a_single_client_across_threads():
    provider = _SlowCreateMock()
    results = _run_together(provider.get_client)

    assert len(provider.created) == 1
    assert len(results) == THREADS and all(client is provider.created[0] for client in results)
    assert provider.get_client() is provider.created[0]


def test_chain_creates_a_single_provider_client_across_threads(monkeypatch):
    created: list[MockLLMClient] = []

    def slow_create(name, **kwargs):
        time.sleep(0.05)
        client = MockLLMClient(name=name)
        created.append(client)
        return client

    monkeypatch.setattr(chain_module, "create_client", slow_create)
    chain = ModelChain([ModelSpec(provider="mock", model_id="mock")])
    results = _run_together(lambda: chain.client_for("mock"))

    assert len(created) == 1
    assert all(client is created[0] for client in results)
