"""Dense embedders for RAG (GigaChat / Ollama / deterministic)."""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

from ah_memory.relation_normalizer import deterministic_embedding


class RagEmbedder(Protocol):
    name: str

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


class DeterministicRagEmbedder:
    name = "deterministic"

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = max(8, int(dimensions))

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(deterministic_embedding(text, self.dimensions)) for text in texts]


class GigaChatRagEmbedder:
    def __init__(self, inner, *, name: str = "EmbeddingsGigaR") -> None:
        self.inner = inner
        self.name = name

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        payload = [str(t) for t in texts]
        self.inner.warm(payload)
        return [list(self.inner(text)) for text in payload]


class OllamaRagEmbedder:
    name = "ollama"

    def __init__(self, client, *, model: str) -> None:
        self.client = client
        self.model = model
        self.name = model

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(row) for row in self.client.embeddings(list(texts), model=self.model)]


def _in_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def resolve_rag_embedder(*, remote: bool | None = None) -> RagEmbedder:
    """GigaChat embeddings in the live app; deterministic vectors under pytest."""
    if remote is None:
        remote = not _in_pytest()
    if remote:
        live = _try_remote_embedder()
        if live is not None:
            return live
    return DeterministicRagEmbedder()


def _try_remote_embedder() -> RagEmbedder | None:
    from ah_memory.config import load_config

    cfg = load_config()
    model = (cfg.embedding.model or "").strip()
    name = model.lower()
    gigachat_names = {
        "gigachat",
        "embeddingsgigar",
        "embeddings-gigar",
        "embeddings_gigar",
        "embeddings",
        "embeddings-2",
        "embeddings_2",
    }
    if cfg.gigachat.configured and (
        name in gigachat_names or model in {"EmbeddingsGigaR", "Embeddings", "Embeddings-2"}
    ):
        from ah_memory.gigachat_llm import GigaChatClient, GigaChatEmbedder

        api_model = {
            "gigachat": "EmbeddingsGigaR",
            "embeddingsgigar": "EmbeddingsGigaR",
            "embeddings-gigar": "EmbeddingsGigaR",
            "embeddings_gigar": "EmbeddingsGigaR",
            "embeddings": "Embeddings",
            "embeddings-2": "Embeddings-2",
            "embeddings_2": "Embeddings-2",
        }.get(name, model)
        inner = GigaChatEmbedder(
            GigaChatClient(cfg.gigachat),
            model=api_model,
            instruction=None,
        )
        return GigaChatRagEmbedder(inner, name=api_model)

    ollama_names = {"ollama", "nomic-embed-text", "nomic_embed_text"}
    if name in ollama_names or model == cfg.ollama.embedding_model:
        from ah_memory.ollama import OllamaClient

        return OllamaRagEmbedder(
            OllamaClient(cfg.ollama),
            model=cfg.ollama.embedding_model,
        )
    return None
