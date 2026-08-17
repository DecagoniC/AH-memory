"""Dependency-free vector retrieval for benchmark comparison arms."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ah_memory.benchmarks.challenge.protocols import Embedder, SourceDocument


@dataclass(frozen=True)
class RetrievedDocument:
    document: SourceDocument
    score: float


class CosineVectorRetriever:
    """Rank documents by cosine similarity over injected embedder vectors."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        documents: Sequence[SourceDocument],
        *,
        limit: int,
    ) -> tuple[RetrievedDocument, ...]:
        if limit <= 0:
            raise ValueError("retrieval limit must be positive")
        if not documents:
            return ()

        texts = [document.text for document in documents]
        vectors = list(self._embedder.embed((*texts, query)))
        if len(vectors) != len(documents) + 1:
            raise ValueError("embedder returned an unexpected number of vectors")

        normalized = [tuple(float(value) for value in vector) for vector in vectors]
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) != 1 or dimensions == {0}:
            raise ValueError("embedder vectors must have one nonzero shared dimension")

        query_vector = normalized[-1]
        ranked = [
            RetrievedDocument(document, _cosine(vector, query_vector))
            for document, vector in zip(documents, normalized[:-1], strict=True)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return tuple(ranked[:limit])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
