"""Deterministic character n-gram embedder for offline RAG runs."""
from __future__ import annotations

from typing import Sequence


class DeterministicNgramEmbedder:
    """Hash character n-grams into a fixed vector without external models."""

    def __init__(self, *, dimensions: int = 64, ngram: int = 3) -> None:
        if dimensions <= 0 or ngram <= 0:
            raise ValueError("dimensions and ngram must be positive")
        self.dimensions = dimensions
        self.ngram = ngram

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        folded = f" {text.casefold()} "
        width = self.ngram
        for index in range(max(0, len(folded) - width + 1)):
            gram = folded[index : index + width]
            values[hash(gram) % self.dimensions] += 1.0
        return values
