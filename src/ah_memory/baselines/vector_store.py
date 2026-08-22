"""FAISS index of dense embeddings for RAG chunks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np


def _matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(list(vectors), dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("expected a non-empty 2-D embedding matrix")
    faiss.normalize_L2(arr)
    return arr


class FaissVectorStore:
    """Local FAISS vector DB: IndexFlatIP over L2-normalized embeddings (cosine)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._index: faiss.IndexFlatIP | None = None
        self._texts: list[str] = []

    @property
    def dim(self) -> int:
        return int(self._index.d) if self._index is not None else 0

    def count(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def texts(self) -> list[str]:
        return list(self._texts)

    def clear(self) -> None:
        self._index = None
        self._texts = []

    def replace(self, texts: Iterable[str], vectors: Sequence[Sequence[float]]) -> None:
        items = [str(t) for t in texts]
        if len(items) != len(vectors):
            raise ValueError("texts and vectors length mismatch")
        if not items:
            self.clear()
            return
        xb = _matrix(vectors)
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        self._index = index
        self._texts = items
        if self.path is not None:
            self.save(self.path)

    def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 8,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        if self._index is None or not self._texts or top_k <= 0:
            return []
        xq = _matrix([vector])
        k = min(top_k, self.count())
        scores, ids = self._index.search(xq, k)
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if int(idx) < 0:
                continue
            value = float(score)
            if value <= min_score:
                continue
            out.append((self._texts[int(idx)], value))
        return out

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise ValueError("empty FAISS index")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dest))
        sidecar = dest.with_suffix(dest.suffix + ".texts")
        sidecar.write_text("\n\x1e\n".join(self._texts), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        src = Path(path)
        self._index = faiss.read_index(str(src))
        sidecar = src.with_suffix(src.suffix + ".texts")
        raw = sidecar.read_text(encoding="utf-8") if sidecar.is_file() else ""
        self._texts = [part for part in raw.split("\n\x1e\n") if part]
        if self._texts and len(self._texts) != self.count():
            raise ValueError("FAISS index size does not match stored texts")
