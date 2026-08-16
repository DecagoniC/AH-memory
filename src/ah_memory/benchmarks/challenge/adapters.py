"""Adapters from strict challenge schemas to generic benchmark runners."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ah_memory.benchmarks.challenge.protocols import (
    BenchmarkQuery,
    SourceDocument,
)
from ah_memory.benchmarks.challenge.schema import QAItem, RoleCorpusItem
from ah_memory.benchmarks.challenge_evaluation import RoleBenchmarkItem


class CallableEmbedder:
    def __init__(
        self,
        embed: Callable[[list[str]], Sequence[Sequence[float]]],
    ) -> None:
        self._embed = embed

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._embed(list(texts))


def role_benchmark_items(
    corpus: list[RoleCorpusItem],
) -> list[RoleBenchmarkItem]:
    return [
        RoleBenchmarkItem(
            item_id=item.item_id,
            text=item.text,
            expected_roles=(dict(item.expected_roles),),
            noise=item.variant,
        )
        for item in corpus
    ]


def comparison_inputs(
    corpus: list[QAItem],
) -> tuple[list[SourceDocument], list[BenchmarkQuery]]:
    documents: dict[str, SourceDocument] = {}
    queries: list[BenchmarkQuery] = []
    for item in corpus:
        queries.append(
            BenchmarkQuery(
                query_id=item.item_id,
                text=item.question,
                expected_answer=item.answer,
                depth=item.depth,
                proof_path=item.proof_path,
            )
        )
        for document in item.source_documents:
            existing = documents.get(document.uid)
            converted = SourceDocument(document_id=document.uid, text=document.text)
            if existing is not None and existing != converted:
                raise ValueError(f"conflicting source document UID: {document.uid}")
            documents[document.uid] = converted
    return list(documents.values()), queries
