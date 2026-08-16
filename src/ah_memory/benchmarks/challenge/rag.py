"""Vanilla retrieval-augmented generation comparison arm."""
from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from ah_memory.benchmarks.challenge.protocols import (
    AnswerRecord,
    ArmResult,
    BenchmarkQuery,
    ChatBackend,
    SourceDocument,
    StructuredAnswer,
)
from ah_memory.benchmarks.challenge.retrieval import CosineVectorRetriever

AnswerScorer = Callable[[BenchmarkQuery, StructuredAnswer], float]


class VanillaRAG:
    """A graph-free RAG arm with injected retrieval and chat dependencies."""

    def __init__(
        self,
        retriever: CosineVectorRetriever,
        chat_backend: ChatBackend,
        *,
        top_k: int,
        answer_scorer: AnswerScorer | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._retriever = retriever
        self._chat_backend = chat_backend
        self._top_k = top_k
        self._answer_scorer = answer_scorer

    def run(
        self,
        corpus: Sequence[SourceDocument],
        queries: Sequence[BenchmarkQuery],
    ) -> ArmResult:
        records: list[AnswerRecord] = []
        answer_scores: list[float] = []
        for query in queries:
            retrieved = self._retriever.retrieve(
                query.text,
                corpus,
                limit=self._top_k,
            )
            documents = tuple(item.document for item in retrieved)
            answer = self._chat_backend.answer(query, documents)
            if answer.query_id != query.query_id:
                raise ValueError("chat answer query_id does not match the request")
            records.append(AnswerRecord(answer, documents))
            if self._answer_scorer is not None:
                answer_scores.append(float(self._answer_scorer(query, answer)))
            elif query.expected_answer:
                answer_scores.append(
                    float(
                        _normalize(answer.text)
                        == _normalize(query.expected_answer)
                    )
                )

        frozen_records = tuple(records)
        return ArmResult(
            records=frozen_records,
            # M2/M4 ExplainScore requires a complete graph UID trace.
            # A graph-free RAG arm has no such trace by definition.
            explainability=0.0,
            f1=(
                sum(answer_scores) / len(answer_scores)
                if answer_scores
                else 0.0
            ),
        )


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())
