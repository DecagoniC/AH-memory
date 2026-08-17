from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from ah_memory.benchmarks.challenge import (
    AnswerRecord,
    ArmResult,
    BenchmarkQuery,
    CosineVectorRetriever,
    SourceDocument,
    StructuredAnswer,
    VanillaRAG,
    answer_hallucination_rate,
    compare_ah_to_rag,
    evaluate_robustness,
    score_support,
)


class FakeEmbedder:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls.append(tuple(texts))
        return [self.vectors[text] for text in texts]


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[
            tuple[BenchmarkQuery, tuple[SourceDocument, ...]]
        ] = []

    def answer(
        self,
        query: BenchmarkQuery,
        documents: Sequence[SourceDocument],
    ) -> StructuredAnswer:
        frozen_documents = tuple(documents)
        self.calls.append((query, frozen_documents))
        support_ids = (
            (frozen_documents[0].document_id,)
            if query.query_id == "supported"
            else ("not-retrieved",)
        )
        return StructuredAnswer(query.query_id, f"answer:{query.text}", support_ids)


@dataclass
class FixedArm:
    result: ArmResult
    calls: list[tuple[int, int]] = field(default_factory=list)

    def run(self, corpus, queries) -> ArmResult:
        self.calls.append((id(corpus), id(queries)))
        return self.result


@dataclass
class RecordingArm:
    delegate: VanillaRAG
    calls: list[tuple[int, int]] = field(default_factory=list)

    def run(self, corpus, queries) -> ArmResult:
        self.calls.append((id(corpus), id(queries)))
        return self.delegate.run(corpus, queries)


def _inputs() -> tuple[tuple[SourceDocument, ...], tuple[BenchmarkQuery, ...]]:
    return (
        (
            SourceDocument("source-a", "alpha document"),
            SourceDocument("source-b", "beta document"),
            SourceDocument("source-c", "gamma document"),
        ),
        (
            BenchmarkQuery("supported", "alpha query"),
            BenchmarkQuery("unsupported", "beta query"),
        ),
    )


def _rag() -> tuple[VanillaRAG, FakeEmbedder, FakeChat]:
    embedder = FakeEmbedder(
        {
            "alpha document": (1.0, 0.0),
            "beta document": (0.0, 1.0),
            "gamma document": (-1.0, 0.0),
            "alpha query": (1.0, 0.0),
            "beta query": (0.0, 1.0),
        }
    )
    chat = FakeChat()
    rag = VanillaRAG(
        CosineVectorRetriever(embedder),
        chat,
        top_k=2,
        answer_scorer=lambda query, answer: float(query.query_id in answer.text),
    )
    return rag, embedder, chat


def test_cosine_retriever_uses_injected_vectors_and_stable_order() -> None:
    corpus, _ = _inputs()
    embedder = FakeEmbedder(
        {
            "alpha document": (1.0, 0.0),
            "beta document": (0.0, 1.0),
            "gamma document": (1.0, 0.0),
            "query": (1.0, 0.0),
        }
    )

    retrieved = CosineVectorRetriever(embedder).retrieve(
        "query",
        corpus,
        limit=2,
    )

    assert [item.document.document_id for item in retrieved] == [
        "source-a",
        "source-c",
    ]
    assert [item.score for item in retrieved] == pytest.approx([1.0, 1.0])
    assert embedder.calls == [
        (
            "alpha document",
            "beta document",
            "gamma document",
            "query",
        )
    ]


def test_vanilla_rag_is_graph_free_and_returns_structured_support() -> None:
    corpus, queries = _inputs()
    rag, embedder, chat = _rag()

    result = rag.run(corpus, queries)

    assert not hasattr(rag, "graph")
    assert not hasattr(rag, "store")
    assert [record.answer.query_id for record in result.records] == [
        query.query_id for query in queries
    ]
    assert [document.document_id for document in result.records[0].retrieved_documents] == [
        "source-a",
        "source-b",
    ]
    assert [document.document_id for document in result.records[1].retrieved_documents] == [
        "source-b",
        "source-a",
    ]
    assert len(embedder.calls) == len(queries)
    assert len(chat.calls) == len(queries)
    assert result.explainability == 0.0


def test_support_scoring_checks_only_retrieved_source_documents() -> None:
    document = SourceDocument("retrieved", "source text")
    supported = AnswerRecord(
        StructuredAnswer("q", "answer", ("retrieved",)),
        (document,),
    )
    unsupported = AnswerRecord(
        StructuredAnswer("q", "answer", ("outside",)),
        (document,),
    )

    assert score_support(supported).supported
    assert score_support(unsupported).missing_support_ids == ("outside",)
    assert answer_hallucination_rate((supported, unsupported)) == pytest.approx(0.5)


def test_m4_uses_identical_inputs_and_existing_metric_deltas() -> None:
    corpus, queries = _inputs()
    rag, _, _ = _rag()
    rag_arm = RecordingArm(rag)
    ah_records = tuple(
        AnswerRecord(
            StructuredAnswer(query.query_id, "answer", ("source-a",)),
            (corpus[0],),
        )
        for query in queries
    )
    ah_arm = FixedArm(ArmResult(ah_records, explainability=0.9, f1=0.8))

    result = compare_ah_to_rag(ah_arm, rag_arm, corpus, queries)

    assert ah_arm.calls == rag_arm.calls
    assert result.delta_explainability == pytest.approx(0.9)
    assert result.delta_hallucination == pytest.approx(0.5)


def test_m5_returns_four_arms_and_robustness_gain_on_shared_inputs() -> None:
    corpus, queries = _inputs()
    empty_records: tuple[AnswerRecord, ...] = ()
    arms = [
        FixedArm(ArmResult(empty_records, explainability=0.0, f1=f1))
        for f1 in (0.72, 0.48, 0.90, 0.75)
    ]

    result = evaluate_robustness(
        ah_slm=arms[0],
        rag_slm=arms[1],
        ah_llm=arms[2],
        rag_llm=arms[3],
        corpus=corpus,
        queries=queries,
    )

    assert len({arm.calls[0] for arm in arms}) == 1
    assert result.ah_slm is arms[0].result
    assert result.rag_slm is arms[1].result
    assert result.ah_llm is arms[2].result
    assert result.rag_llm is arms[3].result
    assert result.gain == pytest.approx(0.3)
