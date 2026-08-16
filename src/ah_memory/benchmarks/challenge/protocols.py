"""Neutral data contracts for challenge comparison arms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence


@dataclass(frozen=True)
class SourceDocument:
    document_id: Hashable
    text: str


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: Hashable
    text: str
    expected_answer: str = ""
    depth: int = 0
    proof_path: tuple[Hashable, ...] = ()


@dataclass(frozen=True)
class StructuredAnswer:
    query_id: Hashable
    text: str
    support_ids: tuple[Hashable, ...] = ()


@dataclass(frozen=True)
class AnswerRecord:
    answer: StructuredAnswer
    retrieved_documents: tuple[SourceDocument, ...]


@dataclass(frozen=True)
class ArmResult:
    records: tuple[AnswerRecord, ...]
    explainability: float
    f1: float


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class ChatBackend(Protocol):
    def answer(
        self,
        query: BenchmarkQuery,
        documents: Sequence[SourceDocument],
    ) -> StructuredAnswer: ...


class ComparisonArm(Protocol):
    def run(
        self,
        corpus: Sequence[SourceDocument],
        queries: Sequence[BenchmarkQuery],
    ) -> ArmResult: ...
