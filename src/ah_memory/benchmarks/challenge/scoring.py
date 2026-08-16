"""Source-support scoring for structured benchmark answers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from ah_memory.benchmarks.challenge.protocols import AnswerRecord
from ah_memory.benchmarks.challenge_metrics import hallucination_rate


@dataclass(frozen=True)
class SupportScore:
    supported: bool
    matched_support_ids: tuple[Hashable, ...]
    missing_support_ids: tuple[Hashable, ...]


def score_support(record: AnswerRecord) -> SupportScore:
    """Validate declared support IDs against documents retrieved for the answer."""
    retrieved_ids = {
        document.document_id for document in record.retrieved_documents
    }
    matched = tuple(
        support_id
        for support_id in record.answer.support_ids
        if support_id in retrieved_ids
    )
    missing = tuple(
        support_id
        for support_id in record.answer.support_ids
        if support_id not in retrieved_ids
    )
    return SupportScore(
        supported=bool(record.answer.support_ids) and not missing,
        matched_support_ids=matched,
        missing_support_ids=missing,
    )


def support_explainability(records: Sequence[AnswerRecord]) -> float:
    """Fraction of answers whose declared sources are all retrievable."""
    if not records:
        return 0.0
    return sum(score_support(record).supported for record in records) / len(records)


def answer_hallucination_rate(records: Sequence[AnswerRecord]) -> float:
    """Use the challenge metric over retrieved-document support decisions."""
    return hallucination_rate(score_support(record).supported for record in records)
