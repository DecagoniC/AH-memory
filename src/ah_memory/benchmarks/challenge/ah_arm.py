"""AH graph arm for the shared M2/M4 question corpus."""
from __future__ import annotations

from typing import Sequence

from ah_memory.benchmarks.challenge.graph_qa import run_graph_qa
from ah_memory.benchmarks.challenge.protocols import (
    AnswerRecord,
    ArmResult,
    BenchmarkQuery,
    SourceDocument,
    StructuredAnswer,
)
from ah_memory.benchmarks.challenge.schema import QAItem
from ah_memory.benchmarks.challenge_metrics import (
    InferenceEvaluation,
    explain_score,
)


class AHGraphArm:
    """Answer from graph propagation and expose only actually traced fact UIDs."""

    def __init__(self, cases: Sequence[QAItem], *, threshold: float = 0.05) -> None:
        self._cases = {case.item_id: case for case in cases}
        self._threshold = threshold

    def run(
        self,
        corpus: Sequence[SourceDocument],
        queries: Sequence[BenchmarkQuery],
    ) -> ArmResult:
        del corpus  # The AH arm uses the graph materialized from source facts.
        records: list[AnswerRecord] = []
        evaluations: list[InferenceEvaluation] = []
        correct = 0
        for query in queries:
            case = self._cases.get(str(query.query_id))
            if case is None:
                raise KeyError(f"unknown graph QA case: {query.query_id}")
            observation = run_graph_qa(case, threshold=self._threshold)
            actual_fact_uids = tuple(
                uid.removeprefix("L::")
                for uid in observation.trace
                if uid.startswith("L::")
            )
            expected_trace = tuple(f"L::{uid}" for uid in case.proof_path)
            is_correct = _equal(observation.answer, case.answer)
            correct += int(is_correct)
            evaluations.append(
                InferenceEvaluation(
                    correct=is_correct,
                    depth=case.depth,
                    expected_trace=expected_trace,
                    actual_trace=observation.trace,
                )
            )
            fact_documents = tuple(
                SourceDocument(document_id=uid, text=uid)
                for uid in actual_fact_uids
            )
            records.append(
                AnswerRecord(
                    StructuredAnswer(
                        query_id=query.query_id,
                        text=observation.answer,
                        support_ids=actual_fact_uids if observation.answer else (),
                    ),
                    fact_documents,
                )
            )
        count = len(queries) or 1
        return ArmResult(
            records=tuple(records),
            explainability=explain_score(evaluations, d_max=6),
            f1=correct / count,
        )


def _equal(left: str, right: str) -> bool:
    normalize = lambda value: " ".join(value.strip().casefold().split())
    return normalize(left) == normalize(right)
