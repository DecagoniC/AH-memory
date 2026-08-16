from __future__ import annotations

import pytest

from ah_memory.benchmarks.challenge.ah_arm import AHGraphArm
from ah_memory.benchmarks.challenge.adapters import comparison_inputs
from ah_memory.benchmarks.challenge.comparison import compare_ah_to_rag
from ah_memory.benchmarks.challenge.graph_qa import run_graph_qa, to_inference_item
from ah_memory.benchmarks.challenge.protocols import (
    AnswerRecord,
    ArmResult,
    StructuredAnswer,
)
from ah_memory.benchmarks.challenge.schema import (
    QAItem,
    SourceDocument,
    SourceFact,
)
from ah_memory.benchmarks.challenge_evaluation import run_m2_benchmark


def _case(relation: str = "CAUSE", depth: int = 3) -> QAItem:
    facts = tuple(
        SourceFact(
            uid=f"F_{index}",
            subject=f"NODE_{index}",
            relation=relation,
            object=f"NODE_{index + 1}",
        )
        for index in range(depth)
    )
    return QAItem(
        item_id=f"{relation}_{depth}",
        question="Which final node is reached?",
        answer=f"NODE_{depth}",
        depth=depth,
        relation_type=relation,
        proof_path=tuple(fact.uid for fact in facts),
        source_facts=facts,
        source_documents=(
            SourceDocument(
                uid="DOC",
                text=" ".join(
                    f"{fact.subject} {fact.relation} {fact.object}."
                    for fact in facts
                ),
                fact_uids=tuple(fact.uid for fact in facts),
            ),
        ),
    )


@pytest.mark.parametrize("relation", ["FOLLOW", "IS-A", "CAUSE"])
def test_m2_graph_runner_uses_real_directional_factors(relation: str) -> None:
    case = _case(relation)
    report = run_m2_benchmark(
        [to_inference_item(case)],
        lambda _: run_graph_qa(case),
        d_max=6,
    )
    assert report.items[0].correct
    assert report.items[0].trace_complete
    assert report.explain_score == pytest.approx(0.5)


def test_m4_ah_arm_reports_graph_trace_and_positive_explainability() -> None:
    case = _case(depth=2)
    corpus, queries = comparison_inputs([case])

    class UnsupportedArm:
        def run(self, received_corpus, received_queries):
            assert tuple(received_corpus) == tuple(corpus)
            assert tuple(received_queries) == tuple(queries)
            return ArmResult(
                records=(
                    AnswerRecord(
                        StructuredAnswer(queries[0].query_id, "guess", ()),
                        (),
                    ),
                ),
                explainability=0.0,
                f1=0.0,
            )

    result = compare_ah_to_rag(AHGraphArm([case]), UnsupportedArm(), corpus, queries)
    assert result.ah.result.explainability == pytest.approx(2 / 6)
    assert result.delta_explainability > 0.0
    assert result.delta_hallucination > 0.0
