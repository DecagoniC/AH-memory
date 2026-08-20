"""M4: AH vs Vanilla RAG deltas."""
from __future__ import annotations

from ah_memory.eval.gold import build_m4_fixture
from ah_memory.eval.m4 import evaluate_m4


def test_m4_delta_explainability_positive() -> None:
    agent, rag, gold, _ = build_m4_fixture()
    report = evaluate_m4(agent, rag, gold, ticks=6)
    assert report.delta_explainability > 0, report.as_dict()
    # AH must produce at least one faithful traced correct answer
    assert any(i.ah_trace_complete and i.ah_correct for i in report.items), [
        (i.question, i.ah_answer, i.ah_trace[:8]) for i in report.items
    ]


def test_vanilla_rag_has_no_uid_trace() -> None:
    agent, rag, gold, _ = build_m4_fixture()
    r = rag.ask(gold[0].question)
    assert r.trace_uids == []
    assert r.source in {"vanilla_rag", "extractive_rag", "llm_rag", "scripted_rag", "empty_retrieve"}
