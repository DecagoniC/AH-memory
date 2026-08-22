"""M4: AH vs Vanilla RAG deltas."""
from __future__ import annotations

from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.eval.gold import build_m4_fixture
from ah_memory.eval.hypothesis import ah_answer_is_abstain
from ah_memory.eval.m4 import evaluate_m4
from ah_memory.examples.closed_world import closed_world_text


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


def test_m4_corpus_is_chunked_closed_world() -> None:
    text = closed_world_text()
    low = text.lower()
    assert "тиманский" in low
    assert "четласский" in low
    assert "заяц" not in low
    chunks = VanillaRAG(text, top_k=4).chunks
    assert len(chunks) >= 4


def test_ah_does_not_answer_parametric_encyclopedia() -> None:
    agent, _, _, _ = build_m4_fixture()
    reply = agent.ask("Кто такой заяц?", ticks=6)
    blob = reply.answer.lower()
    assert "заяц" not in blob
    assert "зверёк" not in blob and "зверек" not in blob


def test_graph_facts_are_grounded_in_bulletin() -> None:
    from ah_memory.examples.closed_world import (
        build_closed_world_memory,
        extracted_fact_keys,
    )

    keys = extracted_fact_keys(build_closed_world_memory())
    assert len(keys) >= 8
    blob = " ".join(keys).lower()
    assert "тиманский" in blob or "четласский" in blob
