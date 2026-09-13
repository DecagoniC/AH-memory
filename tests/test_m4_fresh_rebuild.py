"""M4 protocol: rebuild the AH graph for every RAG backend."""
from __future__ import annotations

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.eval.gold import openai_hf_gold, openai_hf_text
from ah_memory.eval.m4 import evaluate_m4, evaluate_m4_on_fresh_store, item_payloads
from ah_memory.examples.openai_hf_incident import build_openai_hf_memory


def test_fresh_rebuild_keeps_ah_explain_score_stable() -> None:
    gold = openai_hf_gold()
    corpus = openai_hf_text()
    first = evaluate_m4_on_fresh_store(
        build_openai_hf_memory,
        VanillaRAG(corpus, top_k=4),
        gold,
        ticks=6,
    )
    second = evaluate_m4_on_fresh_store(
        build_openai_hf_memory,
        VanillaRAG(corpus, top_k=4),
        gold,
        ticks=6,
    )
    assert first.explain_ah == second.explain_ah
    assert first.hall_ah == second.hall_ah
    assert [item.ah_answer for item in first.items] == [
        item.ah_answer for item in second.items
    ]
    assert item_payloads(first)[0]["question"] == gold[0].question


def test_shared_agent_is_not_required_for_rebuilt_protocol() -> None:
    gold = openai_hf_gold()
    rag = VanillaRAG(openai_hf_text(), top_k=4)
    shared = Agent(store=build_openai_hf_memory())
    evaluate_m4(shared, rag, gold, ticks=6)
    rebuilt = evaluate_m4_on_fresh_store(
        build_openai_hf_memory,
        rag,
        gold,
        ticks=6,
    )
    assert len(rebuilt.items) == len(gold)
    assert rebuilt.delta_explainability >= 0
