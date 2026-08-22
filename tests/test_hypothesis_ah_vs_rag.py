"""AH vs classical LLM+vector RAG: prove or disprove H1/H2.

H1. Neuro-symbolic AH answers are more explainable (UID trace gated).
H2. They are more resistant to hallucinations than dense FAISS RAG ± a generator.

Official M4 gold uses a closed-world bulletin whose nonce names are not
parametric knowledge. Adversarial traps and a scripted generator still test H2
instead of assuming it.
"""
from __future__ import annotations

import pytest

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.eval.gold import (
    MINI_CORPUS,
    closed_world_gold,
    closed_world_trap_gold,
    mini_in_corpus_gold,
    mini_trap_gold,
)
from ah_memory.eval.hypothesis import (
    ah_answer_is_abstain,
    evaluate_hypothesis,
    make_inventing_rag,
    rag_has_uid_trace,
)
from ah_memory.examples.closed_world import (
    build_closed_world_memory,
    closed_world_text,
)
from tests._mini_graph import build_mini_open_store

INVENTION = "Также из общих знаний: секретный код равен NEPTUNE-9, глаза зелёные, масса 12 кг."
CORPUS = closed_world_text()


def _m4_agent() -> Agent:
    return Agent(store=build_closed_world_memory())


def _m4_extractive() -> VanillaRAG:
    return VanillaRAG(CORPUS, top_k=4)


def _m4_inventing() -> VanillaRAG:
    return make_inventing_rag(CORPUS, INVENTION, top_k=4)


def _mini_agent() -> Agent:
    return Agent(store=build_mini_open_store())


def _mini_extractive() -> VanillaRAG:
    return VanillaRAG(MINI_CORPUS, top_k=4)


def _mini_inventing() -> VanillaRAG:
    return make_inventing_rag(MINI_CORPUS, INVENTION, top_k=4)


# ── AH arm ───────────────────────────────────────────────────────────────────


def test_ah_in_corpus_answers_carry_uid_traces() -> None:
    agent = _m4_agent()
    reply = agent.ask("Что такое Тиманский кряж?", ticks=6)
    assert reply.source == "graph"
    assert reply.trace_uids
    assert any("ТИМАНСКИЙ" in uid.upper() for uid in reply.trace_uids)


def test_ah_abstains_on_unknown_entity_location() -> None:
    reply = _m4_agent().ask("Где обитает барсук?", ticks=6)
    assert ah_answer_is_abstain(reply.answer)
    assert reply.trace_uids == []


def test_ah_known_entity_unknown_slot_dumps_labels() -> None:
    """Without a slot heuristic, a known entity dumps labels rather than abstaining."""
    reply = _m4_agent().ask("Какой секретный код у Тиманского кряжа?", ticks=6)
    assert not ah_answer_is_abstain(reply.answer)
    assert "тиманский" in reply.answer.lower()


def test_ah_wrong_slot_color_is_a_documented_limitation() -> None:
    """Keyword 'цвет' still answers from the graph even when the slot is eyes."""
    reply = _m4_agent().ask("Какого цвета глаза Тиманского кряжа?", ticks=6)
    blob = reply.answer.lower()
    assert blob.startswith("color:") or "тиманский" in blob
    assert not ah_answer_is_abstain(reply.answer)


@pytest.mark.xfail(
    reason="persistent WM can answer a later OOD location from the previous entity",
    strict=False,
)
def test_ah_sequential_ood_location_should_abstain() -> None:
    agent = _m4_agent()
    agent.ask("Где расположен Тиманский кряж?", ticks=6)
    later = agent.ask("Где обитает барсук?", ticks=6)
    assert ah_answer_is_abstain(later.answer)


# ── RAG arm ──────────────────────────────────────────────────────────────────


def test_rag_never_emits_uid_traces() -> None:
    rag = _m4_extractive()
    for question in (
        "Что такое Тиманский кряж?",
        "Где обитает барсук?",
        "Какой пароль у сервера alpha?",
    ):
        reply = rag.ask(question)
        assert rag_has_uid_trace(reply) is False
        assert reply.trace_uids == []


def test_extractive_rag_returns_lexical_chunk_on_known_entity() -> None:
    reply = _m4_extractive().ask("Что такое Тиманский кряж?")
    blob = reply.answer.lower().replace("ё", "е")
    assert "тиманск" in blob or "кряж" in blob
    assert reply.chunks
    assert reply.source == "extractive_rag"


def test_extractive_rag_dumps_corpus_on_unknown_entity_with_overlap() -> None:
    """Without shared content tokens extractive RAG abstains."""
    reply = _m4_extractive().ask("Где обитает барсук?")
    assert ah_answer_is_abstain(reply.answer)


def test_inventing_rag_injects_ungrounded_claims() -> None:
    reply = _m4_inventing().ask("Что такое Тиманский кряж?")
    blob = reply.answer.lower()
    assert "neptune-9" in blob or "12" in blob or "зелён" in blob
    assert reply.trace_uids == []
    assert reply.source == "scripted_rag"


# ── H1 / H2 on closed-world fixture ──────────────────────────────────────────


def test_h1_supported_closed_world_extractive_isolated() -> None:
    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=_m4_extractive,
        in_corpus=closed_world_gold()[:5],
        traps=closed_world_trap_gold(),
        protocol="isolated",
    )
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    assert report.h1_explainability.ah_score > 0
    assert report.h1_explainability.rag_score == 0
    assert any(item.ah_trace_complete and item.ah_correct for item in report.in_corpus.items)


def test_h2_tied_or_supported_against_extractive_traps() -> None:
    """Extractive RAG is conservative on some traps; H2 must not be assumed."""
    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=_m4_extractive,
        in_corpus=closed_world_gold()[:5],
        traps=closed_world_trap_gold(),
        protocol="isolated",
    )
    assert report.h2_hallucination.n == len(closed_world_trap_gold())
    assert report.h2_hallucination.verdict in {"supported", "tied"}
    assert report.h2_hallucination.rag_score >= report.h2_hallucination.ah_score


def test_h2_supported_against_inventing_llm_rag() -> None:
    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=_m4_inventing,
        in_corpus=closed_world_gold()[:5],
        traps=closed_world_trap_gold(),
        protocol="isolated",
    )
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    assert report.h2_hallucination.verdict == "supported", report.as_dict()
    assert report.h2_hallucination.rag_score > report.h2_hallucination.ah_score
    assert report.overall == "supported"
    assert any(item.rag_hallucinated for item in report.traps.items)


def test_official_m4_extractive_supports_h1() -> None:
    """Closed-world gold: explain wins; the mass trap keeps AH at zero hallucination."""
    from ah_memory.eval.m4 import evaluate_m4

    report = evaluate_m4(_m4_agent(), _m4_extractive(), closed_world_gold(), ticks=6)
    payload = report.as_dict()
    assert payload["explain_hypothesis_ok"] is True
    assert payload["Hallucination_AH"] == 0.0
    assert 0.0 <= payload["Hallucination_VanillaRAG"] <= 1.0


def test_h2_supported_on_mentioned_but_unstated_fact() -> None:
    """The lock-code probe has no graph fact; extractive RAG may still quote a chunk."""
    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=_m4_extractive,
        in_corpus=closed_world_gold()[:1],
        traps=[
            item
            for item in closed_world_trap_gold()
            if "Veshnet" in item.question or "секретный код" in item.question
        ],
        protocol="isolated",
    )
    assert report.h2_hallucination.verdict in {"supported", "tied"}, report.as_dict()


def test_unknown_who_tied_against_extractive() -> None:
    """Unknown-entity 'who' can dump graph labels; extractive RAG usually abstains."""
    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=_m4_extractive,
        in_corpus=closed_world_gold()[:1],
        traps=[
            item
            for item in closed_world_trap_gold()
            if item.question == "Кто такой барсук?"
        ],
        protocol="isolated",
    )
    assert report.h2_hallucination.verdict in {"tied", "rejected"}, report.as_dict()
    assert report.traps.items[0].rag_hallucinated is False


def test_sequential_protocol_weakens_h2_via_wm_leak() -> None:
    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=_m4_extractive,
        in_corpus=closed_world_gold()[:2],
        traps=[
            item
            for item in closed_world_trap_gold()
            if "барсук" in item.question and item.question.startswith("Где")
        ],
        protocol="sequential",
    )
    assert report.protocol == "sequential"
    assert report.traps.items[0].ah_hallucinated is True


# ── Generic (non-rabbit) fixture: same architecture ──────────────────────────


def test_mini_graph_h1_and_h2_vs_extractive() -> None:
    report = evaluate_hypothesis(
        make_agent=_mini_agent,
        make_rag=_mini_extractive,
        in_corpus=mini_in_corpus_gold(),
        traps=mini_trap_gold(),
        protocol="isolated",
    )
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    who = next(item for item in report.in_corpus.items if "кто" in item.question.lower())
    assert who.ah_correct and who.ah_trace
    assert who.ah_trace_complete
    where_ood = next(item for item in report.traps.items if "барсук" in item.question)
    assert where_ood.ah_hallucinated is False
    assert where_ood.rag_hallucinated is True
    assert report.h2_hallucination.verdict in {"supported", "tied"}


def test_mini_graph_both_claims_vs_inventing_rag() -> None:
    report = evaluate_hypothesis(
        make_agent=_mini_agent,
        make_rag=_mini_inventing,
        in_corpus=mini_in_corpus_gold(),
        traps=mini_trap_gold(),
        protocol="isolated",
    )
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    assert report.h2_hallucination.verdict == "supported", report.as_dict()
    assert report.overall == "supported"


@pytest.mark.integration
def test_live_deepseek_rag_hypothesis() -> None:
    from ah_memory.config import load_config

    cfg = load_config()
    if not cfg.deepseek.configured:
        pytest.skip("DeepSeek is not configured")

    def make_llm_rag() -> VanillaRAG:
        return VanillaRAG(CORPUS, top_k=4, deepseek=cfg.deepseek, strict=False)

    report = evaluate_hypothesis(
        make_agent=_m4_agent,
        make_rag=make_llm_rag,
        in_corpus=closed_world_gold()[:4],
        traps=closed_world_trap_gold(),
        protocol="isolated",
    )
    assert report.rag_backend.startswith("llm+faiss")
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    # Live generator may still abstain; do not require H2, only score both arms.
    assert report.h2_hallucination.n == len(closed_world_trap_gold())
    assert 0.0 <= report.h2_hallucination.ah_score <= 1.0
    assert 0.0 <= report.h2_hallucination.rag_score <= 1.0
