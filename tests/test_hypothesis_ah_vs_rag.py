"""AH vs classical LLM+vector RAG: prove or disprove H1/H2.

H1. Neuro-symbolic AH answers are more explainable (UID trace gated).
H2. They are more resistant to hallucinations than TF-IDF RAG ± a generator.

Official M4 gold can leave H2 tied against extractive RAG (both abstain on the
moon trap). Adversarial traps and a scripted generator are required to test H2
instead of assuming it.
"""
from __future__ import annotations

import pytest

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.eval.gold import (
    MINI_CORPUS,
    mini_in_corpus_gold,
    mini_trap_gold,
    rabbit_gold,
    rabbit_trap_gold,
)
from ah_memory.eval.hypothesis import (
    ah_answer_is_abstain,
    evaluate_hypothesis,
    make_inventing_rag,
    rag_has_uid_trace,
)
from ah_memory.examples.rabbit import RABBIT_TEXT, build_rabbit_memory
from tests._mini_graph import build_mini_open_store

INVENTION = "Также из общих знаний: секретный код равен NEPTUNE-9, глаза зелёные, масса 12 кг."


def _rabbit_agent() -> Agent:
    return Agent(store=build_rabbit_memory())


def _rabbit_extractive() -> VanillaRAG:
    return VanillaRAG(RABBIT_TEXT, top_k=4)


def _rabbit_inventing() -> VanillaRAG:
    return make_inventing_rag(RABBIT_TEXT, INVENTION, top_k=4)


def _mini_agent() -> Agent:
    return Agent(store=build_mini_open_store())


def _mini_extractive() -> VanillaRAG:
    return VanillaRAG(MINI_CORPUS, top_k=4)


def _mini_inventing() -> VanillaRAG:
    return make_inventing_rag(MINI_CORPUS, INVENTION, top_k=4)


# ── AH arm ───────────────────────────────────────────────────────────────────


def test_ah_in_corpus_answers_carry_uid_traces() -> None:
    agent = _rabbit_agent()
    reply = agent.ask("Кто такой заяц?", ticks=6)
    assert reply.source == "graph"
    assert reply.trace_uids
    assert any(uid in {"M_HARE", "HARE", "M_BEAST"} for uid in reply.trace_uids)


def test_ah_abstains_on_unknown_entity_location() -> None:
    reply = _rabbit_agent().ask("Где обитает барсук?", ticks=6)
    assert ah_answer_is_abstain(reply.answer)
    assert reply.trace_uids == []


def test_ah_abstains_on_unknown_slot_without_color_keyword() -> None:
    reply = _rabbit_agent().ask("Какой секретный код у зайца?", ticks=6)
    assert ah_answer_is_abstain(reply.answer)


def test_ah_abstains_when_color_subject_is_not_in_the_question() -> None:
    """BE_COLORED(FUR) must not answer a color question about a different part."""
    reply = _rabbit_agent().ask("Какого цвета глаза зайца?", ticks=6)
    assert ah_answer_is_abstain(reply.answer)


def test_ah_sequential_ood_location_should_abstain() -> None:
    agent = _rabbit_agent()
    first = agent.ask("Где обитает заяц?", ticks=6)
    later = agent.ask("Где обитает барсук?", ticks=6)
    assert "MEADOW" in first.answer or "луг" in first.answer.lower()
    assert ah_answer_is_abstain(later.answer)


# ── RAG arm ──────────────────────────────────────────────────────────────────


def test_rag_never_emits_uid_traces() -> None:
    rag = _rabbit_extractive()
    for question in ("Кто такой заяц?", "Где обитает барсук?", "Какой пароль у сервера alpha?"):
        reply = rag.ask(question)
        assert rag_has_uid_trace(reply) is False
        assert reply.trace_uids == []


def test_extractive_rag_returns_lexical_chunk_on_known_entity() -> None:
    reply = _rabbit_extractive().ask("Кто такой заяц?")
    assert "заяц" in reply.answer.lower()
    assert reply.chunks
    assert reply.source == "extractive_rag"


def test_extractive_rag_dumps_corpus_on_unknown_entity_with_overlap() -> None:
    """'обитает' matches the rabbit paragraph although the entity is absent."""
    reply = _rabbit_extractive().ask("Где обитает барсук?")
    assert not ah_answer_is_abstain(reply.answer)
    assert "заяц" in reply.answer.lower()


def test_inventing_rag_injects_ungrounded_claims() -> None:
    reply = _rabbit_inventing().ask("Кто такой заяц?")
    blob = reply.answer.lower()
    assert "neptune-9" in blob or "12" in blob or "зелён" in blob
    assert reply.trace_uids == []
    assert reply.source == "scripted_rag"


# ── H1 / H2 on rabbit fixture ────────────────────────────────────────────────


def test_h1_supported_rabbit_extractive_isolated() -> None:
    report = evaluate_hypothesis(
        make_agent=_rabbit_agent,
        make_rag=_rabbit_extractive,
        in_corpus=rabbit_gold()[:5],
        traps=rabbit_trap_gold(),
        protocol="isolated",
    )
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    assert report.h1_explainability.ah_score > 0
    assert report.h1_explainability.rag_score == 0
    assert any(item.ah_trace_complete and item.ah_correct for item in report.in_corpus.items)


def test_h2_tied_or_supported_against_extractive_traps() -> None:
    """Extractive RAG is conservative on some traps; H2 must not be assumed."""
    report = evaluate_hypothesis(
        make_agent=_rabbit_agent,
        make_rag=_rabbit_extractive,
        in_corpus=rabbit_gold()[:5],
        traps=rabbit_trap_gold(),
        protocol="isolated",
    )
    assert report.h2_hallucination.n == len(rabbit_trap_gold())
    assert report.h2_hallucination.verdict in {"supported", "tied"}
    assert report.h2_hallucination.rag_score >= report.h2_hallucination.ah_score


def test_h2_supported_against_inventing_llm_rag() -> None:
    report = evaluate_hypothesis(
        make_agent=_rabbit_agent,
        make_rag=_rabbit_inventing,
        in_corpus=rabbit_gold()[:5],
        traps=rabbit_trap_gold(),
        protocol="isolated",
    )
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    assert report.h2_hallucination.verdict == "supported", report.as_dict()
    assert report.h2_hallucination.rag_score > report.h2_hallucination.ah_score
    assert report.overall == "supported"
    assert any(item.rag_hallucinated for item in report.traps.items)


def test_official_m4_extractive_supports_h1_not_h2() -> None:
    """Same six gold items as постановка §7: explain wins, hall delta is 0 offline."""
    from ah_memory.eval.m4 import evaluate_m4

    report = evaluate_m4(_rabbit_agent(), _rabbit_extractive(), rabbit_gold(), ticks=6)
    payload = report.as_dict()
    assert payload["explain_hypothesis_ok"] is True
    assert payload["hall_hypothesis_ok"] is False
    assert payload["hypothesis_ok"] is False
    assert payload["Hallucination_AH"] == 0.0
    assert payload["Hallucination_VanillaRAG"] == 0.0


def test_h2_unknown_entity_who_both_abstain() -> None:
    """After the WM/slot fix AH no longer dumps an episode title for an unknown who-is."""
    report = evaluate_hypothesis(
        make_agent=_rabbit_agent,
        make_rag=_rabbit_extractive,
        in_corpus=rabbit_gold()[:1],
        traps=[item for item in rabbit_trap_gold() if item.question == "Кто такой барсук?"],
        protocol="isolated",
    )
    assert report.traps.items[0].ah_hallucinated is False
    assert ah_answer_is_abstain(report.traps.items[0].ah_answer)


def test_sequential_ood_location_does_not_reuse_prior_subject() -> None:
    report = evaluate_hypothesis(
        make_agent=_rabbit_agent,
        make_rag=_rabbit_extractive,
        in_corpus=rabbit_gold()[:2],
        traps=[item for item in rabbit_trap_gold() if "барсук" in item.question and item.question.startswith("Где")],
        protocol="sequential",
    )
    assert report.protocol == "sequential"
    assert report.traps.items[0].ah_hallucinated is False
    assert ah_answer_is_abstain(report.traps.items[0].ah_answer)


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
        return VanillaRAG(RABBIT_TEXT, top_k=4, deepseek=cfg.deepseek, strict=False)

    report = evaluate_hypothesis(
        make_agent=_rabbit_agent,
        make_rag=make_llm_rag,
        in_corpus=rabbit_gold()[:4],
        traps=rabbit_trap_gold(),
        protocol="isolated",
    )
    assert report.rag_backend == "llm+tfidf"
    assert report.h1_explainability.verdict == "supported", report.as_dict()
    # Live generator may still abstain; do not require H2, only score both arms.
    assert report.h2_hallucination.n == len(rabbit_trap_gold())
    assert 0.0 <= report.h2_hallucination.ah_score <= 1.0
    assert 0.0 <= report.h2_hallucination.rag_score <= 1.0
