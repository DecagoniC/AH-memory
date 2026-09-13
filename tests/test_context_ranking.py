from __future__ import annotations

import json

from ah_memory.agent import Agent
from ah_memory.context_ranker import GraphContextRanker
from ah_memory.dialogue import DialogueAgent
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def _symbol(store: AHStore, uid: str, label: str) -> None:
    store.ensure_abstract(uid, {label})
    store.ensure_m(f"M_{uid}", label)


def test_prompt_cards_rank_query_match_above_active_noise() -> None:
    store = AHStore()
    _symbol(store, "TARGET", "target entity")
    for index in range(20):
        _symbol(store, f"NOISE_{index}", f"unrelated item {index}")

    agent = Agent(store=store)
    activation = {
        "M_TARGET": 0.1,
        **{f"M_NOISE_{index}": 1.0 for index in range(20)},
    }
    agent.ignition.wm.sync(activation, threshold=0.0)

    cards = agent.perception_context("find target entity", limit=5)
    payloads = [json.loads(card) for card in cards]

    assert len(payloads) == 5
    assert payloads[0]["entity"] == "target entity"
    assert all("uid" not in payload for payload in payloads)


def test_factor_ranking_prefers_query_and_gate_over_old_noise() -> None:
    store = AHStore()
    candidates = [
        FactCandidate(
            f"REL_{index}",
            {"SUBJECT": "ROOT", "OBJECT": f"NOISE_{index}"},
            statement_type="decision",
        )
        for index in range(16)
    ]
    candidates.append(
        FactCandidate(
            "MATCHES",
            {"SUBJECT": "ROOT", "OBJECT": "TARGET"},
            statement_type="assertion",
        )
    )
    Transform(store).apply(
        PerceptionResult(kind="fact", candidates=candidates)
    )
    target_factor = store.list_semantic_factors()[-1]
    target_uid = target_factor.roles["OBJECT"]
    dialogue = DialogueAgent(Agent(store=store))

    compact = dialogue._compact_memory_for_llm(
        {
            "question": "find target",
            "activated_nodes": [target_uid],
            "query_plan": {"factor_gates": {target_factor.uid: 1.0}},
            "timesteps": [{"activation": {target_uid: 0.9}}],
        },
        max_facts=3,
    )

    assert "target" in compact.lower()
    assert "noise 0" not in compact.lower()


def test_compact_facts_use_raw_span_and_humanized_role_heads() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    "BOUNDED_BY",
                    {
                        "SUBJECT": "NORTH_PART",
                        "OBJECT": "VALLEY_STREAM_ALPHA_ONE",
                    },
                    raw_span="north part is bounded by stream alpha one",
                    statement_type="assertion",
                ),
                FactCandidate(
                    "BOUNDED_BY",
                    {
                        "SUBJECT": "MID_PART",
                        "OBJECT": "VALLEY_STREAM_ALPHA_ONE",
                    },
                    raw_span="mid part is bounded by stream alpha one and stream beta two",
                    statement_type="assertion",
                ),
                FactCandidate(
                    "BOUNDED_BY",
                    {
                        "SUBJECT": "MID_PART",
                        "OBJECT": "VALLEY_STREAM_BETA_TWO",
                    },
                    raw_span="mid part is bounded by stream alpha one and stream beta two",
                    statement_type="assertion",
                ),
            ],
        )
    )
    dialogue = DialogueAgent(Agent(store=store))
    labels = {dialogue._label(uid) for uid in store.ah.C}
    assert "alpha one" in labels
    assert "beta two" in labels
    assert "valley stream alpha one" not in labels

    compact = dialogue._compact_memory_for_llm(
        {
            "activated_nodes": list(store.ah.C),
            "events": [],
            "state": {},
        },
        max_facts=12,
    )
    assert compact.count("stream alpha one and stream beta two") == 1
    assert "stream alpha one" in compact
    assert "VALLEY_STREAM_ALPHA_ONE" not in compact


def test_ah_prompt_does_not_repeat_the_same_fact_list() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    "MATCHES",
                    {"SUBJECT": "ROOT", "OBJECT": "TARGET"},
                    statement_type="assertion",
                )
            ],
        )
    )
    dialogue = DialogueAgent(Agent(store=store))
    factor = store.list_semantic_factors()[0]
    target_uid = factor.roles["OBJECT"]
    mem = dialogue._memory_context("find target")
    assert "Факты (с временем добавления)" in mem
    prompt = dialogue._prompt_view(
        mem,
        "decoded answer",
        {
            "question": "find target",
            "activated_nodes": [target_uid],
            "query_plan": {"factor_gates": {factor.uid: 1.0}},
            "timesteps": [{"activation": {target_uid: 0.9}}],
        },
        mode="test",
    )
    assert prompt.count("Факты (с временем добавления)") == 1
    fact_lines = [
        line for line in prompt.splitlines() if line.startswith("• ")
    ]
    assert fact_lines
    assert len(fact_lines) == len(set(fact_lines))


def test_query_reset_drops_stale_evidence_when_plan_is_empty() -> None:
    store = AHStore()
    _symbol(store, "STALE", "stale context")
    agent = Agent(store=store)
    agent.ignition.initialize({"M_STALE": 1.0})
    agent.ignition.wm.sync({"M_STALE": 1.0}, threshold=0.0)

    reply = agent.ask(
        "unmatched question?",
        ticks=1,
        perception=PerceptionResult(
            kind="question",
            candidates=[],
            seed_tokens=[],
        ),
    )

    assert reply.traces
    assert "M_STALE" not in reply.traces[0].evidence


def test_context_ranker_order_is_stable_for_equal_scores() -> None:
    store = AHStore()
    _symbol(store, "B", "beta")
    _symbol(store, "A", "alpha")
    agent = Agent(store=store)
    agent.ignition.wm.sync(
        {"M_B": 0.5, "M_A": 0.5},
        threshold=0.0,
    )

    ranked = GraphContextRanker(store).rank_symbols(
        "",
        agent.ignition.wm.entries(),
    )

    assert [item.uid for item in ranked] == ["M_A", "M_B"]
