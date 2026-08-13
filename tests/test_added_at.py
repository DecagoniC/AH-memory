"""Symbol/fact creation timestamps appear in store metadata and LLM context."""
from __future__ import annotations

from ah_memory.dialogue import DialogueAgent, _format_added_at
from ah_memory.agent import Agent
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def test_ensure_m_stamps_added_at_and_tau() -> None:
    store = AHStore()
    store.ah.tau = 7
    m = store.ensure_m("M_IVAN", "Иван")
    assert m.created_tau == 7
    assert m.added_at
    assert not any(p.name in ("added_at", "created_tau") for p in m.Mt)
    added, tau = store.get_added_at("M_IVAN")
    assert added == m.added_at
    assert tau == 7


def test_abstract_symbol_stamps_added_at() -> None:
    store = AHStore()
    s = store.ensure_abstract("IVAN", {"иван"})
    assert s.added_at
    assert s.created_tau == 0


def test_semantic_factor_metadata_has_added_at() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="PURCHASE",
                    raw_relation="купил",
                    canonical_relation="PURCHASE",
                    roles={"SUBJECT": "FATHER", "OBJECT": "BMW"},
                )
            ],
        )
    )
    factor = store.list_semantic_factors()[0]
    assert factor.metadata.get("added_at")
    assert "created_tau" in factor.metadata
    event = store.list_events()[0]
    assert event.metadata.get("added_at")


def test_llm_context_includes_added_timestamp() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="PURCHASE",
                    raw_relation="купил",
                    canonical_relation="PURCHASE",
                    roles={"SUBJECT": "FATHER", "OBJECT": "BMW"},
                )
            ],
        )
    )
    agent = Agent(store=store)
    dialogue = DialogueAgent(agent)
    factor = store.list_semantic_factors()[0]
    ctx = dialogue._compact_memory_for_llm(
        {
            "activated_nodes": list(factor.variables),
            "events": [],
            "state": {},
        }
    )
    assert "добавлено:" in ctx
    assert "Факты (с временем добавления)" in ctx
    pretty = _format_added_at(str(factor.metadata["added_at"]), int(factor.metadata["created_tau"]))
    assert pretty
    assert pretty.split(",")[0] in ctx or "τ=" in ctx
