from __future__ import annotations

import time

from ah_memory.agent import Agent
from ah_memory.dsl import DSLInterpreter
from ah_memory.gc import collect
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine
from ah_memory.invariants import validate
from ah_memory.perception import JsonLLMPerception
from ah_memory.store import AHStore
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    LinkId,
    Property,
    SecondOrderSymbol,
    Section,
)
from tests._mini_graph import build_mini_open_store

SAMPLE_TEXT = "Сущность — это вид. Сущность обитает в месте."


def test_open_registry_starts_without_domain_relations() -> None:
    store = AHStore()
    assert store.list_relations() == ()


def test_mini_open_extract_and_answer() -> None:
    store = build_mini_open_store()
    validate(store)
    assert len(store.list_semantic_factors()) >= 2
    ans = str(DSLInterpreter(store).execute("answer_who(M_ENTITY)").value)
    assert "вид" in ans


def test_find_roles() -> None:
    store = build_mini_open_store()
    factors = [
        f
        for f in store.list_semantic_factors()
        if f.roles.get("SUBJECT") == "M_ENTITY"
    ]
    assert len(factors) >= 1


def test_dsl_intersect_episodes_and_roles() -> None:
    store = build_mini_open_store()
    dsl = DSLInterpreter(store)
    roles = dsl.execute("findRoles(SUBJECT, M_ENTITY)").value
    episodes = dsl.execute("findLists(kind=Episode)").value
    assert isinstance(roles, list) and roles
    assert isinstance(episodes, list) and episodes
    inter = dsl.execute("intersect(findLists(kind=Episode), findLists(kind=Episode))").value
    assert set(inter) == set(episodes)


def test_ignition_propagates_seeds() -> None:
    store = build_mini_open_store()
    eng = IgnitionEngine(store)
    eng.seed([ActivationSeed("ENTITY", 0.9), ActivationSeed("M_ENTITY", 0.7)])
    hist = []
    for _ in range(6):
        hist.append(",".join(eng.tick().wm))
    joined = "|".join(hist)
    assert "M_ENTITY" in joined or "ENTITY" in joined
    assert any(h for h in hist if h)


def test_gc_removes_orphans_keeps_live() -> None:
    store = AHStore()
    store.add_abstract_symbol(AbstractSymbol(uid="LIVE", R={"TEXT": {"live"}}))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_LIVE", Pr=[Property(name="label", value="live")]))
    store.add_link(
        AssocLink(
            uid="L1",
            id=LinkId.ASSOC.value,
            w=1.0,
            e1=store.m_ref("M_LIVE"),
            e2=store.s_ref("LIVE"),
        )
    )
    store.add_element(Section.C, SecondOrderSymbol(uid="ORPHAN", Pr=[Property(name="label", value="x")]))
    # age past TTL
    store.ah.tau = 100
    store.ah.all_hyper()["ORPHAN"].created_tau = 0  # type: ignore[union-attr]
    report = collect(store, HyperParams(ttl=32))
    assert "ORPHAN" in report.removed_elements
    assert "M_LIVE" in store.ah.C
    assert "LIVE" in store.ah.S


def test_gc_respects_ttl() -> None:
    store = AHStore()
    store.ah.tau = 5
    store.add_element(Section.C, SecondOrderSymbol(uid="NEW", Pr=[Property(name="label", value="n")]))
    report = collect(store, HyperParams(ttl=32))
    assert "NEW" in report.protected_ttl
    assert "NEW" in store.ah.C


def test_agent_ingest_and_ask() -> None:
    import json

    def call_fn(prompt: str) -> str:
        data = json.loads(prompt)
        text = data["text"]
        if "?" in text or text.lower().lstrip().startswith(("кто", "что")):
            return json.dumps(
                {"kind": "question", "candidates": [], "seed_tokens": ["СУЩНОСТЬ"]},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "kind": "fact",
                "candidates": [
                    {
                        "predicate": "IS",
                        "roles": {"SUBJECT": "СУЩНОСТЬ", "OBJECT": "ВИД"},
                        "confidence": 0.9,
                    },
                    {
                        "predicate": "LIVE_IN",
                        "roles": {"SUBJECT": "СУЩНОСТЬ", "LOCATION": "МЕСТЕ"},
                        "confidence": 0.9,
                    },
                ],
                "seed_tokens": ["СУЩНОСТЬ", "ВИД", "МЕСТЕ"],
            },
            ensure_ascii=False,
        )

    agent = Agent(perception=JsonLLMPerception(call_fn, require_grounding=False))
    created = agent.ingest(SAMPLE_TEXT)
    assert len(created.created_n) >= 2
    reply = agent.ask("Кто такая сущность?")
    assert reply.source == "graph"
    assert reply.trace_uids
    assert reply.answer != "неизвестно" or reply.trace_uids


def test_continuous_cycle_trajectory() -> None:
    import json

    def call_fn(prompt: str) -> str:
        data = json.loads(prompt)
        text = data["text"]
        if "?" in text or text.lower().lstrip().startswith(("кто", "что")):
            return json.dumps(
                {"kind": "question", "candidates": [], "seed_tokens": ["СУЩНОСТЬ"]},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "kind": "fact",
                "candidates": [
                    {
                        "predicate": "LIVE_IN",
                        "roles": {"SUBJECT": "СУЩНОСТЬ", "LOCATION": "МЕСТЕ"},
                        "confidence": 0.9,
                    }
                ],
                "seed_tokens": ["СУЩНОСТЬ", "МЕСТЕ"],
            },
            ensure_ascii=False,
        )

    agent = Agent(perception=JsonLLMPerception(call_fn, require_grounding=False))
    r1 = agent.step_message("Сущность обитает в месте.")
    assert r1.answer.startswith("ingested:")
    r2 = agent.step_message("Кто такая сущность?")
    assert r2.traces or r2.trace_uids is not None


def test_ignition_tick_budget() -> None:
    store = build_mini_open_store()
    eng = IgnitionEngine(store)
    eng.seed([ActivationSeed("ENTITY", 0.8), ActivationSeed("M_ENTITY", 0.8)])
    t0 = time.perf_counter()
    eng.tick()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500, f"tick took {elapsed_ms:.1f} ms"
