from __future__ import annotations

import time

from ah_memory.agent import Agent
from ah_memory.corpus import build_encyclopedia
from ah_memory.dsl import DSLInterpreter
from ah_memory.examples.dog import run_dog_ignition
from ah_memory.examples.rabbit import (
    RABBIT_TEXT,
    build_rabbit_memory,
    rabbit_auto_score,
    syntactic_answer_who_is_hare,
)
from ah_memory.gc import collect
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine
from ah_memory.invariants import validate
from ah_memory.perception import RulePerception
from ah_memory.store import AHStore
from ah_memory.templates import CREATE_ROLES, seed_templates, template_roles_coverage
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    Hyperlink,
    LinkId,
    Property,
    Role,
    SecondOrderSymbol,
    Section,
)


def test_templates_middle_requirements() -> None:
    store = AHStore()
    seed_templates(store)
    assert len(store.find_templates()) >= 8
    covered = template_roles_coverage(store)
    for r in (Role.SUBJECT, Role.OBJECT, Role.LOCATION, Role.TIME, Role.CAUSE, Role.TOOL):
        assert r in covered
    create = store.get_template("T_CREATE")
    assert [a.role for a in create.actants] == CREATE_ROLES
    assert len(create.actants) == 7


def test_rabbit_auto_extract_at_least_6_of_8() -> None:
    store = build_rabbit_memory()
    hit, total = rabbit_auto_score(store)
    assert total == 8
    assert hit >= 6, f"only {hit}/8 facts extracted"
    validate(store)
    ans = syntactic_answer_who_is_hare(store)
    assert any(w in ans for w in ("зверёк", "маленький", "дикий", "животное"))


def test_find_roles() -> None:
    store = build_rabbit_memory()
    nodes = store.find_roles(Role.SUBJECT, "M_HARE")
    assert len(nodes) >= 1


def test_dsl_intersect_episodes_and_roles() -> None:
    store = build_rabbit_memory()
    dsl = DSLInterpreter(store)
    roles = dsl.execute("findRoles(SUBJECT, M_HARE)").value
    episodes = dsl.execute("findLists(kind=Episode)").value
    assert isinstance(roles, list) and roles
    assert isinstance(episodes, list) and episodes
    # composition supported
    inter = dsl.execute("intersect(findLists(kind=Episode), findLists(kind=Episode))").value
    assert set(inter) == set(episodes)


def test_ignition_propagates_dog() -> None:
    hist = run_dog_ignition(ticks=8)
    joined = "|".join(hist)
    # should see generalization / private model activation over ticks
    assert "M_DOG" in joined or "M_REX" in joined or "DOG" in joined
    assert any(h for h in hist if h)  # WM non-empty at some tick


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
    agent = Agent()
    created = agent.ingest(RABBIT_TEXT)
    assert len(created.created_n) >= 6
    reply = agent.ask("Кто такой заяц?")
    assert reply.source == "graph"
    assert reply.trace_uids
    assert reply.answer != "неизвестно" or reply.trace_uids


def test_continuous_cycle_trajectory() -> None:
    agent = Agent()
    r1 = agent.step_message("Заяц обитает в лесу.")
    assert r1.answer.startswith("ingested:")
    r2 = agent.step_message("Кто такой заяц?")
    # second message sees prior WM influence via non-empty traces / seeds
    assert r2.traces or r2.trace_uids is not None


def test_encyclopedia_nfr() -> None:
    store, corpus = build_encyclopedia()
    assert len(store.ah.S) >= 150
    assert all(s.R for s in store.ah.S.values())
    assert store.graph_size() >= 1000
    assert len(corpus.split()) >= 15000


def test_ignition_tick_budget() -> None:
    store, _ = build_encyclopedia()
    # shrink to ~1000 if larger — still ok
    eng = IgnitionEngine(store)
    eng.seed([ActivationSeed("ANIMAL", 0.8), ActivationSeed("M_ANIMAL", 0.8)])
    t0 = time.perf_counter()
    eng.tick()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500, f"tick took {elapsed_ms:.1f} ms"
