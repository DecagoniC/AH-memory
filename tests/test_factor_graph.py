"""Tests for factor-graph BP activation."""
from __future__ import annotations

from ah_memory.belief_propagation import BeliefPropagation
from ah_memory.factor_graph import FactorKind, build_factor_graph, collect_variables
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    LinkId,
    Property,
    SecondOrderSymbol,
    Section,
)
from tests._mini_graph import build_mini_open_store


def test_collect_variables_includes_m_and_s() -> None:
    store = build_mini_open_store()
    vars_ = set(collect_variables(store))
    assert "M_ENTITY" in vars_
    assert "ENTITY" in vars_


def test_build_factor_graph_has_hyper_and_pair() -> None:
    store = build_mini_open_store()
    g = build_factor_graph(store, evidence={"M_ENTITY": 3.0}, epsilon=0.05)
    kinds = {f.kind for f in g.factors}
    assert FactorKind.HYPER in kinds
    assert FactorKind.PAIR in kinds
    assert FactorKind.OBS in kinds
    assert FactorKind.PRIOR in kinds


def test_bp_raises_belief_on_seed() -> None:
    store = AHStore()
    store.add_abstract_symbol(AbstractSymbol(uid="A", R={"TEXT": {"a"}}))
    store.add_abstract_symbol(AbstractSymbol(uid="B", R={"TEXT": {"b"}}))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_A", Pr=[Property(name="label", value="a")]))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_B", Pr=[Property(name="label", value="b")]))
    store.add_link(
        AssocLink(
            uid="L1",
            id=LinkId.ASSOC.value,
            w=0.9,
            e1=store.m_ref("M_A"),
            e2=store.m_ref("M_B"),
        )
    )
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[FactCandidate("IS", {"SUBJECT": "A", "OBJECT": "B"})],
            seed_tokens=[],
        )
    )
    g = build_factor_graph(store, evidence={"M_A": 3.5}, epsilon=0.05)
    assert any(f.kind == FactorKind.HYPER for f in g.factors)
    res = BeliefPropagation(rounds=5, damp=0.3).run(g)
    assert "M_A" in res.beliefs
    assert res.beliefs["M_A"] >= res.beliefs.get("A", 0.0)


def test_ignition_bp_sets_x_and_wm() -> None:
    store = build_mini_open_store()
    eng = IgnitionEngine(store, HyperParams())
    eng.seed([ActivationSeed("M_ENTITY", 0.9), ActivationSeed("ENTITY", 0.8)])
    traces = eng.run(4)
    assert traces
    assert store.get_x("M_ENTITY") > 0.0 or any(traces[-1].wm)
