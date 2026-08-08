"""Tests for factor-graph BP activation."""
from __future__ import annotations

from ah_memory.belief_propagation import BeliefPropagation
from ah_memory.examples.dog import build_dog_memory, run_dog_ignition
from ah_memory.examples.rabbit import build_rabbit_memory
from ah_memory.factor_graph import FactorKind, build_factor_graph, collect_variables
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine
from ah_memory.store import AHStore
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    LinkId,
    Property,
    Role,
    SecondOrderSymbol,
    Section,
    Hyperlink,
)
from ah_memory.templates import seed_templates


def test_collect_variables_excludes_hypernodes() -> None:
    store = build_rabbit_memory()
    vars_ = set(collect_variables(store))
    assert "M_HARE" in vars_
    assert "HARE" in vars_
    for n in store.find_hypernodes():
        assert n.uid not in vars_


def test_build_factor_graph_has_hyper_and_pair() -> None:
    store = build_rabbit_memory()
    g = build_factor_graph(store, evidence={"M_HARE": 3.0}, epsilon=0.05)
    kinds = {f.kind for f in g.factors}
    assert FactorKind.HYPER in kinds
    assert FactorKind.PAIR in kinds
    assert FactorKind.OBS in kinds
    assert FactorKind.PRIOR in kinds
    assert any(f.fid.startswith("OBS::M_HARE") for f in g.factors)


def test_bp_raises_belief_on_seed() -> None:
    store = AHStore()
    seed_templates(store)
    store.add_abstract_symbol(AbstractSymbol(uid="HARE", R={"TEXT": {"заяц"}}))
    store.add_abstract_symbol(AbstractSymbol(uid="BEAST", R={"TEXT": {"зверёк"}}))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_HARE", Pr=[Property(name="label", value="заяц")]))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_BEAST", Pr=[Property(name="label", value="зверёк")]))
    store.add_link(
        AssocLink(
            uid="L1",
            id=LinkId.IS_A.value,
            w=0.9,
            e1=store.m_ref("M_HARE"),
            e2=store.m_ref("M_BEAST"),
        )
    )
    store.add_element(
        Section.C,
        Hyperlink(
            uid="N1",
            w=0.8,
            template=store.m_ref("T_IS"),
            fillers={
                Role.SUBJECT: store.m_ref("M_HARE"),
                Role.OBJECT: store.m_ref("M_BEAST"),
            },
        ),
    )
    g = build_factor_graph(store, evidence={"M_HARE": 3.5}, epsilon=0.05)
    res = BeliefPropagation(rounds=3, damp=0.3).run(g)
    assert res.beliefs["M_HARE"] > 0.55
    # IS-A should pull parent up
    assert res.beliefs["M_BEAST"] > res.beliefs.get("HARE", 0.0) * 0.5 or res.beliefs["M_BEAST"] > 0.2


def test_ignition_bp_sets_x_and_wm() -> None:
    store = build_rabbit_memory()
    eng = IgnitionEngine(store, HyperParams(threshold_t=0.4, fg_rounds=2))
    eng.seed([ActivationSeed("HARE", 0.9), ActivationSeed("M_HARE", 0.9)])
    tr = eng.tick()
    assert store.get_x("M_HARE") > 0.3
    assert tr.z_stats.get("n_factors", 0) > 0
    assert isinstance(tr.trace_factors, list)
    assert tr.chains, "expected human activation chains"
    blob = " | ".join(tr.chains)
    assert "M_HARE" in blob or "HARE" in blob
    assert "→" in blob or "seed" in blob


def test_dog_ignition_still_propagates() -> None:
    hist = run_dog_ignition(ticks=6)
    joined = "|".join(hist)
    assert "M_DOG" in joined or "M_REX" in joined or "DOG" in joined
