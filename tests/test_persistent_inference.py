from __future__ import annotations

from ah_memory.belief_propagation import BeliefPropagation
from ah_memory.factor_graph import Factor, FactorGraph, FactorKind
from ah_memory.ignition import IgnitionEngine
from ah_memory.types import Property
from tests._mini_graph import build_mini_open_store


def _pair_graph() -> FactorGraph:
    return FactorGraph(
        variables=["A", "B"],
        factors=[
            Factor(
                fid="F",
                kind=FactorKind.PAIR,
                variables=["A", "B"],
                w=1.0,
                potential_key="assoc",
            )
        ],
    )


def test_bp_state_persists_messages_between_steps() -> None:
    graph = _pair_graph()
    bp = BeliefPropagation(damp=0.0)
    initial = bp.initialize(graph, {"A": 3.0})
    first = bp.step(graph, initial, initial.evidence)
    second = bp.step(graph, first, first.evidence)
    fresh_first = bp.step(graph, bp.initialize(graph, {"A": 3.0}), {"A": 3.0})

    assert first.tick == 1
    assert second.tick == 2
    assert second.variable_to_factor
    assert second.message_history[-1] == second.factor_to_variable
    assert second.factor_to_variable != initial.factor_to_variable
    assert second.activation["B"] >= fresh_first.activation["B"]


def test_same_graph_supports_independent_query_states() -> None:
    graph = _pair_graph()
    bp = BeliefPropagation(damp=0.0)
    a_state = bp.step(graph, bp.initialize(graph, {"A": 3.0}), {"A": 3.0})
    b_state = bp.step(graph, bp.initialize(graph, {"B": 3.0}), {"B": 3.0})

    assert a_state.evidence != b_state.evidence
    assert a_state.activation["A"] > a_state.activation["B"]
    assert b_state.activation["B"] > b_state.activation["A"]


def test_query_factor_gates_suppress_unselected_paths() -> None:
    graph = FactorGraph(
        variables=["A", "B", "C"],
        factors=[
            Factor(
                fid="SELECTED",
                kind=FactorKind.PAIR,
                variables=["A", "B"],
                w=1.0,
                potential_key="assoc",
            ),
            Factor(
                fid="UNSELECTED",
                kind=FactorKind.PAIR,
                variables=["A", "C"],
                w=1.0,
                potential_key="assoc",
            ),
        ],
    )
    bp = BeliefPropagation(damp=0.0)
    state = bp.initialize(graph, {"A": 3.0})

    result = bp.step(
        graph,
        state,
        state.evidence,
        factor_gates={"SELECTED": 1.0},
    )

    assert result.activation["B"] > result.activation["C"]
    assert result.factor_to_variable[("UNSELECTED", "C")] == (0.5, 0.5)


def test_counterfactual_logit_contribution_mode() -> None:
    graph = _pair_graph()
    bp = BeliefPropagation(damp=0.0, contribution_mode="counterfactual_logit")
    state = bp.step(graph, bp.initialize(graph, {"A": 3.0}), {"A": 3.0})
    assert state.trace
    assert any(abs(event.contribution) > 0.0 for event in state.trace)


def test_ignition_reuses_topology_until_structure_changes() -> None:
    store = build_mini_open_store()
    engine = IgnitionEngine(store)
    graph = engine.graph
    engine.tick()
    assert engine.graph is graph

    store.ensure_abstract("NEW_SYMBOL")
    engine.tick()
    assert engine.graph is not graph
    assert "NEW_SYMBOL" in engine.graph.variables


def test_tau_and_properties_do_not_invalidate_topology_revision() -> None:
    store = build_mini_open_store()
    revision = store.ah.revision
    store.ah.tau += 1
    uid = next(
        uid
        for uid, element in store.ah.all_hyper().items()
        if hasattr(element, "Pr")
    )
    store.add_property(uid, Property("note", "dynamic metadata"))
    assert store.ah.revision == revision


def test_simulation_api_does_not_require_ah_x() -> None:
    graph = _pair_graph()
    engine = IgnitionEngine(graph=graph)
    state = engine.initialize({"A": 2.0})
    next_state = engine.tick(state)
    assert next_state.tick == 1
    assert next_state.activation["A"] > 0.0
    assert "A" in next_state.working_memory
    assert next_state.working_memory["A"]["entered_at"] == 1
    assert engine.store is None
