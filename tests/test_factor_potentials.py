from __future__ import annotations

from ah_memory.belief_propagation import BeliefPropagation
from tests._mini_graph import build_mini_open_store
from ah_memory.factor_graph import Factor, FactorGraph, FactorKind, build_factor_graph
from ah_memory.potentials import (
    AssociativePotential,
    BindPotential,
    FollowPotential,
    HypernodePotential,
    IsAPotential,
    PotentialParameters,
)


def test_seven_variable_hyperfactor_uses_every_variable() -> None:
    variables = tuple(f"V{i}" for i in range(7))
    factor = Factor(
        fid="N7",
        kind=FactorKind.HYPER,
        variables=variables,
        w=1.0,
        potential_key="hypernode",
    )
    parameters = PotentialParameters(
        factor_evaluation="exact",
        exact_max_arity=7,
        hypernode_mode="and",
        hypernode_strength=1.0,
    )
    potential = HypernodePotential()
    high = {uid: (0.1, 0.9) for uid in variables}
    low_last = dict(high)
    low_last["V6"] = (0.9, 0.1)

    high_message = potential.message_to(factor, "V0", high, parameters)
    low_message = potential.message_to(factor, "V0", low_last, parameters)

    assert high_message[1] > low_message[1]


def test_auto_mode_is_explicitly_approximate_above_limit() -> None:
    variables = tuple(f"V{i}" for i in range(15))
    factor = Factor("N15", FactorKind.HYPER, variables, potential_key="hypernode")
    parameters = PotentialParameters(
        factor_evaluation="auto",
        exact_max_arity=10,
        hypernode_mode="soft_and",
    )
    message = HypernodePotential().message_to(
        factor,
        "V0",
        {uid: (0.4, 0.6) for uid in variables},
        parameters,
    )
    assert 0.0 <= message[1] <= 1.0
    graph = FactorGraph(variables, [factor])
    bp = BeliefPropagation(potential_parameters=parameters)
    state = bp.step(graph, bp.initialize(graph))
    assert state.factor_evaluation_modes["N15"] == "approximate"


def test_bind_and_assoc_have_independent_strengths() -> None:
    factor = Factor("F", FactorKind.PAIR, ["A", "B"], w=1.0)
    incoming = {"A": (0.1, 0.9), "B": (0.5, 0.5)}
    parameters = PotentialParameters(bind_strength=1.0, association_strength=0.2)

    bind = BindPotential().message_to(factor, "B", incoming, parameters)
    assoc = AssociativePotential().message_to(factor, "B", incoming, parameters)
    assert bind[1] > assoc[1]


def test_is_a_propagates_up_more_than_down() -> None:
    factor = Factor("ISA", FactorKind.PAIR, ["DOG", "ANIMAL"], w=1.0)
    parameters = PotentialParameters(is_a_up_weight=0.9, is_a_down_weight=0.2)
    potential = IsAPotential()

    upward = potential.message_to(
        factor,
        "ANIMAL",
        {"DOG": (0.05, 0.95), "ANIMAL": (0.5, 0.5)},
        parameters,
    )
    downward = potential.message_to(
        factor,
        "DOG",
        {"DOG": (0.5, 0.5), "ANIMAL": (0.05, 0.95)},
        parameters,
    )
    assert upward[1] > downward[1]


def test_follow_is_directional() -> None:
    factor = Factor("FOLLOW", FactorKind.PAIR, ["E1", "E2"], w=1.0)
    parameters = PotentialParameters(
        follow_forward_weight=0.8,
        follow_backward_weight=0.1,
    )
    potential = FollowPotential()
    forward = potential.message_to(
        factor,
        "E2",
        {"E1": (0.05, 0.95), "E2": (0.5, 0.5)},
        parameters,
    )
    backward = potential.message_to(
        factor,
        "E1",
        {"E1": (0.5, 0.5), "E2": (0.05, 0.95)},
        parameters,
    )
    assert forward[1] > backward[1]


def test_factor_graph_keeps_all_incident_variables() -> None:
    graph = FactorGraph(
        variables=[f"V{i}" for i in range(7)],
        factors=[
            Factor(
                "N",
                FactorKind.HYPER,
                [f"V{i}" for i in range(7)],
                potential_key="hypernode",
            )
        ],
    )
    assert len(graph.factor_vars["N"]) == 7
    assert all("N" in graph.var_factors[f"V{i}"] for i in range(7))


def test_existing_episode_follow_links_enter_factor_graph() -> None:
    graph = build_factor_graph(build_mini_open_store(), include_prior=False)
    follow = [factor for factor in graph.factors if factor.potential_key == "follow"]
    assert follow
    assert all(len(factor.variables) == 2 for factor in follow)
