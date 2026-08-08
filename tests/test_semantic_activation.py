from __future__ import annotations

from ah_memory.factor_graph import Factor, FactorGraph, FactorKind
from ah_memory.factor_parameters import FactorParameters
from ah_memory.relations import Relation, RelationProperties
from ah_memory.semantic_activation import (
    ActivationEngine,
    DecayActivation,
    LinearActivation,
    Message,
    SaturatingReLUActivation,
    SigmoidActivation,
)


def _factor(*, symmetric: bool = False, variables=("A", "B")) -> Factor:
    relation = Relation(
        uid="REL_DYNAMIC",
        raw_label="dynamic",
        canonical_label="DYNAMIC",
        arity=len(variables),
        properties=RelationProperties(
            directional=not symmetric,
            symmetric=symmetric,
        ),
    )
    return Factor(
        fid="F_DYNAMIC",
        kind=FactorKind.HYPER,
        variables=variables,
        w=0.8,
        relation=relation,
        parameters=FactorParameters(
            transmission_strength=0.9,
            directionality=0.9,
            selectivity=0.3,
        ),
        confidence=0.9,
        potential_key="semantic",
        metadata={"source_variable": "A"},
    )


def test_directional_factor_transmission_is_asymmetric() -> None:
    factor = _factor()
    assert factor.transmission("A", "B") > factor.transmission("B", "A")


def test_symmetric_factor_transmission_is_equal() -> None:
    factor = _factor(symmetric=True)
    assert factor.transmission("A", "B") == factor.transmission("B", "A")


def test_all_parameterized_activation_functions_are_bounded() -> None:
    factor = _factor()
    seed = Message(0.9, "A", "B", factor.uid, 0)
    for function in (
        LinearActivation(),
        SigmoidActivation(),
        SaturatingReLUActivation(),
        DecayActivation(),
    ):
        output = ActivationEngine(function).propagate(seed, factor, "A", "B")
        assert 0.0 <= output.activation <= 1.0
        assert output.metadata["relation"] == "DYNAMIC"
        assert output.metadata["parameters"]


def test_nary_factor_propagation_keeps_semantic_trace() -> None:
    factor = _factor(variables=("A", "B", "C"))
    graph = FactorGraph(["A", "B", "C"], [factor])
    activation, trace = ActivationEngine().run(
        graph,
        {"A": 1.0},
        timesteps=2,
        threshold=0.05,
        trace=True,
    )
    assert activation["B"] > activation["C"] - 1e-12
    assert {"B", "C"}.issubset(trace.activated_nodes)
    assert trace.activated_factors == ["F_DYNAMIC"]
    assert trace.relations[0]["canonical_label"] == "DYNAMIC"
    assert trace.messages
    assert {
        "source_uid",
        "target_uid",
        "factor_uid",
        "activation",
        "metadata",
    }.issubset(trace.messages[0])
