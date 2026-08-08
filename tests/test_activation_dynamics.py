from __future__ import annotations

import pytest

from ah_memory.activation import (
    ActivationParameters,
    LinearDecayActivation,
    SaturatedReLUActivation,
    SigmoidActivation,
)
from ah_memory.competition import (
    CompetitionParameters,
    GlobalInhibition,
    TopKNormalization,
)
from ah_memory.factor_graph import FactorGraph
from ah_memory.ignition import WorkingMemory


def test_linear_activation_decays_without_input() -> None:
    value = LinearDecayActivation()(
        previous_activation=0.8,
        incoming_signal=0.0,
        evidence=0.0,
        parameters=ActivationParameters(decay=0.25),
    )
    assert value == pytest.approx(0.6)


def test_sigmoid_and_relu_are_continuous_and_bounded() -> None:
    parameters = ActivationParameters(decay=0.1, eta=0.7)
    sigmoid = SigmoidActivation()(0.2, 0.4, 0.3, parameters)
    relu = SaturatedReLUActivation()(0.2, 0.4, 0.3, parameters)
    assert 0.0 < sigmoid < 1.0
    assert 0.0 <= relu <= 1.0


def test_global_inhibition_reduces_spread() -> None:
    graph = FactorGraph(variables=["A", "B", "C"], factors=[])
    activation = {"A": 0.9, "B": 0.8, "C": 0.7}
    inhibited = GlobalInhibition().apply(
        activation,
        graph,
        CompetitionParameters(enabled=True, strength=0.3),
    )
    assert all(inhibited[uid] < activation[uid] for uid in activation)


def test_top_k_competition_keeps_only_winners() -> None:
    graph = FactorGraph(variables=["A", "B", "C"], factors=[])
    activation = {"A": 0.9, "B": 0.8, "C": 0.7}
    result = TopKNormalization().apply(
        activation,
        graph,
        CompetitionParameters(enabled=True, top_k=2),
    )
    assert result["A"] > 0.0
    assert result["B"] > 0.0
    assert result["C"] == 0.0


def test_working_memory_keeps_score_entry_time_and_support() -> None:
    wm = WorkingMemory()
    wm.sync(
        {"A": 0.8, "B": 0.3},
        tick=3,
        threshold=0.5,
        support={"A": ["F1"]},
    )
    wm.sync(
        {"A": 0.7, "C": 0.9},
        tick=4,
        threshold=0.5,
        support={"A": ["F2"], "C": ["F3"]},
    )
    entries = {entry.uid: entry for entry in wm.entries()}
    assert entries["A"].entered_at == 3
    assert entries["A"].activation == 0.7
    assert entries["A"].support == ["F2"]
    assert entries["C"].entered_at == 4
