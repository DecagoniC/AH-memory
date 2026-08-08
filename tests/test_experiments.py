from __future__ import annotations

import csv

from ah_memory.benchmarks.metrics import calculate_metrics
from ah_memory.benchmarks.synthetic import (
    all_scenarios,
    chain,
    competing_concepts,
)
from ah_memory.experiment import ExperimentConfig, ExperimentRunner, grid_search


def test_all_synthetic_scenarios_run_reproducibly() -> None:
    config = ExperimentConfig.from_mapping(
        {"inference": {"max_ticks": 6, "threshold": 0.5}}
    )
    first = ExperimentRunner(config)
    second = ExperimentRunner(config)
    for scenario in all_scenarios():
        result_a = first.run(
            scenario.name,
            scenario.graph,
            scenario.evidence,
            relevant=scenario.relevant,
            irrelevant=scenario.irrelevant,
        )
        result_b = second.run(
            scenario.name,
            scenario.graph,
            scenario.evidence,
            relevant=scenario.relevant,
            irrelevant=scenario.irrelevant,
        )
        assert result_a.state.activation_history == result_b.state.activation_history
        assert result_a.metrics == result_b.metrics


def test_chain_propagates_and_converges_measure_is_finite() -> None:
    scenario = chain()
    result = ExperimentRunner(
        ExperimentConfig.from_mapping(
            {
                "inference": {
                    "max_ticks": 12,
                    "threshold": 0.3,
                    "evidence_decay": 0.7,
                }
            }
        )
    ).run(
        scenario.name,
        scenario.graph,
        scenario.evidence,
        relevant=scenario.relevant,
    )
    assert result.state.activation["B"] > 0.0
    assert result.metrics["peak_activation"] > 0.0
    assert result.metrics["convergence"] >= 0.0


def test_competing_concepts_prefers_dog() -> None:
    scenario = competing_concepts()
    result = ExperimentRunner(
        ExperimentConfig.from_mapping(
            {"inference": {"max_ticks": 8, "threshold": 0.3}}
        )
    ).run(
        scenario.name,
        scenario.graph,
        scenario.evidence,
        relevant=scenario.relevant,
        irrelevant=scenario.irrelevant,
    )
    assert result.state.activation["DOG"] > result.state.activation["CAT"]
    assert result.state.activation["DOG"] > result.state.activation["HORSE"]


def test_metrics_detect_spread_and_oscillation() -> None:
    metrics = calculate_metrics(
        [
            {"A": 0.0, "B": 0.0},
            {"A": 1.0, "B": 0.2},
            {"A": 0.2, "B": 0.8},
            {"A": 0.9, "B": 0.1},
        ],
        threshold=0.5,
        relevant={"A"},
        irrelevant={"B"},
    )
    assert metrics["spread"] == 1.0
    assert metrics["oscillation"] > 0.0
    assert metrics["selectivity"] > 1.0


def test_grid_search_writes_stable_csv_schema(tmp_path) -> None:
    output = tmp_path / "grid.csv"
    rows = grid_search(
        chain,
        output,
        base=ExperimentConfig.from_mapping(
            {"inference": {"max_ticks": 2}}
        ),
        activation_types=["linear"],
        decays=[0.1],
        thresholds=[0.5],
        factor_strengths=[0.4],
    )
    assert len(rows) == 1
    with output.open(encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    assert "propagation_latency" in columns
    assert "selectivity" in columns
    assert output.with_suffix(".json").exists()
