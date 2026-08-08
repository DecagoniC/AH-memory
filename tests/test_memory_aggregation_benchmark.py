from __future__ import annotations

from pathlib import Path

from ah_memory.benchmarks.aggregation import (
    format_trace,
    load_scenarios,
    run_scenario,
    run_suite,
)

SCENARIOS = Path(__file__).parents[1] / "benchmarks" / "memory_aggregation"


def test_ten_memory_aggregation_scenarios_have_required_schema() -> None:
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) == 10
    for scenario in scenarios:
        assert {
            "input",
            "expected_events",
            "expected_relations",
            "expected_state",
        }.issubset(scenario)


def test_normalized_mode_recovers_synonyms_and_state() -> None:
    scenario = next(
        item for item in load_scenarios(SCENARIOS)
        if item["name"] == "07_synonyms"
    )
    fixed = run_scenario(scenario, "fixed")
    normalized = run_scenario(scenario, "normalized")
    assert normalized.metrics["relation_normalization_accuracy"] == 1.0
    assert normalized.metrics["event_extraction_accuracy"] == 1.0
    assert normalized.metrics["state_accuracy"] == 1.0
    assert fixed.metrics["state_accuracy"] < normalized.metrics["state_accuracy"]


def test_three_research_modes_are_comparable() -> None:
    results = {
        mode: run_suite(SCENARIOS, mode)
        for mode in ("fixed", "normalized", "learned")
    }
    expected_metrics = {
        "relation_normalization_accuracy",
        "event_extraction_accuracy",
        "state_accuracy",
        "retrieval_accuracy",
        "activation_precision",
        "activation_recall",
        "path_accuracy",
        "recall_at_k",
        "mrr",
        "propagation_latency",
    }
    for result in results.values():
        assert expected_metrics.issubset(result["aggregate"])
    assert (
        results["normalized"]["aggregate"]["state_accuracy"]
        >= results["fixed"]["aggregate"]["state_accuracy"]
    )
    assert (
        results["learned"]["aggregate"]["activation_precision"]
        != results["normalized"]["aggregate"]["activation_precision"]
    )


def test_trace_mode_reports_factor_message_details() -> None:
    scenario = load_scenarios(SCENARIOS)[0]
    result = run_scenario(scenario, "learned", trace=True)
    rendered = format_trace(result)
    assert "source" not in rendered or "message" in rendered
    assert "factor=" in rendered
    assert "relation=" in rendered
    assert "parameters=" in rendered
