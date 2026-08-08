"""Run synthetic ignition benchmarks or a parameter grid search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ah_memory.benchmarks.synthetic import all_scenarios, chain
from ah_memory.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    grid_search,
    load_experiment_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/experiments.json"))
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    config = (
        load_experiment_config(args.config)
        if args.config
        else ExperimentConfig()
    )
    if args.grid:
        rows = grid_search(
            chain,
            args.output.with_suffix(".csv"),
            base=config,
        )
        print(json.dumps({"runs": len(rows), "output": str(args.output)}, ensure_ascii=False))
        return

    runner = ExperimentRunner(config)
    summaries = []
    for scenario in all_scenarios():
        result = runner.run(
            scenario.name,
            scenario.graph,
            scenario.evidence,
            relevant=scenario.relevant,
            irrelevant=scenario.irrelevant,
        )
        summaries.append(result.summary())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
