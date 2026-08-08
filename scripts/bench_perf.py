"""Measure construction and tick costs over synthetic graph sizes."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ah_memory.benchmarks.synthetic import scale_chain
from ah_memory.experiment import ExperimentConfig, ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/performance.csv"))
    parser.add_argument("--sizes", default="100,500,1000,5000,10000")
    parser.add_argument("--arities", default="2,4,7")
    args = parser.parse_args()

    rows = []
    for size in (int(value) for value in args.sizes.split(",")):
        for arity in (int(value) for value in args.arities.split(",")):
            build_started = time.perf_counter()
            scenario = scale_chain(size, arity)
            build_ms = (time.perf_counter() - build_started) * 1000.0
            runner = ExperimentRunner(
                ExperimentConfig.from_mapping(
                    {
                        "inference": {
                            "max_ticks": 1,
                            "factor_evaluation": "approximate",
                        }
                    }
                )
            )
            result = runner.run(
                scenario.name,
                scenario.graph,
                scenario.evidence,
                relevant=scenario.relevant,
            )
            rows.append(
                {
                    "nodes": size,
                    "arity": arity,
                    "factors": len(scenario.graph.factors),
                    "graph_construction_ms": build_ms,
                    "bp_step_ms": result.state.timings_ms.get("bp_step", 0.0),
                    "activation_update_ms": result.state.timings_ms.get(
                        "activation_update",
                        0.0,
                    ),
                    "total_tick_ms": result.state.timings_ms.get(
                        "total_tick",
                        result.total_run_ms,
                    ),
                }
            )
            print(rows[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
