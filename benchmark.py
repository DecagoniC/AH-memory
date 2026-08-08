"""Compare fixed, normalized and learned open-relation modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ah_memory.benchmarks.aggregation import (
    format_trace,
    load_scenarios,
    run_scenario,
    run_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fixed", "normalized", "learned", "all"),
        default="all",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("benchmarks/memory_aggregation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/memory_aggregation.json"),
    )
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    modes = (
        ("fixed", "normalized", "learned")
        if args.mode == "all"
        else (args.mode,)
    )
    results = [
        run_suite(args.scenarios, mode, trace=args.trace)
        for mode in modes
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {"mode": result["mode"], **result["aggregate"]}
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.trace:
        scenario = load_scenarios(args.scenarios)[0]
        for mode in modes:
            print(f"\n=== {mode}: {scenario['name']} ===")
            print(format_trace(run_scenario(scenario, mode, trace=True)))


if __name__ == "__main__":
    main()
