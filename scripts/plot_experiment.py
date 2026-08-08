"""Render activation history as a text heatmap or optional PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ah_memory.benchmarks.synthetic import all_scenarios
from ah_memory.experiment import ExperimentRunner

BLOCKS = " ░▒▓█"


def text_heatmap(history: list[dict[str, float]]) -> str:
    nodes = sorted(set().union(*(snapshot.keys() for snapshot in history)))
    width = max((len(uid) for uid in nodes), default=1)
    lines = [" " * (width + 2) + " ".join(f"t{i}" for i in range(len(history)))]
    for uid in nodes:
        cells = []
        for snapshot in history:
            value = min(1.0, max(0.0, snapshot.get(uid, 0.0)))
            cells.append(BLOCKS[min(4, int(round(value * 4)))])
        lines.append(f"{uid:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


def save_png(history: list[dict[str, float]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib to use --png") from exc
    nodes = sorted(set().union(*(snapshot.keys() for snapshot in history)))
    matrix = [[snapshot.get(uid, 0.0) for snapshot in history] for uid in nodes]
    figure, axis = plt.subplots(figsize=(10, max(3, len(nodes) * 0.35)))
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_yticks(range(len(nodes)), labels=nodes)
    axis.set_xlabel("tick")
    axis.set_ylabel("variable")
    figure.colorbar(image, ax=axis, label="activation")
    figure.tight_layout()
    figure.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="chain")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    scenario = next(
        item for item in all_scenarios() if item.name == args.scenario
    )
    result = ExperimentRunner().run(
        scenario.name,
        scenario.graph,
        scenario.evidence,
        relevant=scenario.relevant,
        irrelevant=scenario.irrelevant,
    )
    history = result.state.activation_history
    print(text_heatmap(history))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        save_png(history, args.png)


if __name__ == "__main__":
    main()
