"""EntityResolutionBenchmark runner: sweep thresholds, export reports."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ah_memory.benchmarks.entity_resolution.cases import (
    EntityResolutionCase,
    SymbolSpec,
)
from ah_memory.benchmarks.entity_resolution.dataset import (
    DATASET_VERSION,
    control_cases,
    control_symbols,
)
from ah_memory.benchmarks.entity_resolution.evaluator import evaluate_case
from ah_memory.benchmarks.entity_resolution.generator import build_resolution_store
from ah_memory.benchmarks.entity_resolution.metrics import (
    DEFAULT_THRESHOLDS,
    CaseOutcome,
    MetricBundle,
    aggregate_outcomes,
    by_case_type,
    find_optimal_threshold,
)
from ah_memory.benchmarks.entity_resolution.resolvers import (
    EmbeddingResolver,
    SymbolResolver,
    default_resolvers,
    make_embed_fn,
)
from ah_memory.relation_normalizer import deterministic_embedding

EmbeddingFn = Callable[[str], Sequence[float]]


@dataclass
class EntityResolutionBenchmark:
    symbols: list[SymbolSpec] = field(default_factory=control_symbols)
    cases: list[EntityResolutionCase] = field(default_factory=control_cases)
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    embedding_name: str = "deterministic_ngram"
    embed_fn: EmbeddingFn | None = None
    dimensions: int = 64

    def run(
        self,
        *,
        resolvers: dict[str, SymbolResolver] | None = None,
        thresholds: Sequence[float] | None = None,
        run_activation: bool = True,
    ) -> dict[str, Any]:
        embed = self.embed_fn or (
            lambda text: deterministic_embedding(text, self.dimensions)
        )
        # Prefetch GigaChat (or any warm()-capable) embeddings once.
        if hasattr(embed, "warm"):
            texts: list[str] = []
            for symbol in self.symbols:
                texts.append(symbol.name)
                texts.extend(symbol.aliases)
            for case in self.cases:
                texts.append(case.mention)
            try:
                embed.warm(texts)  # type: ignore[union-attr]
            except Exception:
                pass
        resolver_map = resolvers or default_resolvers(
            embed, dimensions=self.dimensions
        )
        thr_list = tuple(thresholds or self.thresholds)
        store, uid_map = build_resolution_store(self.symbols, include_facts=True)
        embed_resolver = EmbeddingResolver(embed, dimensions=self.dimensions)

        summary_by_resolver: dict[str, Any] = {}
        all_case_rows: list[dict[str, Any]] = []
        all_traces: list[dict[str, Any]] = []
        threshold_tables: dict[str, list[dict[str, Any]]] = {}

        for name, resolver in resolver_map.items():
            sweep: dict[float, MetricBundle] = {}
            # Use mid threshold for per-case dump / activation
            report_threshold = 0.85 if 0.85 in thr_list else thr_list[len(thr_list) // 2]
            detail_outcomes: list[CaseOutcome] = []

            for thr in thr_list:
                outcomes = [
                    evaluate_case(
                        case,
                        resolver,
                        self.symbols,
                        threshold=float(thr),
                        store=store,
                        uid_map=uid_map,
                        embed_resolver=embed_resolver,
                        run_activation=run_activation and abs(thr - report_threshold) < 1e-9,
                    )
                    for case in self.cases
                ]
                sweep[float(thr)] = aggregate_outcomes(outcomes)
                if abs(thr - report_threshold) < 1e-9:
                    detail_outcomes = outcomes

            optimal = find_optimal_threshold(sweep)
            by_type = by_case_type(detail_outcomes)
            summary_by_resolver[name] = {
                "at_threshold": report_threshold,
                "metrics": aggregate_outcomes(detail_outcomes).to_dict(),
                "by_case_type": {
                    key: bundle.to_dict() for key, bundle in by_type.items()
                },
                "optimal": optimal,
                "false_merge_rate": aggregate_outcomes(detail_outcomes).false_merge_rate,
            }
            threshold_tables[name] = [
                {"threshold": thr, **bundle.to_dict()}
                for thr, bundle in sorted(sweep.items())
            ]
            for outcome in detail_outcomes:
                row = outcome.to_dict()
                row["resolver"] = name
                row["threshold"] = report_threshold
                all_case_rows.append(row)
                if outcome.activation_trace:
                    all_traces.append(
                        {
                            "case_id": outcome.case_id,
                            "mention": outcome.mention,
                            "resolved_uid": outcome.predicted_uid,
                            "expected_uid": outcome.expected_uid,
                            "similarity": outcome.similarity,
                            "resolver": name,
                            "activation_trace": outcome.activation_trace,
                            "correct": outcome.correct,
                            "activation_ok": outcome.activation_ok,
                        }
                    )

        return {
            "dataset": DATASET_VERSION,
            "embedding": {
                "model": self.embedding_name,
                "dimensions": self.dimensions,
            },
            "cases": len(self.cases),
            "symbols": len(self.symbols),
            "resolvers": summary_by_resolver,
            "threshold_sweep": threshold_tables,
            "case_rows": all_case_rows,
            "activation_traces": all_traces,
        }


def run_entity_resolution_benchmark(
    *,
    output_dir: str | Path | None = "results/entity_resolution",
    embedding_name: str = "deterministic_ngram",
    embed_fn: EmbeddingFn | None = None,
    dimensions: int = 64,
    thresholds: Sequence[float] | None = None,
    run_activation: bool = True,
    resolvers: dict[str, SymbolResolver] | None = None,
) -> dict[str, Any]:
    bench = EntityResolutionBenchmark(
        embedding_name=embedding_name,
        embed_fn=embed_fn,
        dimensions=dimensions,
    )
    report = bench.run(
        resolvers=resolvers,
        thresholds=thresholds,
        run_activation=run_activation,
    )
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dataset": report["dataset"],
                    "embedding": report["embedding"],
                    "cases": report["cases"],
                    "resolvers": report["resolvers"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "threshold_sweep.json").write_text(
            json.dumps(report["threshold_sweep"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (root / "cases.json").write_text(
            json.dumps(report["case_rows"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (root / "activation_traces.json").write_text(
            json.dumps(report["activation_traces"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "ENTITY RESOLUTION BENCHMARK",
        "",
        f"Dataset: {report['dataset']}",
        f"Cases: {report['cases']}",
        f"Embedding: {report['embedding']['model']}",
        "",
        f"{'':20} {'Accuracy':>9} {'Precision':>9} {'Recall':>8} {'F1':>8} {'FMerge':>8}",
        "-" * 70,
    ]
    for name, block in report["resolvers"].items():
        m = block["metrics"]
        lines.append(
            f"{name:20} {m['accuracy']*100:8.1f}% {m['precision']*100:8.1f}% "
            f"{m['recall']*100:7.1f}% {m['f1']*100:7.1f}% {m['false_merge_rate']*100:7.1f}%"
        )
    lines.append("")
    lines.append("False Merge Rate / Best thresholds:")
    for name, block in report["resolvers"].items():
        opt = block.get("optimal") or {}
        lines.append(
            f"  {name:16} FMerge={block['false_merge_rate']*100:.1f}%  "
            f"F1-optimal={opt.get('f1_optimal')}  "
            f"Safety-optimal={opt.get('safety_optimal')}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Entity resolution benchmark")
    parser.add_argument(
        "--output",
        default="results/entity_resolution",
        help="Output directory for JSON reports",
    )
    parser.add_argument(
        "--embedding-name",
        default=None,
        help="Label for embedding backend (default: config.yaml embedding.model)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimensions override",
    )
    parser.add_argument(
        "--resolver",
        action="append",
        dest="resolvers",
        help="Limit to resolver name(s); repeatable (exact/morphology/embedding/hybrid)",
    )
    parser.add_argument(
        "--no-activation",
        action="store_true",
        help="Skip activation engine probes",
    )
    args = parser.parse_args(argv)
    safety = 0.94
    margin = 0.05
    try:
        from ah_memory.config import load_config

        cfg = load_config()
        model = args.embedding_name or cfg.embedding.model
        dims = args.dimensions or cfg.embedding.dimensions
        safety = cfg.identity.safety_threshold
        margin = cfg.identity.margin
    except Exception:
        model = args.embedding_name or "deterministic_ngram"
        dims = args.dimensions or 64
    name, embed_fn, dims = make_embed_fn(model, dimensions=dims)
    resolver_map = default_resolvers(
        embed_fn,
        dimensions=dims,
        safety_threshold=safety,
        margin=margin,
    )
    if args.resolvers:
        wanted = {r.lower() for r in args.resolvers}
        resolver_map = {k: v for k, v in resolver_map.items() if k in wanted}
    report = run_entity_resolution_benchmark(
        output_dir=args.output,
        embedding_name=name,
        embed_fn=embed_fn,
        dimensions=dims,
        run_activation=not args.no_activation,
        resolvers=resolver_map,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
