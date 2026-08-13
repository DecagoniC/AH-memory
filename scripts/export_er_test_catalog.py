"""Export entity-resolution / paraphrase test catalogs for human review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ah_memory.benchmarks.entity_resolution.dataset import (
    DATASET_VERSION,
    control_cases,
    control_graph_facts,
    control_symbols,
)

OUT_DIR = ROOT / "results" / "entity_resolution"


def _paraphrase_tables() -> dict:
    # Keep in sync with tests/test_identity_paraphrase.py
    from test_identity_paraphrase import (
        MINI_FACTS,
        NEGATIVE_PARAPHRASES,
        PARAPHRASE_QUERIES,
    )

    return {
        "mini_facts": [
            {"subject": s, "relation": r, "object": o} for s, r, o in MINI_FACTS
        ],
        "paraphrase_queries": [
            {"query": q, "mention": m, "expected_canonical": c}
            for q, m, c in PARAPHRASE_QUERIES
        ],
        "negative_paraphrases": [
            {"mention": m, "must_not_merge_with": d} for m, d in NEGATIVE_PARAPHRASES
        ],
    }


def build_catalog() -> dict:
    symbols = [s.to_dict() for s in control_symbols()]
    cases = [c.to_dict() for c in control_cases()]
    by_type: dict[str, list] = {}
    for case in cases:
        by_type.setdefault(case["case_type"], []).append(case)
    return {
        "dataset": DATASET_VERSION,
        "symbols": symbols,
        "graph_facts": [
            {"subject": a, "relation": b, "object": c}
            for a, b, c in control_graph_facts()
        ],
        "cases_total": len(cases),
        "cases_by_type": {k: len(v) for k, v in sorted(by_type.items())},
        "cases": cases,
        "pytest_modules": [
            "tests/test_entity_resolution.py",
            "tests/test_identity_ingest.py",
            "tests/test_identity_paraphrase.py",
        ],
        "paraphrase_suite": _paraphrase_tables(),
    }


def to_markdown(catalog: dict) -> str:
    lines = [
        f"# Entity Resolution / Identity — каталог тестов",
        "",
        f"Dataset: `{catalog['dataset']}`",
        f"Всего ER-кейсов: **{catalog['cases_total']}**",
        "",
        "## По типам",
        "",
    ]
    for key, n in catalog["cases_by_type"].items():
        lines.append(f"- `{key}`: {n}")
    lines += ["", "## Символы графа", "", "| uid | name | aliases | kind |", "|---|---|---|---|"]
    for s in catalog["symbols"]:
        aliases = ", ".join(s.get("aliases") or [])
        lines.append(
            f"| `{s['uid']}` | {s['name']} | {aliases or '—'} | {s['kind']} |"
        )
    lines += ["", "## Факты", ""]
    for f in catalog["graph_facts"]:
        lines.append(f"- `{f['subject']}` —{f['relation']}→ `{f['object']}`")

    lines += ["", "## ER cases", ""]
    for case in catalog["cases"]:
        lines.append(
            f"### `{case['case_id']}` ({case['case_type']})"
        )
        lines.append(f"- mention: **{case['mention']}**")
        lines.append(f"- target_uid: `{case['target_uid']}`")
        lines.append(f"- candidates: `{', '.join(case['candidate_uids'])}`")
        lines.append(f"- expected_relation: `{case['expected_relation']}`")
        if case.get("context"):
            lines.append(f"- context: _{case['context']}_")
        if case.get("notes"):
            lines.append(f"- notes: {case['notes']}")
        lines.append("")

    para = catalog["paraphrase_suite"]
    lines += ["## Mini-graph facts (pytest paraphrase)", ""]
    for f in para["mini_facts"]:
        lines.append(f"- {f['subject']} —{f['relation']}→ {f['object']}")
    lines += ["", "## Paraphrase queries", "", "| query | mention | expected |", "|---|---|---|"]
    for row in para["paraphrase_queries"]:
        lines.append(
            f"| {row['query']} | {row['mention']} | {row['expected_canonical']} |"
        )
    lines += ["", "## Negative paraphrases", ""]
    for row in para["negative_paraphrases"]:
        lines.append(
            f"- `{row['mention']}` ≠ `{row['must_not_merge_with']}`"
        )
    lines += [
        "",
        "## Pytest modules",
        "",
        *[f"- `{p}`" for p in catalog["pytest_modules"]],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    json_path = OUT_DIR / "test_catalog.json"
    md_path = OUT_DIR / "test_catalog.md"
    json_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(to_markdown(catalog), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Cases: {catalog['cases_total']}")


if __name__ == "__main__":
    main()
