from __future__ import annotations

import json

from ah_memory.benchmarks.challenge.ngram import DeterministicNgramEmbedder
from ah_memory.benchmarks.challenge.runner import main


def test_ngram_embedder_is_deterministic_and_fixed_width() -> None:
    embedder = DeterministicNgramEmbedder(dimensions=8, ngram=3)
    first = embedder.embed(["alpha document", "beta"])
    second = embedder.embed(["alpha document", "beta"])
    assert first == second
    assert {len(vector) for vector in first} == {8}
    assert first[0] != first[1]


def test_offline_cli_writes_summary_and_jsonl(tmp_path) -> None:
    exit_code = main(["--root", str(tmp_path)])
    runs = list(tmp_path.iterdir())
    assert exit_code == 0
    assert len(runs) == 1
    summary = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "offline"
    assert "explain_score" in summary["m2"]
    assert "items" not in summary["m2"] or isinstance(summary["m2"]["items"], int)
    assert summary["m3"]["gc_efficiency"] == 1.0
    assert "delta_explainability" in summary["m4"]
    assert (runs[0] / "m2_items.jsonl").exists()
    assert (runs[0] / "m4_ah.jsonl").exists()
    assert (runs[0] / "m4_rag.jsonl").exists()
