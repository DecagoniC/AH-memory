"""Collect M1 sample, graph dump, and availability flags for the submission bundle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ah_memory.benchmarks.challenge.adapters import role_benchmark_items
from ah_memory.benchmarks.challenge.loader import load_role_corpus
from ah_memory.benchmarks.challenge.role_baseline import UngatedLLMPerception
from ah_memory.benchmarks.challenge_evaluation import run_m1_benchmark
from ah_memory.config import load_config
from ah_memory.deepseek import DeepSeekClient, DeepSeekHybridPerception
from ah_memory.examples.closed_world import (
    build_closed_world_memory,
    closed_world_auto_score,
    closed_world_text,
    extracted_fact_keys,
)
from ah_memory.graph_export import dump_ah_json, dump_graph
from ah_memory.perception import SeedPerception


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "submission"
OUT.mkdir(parents=True, exist_ok=True)


def _shrink(report: dict) -> dict:
    for item in report.get("items", []):
        item.pop("raw_output", None)
    return report


def _score_only(report) -> dict:
    payload = report.to_dict()
    return payload["score"]


def main() -> None:
    store = build_closed_world_memory()
    corpus_path = OUT / "corpus.txt"
    corpus_path.write_text(closed_world_text())
    dump_path = OUT / "ah_dump.json"
    dump_path.write_text(
        json.dumps(dump_ah_json(store), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    graph_path = OUT / "graph_structure.json"
    graph = dump_graph(store, mode="hyper")
    graph_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hit, total = closed_world_auto_score(store)
    meta = {
        "fact_hit": hit,
        "fact_total": total,
        "extracted_keys": sorted(extracted_fact_keys(store)),
        "stats": graph["stats"],
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "hyperedges": len(graph["hyperedges"]),
    }
    (OUT / "dump_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    m4_src = ROOT / "data" / "m4_report.json"
    if m4_src.exists():
        shutil.copyfile(m4_src, OUT / "m4_rabbit_llm.json")

    roles = role_benchmark_items(load_role_corpus())
    seed = run_m1_benchmark(SeedPerception(), roles, model="seed")
    seed_payload = _shrink(seed.to_dict())
    (OUT / "m1_seed_full.json").write_text(
        json.dumps({"score": seed_payload["score"], "n": len(roles)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg = load_config()
    flags = {
        "deepseek": bool(cfg.deepseek.configured),
        "gigachat": bool(cfg.gigachat.configured),
        "ollama": False,
    }
    try:
        from ah_memory.ollama import is_ollama_available

        flags["ollama"] = is_ollama_available(cfg.ollama)
    except Exception as exc:  # noqa: BLE001
        flags["ollama_error"] = f"{type(exc).__name__}: {exc}"

    live: dict = {"flags": flags}
    if cfg.deepseek.configured:
        subset = roles[:8] + roles[25:29] + roles[50:54]
        ah = DeepSeekHybridPerception(cfg.deepseek, fallback=False)
        client = DeepSeekClient(cfg.deepseek)
        rag = UngatedLLMPerception(lambda messages: client.chat(messages, json_mode=True), backend="deepseek_ungated")
        ah_rep = run_m1_benchmark(ah, subset, model=cfg.deepseek.model)
        rag_rep = run_m1_benchmark(rag, subset, model=f"{cfg.deepseek.model}+ungated")
        ah_d = _shrink(ah_rep.to_dict())
        rag_d = _shrink(rag_rep.to_dict())
        (OUT / "m1_deepseek_gated.json").write_text(
            json.dumps(ah_d, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT / "m1_deepseek_ungated.json").write_text(
            json.dumps(rag_d, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        live["m1_sample"] = {
            "n": len(subset),
            "item_ids": [str(it.item_id) for it in subset],
            "ah_gated": ah_d["score"],
            "rag_ungated": rag_d["score"],
        }

    (OUT / "live_flags.json").write_text(
        json.dumps(live, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"meta": meta, "seed_f1": seed_payload["score"]["weighted_f1"], "live": live}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
