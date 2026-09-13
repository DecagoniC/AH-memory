"""Compare qwen3:8b (Ollama) vs DeepSeek vs GigaChat on fresh HF corpus metrics.

  python scripts/compare_providers_fresh.py
  python scripts/compare_providers_fresh.py --m1-limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ah_memory.baselines.vanilla_rag import VanillaRAG  # noqa: E402
from ah_memory.benchmarks.challenge.adapters import role_benchmark_items  # noqa: E402
from ah_memory.benchmarks.challenge.loader import load_role_corpus  # noqa: E402
from ah_memory.benchmarks.challenge.role_baseline import UngatedLLMPerception  # noqa: E402
from ah_memory.benchmarks.challenge_evaluation import run_m1_benchmark  # noqa: E402
from ah_memory.benchmarks.challenge_metrics import robustness_gain  # noqa: E402
from ah_memory.config import load_config  # noqa: E402
from ah_memory.deepseek import DeepSeekClient, DeepSeekHybridPerception  # noqa: E402
from ah_memory.eval.gold import openai_hf_gold  # noqa: E402
from ah_memory.eval.m4 import evaluate_m4  # noqa: E402
from ah_memory.examples.openai_hf_incident import (  # noqa: E402
    build_openai_hf_memory,
    openai_hf_text,
)
from ah_memory.gigachat_llm import GigaChatClient, HybridPerception  # noqa: E402
from ah_memory.ollama import (  # noqa: E402
    OllamaClient,
    OllamaHybridPerception,
    is_ollama_available,
)
from ah_memory.agent import Agent  # noqa: E402


FRESH_ROLES = ROOT / "benchmarks" / "fresh" / "openai_hf_2026" / "m1_roles.jsonl"


def _chat_json(client: Any):
    def chat(messages: list[dict[str, str]]) -> str:
        return client.chat(messages, json_mode=True)

    return chat


def _probe(name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = fn()
        return {"ok": True, "detail": out, "sec": round(time.perf_counter() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)[:400], "sec": round(time.perf_counter() - t0, 2)}


def run_probes(cfg) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    probes["ollama"] = _probe(
        "ollama",
        lambda: (
            is_ollama_available(cfg.ollama),
            OllamaClient(cfg.ollama).chat(
                [{"role": "user", "content": "Say only: ok"}],
                json_mode=False,
            )[:120],
        ),
    )
    probes["deepseek"] = _probe(
        "deepseek",
        lambda: DeepSeekClient(cfg.deepseek).chat(
            [{"role": "user", "content": 'Reply JSON: {"ok":true}'}],
            json_mode=True,
        )[:120],
    )
    probes["gigachat"] = _probe(
        "gigachat",
        lambda: GigaChatClient(cfg.gigachat).chat(
            [{"role": "user", "content": 'Ответь JSON: {"ok":true}'}],
            json_mode=True,
        )[:120],
    )
    return probes


def run_m4(cfg) -> dict[str, Any]:
    store = build_openai_hf_memory()
    corpus = openai_hf_text()
    gold = openai_hf_gold()
    agent = Agent(store=store)
    backends: dict[str, VanillaRAG] = {
        "extractive+nomic": VanillaRAG(corpus, top_k=4),
        "ollama:qwen3:8b": VanillaRAG(
            corpus, top_k=4, chat_client=OllamaClient(cfg.ollama)
        ),
        "deepseek-chat": VanillaRAG(
            corpus, top_k=4, chat_client=DeepSeekClient(cfg.deepseek)
        ),
        "gigachat-2-pro": VanillaRAG(
            corpus, top_k=4, chat_client=GigaChatClient(cfg.gigachat)
        ),
    }
    out: dict[str, Any] = {}
    for name, rag in backends.items():
        t0 = time.perf_counter()
        report = evaluate_m4(agent, rag, gold, ticks=6)
        payload = report.as_dict()
        payload["rag_backend"] = rag.backend
        payload["sec"] = round(time.perf_counter() - t0, 2)
        out[name] = payload
    return out


def run_m1_m5(cfg, *, limit: int) -> dict[str, Any]:
    roles_all = role_benchmark_items(load_role_corpus(FRESH_ROLES))
    roles = roles_all[: max(1, limit)]
    ollama_client = OllamaClient(cfg.ollama)
    deepseek_client = DeepSeekClient(cfg.deepseek)
    giga_client = GigaChatClient(cfg.gigachat)

    perceptions = {
        "ah_ollama_8b": OllamaHybridPerception(cfg.ollama, fallback=False),
        "ah_deepseek": DeepSeekHybridPerception(cfg.deepseek, fallback=False),
        "ah_gigachat": HybridPerception(cfg.gigachat, fallback=False),
        "rag_ollama_8b": UngatedLLMPerception(
            _chat_json(ollama_client), backend="ollama_ungated"
        ),
        "rag_deepseek": UngatedLLMPerception(
            _chat_json(deepseek_client), backend="deepseek_ungated"
        ),
        "rag_gigachat": UngatedLLMPerception(
            _chat_json(giga_client), backend="gigachat_ungated"
        ),
    }

    m1: dict[str, Any] = {}
    for name, perception in perceptions.items():
        t0 = time.perf_counter()
        result = run_m1_benchmark(perception, roles, model=name)
        d = result.to_dict()
        m1[name] = {
            "weighted_f1": d["score"]["weighted_f1"],
            "normalized_weighted_f1": d["score"]["normalized_weighted_f1"],
            "sec": round(time.perf_counter() - t0, 2),
            "n": len(roles),
        }

    def _f1(key: str) -> float:
        return float(m1[key]["normalized_weighted_f1"])

    return {
        "n_roles": len(roles),
        "m1": m1,
        "m5_vs_gigachat": {
            "robustness_gain": robustness_gain(
                ah_slm_f1=_f1("ah_ollama_8b"),
                rag_slm_f1=_f1("rag_ollama_8b"),
                ah_llm_f1=_f1("ah_gigachat"),
                rag_llm_f1=_f1("rag_gigachat"),
            ),
            "models": {"slm": cfg.ollama.model, "llm": cfg.gigachat.model},
        },
        "m5_vs_deepseek": {
            "robustness_gain": robustness_gain(
                ah_slm_f1=_f1("ah_ollama_8b"),
                rag_slm_f1=_f1("rag_ollama_8b"),
                ah_llm_f1=_f1("ah_deepseek"),
                rag_llm_f1=_f1("rag_deepseek"),
            ),
            "models": {"slm": cfg.ollama.model, "llm": cfg.deepseek.model},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-limit", type=int, default=20, help="role items for live M1/M5")
    ap.add_argument("--skip-m1", action="store_true")
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "data" / "provider_compare_openai_hf.json",
    )
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.deepseek.configured:
        print("ERROR: DeepSeek not configured", file=sys.stderr)
        return 2
    if not cfg.gigachat.configured:
        print("ERROR: GigaChat not configured", file=sys.stderr)
        return 2
    if not is_ollama_available(cfg.ollama):
        print("ERROR: Ollama not available", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "models": {
            "slm": cfg.ollama.model,
            "deepseek": cfg.deepseek.model,
            "gigachat": cfg.gigachat.model,
            "embed": cfg.ollama.embedding_model,
        },
        "probes": run_probes(cfg),
    }
    print(json.dumps({"probes": report["probes"]}, ensure_ascii=False, indent=2), flush=True)
    if not all(report["probes"][k]["ok"] for k in ("ollama", "deepseek", "gigachat")):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"probe failed; wrote {args.out}", flush=True)
        return 1

    if not args.skip_m4:
        print("running M4…", flush=True)
        report["m4"] = run_m4(cfg)
        print(json.dumps(report["m4"], ensure_ascii=False, indent=2), flush=True)

    if not args.skip_m1:
        print(f"running M1/M5 on {args.m1_limit} roles…", flush=True)
        report["m1_m5"] = run_m1_m5(cfg, limit=args.m1_limit)
        print(json.dumps(report["m1_m5"], ensure_ascii=False, indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
