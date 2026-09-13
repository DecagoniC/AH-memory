"""Ingest one HF-incident graph with LLM or SLM; both models answer from it.

  python scripts/compare_shared_graph.py
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

from ah_memory.agent import Agent  # noqa: E402
from ah_memory.compare import COMPARE_GENERATION_SYSTEM  # noqa: E402
from ah_memory.config import load_config  # noqa: E402
from ah_memory.deepseek import DeepSeekClient, DeepSeekHybridPerception  # noqa: E402
from ah_memory.eval.corpus_ingest import ingest_text_batches, split_batches  # noqa: E402
from ah_memory.eval.gold import openai_hf_gold  # noqa: E402
from ah_memory.eval.hypothesis import ah_answer_is_abstain  # noqa: E402
from ah_memory.eval.m4 import GoldItem  # noqa: E402
from ah_memory.examples.openai_hf_incident import openai_hf_text  # noqa: E402
from ah_memory.ollama import (  # noqa: E402
    OllamaClient,
    OllamaHybridPerception,
    is_ollama_available,
)
from ah_memory.perception import SeedPerception  # noqa: E402
from ah_memory.store import AHStore  # noqa: E402


def _keyword_hit(answer: str, keywords: list[str]) -> bool:
    blob = answer.lower().replace("ё", "е")
    return any(k.lower().replace("ё", "е") in blob for k in keywords if k)


def _chat(client: Any, question: str, context: str) -> str:
    if not context.strip() or context.strip() == "неизвестно":
        return "неизвестно"
    return client.chat(
        [
            {"role": "system", "content": COMPARE_GENERATION_SYSTEM},
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос: {question}",
            },
        ],
        json_mode=False,
    ).strip()


def _graph_context(agent: Agent, question: str, *, ticks: int) -> tuple[Any, str]:
    reply = agent.ask(question, ticks=ticks)
    cards = agent.perception_context(question, limit=16)
    parts: list[str] = []
    if reply.answer and reply.answer.strip():
        parts.append(f"Graph answer: {reply.answer.strip()}")
    if reply.trace_uids:
        parts.append("Activated: " + ", ".join(reply.trace_uids[:24]))
    if cards:
        parts.append("Facts:\n" + "\n".join(cards))
    context = "\n\n".join(parts).strip() or "неизвестно"
    return reply, context


def _score(answer: str, gold: GoldItem, *, trace_ok: bool, d_max: int) -> dict[str, Any]:
    correct = _keyword_hit(answer, gold.answer_keywords)
    abstain = ah_answer_is_abstain(answer)
    trap = gold.evidence_spans == ["__none__"]
    hall = (not abstain) if trap else (not correct and not abstain)
    explain = (1.0 if correct else 0.0) * (gold.d / d_max) * (1.0 if trace_ok else 0.0)
    return {
        "answer": answer,
        "correct": correct,
        "abstain": abstain,
        "hallucinated": hall,
        "explain": round(explain, 4),
    }


def _graph_stats(store: AHStore) -> dict[str, int]:
    return {
        "S": len(store.ah.S),
        "C": len(store.ah.C),
        "L": len(store.ah.L),
        "semantic_factors": len(store.semantic_factors),
        "events": len(store.events),
        "graph_size": store.graph_size(),
    }


def ingest_graph(cfg, *, builder: str, max_chars: int, repair: bool) -> dict[str, Any]:
    text = openai_hf_text()
    batches = split_batches(text, max_chars=max_chars)
    if builder == "llm":
        perception = DeepSeekHybridPerception(cfg.deepseek, fallback=False)
        builder_name = cfg.deepseek.model
    elif builder == "slm":
        perception = OllamaHybridPerception(cfg.ollama, fallback=False)
        builder_name = cfg.ollama.model
    else:
        raise ValueError(f"unknown builder: {builder}")
    agent = Agent(store=AHStore(), perception=perception)
    t0 = time.perf_counter()
    records = ingest_text_batches(agent, batches, repair_uncovered=repair)
    ingest_sec = round(time.perf_counter() - t0, 2)
    n_cand = sum(len(r.candidates) for r in records)
    coverage = [r.coverage_ratio for r in records]
    return {
        "builder": builder,
        "builder_model": builder_name,
        "batches": len(batches),
        "candidates": n_cand,
        "mean_coverage": round(sum(coverage) / max(1, len(coverage)), 4),
        "ingest_sec": ingest_sec,
        "graph": _graph_stats(agent.store),
        "agent": agent,
    }


def answer_from_graph(
    agent: Agent,
    clients: dict[str, Any],
    gold: list[GoldItem],
    *,
    ticks: int,
) -> dict[str, Any]:
    agent.perception = SeedPerception()
    d_max = max((g.d for g in gold), default=1) or 1
    items: list[dict[str, Any]] = []
    for g in gold:
        raw, context = _graph_context(agent, g.question, ticks=ticks)
        trace_ok = bool(raw.trace_uids) if not g.gold_trace else True
        if g.gold_trace:
            tset = {u.lower() for u in raw.trace_uids}
            expanded = set(tset)
            for uid in list(tset):
                if uid.startswith("m_"):
                    expanded.add(uid[2:])
                else:
                    expanded.add(f"m_{uid}")
            trace_ok = all(
                g_uid.lower() in expanded or g_uid.lower().removeprefix("m_") in expanded
                for g_uid in g.gold_trace
            )
        row: dict[str, Any] = {
            "question": g.question,
            "graph_raw": _score(raw.answer, g, trace_ok=trace_ok, d_max=d_max),
            "context_chars": len(context),
            "trace": list(raw.trace_uids)[:16],
            "trace_complete": trace_ok,
        }
        row["graph_raw"]["answer"] = raw.answer
        for name, client in clients.items():
            t0 = time.perf_counter()
            generated = _chat(client, g.question, context)
            scored = _score(generated, g, trace_ok=trace_ok, d_max=d_max)
            scored["sec"] = round(time.perf_counter() - t0, 2)
            row[name] = scored
        items.append(row)

    def _mean(field: str, arm: str) -> float:
        return round(
            sum(float(item[arm][field]) for item in items) / max(1, len(items)),
            4,
        )

    arms = ["graph_raw", *clients]
    summary = {
        arm: {
            "correct": _mean("correct", arm),
            "hallucinated": _mean("hallucinated", arm),
            "explain": _mean("explain", arm),
            "abstain": _mean("abstain", arm),
        }
        for arm in arms
    }
    return {"d_max": d_max, "n": len(items), "summary": summary, "items": items}


def run_one(cfg, *, builder: str, max_chars: int, repair: bool, ticks: int) -> dict[str, Any]:
    print(f"ingesting graph with {builder}…", flush=True)
    packed = ingest_graph(cfg, builder=builder, max_chars=max_chars, repair=repair)
    agent = packed.pop("agent")
    print(json.dumps({k: packed[k] for k in packed}, ensure_ascii=True), flush=True)
    clients = {
        "slm": OllamaClient(cfg.ollama),
        "llm": DeepSeekClient(cfg.deepseek),
    }
    print("answering from the shared graph…", flush=True)
    qa = answer_from_graph(agent, clients, openai_hf_gold(), ticks=ticks)
    print(json.dumps(qa["summary"], ensure_ascii=True, indent=2), flush=True)
    return {**packed, "qa": qa}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1400)
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "data" / "shared_graph_llm_slm.json",
    )
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config()
    if not cfg.deepseek.configured:
        print("ERROR: DeepSeek not configured", file=sys.stderr)
        return 2
    if not is_ollama_available(cfg.ollama):
        print("ERROR: Ollama not available", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "protocol": (
            "One graph per builder. Both SLM and LLM generate answers "
            "from that same graph context only (no RAG corpus)."
        ),
        "models": {"slm": cfg.ollama.model, "llm": cfg.deepseek.model},
        "runs": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for builder in ("llm", "slm"):
        report["runs"][builder] = run_one(
            cfg,
            builder=builder,
            max_chars=args.max_chars,
            repair=not args.no_repair,
            ticks=args.ticks,
        )
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"checkpoint {args.out} after {builder}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
