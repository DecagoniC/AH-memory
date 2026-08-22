"""Run M4 comparison: AH-memory vs Vanilla RAG.

  python scripts/eval_m4.py          # extractive if no key
  python scripts/eval_m4.py --llm    # DeepSeek LLM + FAISS RAG
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ah_memory.eval.gold import build_m4_fixture  # noqa: E402
from ah_memory.eval.m4 import evaluate_m4  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="RAG answers via DeepSeek if key set")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "m4_report.json")
    args = ap.parse_args()

    agent, rag, gold, _ = build_m4_fixture(use_llm=args.llm)
    print(f"RAG backend: {rag.backend}")

    report = evaluate_m4(agent, rag, gold, ticks=args.ticks)
    summary = report.as_dict()
    detail = [
        {
            "q": i.question,
            "ah": i.ah_answer,
            "rag": i.rag_answer,
            "ah_correct": i.ah_correct,
            "rag_correct": i.rag_correct,
            "ah_trace_complete": i.ah_trace_complete,
            "ah_hall": i.ah_hallucinated,
            "rag_hall": i.rag_hallucinated,
            "ah_explain": i.ah_explain,
            "trace": i.ah_trace[:16],
        }
        for i in report.items
    ]
    payload = {"summary": summary, "items": detail, "rag_backend": rag.backend}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0 if summary.get("explain_hypothesis_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
