"""Compare AH-memory vs DeepSeek LLM + vector RAG (M4).

  python scripts/compare_ah_vs_rag.py
  python scripts/compare_ah_vs_rag.py -q "Кто такой заяц?"
  python scripts/compare_ah_vs_rag.py --m4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ah_memory.compare import CompareEngine  # noqa: E402
from ah_memory.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="AH vs БЯМ+RAG")
    ap.add_argument("-q", "--question", type=str, default=None)
    ap.add_argument("--m4", action="store_true", help="полный прогон gold M4")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "compare_report.json")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.deepseek.configured:
        print("ERROR: DEEPSEEK_API_KEY не задан (.env)", file=sys.stderr)
        return 2

    engine = CompareEngine.from_rabbit(cfg.deepseek, ticks=args.ticks)
    print(f"RAG backend: {engine.rag.backend}  model={cfg.deepseek.model}")

    payload: dict = {"model": cfg.deepseek.model, "rag_backend": engine.rag.backend}

    if args.question:
        turn = engine.ask(args.question)
        payload["turn"] = turn.as_dict()
        print("\n=== АГ-память ===")
        print(turn.ah_answer)
        print("trace:", ", ".join(turn.ah_trace_uids[:12]) or "—")
        print("\n=== БЯМ + RAG ===")
        print(turn.rag_answer)
        print("chunks:", len(turn.rag_chunks))

    if args.m4 or not args.question:
        report = engine.run_m4()
        payload["m4"] = {
            "summary": report.as_dict(),
            "items": [
                {
                    "q": i.question,
                    "ah": i.ah_answer,
                    "rag": i.rag_answer,
                    "ah_correct": i.ah_correct,
                    "rag_correct": i.rag_correct,
                    "ah_trace_complete": i.ah_trace_complete,
                    "ah_hall": i.ah_hallucinated,
                    "rag_hall": i.rag_hallucinated,
                    "ah_explain": round(i.ah_explain, 4),
                    "trace": i.ah_trace[:16],
                    "rag_chunks": i.rag_chunks[:2],
                }
                for i in report.items
            ],
        }
        print("\n=== M4 summary ===")
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    summary = (payload.get("m4") or {}).get("summary") or {}
    return 0 if summary.get("explain_hypothesis_ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
