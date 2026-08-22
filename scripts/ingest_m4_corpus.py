"""Fetch or ingest the M4 closed-world corpus.

  python scripts/ingest_m4_corpus.py --fetch
  python scripts/ingest_m4_corpus.py --ingest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ah_memory.agent import Agent  # noqa: E402
from ah_memory.config import load_config  # noqa: E402
from ah_memory.deepseek import DeepSeekHybridPerception  # noqa: E402
from ah_memory.eval.corpus_ingest import (  # noqa: E402
    ingest_text_batches,
    records_to_payload,
    split_batches,
)
from ah_memory.examples.closed_world import closed_world_text  # noqa: E402
from ah_memory.store import AHStore  # noqa: E402

CORPUS_PATH = ROOT / "benchmarks" / "m4" / "closed_world.txt"
FACTS_PATH = ROOT / "benchmarks" / "m4" / "closed_world_facts.json"
WIKI_API = "https://ru.wikipedia.org/w/api.php"
WIKI_TITLE = "Тиманский кряж"


def fetch_wikipedia() -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": WIKI_TITLE,
    }
    headers = {"User-Agent": "AH-memory-m4-corpus/1.0 (local benchmark ingest)"}
    with httpx.Client(timeout=60.0, headers=headers) as client:
        data = client.get(WIKI_API, params=params).json()
    pages = data["query"]["pages"]
    extract = next(iter(pages.values()))["extract"]
    extract = extract.replace("\u0301", "")
    cut = re.split(r"\n\s*\n(?=См\. также|Примечания|Литература|Ссылки\b)", extract, maxsplit=1)
    body = cut[0].strip()
    header = (
        f"Источник: русская Википедия, статья «{WIKI_TITLE}» "
        "(CC BY-SA 4.0), выгрузка TextExtracts без разделов литературы и ссылок.\n"
    )
    return header + "\n" + body + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="скачать текст в closed_world.txt")
    ap.add_argument("--ingest", action="store_true", help="прогнать батчи через LLM → граф")
    ap.add_argument("--max-chars", type=int, default=700)
    args = ap.parse_args()
    if not args.fetch and not args.ingest:
        args.fetch = True
        args.ingest = True

    if args.fetch:
        text = fetch_wikipedia()
        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CORPUS_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {CORPUS_PATH} ({len(text)} chars)")

    if not args.ingest:
        return 0

    cfg = load_config()
    if not cfg.deepseek.configured:
        print("ERROR: DEEPSEEK_API_KEY не задан", file=sys.stderr)
        return 2
    text = closed_world_text()
    batches = split_batches(text, max_chars=args.max_chars)
    print(f"batches: {len(batches)}")
    agent = Agent(
        store=AHStore(),
        perception=DeepSeekHybridPerception(cfg.deepseek, fallback=False),
    )
    records = ingest_text_batches(agent, batches)
    payload = records_to_payload(records, source=str(CORPUS_PATH.as_posix()))
    payload["graph"] = {
        "semantic_factors": len(agent.store.semantic_factors),
        "events": len(agent.store.events),
        "S": len(agent.store.ah.S),
        "graph_size": agent.store.graph_size(),
    }
    coverage = payload["coverage"]
    if coverage["uncovered"]:
        print(
            "ERROR: incomplete extraction: "
            f"{coverage['covered']}/{coverage['segments']} segments covered",
            file=sys.stderr,
        )
        print(
            json.dumps(coverage["uncovered"], ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 3
    FACTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    from ah_memory.graph_library import remember_fixture

    remember_fixture("closed-world", agent.store, source_text=closed_world_text())
    print(json.dumps(payload["graph"], ensure_ascii=False))
    print(f"coverage: {coverage['covered']}/{coverage['segments']}")
    print(f"candidates: {payload['n_candidates']}")
    print(f"wrote {FACTS_PATH}")
    return 0 if payload["n_candidates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
