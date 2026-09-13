"""CLI for resumable streaming text-to-graph ingestion."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ah_memory.agent import Agent
from ah_memory.config import load_config
from ah_memory.deepseek import DeepSeekHybridPerception
from ah_memory.gigachat_llm import HybridPerception
from ah_memory.identity import build_identity_service
from ah_memory.perception import SeedPerception
from ah_memory.store import AHStore
from ah_memory.stream_ingest import (
    IngestJournal,
    StreamIngestConfig,
    StreamingGraphIngestor,
)


def _build_agent(store: AHStore | None = None) -> Agent:
    config = load_config()
    target = store or AHStore()
    if not config.agent.use_llm:
        perception = SeedPerception()
    elif (
        config.agent.llm_provider == "deepseek"
        and config.deepseek.configured
    ):
        perception = DeepSeekHybridPerception(
            config.deepseek,
            fallback=config.agent.fallback_rules,
        )
    elif (
        config.agent.llm_provider == "gigachat"
        and config.gigachat.configured
    ):
        perception = HybridPerception(
            config.gigachat,
            fallback=config.agent.fallback_rules,
        )
    else:
        perception = SeedPerception()
    identity = build_identity_service(
        target,
        enabled=config.identity.enabled,
        use_embeddings=False,
        safety_threshold=config.identity.safety_threshold,
        margin=config.identity.margin,
    )
    return Agent(
        store=target,
        perception=perception,
        identity=identity,
    )


def _print_summary(summary) -> None:
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


def _start(args: argparse.Namespace) -> int:
    source = Path(args.source)
    journal = IngestJournal(args.journal)
    config = StreamIngestConfig(
        max_batch_tokens=args.max_batch_tokens,
        source_block_chars=args.source_block_chars,
        repair_uncovered=not args.no_repair,
        strict_coverage=args.strict_coverage,
        max_attempts=args.max_attempts,
        context_card_limit=args.context_card_limit,
        batch_page_size=args.batch_page_size,
    )
    agent = _build_agent()
    ingestor = StreamingGraphIngestor(agent, journal, config)
    checkpoint = Path(args.output or source.with_suffix(".ah.json"))
    job_id = ingestor.create_job_from_path(
        source,
        checkpoint_path=checkpoint,
    )
    print(f"job_id={job_id}")
    _print_summary(ingestor.run(job_id, limit=args.limit))
    return 0


def _resume(args: argparse.Namespace) -> int:
    journal = IngestJournal(args.journal)
    job = journal.job(args.job_id)
    config = StreamIngestConfig(**job["config"])
    checkpoint = Path(job["checkpoint_path"])
    store = (
        StreamingGraphIngestor.load_checkpoint(checkpoint)
        if checkpoint.exists()
        else AHStore()
    )
    ingestor = StreamingGraphIngestor(
        _build_agent(store),
        journal,
        config,
    )
    _print_summary(ingestor.run(args.job_id, limit=args.limit))
    return 0


def _status(args: argparse.Namespace) -> int:
    journal = IngestJournal(args.journal)
    _print_summary(journal.summary(args.job_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream arbitrarily large UTF-8 text into an AH graph",
    )
    parser.add_argument(
        "--journal",
        default=".ah-ingest.sqlite",
        help="SQLite job journal",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="plan and run a new job")
    start.add_argument("source", help="UTF-8 source text file")
    start.add_argument("--output", help="graph checkpoint JSON")
    start.add_argument("--max-batch-tokens", type=int, default=1800)
    start.add_argument("--source-block-chars", type=int, default=65536)
    start.add_argument("--max-attempts", type=int, default=3)
    start.add_argument("--context-card-limit", type=int, default=24)
    start.add_argument("--batch-page-size", type=int, default=64)
    start.add_argument("--strict-coverage", action="store_true")
    start.add_argument("--no-repair", action="store_true")
    start.add_argument("--limit", type=int)
    start.set_defaults(handler=_start)

    resume = subparsers.add_parser("resume", help="resume a staged job")
    resume.add_argument("job_id")
    resume.add_argument("--limit", type=int)
    resume.set_defaults(handler=_resume)

    status = subparsers.add_parser("status", help="show job status")
    status.add_argument("job_id")
    status.set_defaults(handler=_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
