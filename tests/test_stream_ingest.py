from __future__ import annotations

import io
import re

from ah_memory.agent import Agent
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.stream_ingest import (
    IngestJournal,
    StreamIngestConfig,
    StreamingGraphIngestor,
    iter_source_blocks,
)


class StaticFactPerception:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, text: str, wm_context=None) -> PerceptionResult:
        self.calls += 1
        candidates = [
            FactCandidate(
                predicate="CONTAINS",
                canonical_relation="CONTAINS",
                raw_relation="содержит",
                raw_span=match.group(0),
                roles={
                    "SUBJECT": match.group(1),
                    "OBJECT": match.group(2),
                },
            )
            for match in re.finditer(
                r"(Объект-\d+) содержит (Элемент-\d+)",
                text,
            )
        ]
        return PerceptionResult(
            kind="fact",
            candidates=candidates,
            seed_tokens=[],
            meta={"backend": "static"},
        )


class EmptyPerception:
    def parse(self, text: str, wm_context=None) -> PerceptionResult:
        return PerceptionResult(
            kind="message",
            candidates=[],
            seed_tokens=[],
            meta={"backend": "empty"},
        )


def _large_fixture(size: int = 40) -> str:
    return "\n\n".join(
        f"Объект-{index} содержит Элемент-{index}."
        for index in range(size)
    )


def test_source_reader_is_bounded_and_preserves_text() -> None:
    source = "слово " * 1000
    blocks = list(
        iter_source_blocks(
            io.StringIO(source),
            max_chars=1024,
            read_size=128,
        )
    )

    assert len(blocks) > 1
    assert max(len(text) for _, text in blocks) <= 1024
    assert "".join(text for _, text in blocks) == source
    assert [start for start, _ in blocks] == [
        sum(len(text) for _, text in blocks[:index])
        for index in range(len(blocks))
    ]


def test_stream_ingest_resumes_and_commits_idempotently(tmp_path) -> None:
    journal = IngestJournal(tmp_path / "jobs.sqlite")
    checkpoint = tmp_path / "graph.json"
    config = StreamIngestConfig(
        max_batch_tokens=32,
        source_block_chars=1024,
        strict_coverage=True,
    )
    first_agent = Agent(perception=StaticFactPerception())
    first = StreamingGraphIngestor(first_agent, journal, config)
    job_id = first.create_job_from_text(
        _large_fixture(),
        source_name="fixture",
        checkpoint_path=checkpoint,
    )

    planned = journal.summary(job_id)
    assert planned.segments == 40
    assert planned.batches > 1

    partial = first.run(job_id, limit=1)
    assert partial.committed == 1
    assert partial.pending == planned.batches - 1
    assert checkpoint.exists()

    restored = StreamingGraphIngestor.load_checkpoint(checkpoint)
    second_agent = Agent(
        store=restored,
        perception=StaticFactPerception(),
    )
    resumed = StreamingGraphIngestor(
        second_agent,
        journal,
        config,
    )
    completed = resumed.run(job_id)

    assert completed.status == "complete"
    assert completed.committed == planned.batches
    assert completed.coverage_ratio == 1.0
    assert len(second_agent.store.list_semantic_factors()) == 40
    size = second_agent.store.graph_size()

    repeated = resumed.run(job_id)
    assert repeated.status == "complete"
    assert second_agent.store.graph_size() == size
    assert any(
        factor.metadata.get("candidate_metadata", {})
        .get("provenance", {})
        .get("job_id")
        == job_id
        for factor in second_agent.store.list_semantic_factors()
    )


def test_strict_coverage_stages_failure_without_graph_mutation(
    tmp_path,
) -> None:
    journal = IngestJournal(tmp_path / "jobs.sqlite")
    agent = Agent(perception=EmptyPerception())
    ingestor = StreamingGraphIngestor(
        agent,
        journal,
        StreamIngestConfig(
            max_batch_tokens=32,
            source_block_chars=1024,
            strict_coverage=True,
            max_attempts=1,
        ),
    )
    job_id = ingestor.create_job_from_text(
        "Объект-1 содержит Элемент-1.",
        checkpoint_path=tmp_path / "graph.json",
    )

    result = ingestor.run(job_id)

    assert result.status == "partial"
    assert result.failed == 1
    assert len(agent.store.list_semantic_factors()) == 0


def test_commit_retry_reuses_staged_candidates(tmp_path) -> None:
    class FailCheckpointOnce(StreamingGraphIngestor):
        failures = 1

        def _save_checkpoint(self, job_id: str) -> None:
            if self.failures:
                self.failures -= 1
                raise OSError("simulated checkpoint failure")
            super()._save_checkpoint(job_id)

    perception = StaticFactPerception()
    journal = IngestJournal(tmp_path / "jobs.sqlite")
    ingestor = FailCheckpointOnce(
        Agent(perception=perception),
        journal,
        StreamIngestConfig(
            max_batch_tokens=32,
            source_block_chars=1024,
            max_attempts=3,
        ),
    )
    job_id = ingestor.create_job_from_text(
        "Объект-1 содержит Элемент-1.",
        checkpoint_path=tmp_path / "graph.json",
    )

    first = ingestor.run(job_id, limit=1)
    assert first.extracted == 1
    assert perception.calls == 1
    assert len(ingestor.agent.store.list_semantic_factors()) == 0

    second = ingestor.run(job_id)
    assert second.status == "complete"
    assert second.committed == 1
    assert perception.calls == 1
