"""Resumable, disk-backed streaming text-to-graph ingestion."""
from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from ah_memory.agent import Agent
from ah_memory.eval.corpus_ingest import (
    candidate_to_dict,
    dict_to_candidate,
    prepare_text_batch,
    split_atomic_segments,
)
from ah_memory.graph_export import dump_ah_json
from ah_memory.morph import seeds_from_roles
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store_codec import restore_store
from ah_memory.store_codec import snapshot_store
from ah_memory.types import Section

_TOKEN = re.compile(r"[\w-]+", re.UNICODE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class StreamIngestConfig:
    max_batch_tokens: int = 1800
    source_block_chars: int = 65536
    repair_uncovered: bool = True
    strict_coverage: bool = False
    max_attempts: int = 3
    context_card_limit: int = 24
    batch_page_size: int = 64
    transactional_batches: bool = True

    def __post_init__(self) -> None:
        if self.max_batch_tokens < 32:
            raise ValueError("max_batch_tokens must be >= 32")
        if self.source_block_chars < 1024:
            raise ValueError("source_block_chars must be >= 1024")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.context_card_limit < 0:
            raise ValueError("context_card_limit must be >= 0")
        if self.batch_page_size < 1:
            raise ValueError("batch_page_size must be >= 1")


@dataclass(frozen=True)
class IngestJobSummary:
    job_id: str
    status: str
    source_name: str
    documents: int
    segments: int
    batches: int
    pending: int
    extracted: int
    committed: int
    failed: int
    candidates: int
    coverage_ratio: float
    checkpoint_path: str


class IngestJournal:
    """SQLite journal for source segments, staged candidates and commits."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    source_name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    chars INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    context TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    batch_id INTEGER,
                    UNIQUE(document_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    fingerprint TEXT NOT NULL,
                    text TEXT NOT NULL,
                    segment_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    backend TEXT,
                    candidates_json TEXT,
                    coverage_json TEXT,
                    created_uids_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_segments_job
                    ON segments(job_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_batches_job_status
                    ON batches(job_id, status, batch_id);
                """
            )

    def create_job(
        self,
        source_name: str,
        config: StreamIngestConfig,
        checkpoint_path: str | Path,
        *,
        job_id: str | None = None,
    ) -> str:
        identifier = job_id or uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO jobs(
                    job_id, status, source_name, config_json,
                    checkpoint_path, created_at, updated_at
                ) VALUES (?, 'planning', ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    source_name,
                    json.dumps(asdict(config), ensure_ascii=False),
                    str(Path(checkpoint_path)),
                    now,
                    now,
                ),
            )
        return identifier

    def begin_document(
        self,
        job_id: str,
        source_name: str,
    ) -> str:
        document_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO documents(
                    document_id, job_id, source_name, checksum, chars, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    job_id,
                    source_name,
                    "",
                    0,
                    _now(),
                ),
            )
        return document_id

    def append_segments(
        self,
        job_id: str,
        document_id: str,
        segments: Iterable[dict[str, Any]],
    ) -> None:
        rows = list(segments)
        if not rows:
            return
        with self._connect() as db:
            db.executemany(
                """
                INSERT INTO segments(
                    job_id, document_id, ordinal, start_offset, end_offset,
                    text, context, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        document_id,
                        int(row["ordinal"]),
                        int(row["start"]),
                        int(row["end"]),
                        str(row["text"]),
                        str(row["context"]),
                        str(row["kind"]),
                    )
                    for row in rows
                ],
            )

    def finish_document(
        self,
        document_id: str,
        *,
        checksum: str,
        chars: int,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE documents SET checksum = ?, chars = ?
                WHERE document_id = ?
                """,
                (checksum, chars, document_id),
            )

    def plan_batches(
        self,
        job_id: str,
        *,
        max_batch_tokens: int,
    ) -> int:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT segment_id, context
                FROM segments
                WHERE job_id = ? AND batch_id IS NULL
                ORDER BY document_id, ordinal
                """,
                (job_id,),
            )
            current: list[sqlite3.Row] = []
            tokens = 0
            planned = 0
            now = _now()

            def write_group(group: list[sqlite3.Row]) -> None:
                nonlocal planned
                if not group:
                    return
                segment_ids = [int(row["segment_id"]) for row in group]
                text = "\n\n".join(
                    str(row["context"]).strip()
                    for row in group
                    if str(row["context"]).strip()
                )
                fingerprint = hashlib.sha256(
                    json.dumps(
                        [segment_ids, text],
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                cursor = db.execute(
                    """
                    INSERT INTO batches(
                        job_id, fingerprint, text, segment_ids_json,
                        status, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        job_id,
                        fingerprint,
                        text,
                        json.dumps(segment_ids),
                        now,
                    ),
                )
                batch_id = int(cursor.lastrowid)
                db.executemany(
                    "UPDATE segments SET batch_id = ? WHERE segment_id = ?",
                    [(batch_id, segment_id) for segment_id in segment_ids],
                )
                planned += 1

            for row in rows:
                count = max(1, len(_TOKEN.findall(str(row["context"]))))
                if current and tokens + count > max_batch_tokens:
                    write_group(current)
                    current = []
                    tokens = 0
                current.append(row)
                tokens += count
            if current:
                write_group(current)
            db.execute(
                "UPDATE jobs SET status = 'pending', updated_at = ? WHERE job_id = ?",
                (now, job_id),
            )
        return planned

    def job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        return result

    def runnable_batches(
        self,
        job_id: str,
        *,
        max_attempts: int,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM batches
            WHERE job_id = ?
              AND (
                status = 'pending'
                OR (status IN ('extracted', 'failed') AND attempts < ?)
              )
            ORDER BY batch_id
        """
        params: list[Any] = [job_id, max_attempts]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute(query, params).fetchall()
            ]

    def batch_segments(self, batch_id: int) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    """
                    SELECT segment_id, document_id, ordinal, start_offset,
                           end_offset, text, kind
                    FROM segments
                    WHERE batch_id = ?
                    ORDER BY ordinal
                    """,
                    (batch_id,),
                ).fetchall()
            ]

    def boundary_context(self, batch_id: int) -> list[str]:
        with self._connect() as db:
            bounds = db.execute(
                """
                SELECT document_id, MIN(ordinal) AS first_ordinal,
                       MAX(ordinal) AS last_ordinal
                FROM segments WHERE batch_id = ? GROUP BY document_id
                """,
                (batch_id,),
            ).fetchall()
            context: list[str] = []
            for bound in bounds:
                rows = db.execute(
                    """
                    SELECT text FROM segments
                    WHERE document_id = ?
                      AND ordinal IN (?, ?)
                    ORDER BY ordinal
                    """,
                    (
                        bound["document_id"],
                        int(bound["first_ordinal"]) - 1,
                        int(bound["last_ordinal"]) + 1,
                    ),
                ).fetchall()
                context.extend(str(row["text"]) for row in rows)
        return context

    def mark_processing(self, batch_id: int) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE batches
                SET status = 'processing', attempts = attempts + 1,
                    error = NULL, updated_at = ?
                WHERE batch_id = ?
                """,
                (_now(), batch_id),
            )

    def stage_extraction(
        self,
        batch_id: int,
        *,
        backend: str,
        candidates: list[dict[str, Any]],
        coverage: dict[str, Any],
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE batches
                SET status = 'extracted', backend = ?, candidates_json = ?,
                    coverage_json = ?, error = NULL, updated_at = ?
                WHERE batch_id = ?
                """,
                (
                    backend,
                    json.dumps(candidates, ensure_ascii=False),
                    json.dumps(coverage, ensure_ascii=False),
                    _now(),
                    batch_id,
                ),
            )

    def mark_committed(
        self,
        batch_id: int,
        created_uids: list[str],
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE batches
                SET status = 'committed', created_uids_json = ?,
                    error = NULL, updated_at = ?
                WHERE batch_id = ?
                """,
                (
                    json.dumps(created_uids, ensure_ascii=False),
                    _now(),
                    batch_id,
                ),
            )

    def mark_failed(self, batch_id: int, error: str) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE batches
                SET status = 'failed', error = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (error[:4000], _now(), batch_id),
            )

    def mark_commit_retry(self, batch_id: int, error: str) -> None:
        """Keep staged candidates so resume does not call the LLM again."""
        with self._connect() as db:
            db.execute(
                """
                UPDATE batches
                SET status = 'extracted', error = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (error[:4000], _now(), batch_id),
            )

    def refresh_job_status(self, job_id: str) -> str:
        with self._connect() as db:
            counts = {
                str(row["status"]): int(row["n"])
                for row in db.execute(
                    """
                    SELECT status, COUNT(*) AS n
                    FROM batches WHERE job_id = ? GROUP BY status
                    """,
                    (job_id,),
                ).fetchall()
            }
            if counts.get("pending", 0) or counts.get("extracted", 0):
                status = "pending"
            elif counts.get("processing", 0):
                status = "running"
            elif counts.get("failed", 0):
                status = "partial"
            else:
                status = "complete"
            db.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status, _now(), job_id),
            )
        return status

    def fail_exhausted(self, job_id: str, max_attempts: int) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE batches
                SET status = 'failed',
                    error = COALESCE(error, 'maximum attempts exceeded'),
                    updated_at = ?
                WHERE job_id = ? AND status = 'extracted' AND attempts >= ?
                """,
                (_now(), job_id, max_attempts),
            )

    def summary(self, job_id: str) -> IngestJobSummary:
        job = self.job(job_id)
        with self._connect() as db:
            documents = int(
                db.execute(
                    "SELECT COUNT(*) FROM documents WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
            segments = int(
                db.execute(
                    "SELECT COUNT(*) FROM segments WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
            rows = db.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM batches WHERE job_id = ? GROUP BY status
                """,
                (job_id,),
            ).fetchall()
            counts = {str(row["status"]): int(row["n"]) for row in rows}
            staged = db.execute(
                """
                SELECT candidates_json, coverage_json FROM batches
                WHERE job_id = ? AND candidates_json IS NOT NULL
                """,
                (job_id,),
            ).fetchall()
        candidate_count = 0
        covered = 0
        coverage_segments = 0
        for row in staged:
            candidate_count += len(json.loads(row["candidates_json"]))
            coverage = json.loads(row["coverage_json"] or "{}")
            coverage_segments += int(coverage.get("segments") or 0)
            covered += int(coverage.get("covered") or 0)
        return IngestJobSummary(
            job_id=job_id,
            status=str(job["status"]),
            source_name=str(job["source_name"]),
            documents=documents,
            segments=segments,
            batches=sum(counts.values()),
            pending=counts.get("pending", 0) + counts.get("processing", 0),
            extracted=counts.get("extracted", 0),
            committed=counts.get("committed", 0),
            failed=counts.get("failed", 0),
            candidates=candidate_count,
            coverage_ratio=(
                covered / coverage_segments
                if coverage_segments
                else 1.0
            ),
            checkpoint_path=str(job["checkpoint_path"]),
        )


def iter_source_blocks(
    reader: TextIO,
    *,
    max_chars: int = 65536,
    read_size: int = 16384,
) -> Iterable[tuple[int, str]]:
    """Yield bounded source blocks while preserving absolute offsets."""
    buffer = ""
    absolute = 0
    while True:
        chunk = reader.read(read_size)
        if chunk:
            buffer += chunk
        while buffer and (len(buffer) > max_chars or not chunk):
            if len(buffer) <= max_chars:
                cut = len(buffer)
            else:
                window = buffer[:max_chars]
                paragraph = window.rfind("\n\n")
                sentence = max(
                    window.rfind(". "),
                    window.rfind("! "),
                    window.rfind("? "),
                    window.rfind("\n"),
                )
                whitespace = window.rfind(" ")
                minimum = max_chars // 2
                if paragraph >= minimum:
                    cut = paragraph + 2
                elif sentence >= minimum:
                    cut = sentence + 1
                elif whitespace >= minimum:
                    cut = whitespace + 1
                else:
                    cut = max_chars
            block = buffer[:cut]
            yield absolute, block
            buffer = buffer[cut:]
            absolute += cut
        if not chunk:
            break


class StreamingGraphIngestor:
    """Plan, stage and commit an arbitrarily large text source."""

    def __init__(
        self,
        agent: Agent,
        journal: IngestJournal,
        config: StreamIngestConfig | None = None,
    ) -> None:
        self.agent = agent
        self.journal = journal
        self.config = config or StreamIngestConfig()

    def create_job_from_text(
        self,
        text: str,
        *,
        source_name: str = "text",
        checkpoint_path: str | Path | None = None,
        job_id: str | None = None,
    ) -> str:
        return self.create_job_from_reader(
            io.StringIO(text),
            source_name=source_name,
            checkpoint_path=checkpoint_path,
            job_id=job_id,
        )

    def create_job_from_path(
        self,
        path: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        job_id: str | None = None,
        encoding: str = "utf-8",
    ) -> str:
        source = Path(path)
        with source.open("r", encoding=encoding) as reader:
            return self.create_job_from_reader(
                reader,
                source_name=str(source),
                checkpoint_path=checkpoint_path,
                job_id=job_id,
            )

    def create_job_from_reader(
        self,
        reader: TextIO,
        *,
        source_name: str,
        checkpoint_path: str | Path | None = None,
        job_id: str | None = None,
    ) -> str:
        checkpoint = Path(
            checkpoint_path
            or Path(tempfile.gettempdir()) / f"ah-ingest-{job_id or uuid.uuid4().hex}.json"
        )
        identifier = self.journal.create_job(
            source_name,
            self.config,
            checkpoint,
            job_id=job_id,
        )
        document_id = self.journal.begin_document(
            identifier,
            source_name,
        )
        digest = hashlib.sha256()
        ordinal = 0
        chars = 0
        for block_start, block in iter_source_blocks(
            reader,
            max_chars=self.config.source_block_chars,
        ):
            digest.update(block.encode("utf-8"))
            chars += len(block)
            rows: list[dict[str, Any]] = []
            for segment in split_atomic_segments(block):
                rows.append(
                    {
                        "ordinal": ordinal,
                        "start": block_start + segment.start,
                        "end": block_start + segment.end,
                        "text": segment.text,
                        "context": segment.context,
                        "kind": segment.kind,
                    }
                )
                ordinal += 1
            self.journal.append_segments(
                identifier,
                document_id,
                rows,
            )
        self.journal.finish_document(
            document_id,
            checksum=digest.hexdigest(),
            chars=chars,
        )
        self.journal.plan_batches(
            identifier,
            max_batch_tokens=self.config.max_batch_tokens,
        )
        return identifier

    def run(
        self,
        job_id: str,
        *,
        limit: int | None = None,
    ) -> IngestJobSummary:
        processed = 0
        while limit is None or processed < limit:
            page_limit = self.config.batch_page_size
            if limit is not None:
                page_limit = min(page_limit, limit - processed)
            batches = self.journal.runnable_batches(
                job_id,
                max_attempts=self.config.max_attempts,
                limit=page_limit,
            )
            if not batches:
                break
            for batch in batches:
                self._process_batch(job_id, batch)
                processed += 1
        self.journal.fail_exhausted(
            job_id,
            self.config.max_attempts,
        )
        self.journal.refresh_job_status(job_id)
        return self.journal.summary(job_id)

    def _process_batch(
        self,
        job_id: str,
        batch: dict[str, Any],
    ) -> None:
        batch_id = int(batch["batch_id"])
        staged = batch["status"] == "extracted"
        rollback_payload: dict[str, Any] | None = None
        try:
            self.journal.mark_processing(batch_id)
            if batch["status"] != "extracted":
                prepared = prepare_text_batch(
                    self.agent,
                    str(batch["text"]),
                    repair_uncovered=self.config.repair_uncovered,
                    wm_context=self._context_for_batch(
                        batch_id,
                        str(batch["text"]),
                    ),
                )
                if (
                    self.config.strict_coverage
                    and prepared.uncovered_segments
                ):
                    raise ValueError(
                        "incomplete coverage: "
                        f"{len(prepared.uncovered_segments)} segments"
                    )
                source_segments = self.journal.batch_segments(batch_id)
                candidates = [
                    self._with_provenance(
                        candidate,
                        job_id=job_id,
                        batch_id=batch_id,
                        source_segments=source_segments,
                    )
                    for candidate in prepared.candidates
                ]
                coverage = {
                    "segments": len(prepared.segments),
                    "covered": sum(
                        bool(segment.get("covered"))
                        for segment in prepared.segments
                    ),
                    "ratio": prepared.coverage_ratio,
                    "uncovered": prepared.uncovered_segments,
                    "repair_attempts": prepared.repair_attempts,
                }
                candidate_payloads = [
                    candidate_to_dict(candidate)
                    for candidate in candidates
                ]
                self.journal.stage_extraction(
                    batch_id,
                    backend=prepared.backend,
                    candidates=candidate_payloads,
                    coverage=coverage,
                )
                staged = True
            else:
                candidate_payloads = json.loads(
                    batch["candidates_json"] or "[]"
                )
            candidates_now = [
                dict_to_candidate(payload)
                for payload in candidate_payloads
            ]
            perception = PerceptionResult(
                kind="fact",
                candidates=candidates_now,
                seed_tokens=seeds_from_roles(candidates_now),
                meta={
                    "backend": "stream_ingest",
                    "job_id": job_id,
                    "batch_id": batch_id,
                },
            )
            if self.config.transactional_batches:
                rollback_payload = snapshot_store(self.agent.store)
            report = self.agent.ingest(
                str(batch["text"]),
                section=Section.C,
                perception=perception,
            )
            if report.skipped:
                raise ValueError(
                    "batch commit skipped candidates: "
                    + "; ".join(report.skipped)
                )
            self._save_checkpoint(job_id)
            self.journal.mark_committed(
                batch_id,
                list(report.created_n),
            )
        except Exception as exc:  # noqa: BLE001
            if rollback_payload is not None:
                self.agent.adopt_store(
                    restore_store(rollback_payload)
                )
            if staged:
                self.journal.mark_commit_retry(batch_id, str(exc))
            else:
                self.journal.mark_failed(batch_id, str(exc))

    def _save_checkpoint(self, job_id: str) -> None:
        job = self.journal.job(job_id)
        path = Path(str(job["checkpoint_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                dump_ah_json(self.agent.store),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _context_for_batch(
        self,
        batch_id: int,
        text: str,
    ) -> list[str]:
        context = self.journal.boundary_context(batch_id)
        limit = self.config.context_card_limit
        if limit <= 0:
            return context
        matched_uids: set[str] = set()
        for token in dict.fromkeys(_TOKEN.findall(text.lower())):
            if len(token) < 3:
                continue
            for symbol in self.agent.store.find_symbols(token):
                matched_uids.add(symbol.uid)
                if len(matched_uids) >= limit:
                    break
            if len(matched_uids) >= limit:
                break
        cards: list[str] = []
        for factor in self.agent.store.list_semantic_factors():
            if not matched_uids.intersection(factor.roles.values()):
                continue
            relation = (
                factor.relation.canonical_label
                if factor.relation is not None
                else "RELATED_TO"
            )
            roles = ", ".join(
                f"{role}={self._label(uid)}"
                for role, uid in factor.roles.items()
            )
            cards.append(f"{relation}: {roles}")
            if len(cards) >= limit:
                break
        return [*context, *cards]

    def _label(self, uid: str) -> str:
        try:
            symbol = self.agent.store.get_symbol(uid)
        except Exception:
            symbol = None
        if symbol is not None:
            for prop in symbol.Pr:
                if prop.name == "label" and prop.value:
                    return prop.value
        return uid.removeprefix("M_").replace("_", " ").lower()

    @staticmethod
    def load_checkpoint(path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return restore_store(payload)

    @staticmethod
    def _with_provenance(
        candidate: FactCandidate,
        *,
        job_id: str,
        batch_id: int,
        source_segments: list[dict[str, Any]],
    ) -> FactCandidate:
        return replace(
            candidate,
            metadata={
                **dict(candidate.metadata),
                "provenance": {
                    "job_id": job_id,
                    "batch_id": batch_id,
                    "segments": source_segments,
                },
            },
        )
