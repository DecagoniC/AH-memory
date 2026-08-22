"""Batch a natural-language corpus through perception → Transform → AH graph."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ah_memory.agent import Agent
from ah_memory.morph import STOP, lemma, seeds_from_roles, slug_uid
from ah_memory.perception import (
    FactCandidate,
    PerceptionResult,
    candidates_from_llm_json,
    gate_candidates,
)
from ah_memory.store import AHStore
from ah_memory.types import Section

_HEADER_SKIP = re.compile(
    r"^(источник:|source:|#|см\. также|примечания|литература|ссылки)\b",
    re.IGNORECASE,
)


@dataclass
class BatchRecord:
    index: int
    text: str
    backend: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    created_n: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    uncovered_segments: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: int = 0
    coverage_ratio: float = 1.0


@dataclass(frozen=True)
class AtomicSegment:
    index: int
    start: int
    end: int
    text: str
    context: str
    kind: str = "sentence"
    group: int | None = None


def split_batches(text: str, *, max_chars: int = 700) -> list[str]:
    """Paragraph batches small enough for one perception call."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    batches: list[str] = []
    buf = ""
    for para in paragraphs:
        first = para.splitlines()[0].strip()
        if (
            _HEADER_SKIP.match(first)
            or first.lower().startswith("источник:")
            or (len(para) < 48 and "\n" not in para)
        ):
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if buf and len(candidate) > max_chars:
            batches.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        batches.append(buf)
    return batches


def split_atomic_segments(text: str) -> list[AtomicSegment]:
    """Split prose and top-level enumerations without using domain vocabulary."""
    segments: list[AtomicSegment] = []
    previous_sentence = ""
    for line_match in re.finditer(r"[^\n]+", text):
        line = line_match.group(0)
        if not line.strip():
            continue
        sentence_ranges = _sentence_ranges(line)
        for local_start, local_end in sentence_ranges:
            sentence = line[local_start:local_end].strip()
            if not sentence:
                continue
            leading = len(line[local_start:local_end]) - len(
                line[local_start:local_end].lstrip()
            )
            absolute_start = line_match.start() + local_start + leading
            colon = _top_level_delimiter(sentence, ":")
            if colon >= 0 and _content_tokens(sentence[colon + 1 :]):
                header = sentence[: colon + 1].strip()
                tail = sentence[colon + 1 :]
                tail_start = absolute_start + colon + 1
                parts = _list_parts(tail)
                for part_start, part_end in parts:
                    raw = tail[part_start:part_end]
                    item = raw.strip(" \t.;,")
                    if not _content_tokens(item):
                        continue
                    trim = raw.find(item)
                    start = tail_start + part_start + max(0, trim)
                    context = "\n".join(
                        part
                        for part in (previous_sentence, f"{header} {item}")
                        if part
                    )
                    segments.append(
                        AtomicSegment(
                            index=len(segments),
                            start=start,
                            end=start + len(item),
                            text=item,
                            context=context,
                            kind="list_item",
                            group=absolute_start + colon,
                        )
                    )
            else:
                parts = _top_level_parts(sentence, {";"})
                for part_start, part_end in parts:
                    raw = sentence[part_start:part_end]
                    item = raw.strip(" \t.;,")
                    if not _content_tokens(item):
                        continue
                    trim = raw.find(item)
                    start = absolute_start + part_start + max(0, trim)
                    segments.append(
                        AtomicSegment(
                            index=len(segments),
                            start=start,
                            end=start + len(item),
                            text=item,
                            context=item,
                            kind=(
                                "clause"
                                if len(parts) > 1
                                else "sentence"
                            ),
                        )
                    )
            previous_sentence = sentence
    return segments


def coverage_audit(
    segments: list[AtomicSegment],
    candidates: Iterable[FactCandidate],
) -> tuple[list[dict[str, Any]], list[AtomicSegment], float]:
    """Map every atomic source segment to grounded candidate spans."""
    candidates_now = list(candidates)
    report: list[dict[str, Any]] = []
    uncovered: list[AtomicSegment] = []
    for segment in segments:
        segment_tokens = _content_tokens(segment.text)
        supporting = [
            index
            for index, candidate in enumerate(candidates_now)
            if _candidate_overlap(candidate, segment)
        ]
        covered_tokens: set[str] = set()
        role_tokens: set[str] = set()
        for index in supporting:
            covered_tokens.update(
                segment_tokens.intersection(
                    _candidate_evidence_tokens(candidates_now[index])
                )
            )
            role_tokens.update(
                segment_tokens.intersection(
                    _candidate_role_tokens(candidates_now[index])
                )
            )
        covered = not segment_tokens or (
            bool(supporting)
            if segment.kind == "sentence"
            else (
                len(role_tokens) / len(segment_tokens) >= 0.5
                if segment.kind == "list_item"
                else len(covered_tokens) / len(segment_tokens) >= 0.6
            )
        )
        if not covered:
            uncovered.append(segment)
        report.append(
            {
                "index": segment.index,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "kind": segment.kind,
                "covered": covered,
                "candidate_indexes": supporting,
            }
        )
    ratio = (
        sum(bool(item["covered"]) for item in report) / len(report)
        if report
        else 1.0
    )
    return report, uncovered, ratio


def candidate_to_dict(candidate: FactCandidate) -> dict[str, Any]:
    return {
        "predicate": candidate.predicate,
        "roles": dict(candidate.roles),
        "raw_span": candidate.raw_span,
        "confidence": candidate.confidence,
        "raw_relation": candidate.raw_relation,
        "canonical_relation": candidate.canonical_relation,
        "statement_type": candidate.statement_type,
        "source": candidate.source,
    }


def dict_to_candidate(payload: dict[str, Any]) -> FactCandidate:
    return FactCandidate(
        predicate=str(payload.get("predicate") or payload.get("canonical_relation") or "RELATED_TO"),
        roles={str(k): str(v) for k, v in dict(payload.get("roles") or {}).items()},
        raw_span=payload.get("raw_span"),
        confidence=float(payload.get("confidence") or 1.0),
        raw_relation=payload.get("raw_relation"),
        canonical_relation=payload.get("canonical_relation"),
        statement_type=payload.get("statement_type") or "assertion",
        source=payload.get("source") or "user",
    )


def ingest_text_batches(
    agent: Agent,
    batches: Iterable[str],
    *,
    section: Section = Section.C,
    repair_uncovered: bool = True,
) -> list[BatchRecord]:
    """Parse, audit and repair each batch before writing accepted facts."""
    records: list[BatchRecord] = []
    for index, batch in enumerate(batches):
        perc = agent.perception.parse(batch, list(agent.ignition.wm.contents()))
        candidates = list(perc.candidates)
        candidates.extend(_schema_repair_from_raw(perc, batch))
        candidates = _deduplicate_candidates(candidates)
        segments = split_atomic_segments(batch)
        segment_report, uncovered, _ = coverage_audit(segments, candidates)
        repair_attempts = 0
        if repair_uncovered:
            for segment in uncovered:
                repair_attempts += 1
                repair = agent.perception.parse(
                    segment.context,
                    list(agent.ignition.wm.contents()),
                )
                candidates.extend(repair.candidates)
                candidates.extend(
                    _schema_repair_from_raw(repair, segment.context)
                )
            candidates = _deduplicate_candidates(candidates)
            segment_report, uncovered, ratio = coverage_audit(
                segments,
                candidates,
            )
            if uncovered:
                candidates.extend(
                    candidate
                    for segment in uncovered
                    if (
                        candidate := _infer_list_candidate(
                            segment,
                            segments,
                            candidates,
                        )
                    )
                    is not None
                )
                candidates = _deduplicate_candidates(candidates)
                segment_report, uncovered, ratio = coverage_audit(
                    segments,
                    candidates,
                )
        else:
            ratio = (
                sum(bool(item["covered"]) for item in segment_report)
                / len(segment_report)
                if segment_report
                else 1.0
            )
        combined = PerceptionResult(
            kind="fact",
            candidates=candidates,
            seed_tokens=seeds_from_roles(candidates),
            meta={
                **dict(perc.meta),
                "coverage_ratio": ratio,
                "repair_attempts": repair_attempts,
            },
        )
        report = agent.ingest(batch, section=section, perception=combined)
        records.append(
            BatchRecord(
                index=index,
                text=batch,
                backend=str((perc.meta or {}).get("backend") or "unknown"),
                candidates=[candidate_to_dict(c) for c in candidates],
                created_n=list(report.created_n),
                skipped=list(report.skipped),
                segments=segment_report,
                uncovered_segments=[
                    {
                        "index": segment.index,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "reason": "no grounded factor after repair",
                    }
                    for segment in uncovered
                ],
                repair_attempts=repair_attempts,
                coverage_ratio=ratio,
            )
        )
    return records


def records_to_payload(records: list[BatchRecord], *, source: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        candidates.extend(record.candidates)
    total_segments = sum(len(record.segments) for record in records)
    covered_segments = sum(
        sum(bool(segment.get("covered")) for segment in record.segments)
        for record in records
    )
    uncovered = [
        {"batch": record.index, **item}
        for record in records
        for item in record.uncovered_segments
    ]
    return {
        "source": source,
        "n_batches": len(records),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "batches": [asdict(record) for record in records],
        "coverage": {
            "segments": total_segments,
            "covered": covered_segments,
            "ratio": (
                covered_segments / total_segments
                if total_segments
                else 1.0
            ),
            "uncovered": uncovered,
        },
    }


def load_fact_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_cached_facts(store: AHStore, payload: dict[str, Any]) -> int:
    """Replay stored FactCandidates through Transform (no LLM)."""
    from ah_memory.perception import PerceptionResult
    from ah_memory.transform import Transform
    from ah_memory.types import Section as StoreSection

    candidates = [dict_to_candidate(item) for item in payload.get("candidates") or []]
    if not candidates:
        return 0
    seeds: list[str] = []
    for candidate in candidates:
        seeds.extend(candidate.roles.values())
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=candidates,
            seed_tokens=list(dict.fromkeys(seeds)),
        ),
        section=StoreSection.C,
    )
    return len(candidates)


def _sentence_ranges(line: str) -> list[tuple[int, int]]:
    boundaries = []
    for boundary in re.finditer(
        r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])",
        line,
    ):
        prefix = line[: boundary.start()].rstrip()
        word = re.search(r"([A-Za-zА-Яа-яЁё]+)\.$", prefix)
        if word is not None and len(word.group(1)) <= 2:
            continue
        boundaries.append(boundary)
    out: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        out.append((start, boundary.start()))
        start = boundary.end()
    out.append((start, len(line)))
    return out


def _top_level_delimiter(text: str, delimiter: str) -> int:
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == delimiter and depth == 0:
            return index
    return -1


def _top_level_parts(
    text: str,
    delimiters: set[str],
) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char in delimiters and depth == 0:
            parts.append((start, index))
            start = index + 1
    parts.append((start, len(text)))
    return parts


def _list_parts(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in _top_level_parts(text, {",", ";"}):
        part = text[start:end]
        depth = 0
        sub_start = 0
        cursor = 0
        for match in re.finditer(
            r"\b(?:и|или|and|or)\b",
            part,
            flags=re.IGNORECASE,
        ):
            for char in part[cursor : match.start()]:
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth = max(0, depth - 1)
            if depth == 0:
                out.append((start + sub_start, start + match.start()))
                sub_start = match.end()
            cursor = match.end()
        out.append((start + sub_start, end))
    return out


def _content_tokens(text: str) -> set[str]:
    return {
        lemma(token)
        for token in re.findall(r"[a-zа-яё0-9]+", text.lower())
        if len(token) > 1 and token not in STOP and lemma(token)
    }


def _candidate_evidence_tokens(
    candidate: FactCandidate,
) -> set[str]:
    evidence = " ".join(
        part
        for part in (
            candidate.raw_span or "",
            candidate.raw_relation or "",
            *(
                value.replace("_", " ")
                for value in candidate.roles.values()
            ),
        )
        if part
    )
    return _content_tokens(evidence)


def _candidate_role_tokens(candidate: FactCandidate) -> set[str]:
    return _content_tokens(
        " ".join(value.replace("_", " ") for value in candidate.roles.values())
    )


def _candidate_overlap(
    candidate: FactCandidate,
    segment: AtomicSegment,
) -> float:
    segment_tokens = _content_tokens(segment.text)
    if not segment_tokens:
        return 1.0
    overlap = len(
        segment_tokens.intersection(_candidate_evidence_tokens(candidate))
    )
    return overlap / len(segment_tokens)


def _deduplicate_candidates(
    candidates: Iterable[FactCandidate],
) -> list[FactCandidate]:
    out: list[FactCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            (
                candidate.canonical_relation
                or candidate.predicate
            ).upper(),
            tuple(sorted(candidate.roles.items())),
            candidate.statement_type,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _schema_repair_from_raw(
    perception: PerceptionResult,
    source_text: str,
) -> list[FactCandidate]:
    """Split one explicitly coordinated role, then re-run normal validation."""
    data = perception.meta.get("llm_raw")
    if not isinstance(data, dict):
        return []
    repaired_payloads: list[dict[str, Any]] = []
    for raw_candidate in data.get("candidates") or []:
        if not isinstance(raw_candidate, dict):
            continue
        roles = raw_candidate.get("roles")
        if not isinstance(roles, dict):
            continue
        compound: list[tuple[str, list[str]]] = []
        for role, value in roles.items():
            if str(role).upper() == "TIME" or not isinstance(value, str):
                continue
            parts = [
                part.strip()
                for part in re.split(
                    r"\s*(?:,|;|\b(?:и|или|and|or)\b)\s*",
                    value,
                    flags=re.IGNORECASE,
                )
                if part.strip()
            ]
            if len(parts) > 1:
                compound.append((str(role), parts))
        if len(compound) != 1:
            continue
        role, parts = compound[0]
        if len(parts) > 4:
            continue
        for part in parts:
            clone = dict(raw_candidate)
            clone["roles"] = {**roles, role: part}
            repaired_payloads.append(clone)
    if not repaired_payloads:
        return []
    repaired = candidates_from_llm_json(
        {
            "kind": data.get("kind", "fact"),
            "candidates": repaired_payloads,
        }
    )
    gated = gate_candidates(
        source_text,
        repaired,
        require_grounding=True,
        allow_open_relations=True,
    )
    assert isinstance(gated, list)
    return gated


def _infer_list_candidate(
    target: AtomicSegment,
    segments: list[AtomicSegment],
    candidates: list[FactCandidate],
) -> FactCandidate | None:
    """Inherit the stable relation/roles of a structurally explicit list."""
    if target.kind != "list_item" or target.group is None:
        return None
    siblings = [
        segment
        for segment in segments
        if segment.group == target.group and segment.index != target.index
    ]
    related = [
        candidate
        for candidate in candidates
        if any(_candidate_overlap(candidate, sibling) > 0.0 for sibling in siblings)
    ]
    relation_counts = Counter(
        (
            candidate.canonical_relation
            or candidate.predicate
        ).upper()
        for candidate in related
    )
    if not relation_counts:
        return None
    relation, count = relation_counts.most_common(1)[0]
    peers = [
        candidate
        for candidate in related
        if (
            candidate.canonical_relation
            or candidate.predicate
        ).upper()
        == relation
    ]
    if count < 2:
        return None
    role_values: dict[str, set[str]] = {}
    for candidate in peers:
        for role, value in candidate.roles.items():
            role_values.setdefault(role, set()).add(value)
    varying = [
        (len(values), role)
        for role, values in role_values.items()
        if role != "SUBJECT" and len(values) > 1
    ]
    if not varying:
        return None
    _, target_role = max(varying)
    stable_roles = {
        role: next(iter(values))
        for role, values in role_values.items()
        if len(values) == 1
    }
    entity_text = re.sub(r"\s*\([^()]*\)\s*$", "", target.text).strip()
    if not entity_text:
        return None
    roles = {
        **stable_roles,
        target_role: slug_uid(entity_text),
    }
    if "SUBJECT" not in roles or len(roles) < 2:
        return None
    prototype = peers[0]
    return FactCandidate(
        predicate=relation,
        roles=roles,
        raw_span=target.text,
        confidence=min(0.8, prototype.confidence),
        raw_relation=prototype.raw_relation or relation,
        canonical_relation=relation,
        statement_type=prototype.statement_type,
        source=prototype.source,
    )
