"""JSONL loaders for the fixed challenge corpora."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypeVar

from ah_memory.benchmarks.challenge.schema import (
    CHALLENGE_DATA_DIR,
    QAItem,
    RoleCorpusItem,
)


T = TypeVar("T")


def _load_jsonl(path: Path, parser: Callable[[dict[str, object]], T]) -> list[T]:
    items: list[T] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("each JSONL record must be an object")
                item = parser(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            item_id = getattr(item, "item_id")
            if item_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {item_id!r}")
            seen_ids.add(item_id)
            items.append(item)
    return items


def load_role_corpus(path: str | Path | None = None) -> list[RoleCorpusItem]:
    """Load and validate the fixed 100-item M1 role corpus."""
    corpus_path = Path(path) if path is not None else CHALLENGE_DATA_DIR / "m1_roles.jsonl"
    items = _load_jsonl(corpus_path, RoleCorpusItem.from_dict)
    if len(items) != 100:
        raise ValueError(f"M1 corpus must contain exactly 100 items; got {len(items)}")
    return items


def load_qa_corpus(path: str | Path | None = None) -> list[QAItem]:
    """Load and validate the fixed 20-item M2/M4 QA corpus."""
    corpus_path = Path(path) if path is not None else CHALLENGE_DATA_DIR / "m2_m4_qa.jsonl"
    items = _load_jsonl(corpus_path, QAItem.from_dict)
    if len(items) != 20:
        raise ValueError(f"M2/M4 corpus must contain exactly 20 items; got {len(items)}")
    depths = {item.depth for item in items}
    if depths != set(range(1, 7)):
        raise ValueError(f"M2/M4 corpus must cover depths 1..6; got {sorted(depths)}")
    relations = {item.relation_type for item in items}
    if relations != {"FOLLOW", "IS-A", "CAUSE"}:
        raise ValueError(
            "M2/M4 corpus must cover FOLLOW, IS-A, and CAUSE relations"
        )
    return items
