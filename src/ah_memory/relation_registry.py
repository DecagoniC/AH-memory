"""Dynamic canonical relation registry."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from ah_memory.relations import (
    Relation,
    canonicalize_label,
)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class RelationRegistry:
    def __init__(self, relations: Iterable[Relation] = ()) -> None:
        self._relations: dict[str, Relation] = {}
        for relation in relations:
            self.register_relation(relation)

    def register_relation(self, relation: Relation, *, replace: bool = False) -> Relation:
        canonical = canonicalize_label(relation.canonical_label)
        existing = self._relations.get(canonical)
        if existing is not None and not replace:
            return existing
        normalized = Relation(
            uid=relation.uid or f"REL_{canonical}",
            raw_label=relation.raw_label,
            canonical_label=canonical,
            embedding=tuple(relation.embedding) if relation.embedding is not None else None,
            arity=max(1, int(relation.arity)),
            properties=relation.properties,
            metadata=dict(relation.metadata),
        )
        self._relations[canonical] = normalized
        return normalized

    def get_relation(self, canonical_label: str) -> Relation | None:
        return self._relations.get(canonicalize_label(canonical_label))

    def require_relation(self, canonical_label: str) -> Relation:
        relation = self.get_relation(canonical_label)
        if relation is None:
            raise KeyError(canonical_label)
        return relation

    def list_relations(self) -> tuple[Relation, ...]:
        return tuple(
            self._relations[key] for key in sorted(self._relations)
        )

    def find_similar_relations(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 5,
        min_similarity: float = -1.0,
    ) -> list[tuple[Relation, float]]:
        scored = [
            (relation, cosine_similarity(embedding, relation.embedding))
            for relation in self._relations.values()
            if relation.embedding is not None
        ]
        scored = [item for item in scored if item[1] >= min_similarity]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, limit)]

    def to_dict(self) -> dict[str, Any]:
        return {
            relation.canonical_label: relation.to_dict()
            for relation in self.list_relations()
        }


def default_relation_registry() -> RelationRegistry:
    """Return an empty open registry; applications register their own relations."""
    return RelationRegistry()
