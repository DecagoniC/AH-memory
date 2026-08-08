"""Open semantic relation and event data models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, TypeAlias

Vector: TypeAlias = tuple[float, ...]


def canonicalize_label(label: str) -> str:
    cleaned = "".join(
        char if char.isalnum() else "_"
        for char in label.strip().upper().replace("Ё", "Е")
    )
    return "_".join(part for part in cleaned.split("_") if part) or "RELATED_TO"


@dataclass(frozen=True)
class RelationProperties:
    directional: bool = False
    symmetric: bool = False
    transitive: bool = False
    temporal: bool = False
    causal: bool = False
    state_changing: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class Relation:
    uid: str
    raw_label: str
    canonical_label: str
    arity: int = 2
    properties: RelationProperties = field(default_factory=RelationProperties)
    embedding: Vector | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "raw_label": self.raw_label,
            "canonical_label": self.canonical_label,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "arity": self.arity,
            "properties": self.properties.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationContext:
    text: str = ""
    subject_uid: str | None = None
    object_uid: str | None = None
    roles: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "subject_uid": self.subject_uid,
            "object_uid": self.object_uid,
            "roles": dict(self.roles),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NormalizedRelation:
    raw_label: str
    canonical_label: str
    confidence: float
    properties: RelationProperties
    embedding: Vector | None = None
    strategy: str = "unknown"
    created: bool = False

    def to_relation(self, *, arity: int = 2) -> Relation:
        canonical = canonicalize_label(self.canonical_label)
        return Relation(
            uid=f"REL_{canonical}",
            raw_label=self.raw_label,
            canonical_label=canonical,
            embedding=self.embedding,
            arity=arity,
            properties=self.properties,
            metadata={
                "normalization_confidence": self.confidence,
                "normalization_strategy": self.strategy,
                "created_by_normalizer": self.created,
            },
        )


@dataclass(frozen=True)
class NodeRef:
    uid: str
    role: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"uid": self.uid, "role": self.role}


@dataclass(frozen=True)
class Event:
    uid: str
    predicate: Relation
    arguments: Mapping[str, NodeRef]
    timestamp: str | None = None
    confidence: float = 1.0
    raw_span: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "predicate": self.predicate.to_dict(),
            "arguments": {
                role: node.to_dict() for role, node in self.arguments.items()
            },
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "raw_span": self.raw_span,
            "metadata": dict(self.metadata),
        }
