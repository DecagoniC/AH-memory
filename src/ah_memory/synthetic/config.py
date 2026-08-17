"""Configuration for synthetic world generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


DEFAULT_RELATION_TYPES: tuple[str, ...] = (
    "KNOW",
    "WORKS_FOR",
    "LIVES_IN",
    "LOCATED_IN",
    "OWNS",
    "PURCHASE",
    "SELL",
    "MOVE",
    "VISITS",
    "USES",
    "CREATED",
    "CAUSE",
    "FOLLOW",
    "PART_OF",
    "IS_A",
)


@dataclass(frozen=True)
class SyntheticGraphConfig:
    num_entities: int = 100
    num_events: int = 200
    num_factors: int = 500
    max_hop_depth: int = 3
    distractor_ratio: float = 0.3
    num_queries: int = 50
    random_seed: int = 42
    num_persons: int = 0
    num_companies: int = 0
    num_places: int = 0
    num_objects: int = 0
    num_documents: int = 0
    relation_types: tuple[str, ...] = DEFAULT_RELATION_TYPES
    preset: str = "custom"

    def __post_init__(self) -> None:
        if self.random_seed is None:
            raise ValueError("random_seed is required")
        if self.num_entities < 4:
            raise ValueError("num_entities must be >= 4")
        if not 0.0 <= self.distractor_ratio <= 1.0:
            raise ValueError("distractor_ratio must be in [0, 1]")
        if self.max_hop_depth < 1:
            raise ValueError("max_hop_depth must be >= 1")

    def resolved_counts(self) -> dict[str, int]:
        """Distribute entity budget across types when per-type counts are zero."""
        persons = self.num_persons
        companies = self.num_companies
        places = self.num_places
        objects = self.num_objects
        documents = self.num_documents
        explicit = persons + companies + places + objects + documents
        if explicit <= 0:
            n = self.num_entities
            persons = max(2, int(n * 0.30))
            companies = max(1, int(n * 0.15))
            places = max(2, int(n * 0.20))
            objects = max(2, int(n * 0.25))
            documents = max(1, n - persons - companies - places - objects)
            if documents < 1:
                documents = 1
                objects = max(1, objects - 1)
        return {
            "Person": persons,
            "Company": companies,
            "Place": places,
            "Object": objects,
            "Document": documents,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["relation_types"] = list(self.relation_types)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SyntheticGraphConfig:
        raw = dict(data or {})
        relations = raw.get("relation_types")
        if relations is not None:
            raw["relation_types"] = tuple(str(item) for item in relations)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in raw.items() if key in known})


def merge_config(
    base: SyntheticGraphConfig,
    overrides: Mapping[str, Any] | None = None,
) -> SyntheticGraphConfig:
    data = base.to_dict()
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in data:
                data[key] = value
    return SyntheticGraphConfig.from_mapping(data)
