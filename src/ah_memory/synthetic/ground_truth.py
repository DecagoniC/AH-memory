"""Ground-truth dataset assembly for synthetic worlds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ah_memory.synthetic.config import SyntheticGraphConfig
from ah_memory.synthetic.entities import Entity
from ah_memory.synthetic.events import SyntheticEvent, SyntheticFactor, WorldState


@dataclass(frozen=True)
class SyntheticQuery:
    query_id: str
    question: str
    answer: str
    answer_uid: str
    answer_type: str
    category: str
    required_depth: int
    proof_path: tuple[str, ...]
    required_nodes: tuple[str, ...]
    distractor_factor_uids: tuple[str, ...] = ()
    seed_uids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "answer": self.answer,
            "answer_uid": self.answer_uid,
            "answer_type": self.answer_type,
            "category": self.category,
            "required_depth": self.required_depth,
            "proof_path": list(self.proof_path),
            "required_nodes": list(self.required_nodes),
            "distractor_factor_uids": list(self.distractor_factor_uids),
            "seed_uids": list(self.seed_uids),
        }


@dataclass(frozen=True)
class SyntheticDocument:
    uid: str
    text: str
    factor_uids: tuple[str, ...]
    paraphrases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "text": self.text,
            "factor_uids": list(self.factor_uids),
            "paraphrases": list(self.paraphrases),
        }


@dataclass
class SyntheticWorld:
    config: SyntheticGraphConfig
    entities: list[Entity]
    factors: list[SyntheticFactor]
    events: list[SyntheticEvent]
    documents: list[SyntheticDocument]
    queries: list[SyntheticQuery]
    world_state: WorldState
    generation_time_sec: float = 0.0
    proof_chains: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def entity_map(self) -> dict[str, Entity]:
        return {entity.uid: entity for entity in self.entities}

    def factor_map(self) -> dict[str, SyntheticFactor]:
        return {factor.uid: factor for factor in self.factors}

    def stats(self) -> dict[str, int]:
        relation_edges = sum(max(0, len(factor.arguments) - 1) for factor in self.factors)
        return {
            "entities": len(self.entities),
            "factors": len(self.factors),
            "relations": relation_edges,
            "events": len(self.events),
            "documents": len(self.documents),
            "queries": len(self.queries),
            "proof_paths": sum(1 for query in self.queries if query.proof_path),
        }

    def to_ground_truth_dict(self) -> dict[str, Any]:
        return {
            "world_state": self.world_state.to_dict(),
            "queries": [query.to_dict() for query in self.queries],
            "proof_chains": list(self.proof_chains),
            "factor_uids": [factor.uid for factor in self.factors],
            "entity_uids": [entity.uid for entity in self.entities],
        }

    def to_summary(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            "config": self.config.to_dict(),
            "stats": stats,
            "generation_time_sec": self.generation_time_sec,
            "seed": self.config.random_seed,
            "metadata": dict(self.metadata),
        }
