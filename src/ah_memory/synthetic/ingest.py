"""Load a synthetic world into a real AHStore via Transform."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ah_memory.factor_parameters import (
    FactorParameterGenerator,
    RuleBasedParameterGenerator,
)
from ah_memory.morph import slug_uid
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.synthetic.ground_truth import SyntheticWorld
from ah_memory.synthetic.relations import map_roles_to_ah
from ah_memory.transform import Transform
from ah_memory.types import Property


@dataclass
class IngestResult:
    store: AHStore
    uid_map: dict[str, str] = field(default_factory=dict)
    factor_map: dict[str, str] = field(default_factory=dict)
    ingested_factors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid_map": dict(self.uid_map),
            "factor_map": dict(self.factor_map),
            "ingested_factors": self.ingested_factors,
            "stats": {
                "semantic_factors": len(self.store.semantic_factors),
                "events": len(self.store.events),
                "S": len(self.store.ah.S),
                "graph_size": self.store.graph_size(),
            },
        }


def synthetic_to_ah_uid(synthetic_uid: str) -> str:
    bare = slug_uid(synthetic_uid)
    return bare if bare.startswith("M_") else f"M_{bare}"


def ingest_world(
    world: SyntheticWorld,
    store: AHStore | None = None,
    *,
    parameter_generator: FactorParameterGenerator | None = None,
    include_distractors: bool = True,
    identity=None,
) -> IngestResult:
    store = store or AHStore()
    transform = Transform(
        store,
        parameter_generator=parameter_generator or RuleBasedParameterGenerator(),
        identity=identity,
        dedup_semantic_factors=False,
    )
    uid_map: dict[str, str] = {}
    factor_map: dict[str, str] = {}

    # Materialize entities with stable synthetic UIDs and display-name labels.
    for entity in world.entities:
        ah_uid = synthetic_to_ah_uid(entity.uid)
        bare = ah_uid[2:] if ah_uid.startswith("M_") else ah_uid
        forms = {entity.name.lower(), entity.uid.lower(), bare.lower()}
        store.ensure_abstract(bare, forms)
        store.ensure_m(ah_uid, entity.name)
        try:
            store.edit_property(ah_uid, Property(name="label", value=entity.name))
        except Exception:
            try:
                store.add_property(ah_uid, Property(name="label", value=entity.name))
            except Exception:
                pass
        uid_map[entity.uid] = ah_uid

    candidates: list[FactCandidate] = []
    ordered_factors = sorted(
        world.factors,
        key=lambda factor: (factor.timestamp, factor.uid),
    )
    for factor in ordered_factors:
        if factor.properties.get("distractor") and not include_distractors:
            continue
        mapped_args = {
            role: uid_map.get(value, synthetic_to_ah_uid(value))
            for role, value in factor.arguments.items()
        }
        # Transform resolves values again; pass bare synthetic-style tokens.
        role_values = {
            role: (
                value[2:] if str(value).startswith("M_") else str(value)
            )
            for role, value in mapped_args.items()
        }
        ah_roles = map_roles_to_ah(factor.type, role_values)
        if len(ah_roles) < 2:
            continue
        before = set(store.semantic_factors)
        candidates.append(
            FactCandidate(
                predicate=factor.type,
                raw_relation=factor.type.lower(),
                canonical_relation=factor.type,
                roles=ah_roles,
                confidence=float(factor.weight),
                raw_span=factor.uid,
            )
        )
        # Ingest one-by-one to keep factor_map aligned.
        transform.apply(
            PerceptionResult(kind="fact", candidates=[candidates[-1]])
        )
        created = set(store.semantic_factors) - before
        if created:
            factor_map[factor.uid] = next(iter(created))
        elif store.semantic_factors:
            # fallback: last inserted
            factor_map[factor.uid] = next(reversed(store.semantic_factors))

    return IngestResult(
        store=store,
        uid_map=uid_map,
        factor_map=factor_map,
        ingested_factors=len(factor_map),
    )
