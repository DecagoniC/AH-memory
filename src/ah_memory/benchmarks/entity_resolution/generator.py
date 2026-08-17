"""Load control symbols into a real AHStore for activation coupling."""
from __future__ import annotations

from ah_memory.benchmarks.entity_resolution.cases import SymbolSpec
from ah_memory.benchmarks.entity_resolution.dataset import (
    control_graph_facts,
    control_symbols,
)
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform
from ah_memory.types import Property


def build_resolution_store(
    symbols: list[SymbolSpec] | None = None,
    *,
    include_facts: bool = True,
) -> tuple[AHStore, dict[str, str]]:
    """
    Materialize control symbols as AH m-nodes.

    Returns store and mapping catalog_uid → AH uid (M_...).
    """
    store = AHStore()
    transform = Transform(store)
    catalog = symbols or control_symbols()
    uid_map: dict[str, str] = {}

    for spec in catalog:
        # Stable AH bare name from catalog name (Cyrillic ok via morph slug on ingest)
        bare = spec.name.upper().replace(" ", "_").replace("-", "_")
        # Keep Latin brand UIDs readable
        if spec.name.isascii():
            bare = spec.name.upper()
        ah_m = f"M_{bare}"
        forms = {spec.name.lower(), *(a.lower() for a in spec.aliases), spec.uid.lower()}
        store.ensure_abstract(bare, forms)
        store.ensure_m(ah_m, spec.name)
        try:
            store.edit_property(ah_m, Property(name="label", value=spec.name))
        except Exception:
            pass
        try:
            store.add_property(
                ah_m,
                Property(name="catalog_uid", value=spec.uid),
                meta=True,
            )
        except Exception:
            pass
        uid_map[spec.uid] = ah_m

    if include_facts:
        candidates = []
        for subj, rel, obj in control_graph_facts():
            if subj not in uid_map or obj not in uid_map:
                continue
            # Pass catalog names so Transform resolves consistently
            subj_name = next(s.name for s in catalog if s.uid == subj)
            obj_name = next(s.name for s in catalog if s.uid == obj)
            candidates.append(
                FactCandidate(
                    predicate=rel,
                    raw_relation=rel.lower(),
                    canonical_relation=rel,
                    roles={"SUBJECT": subj_name, "OBJECT": obj_name},
                    confidence=1.0,
                )
            )
        if candidates:
            transform.apply(PerceptionResult(kind="fact", candidates=candidates))

    return store, uid_map
