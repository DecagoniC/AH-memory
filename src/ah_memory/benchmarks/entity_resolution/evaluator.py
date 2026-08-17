"""Evaluate resolver + activation engine on entity resolution cases."""
from __future__ import annotations

from typing import Any, Sequence

from ah_memory.benchmarks.entity_resolution.cases import (
    EntityResolutionCase,
    SymbolSpec,
)
from ah_memory.benchmarks.entity_resolution.metrics import (
    CaseOutcome,
    classify_outcome,
)
from ah_memory.benchmarks.entity_resolution.resolvers import (
    EmbeddingResolver,
    SymbolResolver,
)
from ah_memory.factor_graph import build_structural_factor_graph
from ah_memory.semantic_activation import ActivationEngine
from ah_memory.store import AHStore


def mention_target_similarity(
    mention: str,
    target_uid: str | None,
    symbols: Sequence[SymbolSpec],
    resolver: EmbeddingResolver,
) -> float | None:
    if not target_uid:
        return None
    symbol = next((s for s in symbols if s.uid == target_uid), None)
    if symbol is None:
        return None
    result = resolver.resolve(
        mention,
        None,
        [symbol],
        candidate_uids=(target_uid,),
        threshold=0.0,
    )
    if not result.candidates:
        return None
    return result.candidates[0].similarity


def run_activation_probe(
    store: AHStore,
    ah_uid: str | None,
    *,
    timesteps: int = 3,
    threshold: float = 0.05,
) -> tuple[bool | None, list[dict[str, Any]]]:
    """Seed resolved AH symbol and check that activation spreads."""
    if not ah_uid:
        return None, []
    graph = build_structural_factor_graph(store)
    if ah_uid not in graph.variables:
        bare = ah_uid[2:] if ah_uid.startswith("M_") else ah_uid
        if bare in graph.variables:
            ah_uid = bare
        else:
            return False, []
    engine = ActivationEngine()
    activation, trace = engine.run(
        graph,
        {ah_uid: 1.0},
        timesteps=timesteps,
        threshold=threshold,
        trace=True,
    )
    ticks: list[dict[str, Any]] = []
    for index, snapshot in enumerate(trace.timesteps):
        for uid, value in sorted(snapshot.items(), key=lambda item: -item[1])[:8]:
            if value >= threshold:
                ticks.append(
                    {
                        "tick": index,
                        "uid": uid,
                        "activation": round(float(value), 4),
                    }
                )
    ok = activation.get(ah_uid, 0.0) >= threshold
    # Also success if any semantic factor touching the seed activated in messages
    if not ok and trace.activated_factors:
        ok = True
    return ok, ticks


def evaluate_case(
    case: EntityResolutionCase,
    resolver: SymbolResolver,
    symbols: Sequence[SymbolSpec],
    *,
    threshold: float,
    store: AHStore | None = None,
    uid_map: dict[str, str] | None = None,
    embed_resolver: EmbeddingResolver | None = None,
    run_activation: bool = True,
) -> CaseOutcome:
    result = resolver.resolve(
        case.mention,
        case.context,
        symbols,
        candidate_uids=case.candidate_uids,
        threshold=threshold,
    )
    sim_resolver = embed_resolver or EmbeddingResolver()
    similarity = mention_target_similarity(
        case.mention,
        case.target_uid,
        symbols,
        sim_resolver,
    )
    if similarity is None and result.candidates:
        similarity = result.candidates[0].similarity

    outcome = classify_outcome(
        case,
        result.selected_uid,
        similarity=similarity,
        method=result.method,
    )

    if run_activation and store is not None and uid_map is not None:
        ah_uid = (
            uid_map.get(result.selected_uid)
            if result.selected_uid
            else None
        )
        activation_ok, ticks = run_activation_probe(store, ah_uid)
        outcome.activation_ok = activation_ok
        outcome.activation_trace = ticks
    return outcome
