"""Evaluate activation over a synthetic ground-truth dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ah_memory.factor_graph import build_structural_factor_graph
from ah_memory.factor_parameters import (
    FactorParameterGenerator,
    RuleBasedParameterGenerator,
)
from ah_memory.semantic_activation import ActivationEngine, ActivationFunction
from ah_memory.store import AHStore
from ah_memory.synthetic.ground_truth import SyntheticQuery, SyntheticWorld
from ah_memory.synthetic.ingest import IngestResult, synthetic_to_ah_uid


@dataclass
class QueryEvalResult:
    query_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    answer_correct: bool
    category: str
    required_depth: int
    proof_path: list[str]
    activated_path: list[str]
    activated_nodes: list[str]
    missing_nodes: list[str]
    extra_nodes: list[str]
    path_precision: float
    path_recall: float
    path_f1: float
    hop_efficiency: float
    activation_noise: float
    ticks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "answer": self.expected_answer,
            "predicted_answer": self.predicted_answer,
            "answer_correct": self.answer_correct,
            "category": self.category,
            "required_depth": self.required_depth,
            "ground_truth_path": list(self.proof_path),
            "activated_path": list(self.activated_path),
            "activated_nodes": list(self.activated_nodes),
            "missing_nodes": list(self.missing_nodes),
            "extra_nodes": list(self.extra_nodes),
            "path_precision": self.path_precision,
            "path_recall": self.path_recall,
            "path_f1": self.path_f1,
            "hop_efficiency": self.hop_efficiency,
            "activation_noise": self.activation_noise,
            "ticks": list(self.ticks),
            "number_of_ticks": len(self.ticks),
        }


@dataclass
class BenchmarkReport:
    aggregate: dict[str, float]
    results: list[QueryEvalResult]
    activation_name: str = "linear"

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": dict(self.aggregate),
            "activation_name": self.activation_name,
            "results": [result.to_dict() for result in self.results],
        }


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _map_proof_to_store(
    proof_path: Sequence[str],
    factor_map: Mapping[str, str],
) -> list[str]:
    return [factor_map[uid] for uid in proof_path if uid in factor_map]


def _map_nodes_to_store(
    nodes: Sequence[str],
    uid_map: Mapping[str, str],
) -> list[str]:
    mapped: list[str] = []
    for uid in nodes:
        if uid in uid_map:
            mapped.append(uid_map[uid])
        else:
            mapped.append(synthetic_to_ah_uid(uid))
    return mapped


def _activation_ticks(
    propagation: Mapping[str, Any],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    ticks: list[dict[str, Any]] = []
    for index, snapshot in enumerate(propagation.get("timesteps") or []):
        activated = [
            uid
            for uid, value in snapshot.items()
            if float(value) >= threshold
        ]
        ticks.append({"tick": index, "activated": activated})
    return ticks


def _predict_answer(
    activation: Mapping[str, float],
    *,
    seed_uids: Sequence[str],
    candidate_uids: Sequence[str],
    answer_type: str,
    entity_names: Mapping[str, str],
    threshold: float,
) -> tuple[str, str]:
    """Pick highest-activated non-seed entity among candidate endpoints."""
    seed_set = set(seed_uids)
    candidate_set = [uid for uid in candidate_uids if uid not in seed_set]
    pool = candidate_set or [
        uid
        for uid, score in activation.items()
        if score >= threshold
        and uid not in seed_set
        and not uid.startswith(("PRIOR::", "SF::"))
    ]
    if not pool:
        return ("", "")
    best_uid = max(pool, key=lambda uid: (activation.get(uid, 0.0), uid))
    best_score = activation.get(best_uid, 0.0)
    if answer_type == "boolean":
        return ("да" if best_score >= threshold else "нет", best_uid)
    if best_score < threshold and candidate_set:
        # still return top candidate for diagnostics
        return entity_names.get(best_uid, best_uid), best_uid
    return entity_names.get(best_uid, best_uid), best_uid


def evaluate_query(
    store: AHStore,
    query: SyntheticQuery,
    *,
    ingest: IngestResult,
    world: SyntheticWorld,
    engine: ActivationEngine,
    timesteps: int = 4,
    threshold: float = 0.05,
) -> QueryEvalResult:
    graph = build_structural_factor_graph(store)
    seed_synth = query.seed_uids or query.required_nodes[:1]
    seed_ah = _map_nodes_to_store(seed_synth, ingest.uid_map)
    evidence = {uid: 1.0 for uid in seed_ah if uid in graph.variables or True}
    # Keep only variables that exist in the store graph.
    var_set = set(graph.variables)
    evidence = {uid: 1.0 for uid in seed_ah if uid in var_set}
    if not evidence and seed_ah:
        # try bare forms
        for uid in seed_ah:
            bare = uid[2:] if uid.startswith("M_") else uid
            if bare in var_set:
                evidence[bare] = 1.0
            if uid in var_set:
                evidence[uid] = 1.0

    activation, propagation = engine.run(
        graph,
        evidence,
        timesteps=timesteps,
        threshold=threshold,
        trace=True,
    )
    prop_dict = propagation.to_dict()
    ticks = _activation_ticks(prop_dict, threshold=threshold)

    proof_store = _map_proof_to_store(query.proof_path, ingest.factor_map)
    required_store = _map_nodes_to_store(query.required_nodes, ingest.uid_map)
    activated_nodes = [
        uid
        for uid, score in activation.items()
        if score >= threshold and not uid.startswith(("SF::", "PRIOR::"))
    ]
    activated_node_set = set(activated_nodes) | set(evidence)
    # Path = factors that actually carried above-threshold messages between
    # activated endpoints (not every factor touched by global spread).
    activated_path: list[str] = []
    seen_factors: set[str] = set()
    for message in propagation.messages:
        fid = str(message.get("factor_uid") or "")
        if not fid or fid in seen_factors:
            continue
        msg_act = float(message.get("activation") or 0.0)
        src = str(message.get("source_uid") or "")
        tgt = str(message.get("target_uid") or "")
        if msg_act < threshold:
            continue
        if src in activated_node_set and tgt in activated_node_set:
            activated_path.append(fid)
            seen_factors.add(fid)
    if not activated_path:
        # Fallback: semantic factors with >=2 activated endpoints.
        for factor in store.list_semantic_factors():
            endpoints = [uid for uid in factor.variables if uid in activated_node_set]
            if len(endpoints) >= 2:
                activated_path.append(factor.uid)

    proof_set = set(proof_store)
    active_set = set(activated_path)
    correct_active = proof_set & active_set
    precision = (
        len(correct_active) / len(active_set) if active_set else 0.0
    )
    recall = (
        len(correct_active) / len(proof_set) if proof_set else 0.0
    )
    optimal = max(1, len(proof_store))
    actual = max(1, len(activated_path))
    hop_efficiency = min(1.0, optimal / actual)
    noise_set = active_set - proof_set
    activation_noise = (
        len(noise_set) / len(active_set) if active_set else 0.0
    )

    # entity name lookup from ingest map reverse + world
    name_by_ah = {
        ah: world.entity_map()[synth].name
        for synth, ah in ingest.uid_map.items()
        if synth in world.entity_map()
    }
    candidate_uids = [
        uid for uid in required_store if uid not in set(seed_ah)
    ] or required_store
    predicted_name, predicted_uid = _predict_answer(
        activation,
        seed_uids=seed_ah,
        candidate_uids=candidate_uids,
        answer_type=query.answer_type,
        entity_names=name_by_ah,
        threshold=threshold,
    )
    expected_ah = ingest.uid_map.get(
        query.answer_uid, synthetic_to_ah_uid(query.answer_uid)
    )
    if query.answer_type == "boolean":
        # Hierarchy: correct if parent endpoint is activated via proof factor.
        answer_correct = (
            predicted_name.lower() == query.answer.lower()
            and (
                expected_ah in activated_nodes
                or bool(set(proof_store) & set(activated_path))
            )
        )
    else:
        answer_correct = predicted_uid == expected_ah or (
            bool(predicted_name)
            and predicted_name.lower() == query.answer.lower()
        )

    missing = [uid for uid in required_store if uid not in set(activated_nodes) | active_set]
    extra = [
        uid
        for uid in activated_nodes
        if uid not in set(required_store) and uid not in set(seed_ah)
    ]

    return QueryEvalResult(
        query_id=query.query_id,
        question=query.question,
        expected_answer=query.answer,
        predicted_answer=predicted_name,
        answer_correct=answer_correct,
        category=query.category,
        required_depth=query.required_depth,
        proof_path=proof_store,
        activated_path=activated_path,
        activated_nodes=activated_nodes,
        missing_nodes=missing,
        extra_nodes=extra,
        path_precision=precision,
        path_recall=recall,
        path_f1=_f1(precision, recall),
        hop_efficiency=hop_efficiency,
        activation_noise=activation_noise,
        ticks=ticks,
    )


def run_benchmark(
    store: AHStore,
    world: SyntheticWorld,
    ingest: IngestResult,
    *,
    limit: int | None = None,
    activation_function: ActivationFunction | None = None,
    parameter_generator: FactorParameterGenerator | None = None,
    timesteps: int = 4,
    threshold: float = 0.05,
    activation_name: str = "linear",
) -> BenchmarkReport:
    generator = parameter_generator or RuleBasedParameterGenerator()
    engine = ActivationEngine(
        activation_function=activation_function,
        parameter_generator=generator,
    )
    queries = world.queries[:limit] if limit is not None else world.queries
    results = [
        evaluate_query(
            store,
            query,
            ingest=ingest,
            world=world,
            engine=engine,
            timesteps=timesteps,
            threshold=threshold,
        )
        for query in queries
    ]
    n = len(results) or 1
    aggregate = {
        "answer_accuracy": sum(1.0 for r in results if r.answer_correct) / n,
        "path_precision": sum(r.path_precision for r in results) / n,
        "path_recall": sum(r.path_recall for r in results) / n,
        "path_f1": sum(r.path_f1 for r in results) / n,
        "hop_efficiency": sum(r.hop_efficiency for r in results) / n,
        "activation_noise": sum(r.activation_noise for r in results) / n,
        "queries": float(len(results)),
    }
    return BenchmarkReport(
        aggregate=aggregate,
        results=results,
        activation_name=activation_name,
    )


def proof_view(
    world: SyntheticWorld,
    query_id: str,
    *,
    ingest: IngestResult | None = None,
) -> dict[str, Any]:
    query = next((q for q in world.queries if q.query_id == query_id), None)
    if query is None:
        raise KeyError(query_id)
    entity_map = world.entity_map()
    factor_map = world.factor_map()
    path_nodes: list[dict[str, Any]] = []
    for uid in query.required_nodes:
        entity = entity_map.get(uid)
        path_nodes.append(
            {
                "uid": uid,
                "ah_uid": (
                    ingest.uid_map.get(uid, synthetic_to_ah_uid(uid))
                    if ingest
                    else synthetic_to_ah_uid(uid)
                ),
                "name": entity.name if entity else uid,
                "type": entity.type if entity else "Unknown",
            }
        )
    steps: list[dict[str, Any]] = []
    for factor_uid in query.proof_path:
        factor = factor_map.get(factor_uid)
        if factor is None:
            continue
        steps.append(
            {
                "factor_uid": factor_uid,
                "ah_factor_uid": (
                    ingest.factor_map.get(factor_uid) if ingest else None
                ),
                "type": factor.type,
                "arguments": {
                    role: {
                        "uid": value,
                        "name": entity_map[value].name
                        if value in entity_map
                        else value,
                    }
                    for role, value in factor.arguments.items()
                },
            }
        )
    distractors = [
        factor_map[uid].to_dict()
        for uid in query.distractor_factor_uids
        if uid in factor_map
    ]
    return {
        "query": query.to_dict(),
        "nodes": path_nodes,
        "steps": steps,
        "distractors": distractors,
    }
