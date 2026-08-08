"""Deterministic memory-aggregation benchmark and comparison modes."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ah_memory.factor_graph import build_structural_factor_graph
from ah_memory.factor_parameters import (
    EmbeddingParameterGenerator,
    FixedParameterGenerator,
    RuleBasedParameterGenerator,
)
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.relation_normalizer import (
    EmbeddingNormalizer,
    ExactNormalizer,
    RelationNormalizer,
)
from ah_memory.semantic_activation import ActivationEngine, PropagationTrace
from ah_memory.store import AHStore
from ah_memory.transform import Transform

BenchmarkMode = Literal["fixed", "normalized", "learned"]

_VERBS = (
    "приобрёл",
    "приобрел",
    "реализовал",
    "продала",
    "продал",
    "купила",
    "купил",
)
_FIXED = {
    "купил": "PURCHASE",
    "купила": "PURCHASE",
    "продал": "SELL",
    "продала": "SELL",
}
_ENTITY_ALIASES = {
    "отец": "FATHER",
    "папа": "FATHER",
    "мать": "MOTHER",
    "мама": "MOTHER",
}
_TEMPORAL_MARKERS = ("потом", "затем", "после", "позже")


@dataclass(frozen=True)
class ExtractedEvent:
    raw_relation: str
    canonical_relation: str
    subject: str
    object: str
    temporal_after: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregationResult:
    scenario: str
    mode: BenchmarkMode
    metrics: dict[str, float]
    events: list[dict[str, Any]]
    state: dict[str, Any]
    trace: dict[str, Any]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkEventExtractor:
    """Benchmark-only deterministic parser; production extraction remains LLM-based."""

    def __init__(self, mode: BenchmarkMode, store: AHStore) -> None:
        self.mode = mode
        self.normalizer = RelationNormalizer(
            store.relations,
            (
                ExactNormalizer(),
                EmbeddingNormalizer(similarity_threshold=0.35),
            ),
        )

    def extract(self, text: str) -> list[ExtractedEvent]:
        current_subject: str | None = None
        events: list[ExtractedEvent] = []
        for sentence in re.split(r"[.!?]+", text):
            clause = " ".join(sentence.strip().lower().replace("ё", "е").split())
            if not clause:
                continue
            verb = next(
                (
                    candidate
                    for candidate in _VERBS
                    if re.search(rf"\b{re.escape(candidate.replace('ё', 'е'))}\b", clause)
                ),
                None,
            )
            if verb is None:
                continue
            normalized_verb = verb.replace("ё", "е")
            match = re.search(rf"\b{re.escape(normalized_verb)}\b", clause)
            if match is None:
                continue
            prefix = clause[: match.start()].strip()
            suffix = clause[match.end() :].strip()
            prefix_tokens = [
                token
                for token in re.findall(r"[a-zа-я0-9]+", prefix)
                if token not in {*_TEMPORAL_MARKERS, "сначала", "этого", "снова"}
            ]
            if prefix_tokens and prefix_tokens[-1] not in {"он", "она"}:
                current_subject = self._entity(prefix_tokens[-1])
            if current_subject is None:
                continue
            object_tokens = re.findall(r"[a-zа-я0-9]+", suffix)
            if not object_tokens:
                continue
            object_uid = self._entity(object_tokens[0])
            canonical = self._canonical(verb)
            events.append(
                ExtractedEvent(
                    raw_relation=verb,
                    canonical_relation=canonical,
                    subject=current_subject,
                    object=object_uid,
                    temporal_after=any(marker in clause for marker in _TEMPORAL_MARKERS),
                )
            )
        return events

    def _canonical(self, raw_relation: str) -> str:
        if self.mode == "fixed":
            return _FIXED.get(raw_relation, raw_relation.upper())
        return self.normalizer.normalize(raw_relation).canonical_label

    @staticmethod
    def _entity(raw: str) -> str:
        cleaned = raw.strip().upper().replace("Ё", "Е")
        return _ENTITY_ALIASES.get(raw.lower().replace("ё", "е"), cleaned)


def load_scenarios(directory: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(directory).glob("*.json"))
    ]


def run_scenario(
    scenario: dict[str, Any],
    mode: BenchmarkMode,
    *,
    trace: bool = False,
) -> AggregationResult:
    started = time.perf_counter()
    store = AHStore()
    extractor = BenchmarkEventExtractor(mode, store)
    extracted = extractor.extract(str(scenario["input"]))
    if mode == "fixed":
        generator = FixedParameterGenerator()
    elif mode == "normalized":
        generator = RuleBasedParameterGenerator()
    else:
        generator = EmbeddingParameterGenerator(seed=42)
    transform = Transform(store, parameter_generator=generator)
    candidates = [
        FactCandidate(
            predicate=event.canonical_relation,
            raw_relation=event.raw_relation,
            canonical_relation=event.canonical_relation,
            roles={"SUBJECT": event.subject, "OBJECT": event.object},
            confidence=1.0,
        )
        for event in extracted
    ]
    transform.apply(PerceptionResult(kind="fact", candidates=candidates))
    graph = build_structural_factor_graph(store)
    expected_state = dict(scenario.get("expected_state") or {})
    relevant = _currently_owned(expected_state)
    subjects = {
        event["subject"] for event in scenario.get("expected_events", [])
    }
    evidence = {f"M_{subject}": 1.0 for subject in subjects}
    activation, propagation = ActivationEngine(
        parameter_generator=generator,
    ).run(
        graph,
        evidence,
        timesteps=4,
        threshold=0.05,
        trace=trace,
    )
    metrics = _metrics(
        scenario,
        extracted,
        store,
        activation,
        propagation,
        relevant,
    )
    return AggregationResult(
        scenario=str(scenario.get("name", "scenario")),
        mode=mode,
        metrics=metrics,
        events=[event.to_dict() for event in extracted],
        state=store.state.to_dict(),
        trace=propagation.to_dict(),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def run_suite(
    directory: str | Path,
    mode: BenchmarkMode,
    *,
    trace: bool = False,
) -> dict[str, Any]:
    results = [
        run_scenario(scenario, mode, trace=trace)
        for scenario in load_scenarios(directory)
    ]
    metric_names = sorted(
        set().union(*(result.metrics.keys() for result in results))
    )
    aggregate: dict[str, float] = {}
    for metric in metric_names:
        values = [
            result.metrics[metric]
            for result in results
            if math.isfinite(result.metrics[metric])
        ]
        aggregate[metric] = (
            sum(values) / len(values) if values else math.inf
        )
    aggregate["latency_ms"] = sum(result.latency_ms for result in results) / len(
        results
    )
    return {
        "mode": mode,
        "aggregate": aggregate,
        "scenarios": [result.to_dict() for result in results],
    }


def format_trace(result: AggregationResult) -> str:
    lines: list[str] = []
    timesteps = result.trace.get("timesteps") or []
    for index, snapshot in enumerate(timesteps):
        lines.append(f"t={index}")
        for uid, value in sorted(snapshot.items(), key=lambda item: -item[1]):
            if value > 0.001:
                lines.append(f"{uid}: {value:.3f}")
    for message in result.trace.get("messages") or []:
        metadata = message.get("metadata") or {}
        lines.append(
            "message "
            f"{message['source_uid']} -> {message['target_uid']} "
            f"factor={message['factor_uid']} "
            f"relation={metadata.get('relation')} "
            f"weight={metadata.get('weight')} "
            f"confidence={metadata.get('confidence')} "
            f"before={metadata.get('activation_before')} "
            f"after={metadata.get('activation_after')} "
            f"parameters={metadata.get('parameters')}"
        )
    return "\n".join(lines)


def _metrics(
    scenario: dict[str, Any],
    extracted: list[ExtractedEvent],
    store: AHStore,
    activation: dict[str, float],
    trace: PropagationTrace,
    relevant: set[str],
) -> dict[str, float]:
    expected_events = list(scenario.get("expected_events") or [])
    expected_relations = set(scenario.get("expected_relations") or [])
    actual_triples = {
        (event.canonical_relation, event.subject, event.object)
        for event in extracted
    }
    expected_triples = {
        (event["relation"], event["subject"], event["object"])
        for event in expected_events
    }
    event_accuracy = len(actual_triples & expected_triples) / max(
        1,
        len(expected_triples),
    )
    actual_relations = {event.canonical_relation for event in extracted}
    if any(event.temporal_after for event in extracted):
        actual_relations.add("AFTER")
    relation_accuracy = len(actual_relations & expected_relations) / max(
        1,
        len(expected_relations),
    )
    state_accuracy = _state_accuracy(
        store.state,
        dict(scenario.get("expected_state") or {}),
    )

    ranking = [
        uid
        for uid, _ in sorted(
            activation.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not uid.startswith("M_FATHER") and not uid.startswith("M_MOTHER")
    ]
    relevant_graph = {f"M_{uid}" for uid in relevant}
    k = max(1, len(relevant_graph))
    recall_at_k = len(set(ranking[:k]) & relevant_graph) / max(
        1,
        len(relevant_graph),
    )
    reciprocal_ranks = [
        1.0 / (ranking.index(uid) + 1)
        for uid in relevant_graph
        if uid in ranking
    ]
    mrr = max(reciprocal_ranks, default=0.0)
    total_activation = sum(activation.values())
    relevant_activation = sum(activation.get(uid, 0.0) for uid in relevant_graph)
    activation_precision = relevant_activation / max(1e-12, total_activation)
    activated = {uid for uid, value in activation.items() if value >= 0.05}
    activation_recall = len(activated & relevant_graph) / max(1, len(relevant_graph))
    traced_relations = {
        relation["canonical_label"] for relation in trace.relations
    }
    path_accuracy = len(traced_relations & expected_relations) / max(
        1,
        len(expected_relations),
    )
    propagation_latency = 0.0 if not relevant_graph else math.inf
    if relevant_graph:
        for timestep, snapshot in enumerate(trace.timesteps):
            if any(snapshot.get(uid, 0.0) >= 0.05 for uid in relevant_graph):
                propagation_latency = float(timestep)
                break
    return {
        "relation_normalization_accuracy": relation_accuracy,
        "event_extraction_accuracy": event_accuracy,
        "state_accuracy": state_accuracy,
        "retrieval_accuracy": recall_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "activation_precision": activation_precision,
        "activation_recall": activation_recall,
        "path_accuracy": path_accuracy,
        "propagation_latency": propagation_latency,
    }


def _currently_owned(expected_state: dict[str, Any]) -> set[str]:
    return {
        object_uid
        for objects in (expected_state.get("owns") or {}).values()
        for object_uid, owned in objects.items()
        if owned
    }


def _state_accuracy(state, expected: dict[str, Any]) -> float:
    checks: list[bool] = []
    for owner, objects in (expected.get("owns") or {}).items():
        checks.extend(
            state.owns(owner, object_uid) is bool(value)
            for object_uid, value in objects.items()
        )
    for owner, value in (expected.get("last_purchase") or {}).items():
        checks.append(state.last_purchase(owner) == value)
    for owner, values in (expected.get("purchase_history") or {}).items():
        checks.append(state.purchase_history(owner) == list(values))
    return sum(checks) / max(1, len(checks))
