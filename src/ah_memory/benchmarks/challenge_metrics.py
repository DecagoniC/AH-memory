"""Metrics from the AH-memory challenge evaluation specification."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, Protocol, Sequence

from ah_memory.morph import slug_uid


REQUIRED_ROLES = frozenset({"SUBJECT", "OBJECT", "LOCATION"})
DEFAULT_ROLE_WEIGHTS: Mapping[str, float] = {
    "SUBJECT": 2.0,
    "OBJECT": 2.0,
}


@dataclass(frozen=True)
class RoleActant:
    """One role value tied to a corpus item."""

    item_id: Hashable
    role: str
    value: str


@dataclass(frozen=True)
class RoleScore:
    correct: int
    predicted: int
    expected: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RoleExtractionScore:
    by_role: Mapping[str, RoleScore]
    weighted_f1: float
    normalized_weighted_f1: float


@dataclass(frozen=True)
class RoleCorpusItem:
    item_id: Hashable
    text: str
    expected_roles: Sequence[Mapping[str, str]]


class RolePerception(Protocol):
    def parse(self, text: str, wm_context: list[str] | None = None) -> Any: ...


@dataclass(frozen=True)
class InferenceEvaluation:
    correct: bool
    depth: int
    expected_trace: Sequence[str]
    actual_trace: Sequence[str]

    @property
    def trace_complete(self) -> bool:
        expected = iter(self.expected_trace)
        target = next(expected, None)
        if target is None:
            return True
        for uid in self.actual_trace:
            if uid != target:
                continue
            target = next(expected, None)
            if target is None:
                return True
        return False


@dataclass(frozen=True)
class ComparisonScore:
    delta_explainability: float
    delta_hallucination: float


def normalize_role_value(value: str) -> str:
    """Score role values with the same UID contract as perception."""
    return slug_uid(str(value))


def role_extraction_score(
    predicted: Iterable[RoleActant],
    expected: Iterable[RoleActant],
    *,
    role_weights: Mapping[str, float] | None = None,
    required_roles: Iterable[str] = REQUIRED_ROLES,
) -> RoleExtractionScore:
    """Calculate per-role precision/recall/F1 and weighted macro F1 (M1)."""
    predicted_counts = Counter(
        (item.item_id, item.role.upper(), normalize_role_value(item.value))
        for item in predicted
    )
    expected_counts = Counter(
        (item.item_id, item.role.upper(), normalize_role_value(item.value))
        for item in expected
    )
    roles = (
        {role for _, role, _ in predicted_counts}
        | {role for _, role, _ in expected_counts}
        | {role.upper() for role in required_roles}
    )
    weights = {
        role.upper(): float(weight)
        for role, weight in (role_weights or DEFAULT_ROLE_WEIGHTS).items()
    }
    by_role: dict[str, RoleScore] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for role in sorted(roles):
        predicted_for_role = Counter(
            {
                (item_id, value): count
                for (item_id, item_role, value), count in predicted_counts.items()
                if item_role == role
            }
        )
        expected_for_role = Counter(
            {
                (item_id, value): count
                for (item_id, item_role, value), count in expected_counts.items()
                if item_role == role
            }
        )
        correct = sum((predicted_for_role & expected_for_role).values())
        predicted_total = sum(predicted_for_role.values())
        expected_total = sum(expected_for_role.values())
        precision = correct / predicted_total if predicted_total else 0.0
        recall = correct / expected_total if expected_total else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        by_role[role] = RoleScore(
            correct=correct,
            predicted=predicted_total,
            expected=expected_total,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        weight = weights.get(role, 1.0)
        weighted_sum += weight * f1
        total_weight += weight
    return RoleExtractionScore(
        by_role=by_role,
        weighted_f1=weighted_sum,
        normalized_weighted_f1=(
            weighted_sum / total_weight if total_weight else 0.0
        ),
    )


def evaluate_role_corpus(
    perception: RolePerception,
    corpus: Sequence[RoleCorpusItem],
) -> RoleExtractionScore:
    """Run a perception backend and score its role actants end to end."""
    predicted: list[RoleActant] = []
    expected: list[RoleActant] = []
    for item in corpus:
        result = perception.parse(item.text)
        for candidate in result.candidates:
            predicted.extend(
                RoleActant(item.item_id, role, value)
                for role, value in candidate.roles.items()
            )
        for role_map in item.expected_roles:
            expected.extend(
                RoleActant(item.item_id, role, value)
                for role, value in role_map.items()
            )
    return role_extraction_score(predicted, expected)


def explain_score(
    evaluations: Sequence[InferenceEvaluation],
    *,
    d_max: int | None = None,
) -> float:
    """Calculate depth-weighted, trace-gated explanation score (M2)."""
    if not evaluations:
        return 0.0
    maximum = d_max if d_max is not None else max(item.depth for item in evaluations)
    if maximum <= 0:
        raise ValueError("d_max must be positive")
    if any(item.depth < 0 or item.depth > maximum for item in evaluations):
        raise ValueError("inference depth must be between zero and d_max")
    return sum(
        float(item.correct) * (item.depth / maximum) * float(item.trace_complete)
        for item in evaluations
    ) / len(evaluations)


def gc_efficiency(orphan_nodes_before: int, orphan_nodes_after: int) -> float:
    """Calculate the removed orphan fraction (M3)."""
    if orphan_nodes_before < 0 or orphan_nodes_after < 0:
        raise ValueError("orphan counts cannot be negative")
    if orphan_nodes_after > orphan_nodes_before:
        raise ValueError("orphan count after GC cannot exceed the initial count")
    if orphan_nodes_before == 0:
        return 1.0 if orphan_nodes_after == 0 else 0.0
    return 1.0 - orphan_nodes_after / orphan_nodes_before


def hallucination_rate(supported: Iterable[bool]) -> float:
    """Fraction of answers unsupported by any retrievable source."""
    values = list(supported)
    if not values:
        return 0.0
    return sum(not value for value in values) / len(values)


def comparison_score(
    *,
    ah_explain_score: float,
    rag_explain_score: float,
    ah_hallucination: float,
    rag_hallucination: float,
) -> ComparisonScore:
    """Calculate the two independently specified AH-vs-RAG deltas (M4)."""
    return ComparisonScore(
        delta_explainability=ah_explain_score - rag_explain_score,
        delta_hallucination=rag_hallucination - ah_hallucination,
    )


def robustness_gain(
    *,
    ah_slm_f1: float,
    rag_slm_f1: float,
    ah_llm_f1: float,
    rag_llm_f1: float,
) -> float:
    """Calculate the relative SLM advantage over the LLM advantage (M5)."""
    if rag_slm_f1 <= 0.0 or rag_llm_f1 <= 0.0:
        raise ValueError("Vanilla RAG F1 denominators must be positive")
    return ah_slm_f1 / rag_slm_f1 - ah_llm_f1 / rag_llm_f1


def total_score(
    *,
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    m5: float,
) -> float:
    """Calculate the organizer's weighted total once scalar M4 is supplied."""
    return 0.20 * m1 + 0.30 * m2 + 0.15 * m3 + 0.20 * m4 + 0.15 * m5
