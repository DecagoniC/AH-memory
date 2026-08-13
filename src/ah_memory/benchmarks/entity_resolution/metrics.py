"""Metrics for entity resolution (Accuracy, P/R/F1, False Merge)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ah_memory.benchmarks.entity_resolution.cases import CaseType, EntityResolutionCase


DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
)


@dataclass
class CaseOutcome:
    case_id: str
    case_type: str
    mention: str
    expected_uid: str | None
    predicted_uid: str | None
    correct: bool
    false_merge: bool
    false_positive: bool
    false_negative: bool
    similarity: float | None = None
    method: str = ""
    activation_ok: bool | None = None
    activation_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "mention": self.mention,
            "expected_uid": self.expected_uid,
            "resolved_uid": self.predicted_uid,
            "correct": self.correct,
            "false_merge": self.false_merge,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "similarity": self.similarity,
            "method": self.method,
            "activation_ok": self.activation_ok,
            "activation_trace": list(self.activation_trace),
        }


@dataclass
class MetricBundle:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_merge_rate: float
    false_positive_rate: float
    false_negative_rate: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_merge_rate": self.false_merge_rate,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "n": self.n,
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classify_outcome(
    case: EntityResolutionCase,
    predicted_uid: str | None,
    *,
    similarity: float | None = None,
    method: str = "",
) -> CaseOutcome:
    expected = case.target_uid
    correct = predicted_uid == expected

    # False merge: selected a wrong existing symbol (identity error).
    false_merge = False
    if case.case_type in {
        CaseType.NEGATIVE,
        CaseType.SEMANTIC_NEAR,
        CaseType.AMBIGUOUS,
        CaseType.HOLD_OUT,
    }:
        if expected is None:
            false_merge = predicted_uid is not None
        else:
            false_merge = (
                predicted_uid is not None and predicted_uid != expected
            )
        # notes may mark a distractor that must never be chosen
        if case.notes.startswith("must_not_merge_with="):
            bad = case.notes.split("=", 1)[1]
            if predicted_uid == bad:
                false_merge = True
                correct = False
        if case.notes.startswith("force_reject="):
            false_merge = predicted_uid is not None
            correct = predicted_uid is None
        elif case.notes.startswith("ambiguous_without_context"):
            false_merge = predicted_uid is not None
            correct = predicted_uid is None

    # Binary IR view: positive cases have a target; negatives with None are reject.
    is_positive = expected is not None
    false_positive = (not is_positive and predicted_uid is not None) or (
        is_positive and predicted_uid is not None and predicted_uid != expected
    )
    false_negative = is_positive and predicted_uid is None

    return CaseOutcome(
        case_id=case.case_id,
        case_type=case.case_type.value,
        mention=case.mention,
        expected_uid=expected,
        predicted_uid=predicted_uid,
        correct=correct,
        false_merge=false_merge,
        false_positive=false_positive,
        false_negative=false_negative,
        similarity=similarity,
        method=method,
    )


def aggregate_outcomes(outcomes: Iterable[CaseOutcome]) -> MetricBundle:
    rows = list(outcomes)
    n = len(rows)
    if n == 0:
        return MetricBundle(0, 0, 0, 0, 0, 0, 0, 0)
    tp = sum(
        1
        for o in rows
        if o.expected_uid is not None and o.predicted_uid == o.expected_uid
    )
    fp = sum(1 for o in rows if o.false_positive)
    fn = sum(1 for o in rows if o.false_negative)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    neg = [
        o
        for o in rows
        if o.case_type in {"NEGATIVE", "SEMANTIC_NEAR", "AMBIGUOUS", "HOLD_OUT"}
    ]
    false_merge_rate = _safe_div(
        sum(1 for o in neg if o.false_merge),
        len(neg),
    )
    return MetricBundle(
        accuracy=_safe_div(sum(1 for o in rows if o.correct), n),
        precision=precision,
        recall=recall,
        f1=f1,
        false_merge_rate=false_merge_rate,
        false_positive_rate=_safe_div(fp, n),
        false_negative_rate=_safe_div(fn, n),
        n=n,
    )


def by_case_type(outcomes: list[CaseOutcome]) -> dict[str, MetricBundle]:
    groups: dict[str, list[CaseOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault(outcome.case_type, []).append(outcome)
    return {key: aggregate_outcomes(rows) for key, rows in sorted(groups.items())}


def find_optimal_threshold(
    sweep: dict[float, MetricBundle],
    *,
    max_false_merge: float = 0.01,
) -> dict[str, Any]:
    if not sweep:
        return {"f1_optimal": None, "safety_optimal": None}
    f1_thr = max(sweep.items(), key=lambda item: (item[1].f1, -item[0]))[0]
    safe = [
        (thr, bundle)
        for thr, bundle in sweep.items()
        if bundle.false_merge_rate <= max_false_merge
    ]
    if safe:
        safety_thr = max(safe, key=lambda item: (item[1].f1, -item[0]))[0]
    else:
        # fallback: lowest false merge, then best F1
        safety_thr = min(
            sweep.items(),
            key=lambda item: (item[1].false_merge_rate, -item[1].f1, item[0]),
        )[0]
    return {
        "f1_optimal": f1_thr,
        "safety_optimal": safety_thr,
        "f1_optimal_metrics": sweep[f1_thr].to_dict(),
        "safety_optimal_metrics": sweep[safety_thr].to_dict(),
    }
