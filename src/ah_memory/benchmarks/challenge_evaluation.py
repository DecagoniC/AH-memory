"""End-to-end runners for the challenge role and inference metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence

from ah_memory.benchmarks.challenge_metrics import (
    InferenceEvaluation,
    RoleActant,
    RoleExtractionScore,
    explain_score,
    normalize_role_value,
    robustness_gain,
    role_extraction_score,
)


class PerceptionBackend(Protocol):
    def parse(self, text: str, wm_context: list[str] | None = None) -> Any: ...


@dataclass(frozen=True)
class RoleBenchmarkItem:
    item_id: Hashable
    text: str
    expected_roles: Sequence[Mapping[str, str]]
    noise: str = "clean"


@dataclass
class RoleItemResult:
    item_id: str
    text: str
    noise: str
    expected_roles: list[dict[str, str]]
    predicted_roles: list[dict[str, str]]
    raw_output: Any = None
    validation_errors: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class M1BenchmarkReport:
    model: str
    score: RoleExtractionScore
    items: list[RoleItemResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "score": {
                "weighted_f1": self.score.weighted_f1,
                "normalized_weighted_f1": self.score.normalized_weighted_f1,
                "by_role": {
                    role: asdict(value)
                    for role, value in self.score.by_role.items()
                },
            },
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class InferenceBenchmarkItem:
    item_id: str
    question: str
    expected_answer: str
    depth: int
    expected_trace: Sequence[str]


@dataclass(frozen=True)
class InferenceObservation:
    answer: str
    trace: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class InferenceItemResult:
    item_id: str
    question: str
    expected_answer: str
    actual_answer: str
    depth: int
    expected_trace: list[str]
    actual_trace: list[str]
    correct: bool
    trace_complete: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class M2BenchmarkReport:
    explain_score: float
    d_max: int
    items: list[InferenceItemResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "explain_score": self.explain_score,
            "d_max": self.d_max,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class M5RoleReport:
    ah_slm: M1BenchmarkReport
    rag_slm: M1BenchmarkReport
    ah_llm: M1BenchmarkReport
    rag_llm: M1BenchmarkReport
    robustness_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ah_slm": self.ah_slm.to_dict(),
            "rag_slm": self.rag_slm.to_dict(),
            "ah_llm": self.ah_llm.to_dict(),
            "rag_llm": self.rag_llm.to_dict(),
            "robustness_gain": self.robustness_gain,
        }


def run_m1_benchmark(
    perception: PerceptionBackend,
    corpus: Sequence[RoleBenchmarkItem],
    *,
    model: str,
) -> M1BenchmarkReport:
    """Run one perception request per corpus item and retain audit logs."""
    predicted: list[RoleActant] = []
    expected: list[RoleActant] = []
    item_results: list[RoleItemResult] = []
    for item in corpus:
        expected_maps = [dict(roles) for roles in item.expected_roles]
        expected.extend(
            RoleActant(item.item_id, role, normalize_role_value(value))
            for roles in expected_maps
            for role, value in roles.items()
        )
        try:
            result = perception.parse(item.text)
            predicted_maps = [
                dict(candidate.roles) for candidate in result.candidates
            ]
            predicted.extend(
                RoleActant(item.item_id, role, value)
                for roles in predicted_maps
                for role, value in roles.items()
            )
            meta = dict(getattr(result, "meta", {}) or {})
            item_results.append(
                RoleItemResult(
                    item_id=str(item.item_id),
                    text=item.text,
                    noise=item.noise,
                    expected_roles=expected_maps,
                    predicted_roles=predicted_maps,
                    raw_output=meta.get("llm_raw"),
                    validation_errors=list(meta.get("validation_errors") or ()),
                )
            )
        except Exception as exc:  # noqa: BLE001 - benchmark must log failed items
            item_results.append(
                RoleItemResult(
                    item_id=str(item.item_id),
                    text=item.text,
                    noise=item.noise,
                    expected_roles=expected_maps,
                    predicted_roles=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return M1BenchmarkReport(
        model=model,
        score=role_extraction_score(predicted, expected),
        items=item_results,
    )


def run_m2_benchmark(
    corpus: Sequence[InferenceBenchmarkItem],
    answer: Callable[[InferenceBenchmarkItem], InferenceObservation],
    *,
    d_max: int = 6,
) -> M2BenchmarkReport:
    """Run QA cases and gate each answer by its actual ordered UID trace."""
    evaluations: list[InferenceEvaluation] = []
    results: list[InferenceItemResult] = []
    for item in corpus:
        observation = answer(item)
        correct = _answer_equal(observation.answer, item.expected_answer)
        evaluation = InferenceEvaluation(
            correct=correct,
            depth=item.depth,
            expected_trace=item.expected_trace,
            actual_trace=observation.trace,
        )
        evaluations.append(evaluation)
        results.append(
            InferenceItemResult(
                item_id=item.item_id,
                question=item.question,
                expected_answer=item.expected_answer,
                actual_answer=observation.answer,
                depth=item.depth,
                expected_trace=list(item.expected_trace),
                actual_trace=list(observation.trace),
                correct=correct,
                trace_complete=evaluation.trace_complete,
                metadata=dict(observation.metadata),
            )
        )
    return M2BenchmarkReport(
        explain_score=explain_score(evaluations, d_max=d_max),
        d_max=d_max,
        items=results,
    )


def run_m5_role_benchmark(
    corpus: Sequence[RoleBenchmarkItem],
    *,
    ah_slm: PerceptionBackend,
    rag_slm: PerceptionBackend,
    ah_llm: PerceptionBackend,
    rag_llm: PerceptionBackend,
    model_names: Mapping[str, str],
) -> M5RoleReport:
    """Run the same M1 corpus through the AH/RAG by SLM/LLM matrix."""
    runs = {
        "ah_slm": run_m1_benchmark(
            ah_slm, corpus, model=model_names.get("ah_slm", "ah_slm")
        ),
        "rag_slm": run_m1_benchmark(
            rag_slm, corpus, model=model_names.get("rag_slm", "rag_slm")
        ),
        "ah_llm": run_m1_benchmark(
            ah_llm, corpus, model=model_names.get("ah_llm", "ah_llm")
        ),
        "rag_llm": run_m1_benchmark(
            rag_llm, corpus, model=model_names.get("rag_llm", "rag_llm")
        ),
    }
    return M5RoleReport(
        **runs,
        robustness_gain=robustness_gain(
            ah_slm_f1=runs["ah_slm"].score.normalized_weighted_f1,
            rag_slm_f1=runs["rag_slm"].score.normalized_weighted_f1,
            ah_llm_f1=runs["ah_llm"].score.normalized_weighted_f1,
            rag_llm_f1=runs["rag_llm"].score.normalized_weighted_f1,
        ),
    )


def observation_from_agent_reply(reply: Any) -> InferenceObservation:
    """Adapt an AgentReply without weakening the runtime trace contract."""
    full_trace = dict(getattr(reply, "full_trace", {}) or {})
    return InferenceObservation(
        answer=str(getattr(reply, "answer", full_trace.get("answer", ""))),
        trace=tuple(str(uid) for uid in full_trace.get("activated_factors", ())),
        metadata={
            "source": str(getattr(reply, "source", "")),
            "activated_nodes": list(full_trace.get("activated_nodes", ())),
            "timesteps": list(full_trace.get("timesteps", ())),
        },
    )


def _answer_equal(actual: str, expected: str) -> bool:
    normalize = lambda value: " ".join(str(value).strip().casefold().split())
    return normalize(actual) == normalize(expected)
