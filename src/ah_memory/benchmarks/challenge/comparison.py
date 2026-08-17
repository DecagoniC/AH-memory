"""M4 and M5 orchestration over neutral benchmark arms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ah_memory.benchmarks.challenge.protocols import (
    ArmResult,
    BenchmarkQuery,
    ComparisonArm,
    SourceDocument,
)
from ah_memory.benchmarks.challenge.scoring import answer_hallucination_rate
from ah_memory.benchmarks.challenge_metrics import (
    ComparisonScore,
    comparison_score,
    robustness_gain,
)


@dataclass(frozen=True)
class ArmEvaluation:
    result: ArmResult
    hallucination: float


@dataclass(frozen=True)
class M4ComparisonResult:
    ah: ArmEvaluation
    rag: ArmEvaluation
    score: ComparisonScore

    @property
    def delta_explainability(self) -> float:
        return self.score.delta_explainability

    @property
    def delta_hallucination(self) -> float:
        return self.score.delta_hallucination


@dataclass(frozen=True)
class RobustnessResult:
    ah_slm: ArmResult
    rag_slm: ArmResult
    ah_llm: ArmResult
    rag_llm: ArmResult
    gain: float


def compare_ah_to_rag(
    ah_arm: ComparisonArm,
    rag_arm: ComparisonArm,
    corpus: Sequence[SourceDocument],
    queries: Sequence[BenchmarkQuery],
) -> M4ComparisonResult:
    """Run both arms on the same immutable inputs and calculate M4 deltas."""
    shared_corpus = tuple(corpus)
    shared_queries = tuple(queries)
    ah_result = ah_arm.run(shared_corpus, shared_queries)
    rag_result = rag_arm.run(shared_corpus, shared_queries)
    ah_hallucination = answer_hallucination_rate(ah_result.records)
    rag_hallucination = answer_hallucination_rate(rag_result.records)
    return M4ComparisonResult(
        ah=ArmEvaluation(ah_result, ah_hallucination),
        rag=ArmEvaluation(rag_result, rag_hallucination),
        score=comparison_score(
            ah_explain_score=ah_result.explainability,
            rag_explain_score=rag_result.explainability,
            ah_hallucination=ah_hallucination,
            rag_hallucination=rag_hallucination,
        ),
    )


def evaluate_robustness(
    *,
    ah_slm: ComparisonArm,
    rag_slm: ComparisonArm,
    ah_llm: ComparisonArm,
    rag_llm: ComparisonArm,
    corpus: Sequence[SourceDocument],
    queries: Sequence[BenchmarkQuery],
) -> RobustnessResult:
    """Run the AH/RAG by SLM/LLM four-arm matrix and calculate M5."""
    shared_corpus = tuple(corpus)
    shared_queries = tuple(queries)
    ah_slm_result = ah_slm.run(shared_corpus, shared_queries)
    rag_slm_result = rag_slm.run(shared_corpus, shared_queries)
    ah_llm_result = ah_llm.run(shared_corpus, shared_queries)
    rag_llm_result = rag_llm.run(shared_corpus, shared_queries)
    return RobustnessResult(
        ah_slm=ah_slm_result,
        rag_slm=rag_slm_result,
        ah_llm=ah_llm_result,
        rag_llm=rag_llm_result,
        gain=robustness_gain(
            ah_slm_f1=ah_slm_result.f1,
            rag_slm_f1=rag_slm_result.f1,
            ah_llm_f1=ah_llm_result.f1,
            rag_llm_f1=rag_llm_result.f1,
        ),
    )
