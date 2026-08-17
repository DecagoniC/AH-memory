"""Offline challenge corpora and M4/M5 comparison infrastructure."""

from ah_memory.benchmarks.challenge.loader import load_qa_corpus, load_role_corpus
from ah_memory.benchmarks.challenge.schema import QAItem, RoleCorpusItem

from ah_memory.benchmarks.challenge.comparison import (
    ArmEvaluation,
    M4ComparisonResult,
    RobustnessResult,
    compare_ah_to_rag,
    evaluate_robustness,
)
from ah_memory.benchmarks.challenge.protocols import (
    AnswerRecord,
    ArmResult,
    BenchmarkQuery,
    ChatBackend,
    ComparisonArm,
    Embedder,
    SourceDocument,
    StructuredAnswer,
)
from ah_memory.benchmarks.challenge.ngram import DeterministicNgramEmbedder
from ah_memory.benchmarks.challenge.rag import AnswerScorer, VanillaRAG
from ah_memory.benchmarks.challenge.retrieval import (
    CosineVectorRetriever,
    RetrievedDocument,
)
from ah_memory.benchmarks.challenge.scoring import (
    SupportScore,
    answer_hallucination_rate,
    score_support,
    support_explainability,
)

__all__ = [
    "AnswerRecord",
    "AnswerScorer",
    "ArmEvaluation",
    "ArmResult",
    "BenchmarkQuery",
    "ChatBackend",
    "ComparisonArm",
    "CosineVectorRetriever",
    "DeterministicNgramEmbedder",
    "Embedder",
    "M4ComparisonResult",
    "QAItem",
    "RetrievedDocument",
    "RoleCorpusItem",
    "RobustnessResult",
    "SourceDocument",
    "StructuredAnswer",
    "SupportScore",
    "VanillaRAG",
    "answer_hallucination_rate",
    "compare_ah_to_rag",
    "evaluate_robustness",
    "load_qa_corpus",
    "load_role_corpus",
    "score_support",
    "support_explainability",
]
