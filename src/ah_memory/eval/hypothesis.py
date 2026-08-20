"""H1/H2 protocol: neuro-symbolic AH vs classical LLM+vector RAG.

H1 explainability: on in-corpus items, trace-gated ExplainScore(AH) > ExplainScore(RAG).
    RAG has no UID trace, so official RAG explain is 0 (постановка M4).
H2 hallucination: on trap items, hallucination_rate(RAG) > hallucination_rate(AH).

The runner can isolate a fresh graph per question (fair OOD) or reuse one agent
(sequential WM). Verdicts are supported / tied / rejected — tests may accept a
rejected claim when documenting a real limitation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG, VanillaRAGReply
from ah_memory.eval.m4 import (
    GoldItem,
    ItemScore,
    M4Report,
    _ah_hallucinated,
    _keyword_hit,
    _rag_hallucinated,
    _trace_complete,
    evaluate_m4,
)

Verdict = Literal["supported", "tied", "rejected"]
AgentFactory = Callable[[], Agent]
RagFactory = Callable[[], VanillaRAG]


def inventing_generator(invention: str) -> Callable[[str, list[str]], str]:
    """Stand-in for a BЯМ that mixes retrieved text with ungrounded claims."""

    def generate(question: str, chunks: list[str]) -> str:
        del question
        body = " ".join(chunks).strip() or "нет фрагментов"
        return f"{body} {invention}".strip()

    return generate


def make_inventing_rag(corpus: str, invention: str, *, top_k: int = 4) -> VanillaRAG:
    return VanillaRAG(corpus, top_k=top_k, generator=inventing_generator(invention))


@dataclass(frozen=True)
class ClaimResult:
    name: str
    verdict: Verdict
    delta: float
    ah_score: float
    rag_score: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisReport:
    protocol: str
    rag_backend: str
    h1_explainability: ClaimResult
    h2_hallucination: ClaimResult
    in_corpus: M4Report
    traps: M4Report

    @property
    def overall(self) -> Verdict:
        if (
            self.h1_explainability.verdict == "supported"
            and self.h2_hallucination.verdict == "supported"
        ):
            return "supported"
        if (
            self.h1_explainability.verdict == "rejected"
            and self.h2_hallucination.verdict == "rejected"
        ):
            return "rejected"
        return "tied"

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "rag_backend": self.rag_backend,
            "overall": self.overall,
            "h1_explainability": self.h1_explainability.as_dict(),
            "h2_hallucination": self.h2_hallucination.as_dict(),
            "in_corpus": self.in_corpus.as_dict(),
            "traps": self.traps.as_dict(),
            "in_corpus_items": [_item_dict(item) for item in self.in_corpus.items],
            "trap_items": [_item_dict(item) for item in self.traps.items],
        }


def _item_dict(item: ItemScore) -> dict[str, Any]:
    return {
        "question": item.question,
        "ah_answer": item.ah_answer,
        "rag_answer": item.rag_answer,
        "ah_correct": item.ah_correct,
        "rag_correct": item.rag_correct,
        "ah_trace_complete": item.ah_trace_complete,
        "ah_hallucinated": item.ah_hallucinated,
        "rag_hallucinated": item.rag_hallucinated,
        "ah_explain": item.ah_explain,
        "ah_trace": list(item.ah_trace[:16]),
    }


def _verdict(delta: float, *, higher_is_better: bool = True) -> Verdict:
    if abs(delta) < 1e-12:
        return "tied"
    positive = delta > 0
    if not higher_is_better:
        positive = not positive
    return "supported" if positive else "rejected"


def _empty_report() -> M4Report:
    return M4Report(
        items=[],
        explain_ah=0.0,
        explain_rag=0.0,
        hall_ah=0.0,
        hall_rag=0.0,
        delta_explainability=0.0,
        delta_hallucination=0.0,
        d_max=1,
    )


def evaluate_pair(
    agent: Agent,
    rag: VanillaRAG,
    gold: list[GoldItem],
    *,
    ticks: int = 6,
) -> M4Report:
    if not gold:
        return _empty_report()
    return evaluate_m4(agent, rag, gold, ticks=ticks)


def evaluate_isolated(
    make_agent: AgentFactory,
    make_rag: RagFactory,
    gold: list[GoldItem],
    *,
    ticks: int = 6,
) -> M4Report:
    """Fresh AH graph and RAG index per question (no WM carry-over)."""
    if not gold:
        return _empty_report()
    d_max = max((item.d for item in gold), default=1) or 1
    items: list[ItemScore] = []
    for case in gold:
        one = evaluate_m4(make_agent(), make_rag(), [case], ticks=ticks)
        scored = one.items[0]
        ah_explain = (
            (1.0 if scored.ah_correct else 0.0)
            * (case.d / d_max)
            * (1.0 if scored.ah_trace_complete else 0.0)
        )
        items.append(
            ItemScore(
                question=scored.question,
                ah_answer=scored.ah_answer,
                rag_answer=scored.rag_answer,
                ah_correct=scored.ah_correct,
                rag_correct=scored.rag_correct,
                ah_trace_complete=scored.ah_trace_complete,
                ah_hallucinated=scored.ah_hallucinated,
                rag_hallucinated=scored.rag_hallucinated,
                ah_explain=ah_explain,
                rag_explain=0.0,
                ah_trace=list(scored.ah_trace),
                rag_chunks=list(scored.rag_chunks),
            )
        )
    n = max(1, len(items))
    explain_ah = sum(item.ah_explain for item in items) / n
    hall_ah = sum(1 for item in items if item.ah_hallucinated) / n
    hall_rag = sum(1 for item in items if item.rag_hallucinated) / n
    return M4Report(
        items=items,
        explain_ah=explain_ah,
        explain_rag=0.0,
        hall_ah=hall_ah,
        hall_rag=hall_rag,
        delta_explainability=explain_ah,
        delta_hallucination=hall_rag - hall_ah,
        d_max=d_max,
    )


def evaluate_hypothesis(
    *,
    make_agent: AgentFactory,
    make_rag: RagFactory,
    in_corpus: list[GoldItem],
    traps: list[GoldItem],
    protocol: Literal["isolated", "sequential"] = "isolated",
    ticks: int = 6,
) -> HypothesisReport:
    rag_probe = make_rag()
    if protocol == "isolated":
        in_rep = evaluate_isolated(make_agent, make_rag, in_corpus, ticks=ticks)
        trap_rep = evaluate_isolated(make_agent, make_rag, traps, ticks=ticks)
    else:
        agent = make_agent()
        rag = make_rag()
        in_rep = evaluate_pair(agent, rag, in_corpus, ticks=ticks)
        trap_rep = evaluate_pair(agent, rag, traps, ticks=ticks)
    h1 = ClaimResult(
        name="explainability",
        verdict=_verdict(in_rep.delta_explainability),
        delta=in_rep.delta_explainability,
        ah_score=in_rep.explain_ah,
        rag_score=in_rep.explain_rag,
        n=len(in_rep.items),
    )
    h2 = ClaimResult(
        name="hallucination_resistance",
        verdict=_verdict(trap_rep.delta_hallucination),
        delta=trap_rep.delta_hallucination,
        ah_score=trap_rep.hall_ah,
        rag_score=trap_rep.hall_rag,
        n=len(trap_rep.items),
    )
    return HypothesisReport(
        protocol=protocol,
        rag_backend=getattr(rag_probe, "backend", "unknown"),
        h1_explainability=h1,
        h2_hallucination=h2,
        in_corpus=in_rep,
        traps=trap_rep,
    )


def rag_has_uid_trace(reply: VanillaRAGReply) -> bool:
    return bool(reply.trace_uids)


def ah_answer_is_abstain(answer: str) -> bool:
    return _keyword_hit(answer, ["неизвестно", "нет данных", "не знаю", "unknown"])


def trap_is_hallucinated_ah(agent: Agent, answer: str, trace: list[str], gold: GoldItem) -> bool:
    return _ah_hallucinated(agent, answer, trace, gold)


def trap_is_hallucinated_rag(answer: str, chunks: list[str], gold: GoldItem) -> bool:
    return _rag_hallucinated(answer, chunks, gold)


def trace_contains_gold(trace: list[str], gold_uids: list[str]) -> bool:
    return _trace_complete(trace, gold_uids)
