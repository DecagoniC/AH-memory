"""M4: Δ_explainability and Δ_hallucination vs Vanilla RAG (постановка §7)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG

_TOKEN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", re.UNICODE)


@dataclass
class GoldItem:
    question: str
    """Gold answer keywords (any hit → correct)."""
    answer_keywords: list[str]
    """UID chain that should appear in AH trace (FOLLOW/IS-A/CAUSE path)."""
    gold_trace: list[str]
    """Chain depth d for ExplainScore weight."""
    d: int = 1
    """Corpus substrings that ground a non-hallucinated RAG answer."""
    evidence_spans: list[str] = field(default_factory=list)


@dataclass
class ItemScore:
    question: str
    ah_answer: str
    rag_answer: str
    ah_correct: bool
    rag_correct: bool
    ah_trace_complete: bool
    ah_hallucinated: bool
    rag_hallucinated: bool
    ah_explain: float
    rag_explain: float
    ah_trace: list[str]
    rag_chunks: list[str]


@dataclass
class M4Report:
    items: list[ItemScore]
    explain_ah: float
    explain_rag: float
    hall_ah: float
    hall_rag: float
    delta_explainability: float
    delta_hallucination: float
    d_max: int

    def as_dict(self) -> dict:
        return {
            "ExplainScore_AH": round(self.explain_ah, 4),
            "ExplainScore_VanillaRAG": round(self.explain_rag, 4),
            "Hallucination_AH": round(self.hall_ah, 4),
            "Hallucination_VanillaRAG": round(self.hall_rag, 4),
            "delta_explainability": round(self.delta_explainability, 4),
            "delta_hallucination": round(self.delta_hallucination, 4),
            "n": len(self.items),
            "explain_hypothesis_ok": self.delta_explainability > 0,
            "hall_hypothesis_ok": self.delta_hallucination > 0,
            "hypothesis_ok": self.delta_explainability > 0 and self.delta_hallucination > 0,
        }


def evaluate_m4(
    agent: Agent,
    rag: VanillaRAG,
    gold: list[GoldItem],
    *,
    ticks: int = 6,
) -> M4Report:
    d_max = max((g.d for g in gold), default=1) or 1
    items: list[ItemScore] = []
    for g in gold:
        ah = agent.ask(g.question, ticks=ticks)
        vr = rag.ask(g.question)
        ah_ok = _keyword_hit(ah.answer, g.answer_keywords)
        rag_ok = _keyword_hit(vr.answer, g.answer_keywords)
        ah_trace_ok = _trace_complete(ah.trace_uids, g.gold_trace)
        # M2: zero contribution if no faithful trace
        ah_explain = (1.0 if ah_ok else 0.0) * (g.d / d_max) * (1.0 if ah_trace_ok else 0.0)
        # Vanilla RAG has no UID trace → trace_complete=0 by постановка M4
        rag_explain = 0.0
        ah_hall = _ah_hallucinated(agent, ah.answer, ah.trace_uids, g)
        rag_hall = _rag_hallucinated(vr.answer, vr.chunks, g)
        items.append(
            ItemScore(
                question=g.question,
                ah_answer=ah.answer,
                rag_answer=vr.answer,
                ah_correct=ah_ok,
                rag_correct=rag_ok,
                ah_trace_complete=ah_trace_ok,
                ah_hallucinated=ah_hall,
                rag_hallucinated=rag_hall,
                ah_explain=ah_explain,
                rag_explain=rag_explain,
                ah_trace=list(ah.trace_uids),
                rag_chunks=list(vr.chunks),
            )
        )
    n = max(1, len(items))
    explain_ah = sum(i.ah_explain for i in items) / n
    explain_rag = sum(i.rag_explain for i in items) / n
    hall_ah = sum(1 for i in items if i.ah_hallucinated) / n
    hall_rag = sum(1 for i in items if i.rag_hallucinated) / n
    return M4Report(
        items=items,
        explain_ah=explain_ah,
        explain_rag=explain_rag,
        hall_ah=hall_ah,
        hall_rag=hall_rag,
        delta_explainability=explain_ah - explain_rag,
        delta_hallucination=hall_rag - hall_ah,
        d_max=d_max,
    )


def _norm(s: str) -> str:
    return s.lower().replace("ё", "е")


def _keyword_hit(answer: str, keywords: list[str]) -> bool:
    a = _norm(answer)
    return any(_norm(k) in a for k in keywords if k)


def _trace_complete(trace: list[str], gold: list[str]) -> bool:
    if not gold:
        return bool(trace)
    tset = {_norm(u) for u in trace}
    # accept bare / M_ forms
    expanded = set(tset)
    for u in list(tset):
        if u.startswith("m_"):
            expanded.add(u[2:])
        else:
            expanded.add(f"m_{u}")
    return all(_norm(g) in expanded or _norm(g).removeprefix("m_") in expanded for g in gold)


def _ah_hallucinated(agent: Agent, answer: str, trace: list[str], gold: GoldItem) -> bool:
    """True if answer claims content not grounded in Trace/WM-neighbor facts."""
    if not answer or _norm(answer) in {"неизвестно", "unknown", "не знаю"}:
        return False
    if gold.evidence_spans == ["__none__"]:
        # abstain is correct; any other content = hall
        return not _keyword_hit(answer, ["неизвестно", "нет данных", "не знаю"])
    if not _keyword_hit(answer, gold.answer_keywords):
        return not bool(trace)
    return not _trace_complete(trace, gold.gold_trace) and not bool(trace)


def _rag_hallucinated(answer: str, chunks: list[str], gold: GoldItem) -> bool:
    if not answer or _norm(answer) in {"неизвестно", "unknown", "не знаю"}:
        return False
    # Trap items: any concrete answer is a hallucination
    if gold.evidence_spans == ["__none__"]:
        return not _keyword_hit(answer, ["неизвестно", "нет данных", "не знаю"])
    blob = _norm(" ".join(chunks))
    spans = gold.evidence_spans or gold.answer_keywords
    grounded = any(_norm(s) in blob for s in spans if s)
    if not grounded:
        return True
    ans_toks = [t for t in _TOKEN.findall(_norm(answer)) if len(t) > 3]
    if not ans_toks:
        return False
    in_ctx = sum(1 for t in ans_toks if t in blob)
    return (in_ctx / len(ans_toks)) < 0.35
