"""Graph-backed M2 observations over the fixed challenge QA schema."""
from __future__ import annotations

from ah_memory.benchmarks.challenge.schema import QAItem
from ah_memory.benchmarks.challenge_evaluation import (
    InferenceBenchmarkItem,
    InferenceObservation,
)
from ah_memory.factor_graph import build_structural_factor_graph
from ah_memory.semantic_activation import ActivationEngine
from ah_memory.store import AHStore
from ah_memory.types import AssocLink, SecondOrderSymbol, Section


def to_inference_item(item: QAItem) -> InferenceBenchmarkItem:
    return InferenceBenchmarkItem(
        item_id=item.item_id,
        question=item.question,
        expected_answer=item.answer,
        depth=item.depth,
        expected_trace=tuple(f"L::{uid}" for uid in item.proof_path),
    )


def run_graph_qa(
    item: QAItem,
    *,
    threshold: float = 0.05,
) -> InferenceObservation:
    """Materialize one case, propagate from its first subject, and retain UID trace."""
    store = AHStore()
    node_uids = {
        value
        for fact in item.source_facts
        for value in (fact.subject, fact.object)
    }
    for uid in sorted(node_uids):
        store.add_element(Section.C, SecondOrderSymbol(uid=_m_uid(uid)))
    for fact in item.source_facts:
        store.add_link(
            AssocLink(
                uid=fact.uid,
                id=fact.relation,
                w=1.0,
                e1=store.m_ref(_m_uid(fact.subject)),
                e2=store.m_ref(_m_uid(fact.object)),
            )
        )

    graph = build_structural_factor_graph(store)
    first = next(fact for fact in item.source_facts if fact.uid == item.proof_path[0])
    target_uid = _m_uid(item.answer)
    activation, trace = ActivationEngine().run(
        graph,
        {_m_uid(first.subject): 1.0},
        timesteps=max(1, item.depth),
        threshold=threshold,
        trace=True,
    )
    actual_trace: list[str] = []
    for message in trace.messages:
        factor_uid = str(message.get("factor_uid") or "")
        if (
            factor_uid
            and factor_uid not in actual_trace
            and float(message.get("activation") or 0.0) >= threshold
        ):
            actual_trace.append(factor_uid)
    answer = item.answer if activation.get(target_uid, 0.0) >= threshold else ""
    return InferenceObservation(
        answer=answer,
        trace=tuple(actual_trace),
        metadata={
            "target_uid": target_uid,
            "target_activation": activation.get(target_uid, 0.0),
            "activated_nodes": list(trace.activated_nodes),
            "trace": trace.to_dict(),
        },
    )


def _m_uid(uid: str) -> str:
    return uid if uid.startswith("M_") else f"M_{uid}"
