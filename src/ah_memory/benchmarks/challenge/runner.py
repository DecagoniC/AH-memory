"""CLI for the open challenge metric stand."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ah_memory.benchmarks.challenge.adapters import comparison_inputs, role_benchmark_items
from ah_memory.benchmarks.challenge.ah_arm import AHGraphArm
from ah_memory.benchmarks.challenge.chat_backend import JsonChatAnswerBackend
from ah_memory.benchmarks.challenge.comparison import compare_ah_to_rag
from ah_memory.benchmarks.challenge.loader import load_qa_corpus, load_role_corpus
from ah_memory.benchmarks.challenge.ngram import DeterministicNgramEmbedder
from ah_memory.benchmarks.challenge.protocols import (
    BenchmarkQuery,
    SourceDocument,
    StructuredAnswer,
)
from ah_memory.benchmarks.challenge.rag import VanillaRAG
from ah_memory.benchmarks.challenge.retrieval import CosineVectorRetriever
from ah_memory.benchmarks.challenge.role_baseline import UngatedLLMPerception
from ah_memory.benchmarks.challenge_evaluation import (
    run_m1_benchmark,
    run_m2_benchmark,
    run_m5_role_benchmark,
)
from ah_memory.benchmarks.challenge.graph_qa import run_graph_qa, to_inference_item
from ah_memory.benchmarks.challenge_metrics import gc_efficiency
from ah_memory.benchmarks.challenge_report import write_challenge_report
from ah_memory.config import load_config
from ah_memory.gc import collect, orphan_uids
from ah_memory.hyperparams import HyperParams
from ah_memory.store import AHStore
from ah_memory.types import AbstractSymbol, AssocLink, LinkId, SecondOrderSymbol, Section


class HeuristicQAChat:
    """Graph-free offline answerer that only reads retrieved document text."""

    def answer(
        self,
        query: BenchmarkQuery,
        documents: list[SourceDocument] | tuple[SourceDocument, ...],
    ) -> StructuredAnswer:
        markers = (" follows ", " is a ", " causes ")
        chosen = ""
        support: list[str] = []
        for document in documents:
            lowered = f" {document.text.casefold()} "
            for marker in markers:
                if marker not in lowered:
                    continue
                chosen = document.text.split(marker.strip(), 1)[-1].strip(" .")
                support.append(str(document.document_id))
        return StructuredAnswer(
            query_id=query.query_id,
            text=chosen,
            support_ids=tuple(support),
        )


def _measure_m3() -> dict[str, Any]:
    store = AHStore()
    store.add_abstract_symbol(AbstractSymbol(uid="LIVE"))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_LIVE"))
    store.add_link(
        AssocLink(
            uid="LIVE_LINK",
            id=LinkId.ASSOC.value,
            w=1.0,
            e1=store.s_ref("LIVE"),
            e2=store.m_ref("M_LIVE"),
        )
    )
    orphans = {f"ORPHAN_{index:03d}" for index in range(200)}
    for uid in orphans:
        store.add_element(Section.C, SecondOrderSymbol(uid=uid))
    before = orphan_uids(store)
    live_before = {"LIVE", "M_LIVE", "LIVE_LINK"}
    store.ah.tau = 50
    report = collect(store, HyperParams(ttl=50))
    after = orphan_uids(store)
    false_deletes = [uid for uid in live_before if uid not in store.ah.S and uid not in store.ah.C and uid not in store.ah.L]
    return {
        "gc_efficiency": gc_efficiency(len(before), len(after)),
        "orphans_before": len(before),
        "orphans_after": len(after),
        "removed_elements": list(report.removed_elements),
        "false_live_deletes": false_deletes,
    }


def _offline_rag() -> VanillaRAG:
    return VanillaRAG(
        CosineVectorRetriever(DeterministicNgramEmbedder()),
        HeuristicQAChat(),
        top_k=6,
    )


def _run_offline() -> tuple[dict[str, Any], dict[str, Any]]:
    qa = load_qa_corpus()
    m2 = run_m2_benchmark(
        [to_inference_item(item) for item in qa],
        lambda item: run_graph_qa(next(case for case in qa if case.item_id == item.item_id)),
        d_max=6,
    )
    corpus, queries = comparison_inputs(qa)
    m4 = compare_ah_to_rag(AHGraphArm(qa), _offline_rag(), corpus, queries)
    m3 = _measure_m3()
    return {
        "mode": "offline",
        "m2": {
            "explain_score": m2.explain_score,
            "d_max": m2.d_max,
            "items": len(m2.items),
            "correct": sum(item.correct for item in m2.items),
            "trace_complete": sum(item.trace_complete for item in m2.items),
        },
        "m3": {
            "gc_efficiency": m3["gc_efficiency"],
            "orphans_before": m3["orphans_before"],
            "orphans_after": m3["orphans_after"],
            "false_live_deletes": m3["false_live_deletes"],
        },
        "m4": {
            "delta_explainability": m4.delta_explainability,
            "delta_hallucination": m4.delta_hallucination,
            "ah_explainability": m4.ah.result.explainability,
            "rag_explainability": m4.rag.result.explainability,
            "ah_hallucination": m4.ah.hallucination,
            "rag_hallucination": m4.rag.hallucination,
        },
    }, {
        "m2/items": [
            {
                **item.to_dict(),
                "metadata": {
                    key: value
                    for key, value in item.metadata.items()
                    if key != "trace"
                },
            }
            for item in m2.items
        ],
        "m3/gc": [m3],
        "m4/ah": [
            {
                "query_id": str(record.answer.query_id),
                "answer": record.answer.text,
                "support_ids": list(record.answer.support_ids),
            }
            for record in m4.ah.result.records
        ],
        "m4/rag": [
            {
                "query_id": str(record.answer.query_id),
                "answer": record.answer.text,
                "support_ids": list(record.answer.support_ids),
            }
            for record in m4.rag.result.records
        ],
    }


def _chat_fn(client: Any):
    def chat(messages: list[dict[str, str]]) -> str:
        return client.chat(messages, json_mode=True)

    return chat


def _run_live() -> tuple[dict[str, Any], dict[str, Any]]:
    from ah_memory.gigachat_llm import GigaChatClient, GigaChatPerception
    from ah_memory.ollama import OllamaClient, OllamaPerception, is_ollama_available

    cfg = load_config()
    if not cfg.gigachat.configured:
        raise RuntimeError("GigaChat credentials are not configured")
    if not is_ollama_available(cfg.ollama):
        raise RuntimeError("Ollama is not available")

    roles = role_benchmark_items(load_role_corpus())
    giga_client = GigaChatClient(cfg.gigachat)
    ollama_client = OllamaClient(cfg.ollama)
    ah_llm = GigaChatPerception(cfg.gigachat)
    ah_slm = OllamaPerception(cfg.ollama)
    rag_llm = UngatedLLMPerception(_chat_fn(giga_client), backend="gigachat_ungated")
    rag_slm = UngatedLLMPerception(_chat_fn(ollama_client), backend="ollama_ungated")
    m5 = run_m5_role_benchmark(
        roles,
        ah_slm=ah_slm,
        rag_slm=rag_slm,
        ah_llm=ah_llm,
        rag_llm=rag_llm,
        model_names={
            "ah_slm": cfg.ollama.model,
            "rag_slm": f"{cfg.ollama.model}+ungated",
            "ah_llm": cfg.gigachat.model,
            "rag_llm": f"{cfg.gigachat.model}+ungated",
        },
    )
    return {
        "mode": "live",
        "m1": {
            "ah_slm": m5.ah_slm.to_dict()["score"],
            "ah_llm": m5.ah_llm.to_dict()["score"],
        },
        "m5": {
            "robustness_gain": m5.robustness_gain,
            "models": {
                "ah_slm": cfg.ollama.model,
                "ah_llm": cfg.gigachat.model,
            },
        },
    }, {
        "m1/ah_slm": m5.ah_slm.to_dict()["items"],
        "m1/rag_slm": m5.rag_slm.to_dict()["items"],
        "m1/ah_llm": m5.ah_llm.to_dict()["items"],
        "m1/rag_llm": m5.rag_llm.to_dict()["items"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the open AH-memory challenge stand")
    parser.add_argument("--live", action="store_true", help="also run GigaChat and Ollama")
    parser.add_argument("--root", default="results/challenge")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    summary, logs = _run_offline()
    if args.live:
        live_summary, live_logs = _run_live()
        summary.update(live_summary)
        logs.update(live_logs)
    summary["elapsed_sec"] = round(time.perf_counter() - started, 3)
    output = write_challenge_report(summary, logs, root=Path(args.root))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
