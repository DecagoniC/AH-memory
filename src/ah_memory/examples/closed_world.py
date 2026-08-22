"""Closed-world M4 fixture: NL corpus + facts extracted by batched perception."""
from __future__ import annotations

from pathlib import Path

from ah_memory.eval.corpus_ingest import apply_cached_facts, load_fact_payload
from ah_memory.store import AHStore

_CORPUS_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "m4" / "closed_world.txt"
)
_FACTS_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "m4" / "closed_world_facts.json"
)

_FALLBACK_TEXT = (
    "Источник: русская Википедия, статья «Тиманский кряж» (CC BY-SA 4.0).\n\n"
    "Тиманский кряж (Тиман) — крупная возвышенность на северо-востоке "
    "Восточно-Европейской равнины.\n"
)


def closed_world_text() -> str:
    """Source bulletin shared by Vanilla RAG and the AH graph."""
    if _CORPUS_PATH.is_file():
        return _CORPUS_PATH.read_text(encoding="utf-8").strip() + "\n"
    return _FALLBACK_TEXT.strip() + "\n"


def closed_world_facts_path() -> Path:
    return _FACTS_PATH


def build_closed_world_memory() -> AHStore:
    store = AHStore()
    if _FACTS_PATH.is_file():
        apply_cached_facts(store, load_fact_payload(_FACTS_PATH))
    from ah_memory.graph_library import remember_fixture

    remember_fixture("closed-world", store, source_text=closed_world_text())
    return store


def extracted_fact_keys(store: AHStore) -> set[str]:
    keys: set[str] = set()
    for factor in store.list_semantic_factors():
        pred = (
            factor.relation.canonical_label.upper()
            if factor.relation is not None
            else ""
        )
        subj = (factor.roles.get("SUBJECT") or "").replace("M_", "")
        obj = (factor.roles.get("OBJECT") or "").replace("M_", "")
        loc = (factor.roles.get("LOCATION") or "").replace("M_", "")
        if pred and subj and obj:
            keys.add(f"{pred}:{subj}:{obj}")
        elif pred and subj and loc:
            keys.add(f"{pred}:{subj}:{loc}")
        elif pred and subj:
            keys.add(f"{pred}:{subj}")
    return keys


def closed_world_auto_score(store: AHStore) -> tuple[int, int]:
    got = extracted_fact_keys(store)
    expected = 0
    if _FACTS_PATH.is_file():
        expected = len(load_fact_payload(_FACTS_PATH).get("candidates") or [])
    return len(got), max(1, expected)
