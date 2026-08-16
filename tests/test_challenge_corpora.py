from __future__ import annotations

from collections import Counter

from ah_memory.benchmarks.challenge.adapters import comparison_inputs
from ah_memory.benchmarks.challenge.generator import corpus_bytes
from ah_memory.benchmarks.challenge.loader import load_qa_corpus, load_role_corpus
from ah_memory.benchmarks.challenge.schema import (
    CHALLENGE_DATA_DIR,
    MANDATORY_ROLES,
    QA_RELATIONS,
    ROLE_VARIANTS,
)


def test_m1_fixed_count_balance_and_atomic_roles() -> None:
    items = load_role_corpus()

    assert len(items) == 100
    assert len({item.item_id for item in items}) == 100
    assert Counter(item.variant for item in items) == {
        variant: 25 for variant in ROLE_VARIANTS
    }
    assert len({item.domain for item in items}) == 5
    for item in items:
        assert set(item.expected_roles) == MANDATORY_ROLES
        assert all(
            isinstance(value, str) and value.strip()
            for value in item.expected_roles.values()
        )


def test_m2_m4_fixed_count_depths_relations_and_exact_proofs() -> None:
    items = load_qa_corpus()

    assert len(items) == 20
    assert len({item.item_id for item in items}) == 20
    assert {item.depth for item in items} == set(range(1, 7))
    assert {item.relation_type for item in items} == QA_RELATIONS
    for item in items:
        facts = {fact.uid: fact for fact in item.source_facts}
        assert len(item.proof_path) == item.depth
        assert tuple(facts) == item.proof_path
        assert all(facts[uid].relation == item.relation_type for uid in item.proof_path)
        for left, right in zip(item.proof_path, item.proof_path[1:]):
            assert facts[left].object == facts[right].subject
        assert facts[item.proof_path[-1]].object == item.answer
        documented = {
            uid
            for document in item.source_documents
            for uid in document.fact_uids
        }
        assert documented == set(item.proof_path)


def test_ah_and_rag_share_the_same_source_documents() -> None:
    items = load_qa_corpus()
    documents, queries = comparison_inputs(items)
    document_ids = {document.document_id for document in documents}
    query_ids = {query.query_id for query in queries}
    assert query_ids == {item.item_id for item in items}
    expected_docs = {
        document.uid
        for item in items
        for document in item.source_documents
    }
    assert document_ids == expected_docs
    for item in items:
        covered = {
            fact_uid
            for document in item.source_documents
            for fact_uid in document.fact_uids
        }
        assert covered == set(item.proof_path)


def test_committed_jsonl_is_byte_reproducible() -> None:
    first = corpus_bytes()
    second = corpus_bytes()

    assert first == second
    assert set(first) == {"m1_roles.jsonl", "m2_m4_qa.jsonl"}
    for filename, generated in first.items():
        assert (CHALLENGE_DATA_DIR / filename).read_bytes() == generated
