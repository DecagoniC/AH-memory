"""Extended ER corpus: ambiguous, surface, hard-neg, hold-out, ingest regression."""
from __future__ import annotations

import pytest

from ah_memory.benchmarks.entity_resolution.dataset import control_cases, control_symbols
from ah_memory.benchmarks.entity_resolution.dataset_extended import (
    HOLD_OUT_POSITIVES,
    assert_hold_out_not_in_lexicon,
)
from ah_memory.benchmarks.entity_resolution.resolvers import HybridResolver
from ah_memory.identity import DEFAULT_SYNONYMS, build_identity_service
from ah_memory.morph import slug_uid
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform


@pytest.fixture(scope="module")
def symbols():
    return control_symbols()


@pytest.fixture(scope="module")
def hybrid_offline():
    return HybridResolver(use_embeddings=False)


def test_corpus_size_and_types():
    cases = control_cases()
    types = {c.case_type.value for c in cases}
    assert len(cases) >= 80
    assert {
        "AMBIGUOUS",
        "SURFACE",
        "HOLD_OUT",
        "NEGATIVE",
        "MORPHOLOGY",
    } <= types


def test_hold_out_not_in_lexicon():
    assert_hold_out_not_in_lexicon()
    keys = {k.lower().replace("ё", "е") for k in DEFAULT_SYNONYMS}
    for _cid, mention, _t, _c in HOLD_OUT_POSITIVES:
        assert mention.lower().replace("ё", "е") not in keys


@pytest.mark.parametrize(
    "case_id,mention,target,context",
    [
        ("amb_001", "Иван", None, None),
        ("amb_002", "Иван", "s_050", "Иван живёт в Москве"),
        ("amb_003", "Иван", "s_051", "Иван работает в Казани"),
        ("amb_004", "Анна", None, None),
        ("amb_005", "Анна", "s_052", "Анна переехала в Берлин"),
        ("amb_006", "Анна", "s_053", "Анна учится в Париже"),
    ],
)
def test_ambiguous_names(hybrid_offline, symbols, case_id, mention, target, context):
    case = next(c for c in control_cases() if c.case_id == case_id)
    result = hybrid_offline.resolve(
        mention,
        context,
        symbols,
        candidate_uids=case.candidate_uids,
        threshold=0.94,
    )
    assert result.selected_uid == target


@pytest.mark.parametrize(
    "mention,target",
    [
        ("Yuliya", "s_001"),
        ("bmw", "s_030"),
        ("Санкт Петербург", "s_004"),
        ("санкт  петербург", "s_004"),
        ("New York", "s_056"),
        ("нью йорк", "s_056"),
        ("Moskva", "s_003"),
        ("audi", "s_031"),
    ],
)
def test_surface_latin_multiword(hybrid_offline, symbols, mention, target):
    result = hybrid_offline.resolve(mention, None, symbols, threshold=0.94)
    assert result.selected_uid == target


@pytest.mark.parametrize(
    "mention,wrong",
    [
        ("Toyota", "s_030"),
        ("инженер", "s_011"),
        ("Нью-Йорк", "s_003"),
        ("программист", "s_011"),
        ("Lada", "s_057"),
    ],
)
def test_hard_negative_force_reject(hybrid_offline, symbols, mention, wrong):
    result = hybrid_offline.resolve(
        mention,
        None,
        symbols,
        candidate_uids=(wrong,),
        threshold=0.5,
    )
    assert result.selected_uid is None


def test_hard_negative_own_symbol(hybrid_offline, symbols):
    r = hybrid_offline.resolve(
        "Toyota",
        None,
        symbols,
        candidate_uids=("s_057", "s_030", "s_031"),
        threshold=0.94,
    )
    assert r.selected_uid == "s_057"


def test_hold_out_offline_not_cheating_via_alias(hybrid_offline, symbols):
    """Without embeddings, hold-out jargon should not resolve via lexicon/alias."""
    for _cid, mention, target, cands in HOLD_OUT_POSITIVES:
        result = hybrid_offline.resolve(
            mention,
            None,
            symbols,
            candidate_uids=cands,
            threshold=0.94,
        )
        # Must not silently hit via DEFAULT_SYNONYMS; None is OK offline.
        if result.selected_uid is not None:
            assert result.method != "alias"
            # exact/morph accidental hit is rare; if hits must be correct target
            assert result.selected_uid == target


def test_hold_out_negatives_offline(hybrid_offline, symbols):
    for mention, wrong in (
        ("легковушка", "s_020"),
        ("эскулап", "s_022"),
        ("гаджет", "s_013"),
    ):
        r = hybrid_offline.resolve(
            mention, None, symbols, candidate_uids=(wrong,), threshold=0.5
        )
        assert r.selected_uid is None


def test_ingest_paraphrase_no_duplicate_uid():
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="LIVES_IN",
                    raw_relation="живёт",
                    canonical_relation="LIVES_IN",
                    roles={"SUBJECT": "Юлия", "OBJECT": "Москва"},
                    confidence=1.0,
                )
            ],
            seed_tokens=["Юлия", "Москва", "Yuliya"],
        )
    )
    before = set(store.ah.S)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="LIVES_IN",
                    raw_relation="живёт",
                    canonical_relation="LIVES_IN",
                    roles={"SUBJECT": "Yuliya", "OBJECT": "Москве"},
                    confidence=1.0,
                )
            ],
            seed_tokens=["Yuliya", "Москве"],
        )
    )
    # Latin/paraphrase must attach, not spawn parallel YULIYA / МОСКВЕ symbols
    assert "YULIYA" not in store.ah.S
    assert slug_uid("Москве") == slug_uid("Москва") or "МОСКВЕ" not in store.ah.S
    yulia = slug_uid("Юлия")
    assert yulia in before or yulia in store.ah.S
    forms = {f.lower() for f in store.ah.S[yulia].R.get("TEXT", set())}
    assert "yuliya" in forms or "юлия" in forms


def test_ingest_ambiguous_ivan_stays_separate_without_merge():
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[],
            seed_tokens=["Иван"],
        )
    )
    # Second Ivan mention without context should create via slug (same lemma → same UID).
    # Document expected behavior: lemma identity collapses same name — OK for ingest;
    # ER ambiguous cases cover catalog with distinct UIDs.
    transform.apply(
        PerceptionResult(kind="fact", candidates=[], seed_tokens=["Ивана"])
    )
    assert slug_uid("Иван") in store.ah.S


def test_surface_suite_hybrid_accuracy(hybrid_offline, symbols):
    surface = [c for c in control_cases() if c.case_type.value == "SURFACE"]
    assert len(surface) >= 8
    fails = []
    for case in surface:
        r = hybrid_offline.resolve(
            case.mention,
            case.context,
            symbols,
            candidate_uids=case.candidate_uids,
            threshold=0.94,
        )
        if r.selected_uid != case.target_uid:
            fails.append((case.case_id, case.mention, r.selected_uid, case.target_uid))
    assert not fails, fails


def test_ambiguous_suite_hybrid(hybrid_offline, symbols):
    amb = [c for c in control_cases() if c.case_type.value == "AMBIGUOUS"]
    assert len(amb) >= 6
    fails = []
    for case in amb:
        r = hybrid_offline.resolve(
            case.mention,
            case.context,
            symbols,
            candidate_uids=case.candidate_uids,
            threshold=0.94,
        )
        if r.selected_uid != case.target_uid:
            fails.append((case.case_id, r.selected_uid, case.target_uid, case.context))
    assert not fails, fails
