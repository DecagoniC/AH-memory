"""Compound kinship mentions: брат → брат максим when full name exists."""
from __future__ import annotations

from ah_memory.identity import build_identity_service
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def test_brother_resolves_to_brother_maxim():
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)

    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="HAVE",
                    raw_relation="есть",
                    canonical_relation="HAVE",
                    roles={"SUBJECT": "Я", "OBJECT": "брат Максим"},
                    raw_span="у меня есть брат Максим",
                )
            ],
            seed_tokens=["Я", "брат", "Максим"],
        )
    )
    assert "БРАТ_МАКСИМ" in store.ah.S
    assert store.ah.S["БРАТ_МАКСИМ"].R["TEXT"]

    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="ENROLL_IN_COLLEGE",
                    raw_relation="поступает в колледж",
                    canonical_relation="ENROLL_IN_COLLEGE",
                    roles={"SUBJECT": "брата", "OBJECT": "колледж"},
                    raw_span="мой брат поступает в колледж",
                )
            ],
            seed_tokens=["брат", "колледж"],
        )
    )

    factors = store.list_semantic_factors()
    enroll = next(
        f for f in factors if f.relation and f.relation.canonical_label == "ENROLL_IN_COLLEGE"
    )
    assert enroll.roles["SUBJECT"] == "M_БРАТ_МАКСИМ"


def test_brother_slug_from_llm_resolves_to_brother_maxim():
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)

    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="HAVE",
                    raw_relation="есть",
                    canonical_relation="HAVE",
                    roles={"SUBJECT": "Я", "OBJECT": "брат Максим"},
                    raw_span="у меня есть брат Максим",
                )
            ],
            seed_tokens=["Я", "брат", "Максим"],
        )
    )

    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="ENROLL_IN_COLLEGE",
                    raw_relation="поступает в колледж",
                    canonical_relation="ENROLL_IN_COLLEGE",
                    roles={"SUBJECT": "БРАТ", "OBJECT": "колледж"},
                    raw_span="мой брат поступает в колледж",
                )
            ],
            seed_tokens=["брат", "колледж"],
        )
    )

    enroll = next(
        f
        for f in store.list_semantic_factors()
        if f.relation and f.relation.canonical_label == "ENROLL_IN_COLLEGE"
    )
    assert enroll.roles["SUBJECT"] == "M_БРАТ_МАКСИМ"
    assert "БРАТ" not in store.ah.S or enroll.roles["SUBJECT"] == "M_БРАТ_МАКСИМ"


def test_compound_role_binds_to_head_symbol() -> None:
    store = AHStore()
    transform = Transform(store)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="HAS_PART",
                    roles={"SUBJECT": "ВОРОН", "OBJECT": "ПЕРО"},
                    canonical_relation="HAS_PART",
                    raw_span="у ворона есть перо",
                )
            ],
            seed_tokens=["ВОРОН", "ПЕРО"],
        )
    )
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="LOCATED_ON",
                    roles={
                        "SUBJECT": "ПИСЬМЕННЫЙ_ПЕРО",
                        "OBJECT": "ПИСЬМЕННЫЙ_СТОЛ",
                    },
                    canonical_relation="LOCATED_ON",
                    raw_span="на письменном столе есть письменное перо",
                )
            ],
            seed_tokens=["ПИСЬМЕННЫЙ_ПЕРО", "ПИСЬМЕННЫЙ_СТОЛ"],
        )
    )

    assert "ПЕРО" in store.ah.S
    assert "M_ПИСЬМЕННЫЙ_ПЕРО" in store.ah.C
    bind = [
        link
        for link in store.ah.L.values()
        if link.id == "BIND"
        and {link.e1.target_uid, link.e2.target_uid}
        == {"M_ПИСЬМЕННЫЙ_ПЕРО", "ПЕРО"}
    ]
    assert bind
    assert not any(
        link.id == "IS-A"
        and {link.e1.target_uid, link.e2.target_uid}
        == {"M_ПИСЬМЕННЫЙ_ПЕРО", "M_ПЕРО"}
        for link in store.ah.L.values()
    )
    crow = next(
        factor
        for factor in store.list_semantic_factors()
        if factor.roles.get("OBJECT") == "M_ПЕРО"
    )
    assert crow.roles["SUBJECT"] in {"M_ВОРОН", "M_ВОРОНА"}
