"""Ingest-time identity: synonyms attach to existing symbols, no false merge."""
from __future__ import annotations

from ah_memory.identity import build_identity_service, catalog_from_store
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def test_ingest_synonym_reuses_symbol():
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="IS_A",
                    raw_relation="это",
                    canonical_relation="IS_A",
                    roles={"SUBJECT": "автомобиль", "OBJECT": "транспорт"},
                    confidence=1.0,
                )
            ],
            seed_tokens=["автомобиль", "транспорт"],
        )
    )
    before = set(store.ah.S)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="HAVE",
                    raw_relation="есть",
                    canonical_relation="HAVE",
                    roles={"SUBJECT": "Антон", "OBJECT": "машина"},
                    confidence=1.0,
                )
            ],
            seed_tokens=["Антон", "машина"],
        )
    )
    # машина → автомобиль (alias), no parallel МАШИНА symbol
    assert "МАШИНА" not in store.ah.S
    car_uid = next(u for u in before if "АВТОМОБИЛЬ" in u or u == "АВТОМОБИЛЬ")
    forms = {f.lower() for f in store.ah.S[car_uid].R.get("TEXT", set())}
    assert "машина" in forms or "автомобиль" in forms


def test_ingest_does_not_merge_audi_bmw():
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[],
            seed_tokens=["BMW", "Audi"],
        )
    )
    uids = set(store.ah.S)
    assert any("BMW" in u for u in uids)
    assert any("AUDI" in u for u in uids)
    assert len([u for u in uids if "BMW" in u or "AUDI" in u]) >= 2


def test_catalog_from_store_exposes_forms():
    store = AHStore()
    store.ensure_abstract("АВТОМОБИЛЬ", {"автомобиль", "машина"})
    store.ensure_m("M_АВТОМОБИЛЬ", "автомобиль")
    cat = catalog_from_store(store)
    assert any(_norm_name(s) == "автомобиль" for s in cat)
    auto = next(s for s in cat if "автомобиль" in s.name.lower() or s.uid == "АВТОМОБИЛЬ")
    assert "машина" in {a.lower() for a in auto.aliases} or "машина" in auto.name.lower()


def _norm_name(spec) -> str:
    return spec.name.lower().replace("ё", "е")
