"""Broader identity tests: paraphrased mentions on a mini graph + tiny synthetic world."""
from __future__ import annotations

import re

import pytest

from ah_memory.benchmarks.entity_resolution.dataset import control_cases, control_symbols
from ah_memory.benchmarks.entity_resolution.resolvers import HybridResolver, _norm
from ah_memory.identity import build_identity_service, catalog_from_store
from ah_memory.morph import slug_uid
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.synthetic import SyntheticGraphGenerator, get_preset, ingest_world
from ah_memory.transform import Transform

# Hand-built mini world: canonical facts + paraphrased queries.
MINI_FACTS: list[tuple[str, str, str]] = [
    ("Юлия", "LIVES_IN", "Москва"),
    ("Юлия", "WORKS_FOR", "врач"),
    ("Антон", "PURCHASE", "автомобиль"),
    ("Антон", "PURCHASE", "BMW"),
    ("Мария", "LIVES_IN", "Казань"),
    ("Сергей", "PURCHASE", "ноутбук"),
    ("Сергей", "PURCHASE", "телефон"),
]

# (query, mention_to_resolve, expected_canonical_name)
PARAPHRASE_QUERIES: list[tuple[str, str, str]] = [
    ("Где живёт Юли?", "Юли", "Юлия"),
    ("Куда переехала Юлию?", "Юлию", "Юлия"),
    ("С кем говорил Антон про Юлией?", "Юлией", "Юлия"),
    ("В каком городе живёт Маше?", "Маше", "Мария"),
    ("Что купила Маша?", "Маша", "Мария"),
    ("Какой город у Марии?", "Марии", "Мария"),
    ("Где работает врач Юлия — в Москве?", "Москве", "Москва"),
    ("Антон был в Москву?", "Москву", "Москва"),
    ("Кто купил машину?", "машину", "автомобиль"),
    ("На какой машине ездит Антон?", "машине", "автомобиль"),
    ("Какое авто приобрёл Антон?", "авто", "автомобиль"),
    ("Какой ноут у Сергея?", "ноут", "ноутбук"),
    ("Какой смартфон купил Сергей?", "смартфон", "телефон"),
    ("Кто живёт в Питере?", "Питере", "Санкт-Петербург"),
    ("Есть ли связь с Петербургом?", "Петербургом", "Санкт-Петербург"),
    ("Сергей говорил с Сергеем?", "Сергеем", "Сергей"),
    ("Передай Антону сообщение", "Антону", "Антон"),
    ("Казань или Казани — одно место", "Казани", "Казань"),
]

NEGATIVE_PARAPHRASES: list[tuple[str, str]] = [
    ("мотоцикл", "автомобиль"),
    ("пациент", "врач"),
    ("Audi", "BMW"),
    ("Санкт-Петербург", "Москва"),
    ("велосипед", "автомобиль"),
    ("ноутбук", "телефон"),
]


def _build_mini_graph() -> tuple[AHStore, Transform]:
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    transform = Transform(store, identity=identity)
    # Seed alias surfaces so lexicon/exact can attach to canons.
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[],
            seed_tokens=[
                "Санкт-Петербург",
                "Петербург",
                "Питер",
                "Мария",
                "Маша",
                "Сергей",
                "Серёжа",
            ],
        )
    )
    for subj, rel, obj in MINI_FACTS:
        transform.apply(
            PerceptionResult(
                kind="fact",
                candidates=[
                    FactCandidate(
                        predicate=rel,
                        raw_relation=rel.lower(),
                        canonical_relation=rel,
                        roles={"SUBJECT": subj, "OBJECT": obj},
                        confidence=1.0,
                    )
                ],
                seed_tokens=[subj, obj],
            )
        )
    return store, transform


def _canonical_bare(name: str) -> str:
    return slug_uid(name)


@pytest.fixture(scope="module")
def mini_store() -> AHStore:
    store, _ = _build_mini_graph()
    return store


def test_control_dataset_has_broad_paraphrases():
    cases = control_cases()
    assert len(cases) >= 80
    types = {c.case_type.value for c in cases}
    assert {"MORPHOLOGY", "SYNONYM", "NEGATIVE", "CONTEXTUAL", "AMBIGUOUS", "SURFACE", "HOLD_OUT"} <= types
    assert any(c.context and "живёт" in c.context for c in cases)
    assert any(c.mention == "ноут" for c in cases)


@pytest.mark.parametrize("query,mention,canonical", PARAPHRASE_QUERIES)
def test_mini_graph_paraphrase_queries(mini_store: AHStore, query, mention, canonical):
    identity = build_identity_service(mini_store, use_embeddings=False)
    hit = identity.resolve_bare_uid(mention, context=query)
    assert hit is not None, f"{mention!r} in {query!r} unresolved"
    expected = _canonical_bare(canonical)
    # Accept either slug of canonical or already-resolved bare equal ignoring case
    assert hit == expected or hit.upper() == expected.upper() or _norm(hit) == _norm(
        expected
    ), f"{mention!r} → {hit!r}, want {expected!r} (query={query!r})"


@pytest.mark.parametrize("mention,distractor", NEGATIVE_PARAPHRASES)
def test_mini_graph_no_false_merge(mini_store: AHStore, mention, distractor):
    identity = build_identity_service(mini_store, use_embeddings=False)
    # Ensure distractor exists
    mini_store.ensure_abstract(slug_uid(distractor), {distractor.lower()})
    mini_store.ensure_m(f"M_{slug_uid(distractor)}", distractor)
    hit = identity.resolve_bare_uid(mention)
    bad = slug_uid(distractor)
    assert hit != bad, f"{mention!r} wrongly merged into {distractor!r}"


def test_hybrid_paraphrase_table_offline():
    symbols = control_symbols()
    hybrid = HybridResolver(use_embeddings=False)
    positives = [
        c
        for c in control_cases()
        if c.target_uid is not None
        and c.case_type.value in {"MORPHOLOGY", "SYNONYM", "CONTEXTUAL"}
    ]
    assert len(positives) >= 30
    failed = []
    for case in positives:
        result = hybrid.resolve(
            case.mention,
            case.context,
            symbols,
            candidate_uids=case.candidate_uids,
            threshold=0.94,
        )
        if result.selected_uid != case.target_uid:
            failed.append((case.case_id, case.mention, result.selected_uid, case.target_uid))
    assert not failed, f"failed paraphrase cases: {failed[:12]}"


def test_tiny_synthetic_morph_paraphrases():
    world = SyntheticGraphGenerator(get_preset("tiny")).generate()
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    result = ingest_world(world, store, identity=identity, include_distractors=False)

    persons = [e for e in world.entities if e.type == "Person"][:8]
    places = [e for e in world.entities if e.type == "Place"][:6]
    assert persons and places

    from ah_memory.morph import _get_morph

    morph = _get_morph()
    syn_to_bare = {
        syn: (ah[2:] if ah.startswith("M_") else ah)
        for syn, ah in result.uid_map.items()
    }
    checked = 0
    for entity in (*persons, *places):
        name = entity.name
        if not re.search(r"[а-яА-ЯёЁ]", name):
            continue
        target = syn_to_bare.get(entity.uid)
        if not target:
            catalog = catalog_from_store(store)
            target = next(
                (s.uid for s in catalog if _norm(s.name) == _norm(name)),
                None,
            )
        if target is None:
            continue
        parses = morph.parse(name.lower().replace("ё", "е"))
        if not parses:
            continue
        for tag in ("gent", "datv", "accs", "ablt", "loct"):
            try:
                inf = parses[0].inflect({tag})
            except Exception:
                continue
            if inf is None:
                continue
            form = inf.word
            if _norm(form) == _norm(name):
                continue
            hit = identity.resolve_bare_uid(form)
            assert hit == target, f"{name}: {form!r} → {hit!r}, want {target!r}"
            checked += 1
            break
    assert checked >= 6, f"too few morph paraphrases checked: {checked}"


def test_tiny_synthetic_query_entity_mentions():
    """Resolve entity names from synthetic questions (exact + lower paraphrase)."""
    world = SyntheticGraphGenerator(get_preset("tiny")).generate()
    store = AHStore()
    identity = build_identity_service(store, use_embeddings=False)
    result = ingest_world(world, store, identity=identity, include_distractors=False)
    assert result.ingested_factors > 0

    name_to_bare: dict[str, str] = {}
    for entity in world.entities:
        ah = result.uid_map.get(entity.uid)
        if not ah:
            continue
        bare = ah[2:] if ah.startswith("M_") else ah
        name_to_bare[_norm(entity.name)] = bare

    samples = 0
    for query in world.queries[:12]:
        answer = query.answer or ""
        for entity in world.entities:
            if entity.name not in query.question and entity.name not in answer:
                continue
            bare = name_to_bare.get(_norm(entity.name))
            if not bare:
                continue
            hit = identity.resolve_bare_uid(entity.name)
            if hit:
                assert hit == bare
                samples += 1
            hit2 = identity.resolve_bare_uid(entity.name.lower())
            if hit2:
                assert hit2 == bare
                samples += 1
    assert samples >= 4
