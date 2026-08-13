"""Regression tests for Entity Resolution benchmark."""
from __future__ import annotations

from ah_memory.benchmarks.entity_resolution.cases import CaseType, ResolutionResult
from ah_memory.benchmarks.entity_resolution.dataset import control_cases, control_symbols
from ah_memory.benchmarks.entity_resolution.evaluator import (
    evaluate_case,
    run_activation_probe,
)
from ah_memory.benchmarks.entity_resolution.generator import build_resolution_store
from ah_memory.benchmarks.entity_resolution.metrics import (
    DEFAULT_THRESHOLDS,
    find_optimal_threshold,
)
from ah_memory.benchmarks.entity_resolution.resolvers import (
    EmbeddingResolver,
    ExactResolver,
    HybridResolver,
    MorphologyResolver,
    default_resolvers,
)
from ah_memory.benchmarks.entity_resolution.runner import EntityResolutionBenchmark


def _yulia_case():
    return next(c for c in control_cases() if c.case_id == "morph_yulia_01")


def test_resolver_interface():
    symbols = control_symbols()
    for resolver in default_resolvers().values():
        assert callable(getattr(resolver, "resolve", None))
        assert getattr(resolver, "name", None)
        result = resolver.resolve("Юлия", None, symbols, threshold=0.5)
        assert isinstance(result, ResolutionResult)
        assert hasattr(result, "selected_uid")
        assert hasattr(result, "confidence")
        assert hasattr(result, "method")
        assert isinstance(result.candidates, list)


def test_yulia_morphology():
    symbols = control_symbols()
    morph = MorphologyResolver()
    hybrid = HybridResolver()
    for mention in ("Юли", "Юлию", "Юлией"):
        r_m = morph.resolve(mention, None, symbols, candidate_uids=("s_001", "s_002"))
        assert r_m.selected_uid == "s_001", mention
        r_h = hybrid.resolve(mention, None, symbols, candidate_uids=("s_001", "s_002"))
        assert r_h.selected_uid == "s_001", mention


def test_synonym_car():
    symbols = control_symbols()
    hybrid = HybridResolver(use_embeddings=False)
    result = hybrid.resolve(
        "машина",
        None,
        symbols,
        candidate_uids=("s_010", "s_020", "s_021"),
        threshold=0.0,
    )
    assert result.selected_uid == "s_010"
    assert result.method in {"alias", "exact", "morphology"}


def test_alias_resolver_and_anti_merge():
    from ah_memory.identity import AliasResolver, GatedEmbeddingResolver, IdentityPolicy

    symbols = control_symbols()
    alias = AliasResolver()
    assert alias.resolve("доктор", None, symbols).selected_uid == "s_011"
    gated = GatedEmbeddingResolver(policy=IdentityPolicy(safety_threshold=0.5))
    # Even with low floor, Audi must not merge onto BMW-only pool
    force = gated.resolve(
        "Audi",
        None,
        symbols,
        candidate_uids=("s_030",),
        threshold=0.5,
    )
    assert force.selected_uid is None


def test_negative_car_motorcycle():
    symbols = control_symbols()
    morph = MorphologyResolver()
    emb = EmbeddingResolver()
    # Must not morph-merge motorcycle → car
    r = morph.resolve(
        "мотоцикл",
        None,
        symbols,
        candidate_uids=("s_010",),
        threshold=0.5,
    )
    assert r.selected_uid is None
    # Force-reject pool: only car present → high threshold must refuse
    r2 = emb.resolve(
        "мотоцикл",
        None,
        symbols,
        candidate_uids=("s_010",),
        threshold=0.98,
    )
    assert r2.selected_uid is None or r2.confidence < 0.98


def test_negative_moscow_spb():
    symbols = control_symbols()
    hybrid = HybridResolver()
    r = hybrid.resolve(
        "Санкт-Петербург",
        None,
        symbols,
        candidate_uids=("s_003", "s_004"),
        threshold=0.85,
    )
    assert r.selected_uid == "s_004"
    force = EmbeddingResolver().resolve(
        "Санкт-Петербург",
        None,
        symbols,
        candidate_uids=("s_003",),
        threshold=0.98,
    )
    assert force.selected_uid is None


def test_negative_doctor_patient():
    symbols = control_symbols()
    morph = MorphologyResolver()
    r = morph.resolve(
        "пациент",
        None,
        symbols,
        candidate_uids=("s_011",),
        threshold=0.5,
    )
    assert r.selected_uid is None
    exact = ExactResolver().resolve(
        "пациент",
        None,
        symbols,
        candidate_uids=("s_011", "s_022"),
    )
    assert exact.selected_uid == "s_022"


def test_threshold_sweep():
    bench = EntityResolutionBenchmark(thresholds=(0.70, 0.80, 0.90))
    report = bench.run(
        resolvers={"embedding": EmbeddingResolver()},
        thresholds=(0.70, 0.80, 0.90),
        run_activation=False,
    )
    sweep = {row["threshold"]: row for row in report["threshold_sweep"]["embedding"]}
    assert set(sweep) >= {0.70, 0.80, 0.90}
    opt = report["resolvers"]["embedding"]["optimal"]
    assert opt["f1_optimal"] is not None
    assert opt["safety_optimal"] is not None
    assert len(DEFAULT_THRESHOLDS) >= 10


def test_activation_after_resolution():
    symbols = control_symbols()
    store, uid_map = build_resolution_store(symbols, include_facts=True)
    case = _yulia_case()
    outcome = evaluate_case(
        case,
        MorphologyResolver(),
        symbols,
        threshold=0.5,
        store=store,
        uid_map=uid_map,
        run_activation=True,
    )
    assert outcome.predicted_uid == "s_001"
    assert outcome.correct is True
    assert outcome.activation_ok is True
    assert outcome.activation_trace
    ah_uid = uid_map["s_001"]
    ok, ticks = run_activation_probe(store, ah_uid)
    assert ok is True
    assert ticks


def test_find_optimal_threshold_modes():
    from ah_memory.benchmarks.entity_resolution.metrics import MetricBundle

    sweep = {
        0.7: MetricBundle(0.8, 0.7, 0.9, 0.79, 0.05, 0.1, 0.1, 10),
        0.85: MetricBundle(0.85, 0.9, 0.8, 0.85, 0.0, 0.05, 0.1, 10),
        0.95: MetricBundle(0.7, 0.95, 0.5, 0.65, 0.0, 0.02, 0.3, 10),
    }
    opt = find_optimal_threshold(sweep, max_false_merge=0.01)
    assert opt["f1_optimal"] == 0.85
    assert opt["safety_optimal"] in {0.85, 0.95}


def test_control_dataset_types():
    types = {c.case_type for c in control_cases()}
    assert CaseType.MORPHOLOGY in types
    assert CaseType.SYNONYM in types
    assert CaseType.NEGATIVE in types
    assert CaseType.CONTEXTUAL in types
    assert any(c.mention == "Юли" and c.target_uid == "s_001" for c in control_cases())
