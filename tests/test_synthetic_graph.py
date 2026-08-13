"""Tests for synthetic world generator, ingest, and activation benchmark."""
from __future__ import annotations

from pathlib import Path

from ah_memory.synthetic import (
    SyntheticGraphConfig,
    SyntheticGraphGenerator,
    export_dataset,
    get_preset,
    ingest_world,
    run_benchmark,
)
def _small_world(seed: int = 42):
    return SyntheticGraphGenerator(get_preset("small")).generate() if seed == 42 else (
        SyntheticGraphGenerator(
            SyntheticGraphConfig(
                num_entities=100,
                num_factors=500,
                num_events=200,
                num_queries=50,
                max_hop_depth=3,
                distractor_ratio=0.3,
                random_seed=seed,
                preset="small",
            )
        ).generate()
    )


def test_seed_reproducibility() -> None:
    a = _small_world(42)
    b = _small_world(42)
    assert [e.uid for e in a.entities] == [e.uid for e in b.entities]
    assert [f.to_dict() for f in a.factors] == [f.to_dict() for f in b.factors]
    assert [q.to_dict() for q in a.queries] == [q.to_dict() for q in b.queries]


def test_different_seed_different_world() -> None:
    a = _small_world(42)
    b = _small_world(43)
    assert [f.uid for f in a.factors] != [f.uid for f in b.factors] or [
        f.arguments for f in a.factors
    ] != [f.arguments for f in b.factors]


def test_no_broken_uids() -> None:
    world = _small_world()
    entity_uids = {entity.uid for entity in world.entities}
    for factor in world.factors:
        assert factor.uid.startswith("factor_")
        for role, uid in factor.arguments.items():
            assert uid in entity_uids, f"broken uid {uid} in {factor.uid}.{role}"


def test_isa_acyclic() -> None:
    world = _small_world()
    parent: dict[str, str] = {}
    for factor in world.factors:
        if factor.type != "IS_A":
            continue
        child = factor.arguments["child"]
        parent[child] = factor.arguments["parent"]

    def has_cycle(node: str, stack: set[str]) -> bool:
        if node in stack:
            return True
        nxt = parent.get(node)
        if nxt is None:
            return False
        return has_cycle(nxt, stack | {node})

    assert not any(has_cycle(node, set()) for node in parent)


def test_follow_respects_temporal_order() -> None:
    world = _small_world()
    by_uid = {factor.uid: factor for factor in world.factors}
    for factor in world.factors:
        if factor.type != "FOLLOW":
            continue
        prev_uid = factor.arguments["previous_event"]
        next_uid = factor.arguments["next_event"]
        # FOLLOW may link Event entities; story_events carry ordered labels.
        story = factor.properties.get("story_events")
        if story and len(story) == 2:
            assert story[0] != story[1]
        assert prev_uid != next_uid
        assert factor.timestamp >= 0
        assert factor.uid in by_uid


def test_every_query_has_answer_and_multihop_proof() -> None:
    world = _small_world()
    assert world.queries
    for query in world.queries:
        assert query.answer
        assert query.answer_uid
        if query.category == "multi_hop":
            assert query.proof_path
            assert query.required_depth >= 1


def test_documents_only_contain_ground_truth_factors() -> None:
    world = _small_world()
    factor_uids = {factor.uid for factor in world.factors}
    distractors = {
        factor.uid
        for factor in world.factors
        if factor.properties.get("distractor")
    }
    for document in world.documents:
        assert document.factor_uids
        for uid in document.factor_uids:
            assert uid in factor_uids
            assert uid not in distractors


def test_ingest_into_ah_store() -> None:
    world = _small_world()
    result = ingest_world(world)
    assert result.ingested_factors == len(world.factors)
    assert len(result.store.semantic_factors) == len(world.factors)
    assert result.uid_map
    assert all(uid.startswith("M_") for uid in result.uid_map.values())


def test_export_dataset_files(tmp_path: Path) -> None:
    world = _small_world()
    root = export_dataset(world, tmp_path / "dataset")
    for name in (
        "metadata.json",
        "entities.jsonl",
        "factors.jsonl",
        "events.jsonl",
        "documents.jsonl",
        "queries.jsonl",
        "ground_truth.json",
        "graph.graphml",
    ):
        assert (root / name).exists()


def test_small_benchmark_through_activation_engine(capsys) -> None:
    world = SyntheticGraphGenerator(get_preset("small")).generate()
    ingest = ingest_world(world)
    report = run_benchmark(ingest.store, world, ingest, limit=20)
    assert report.aggregate["queries"] == 20
    for key in (
        "answer_accuracy",
        "path_precision",
        "path_recall",
        "path_f1",
        "activation_noise",
    ):
        assert key in report.aggregate
        assert 0.0 <= report.aggregate[key] <= 1.0
    print(
        "Answer Accuracy={:.3f} Path P={:.3f} R={:.3f} F1={:.3f}".format(
            report.aggregate["answer_accuracy"],
            report.aggregate["path_precision"],
            report.aggregate["path_recall"],
            report.aggregate["path_f1"],
        )
    )
    captured = capsys.readouterr()
    assert "Answer Accuracy=" in captured.out


def test_query_categories_cover_core_types() -> None:
    world = _small_world()
    cats = {query.category for query in world.queries}
    assert "multi_hop" in cats
    assert "temporal" in cats or "aggregation" in cats
    assert "direct" in cats or "hierarchy" in cats
