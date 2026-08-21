from __future__ import annotations

import json

import pytest

from ah_memory.agent import Agent
from ah_memory.factor_parameters import (
    EmbeddingParameterGenerator,
    FactorParameters,
    RuleBasedParameterGenerator,
)
from ah_memory.graph_export import dump_graph
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.relation_normalizer import (
    EmbeddingNormalizer,
    ExactNormalizer,
    LLMNormalizer,
    RelationNormalizer,
)
from ah_memory.relation_registry import RelationRegistry, default_relation_registry
from ah_memory.relations import Relation, RelationContext, RelationProperties
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def test_arbitrary_relation_is_registered_without_source_change() -> None:
    registry = RelationRegistry()
    normalizer = RelationNormalizer(registry, (ExactNormalizer(),))
    normalized = normalizer.normalize(
        "познакомился с",
        RelationContext(roles={"SUBJECT": "A", "OBJECT": "B"}),
    )
    relation = registry.get_relation(normalized.canonical_label)
    assert relation is not None
    assert relation.raw_label == "познакомился с"
    assert relation.arity == 2


def test_exact_normalizer_preserves_raw_relation() -> None:
    registry = default_relation_registry()
    normalized = RelationNormalizer(
        registry,
        (ExactNormalizer({"linked": "RELATE"}),),
    ).normalize("linked")
    assert normalized.raw_label == "linked"
    assert normalized.canonical_label == "RELATE"
    assert normalized.strategy == "exact"


def test_embedding_normalizer_uses_configurable_threshold() -> None:
    purchase = Relation(
        uid="REL_PURCHASE",
        raw_label="purchase",
        canonical_label="PURCHASE",
        embedding=(1.0, 0.0),
    )
    sell = Relation(
        uid="REL_SELL",
        raw_label="sell",
        canonical_label="SELL",
        embedding=(0.0, 1.0),
    )
    registry = RelationRegistry((purchase, sell))
    vectors = {"приобрёл": (0.9, 0.1)}
    strategy = EmbeddingNormalizer(
        lambda text: vectors[text],
        similarity_threshold=0.8,
    )
    normalized = strategy.normalize("приобрёл", registry)
    assert normalized is not None
    assert normalized.canonical_label == "PURCHASE"


def test_llm_normalizer_does_not_mutate_graph() -> None:
    store = AHStore()
    before = store.ah.revision

    def call(prompt: str) -> str:
        request = json.loads(prompt)
        assert request["raw_relation"] == "учился у"
        return json.dumps(
            {
                "canonical_label": "STUDIED_UNDER",
                "confidence": 0.92,
                "create_new": True,
                "properties": {"directional": True},
            }
        )

    normalizer = RelationNormalizer(store.relations, (LLMNormalizer(call),))
    normalized = normalizer.normalize("учился у")
    assert normalized.canonical_label == "STUDIED_UNDER"
    assert store.ah.revision == before
    assert not store.semantic_factors


def test_transform_stores_raw_and_canonical_relation() -> None:
    store = AHStore()
    report = Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="PURCHASE",
                    raw_relation="приобрёл",
                    canonical_relation="PURCHASE",
                    roles={"SUBJECT": "FATHER", "OBJECT": "BMW"},
                    confidence=0.95,
                )
            ],
        )
    )
    assert report.created_n
    factor = store.list_semantic_factors()[0]
    assert factor.relation.canonical_label == "PURCHASE"
    assert factor.metadata["raw_relation"] == "приобрёл"
    assert factor.variables == ("M_FATHER", "M_BMW")
    assert store.list_events()[0].metadata["raw_relation"] == "приобрёл"


def test_event_factor_can_bind_more_than_two_variables() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="PURCHASE",
                    raw_relation="купил",
                    canonical_relation="PURCHASE",
                    roles={
                        "SUBJECT": "FATHER",
                        "OBJECT": "BMW",
                        "TIME": "T1",
                    },
                )
            ],
        )
    )
    factor = store.list_semantic_factors()[0]
    event = store.list_events()[0]
    assert len(factor.variables) == 3
    assert factor.roles["TIME"] == "M_T1"
    assert event.timestamp == "M_T1"


def test_open_semantic_factor_is_visible_in_exported_graph() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="PURCHASE",
                    raw_relation="купил",
                    canonical_relation="PURCHASE",
                    roles={"SUBJECT": "FATHER", "OBJECT": "BMW"},
                )
            ],
        )
    )
    graph = dump_graph(store)
    factor = store.list_semantic_factors()[0]
    exported = next(item for item in graph["nodes"] if item["id"] == factor.uid)
    spokes = [
        edge
        for edge in graph["edges"]
        if edge["kind"] == "hyper_incidence"
    ]
    assert exported["kind"] == "hyperedge"
    assert graph["hyperedges"][0]["predicate"] == "PURCHASE"
    assert len(spokes) == 2
    assert any(
        edge["from"] == factor.uid and edge["to"] == "M_FATHER"
        for edge in spokes
    )
    assert any(
        edge["from"] == factor.uid and edge["to"] == "M_BMW"
        for edge in spokes
    )
    # S/M + semantic factor + BIND/mesh links
    assert graph["stats"]["graph_size"] >= 3
    assert len(store.list_semantic_factors()) == 1


def test_generators_produce_serializable_relation_parameters() -> None:
    relation = Relation(
        uid="REL_TEMPORAL_CAUSE",
        raw_label="after causing",
        canonical_label="TEMPORAL_CAUSE",
        properties=RelationProperties(
            directional=True,
            temporal=True,
            causal=True,
        ),
    )
    rule = RuleBasedParameterGenerator().generate(relation)
    learned = EmbeddingParameterGenerator(seed=7).generate(relation)
    assert rule.temporal_bias > 0.0
    assert rule.causal_bias > 0.0
    assert set(rule.to_dict()) == set(FactorParameters().to_dict())
    assert all(0.0 <= value <= 1.0 for value in learned.to_dict().values())


def test_relation_specific_parameters_are_data_driven_overrides() -> None:
    relation = Relation(
        uid="REL_CUSTOM",
        raw_label="custom",
        canonical_label="CUSTOM",
    )
    override = FactorParameters(transmission_strength=0.93, persistence=0.88)
    generator = RuleBasedParameterGenerator(
        overrides={"CUSTOM": override}
    )
    assert generator.generate(relation) == override


def test_embedding_projection_can_be_replaced_for_training() -> None:
    generator = EmbeddingParameterGenerator(embedding_dimensions=2, seed=1)
    generator.set_projection([[0.0, 0.0]] * 7, [0.0] * 7)
    relation = Relation(
        uid="REL_X",
        raw_label="x",
        canonical_label="X",
        embedding=(1.0, 0.0),
    )
    parameters = generator.generate(relation)
    assert all(value == pytest.approx(0.5) for value in parameters.to_dict().values())


def test_agent_answer_exposes_complete_structured_trace() -> None:
    class QuestionPerception:
        def parse(self, text: str, wm_context=None) -> PerceptionResult:
            return PerceptionResult(
                kind="question",
                candidates=[],
                seed_tokens=["FATHER"],
            )

    agent = Agent(perception=QuestionPerception())
    agent.transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="PURCHASE",
                    raw_relation="купил",
                    canonical_relation="PURCHASE",
                    roles={"SUBJECT": "FATHER", "OBJECT": "BMW"},
                )
            ],
        )
    )
    reply = agent.ask("Что связано с отцом?", ticks=2)
    assert {
        "answer",
        "activated_nodes",
        "activated_factors",
        "timesteps",
        "relations",
        "state_transitions",
        "final_evidence",
    }.issubset(reply.full_trace)
    assert reply.full_trace["relations"][0]["canonical_label"] == "PURCHASE"
    assert not any(
        uid.startswith("SF::") for uid in reply.full_trace["activated_nodes"]
    )
    assert any(
        uid.startswith("SF::") for uid in reply.full_trace["activated_factors"]
    )
