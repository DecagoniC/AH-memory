from __future__ import annotations

from ah_memory.agent import Agent
from ah_memory.examples.closed_world import build_closed_world_memory
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore


def _resource_question() -> PerceptionResult:
    return PerceptionResult(
        kind="question",
        candidates=[
            FactCandidate(
                predicate="HAS_RESOURCE",
                raw_relation="ресурсы",
                canonical_relation="HAS_RESOURCE",
                roles={
                    "SUBJECT": "ТИМАНСКИЙ_КРЯЖ",
                    "OBJECT": "РЕСУРС",
                },
                statement_type="open_question",
            )
        ],
        seed_tokens=["ТИМАНСКИЙ_КРЯЖ", "РЕСУРС"],
    )


def test_semantic_entry_point_maps_query_relation_to_existing_factors() -> None:
    agent = Agent(store=build_closed_world_memory())

    reply = agent.ask(
        "Какие ресурсы есть на Тиманском кряже?",
        perception=_resource_question(),
    )

    for expected in (
        "россыпи титановых минералов (Ярега)",
        "бокситы (Четласский Камень)",
        "агаты, связанные с девонскими базальтами",
        "месторождения нефти",
        "газа и конденсата (Войвож и Омра)",
        "битумы и горючие сланцы",
        "торф",
        "строительные камни",
    ):
        assert expected in reply.answer
    assert len(reply.answer.split("; ")) == 8
    assert "россыпь титановый" not in reply.answer
    assert "строительный камни" not in reply.answer
    assert "activated:" not in reply.answer
    assert {
        factor.relation.canonical_label
        for factor in agent.store.list_semantic_factors()
        if factor.relation is not None
        and factor.uid in reply.trace_uids
    } == {"HAS_MINERAL", "LOCATED_AT", "ASSOCIATED_WITH"}


def test_ingesting_question_does_not_create_symbols_events_or_factors() -> None:
    store = AHStore()
    agent = Agent(store=store)
    before = (
        len(store.ah.S),
        len(store.ah.all_hyper()),
        len(store.list_events()),
        len(store.list_semantic_factors()),
    )

    report = agent.ingest(
        "Какие ресурсы есть у объекта?",
        perception=_resource_question(),
    )

    assert report.created_n == []
    assert report.seed_uids == []
    assert (
        len(store.ah.S),
        len(store.ah.all_hyper()),
        len(store.list_events()),
        len(store.list_semantic_factors()),
    ) == before


def test_step_message_routes_typed_question_candidates_to_query_planner() -> None:
    perception = _resource_question()

    class StaticPerception:
        def parse(self, text: str, wm_context=None) -> PerceptionResult:
            return perception

    agent = Agent(
        store=build_closed_world_memory(),
        perception=StaticPerception(),
    )

    reply = agent.step_message("Какие ресурсы есть на Тиманском кряже?")

    assert reply.answer != "неизвестно"
    assert reply.full_trace["query_plan"]["factor_gates"]


def test_query_metadata_uses_question_terms_and_longest_entity_anchor() -> None:
    agent = Agent(store=build_closed_world_memory())
    perception = PerceptionResult(
        kind="question",
        candidates=[],
        seed_tokens=["РЕСУРС", "КРЯЖ"],
        meta={
            "llm_raw": {
                "kind": "question",
                "candidates": [],
                "query": {
                    "relation": "has_resource",
                    "target_role": "OBJECT",
                    "cardinality": "many",
                },
            }
        },
    )

    reply = agent.ask(
        "Какие ресурсы есть на Тиманском кряже?",
        perception=perception,
    )
    plan = reply.full_trace["query_plan"]

    assert plan["anchors"] == ["M_ТИМАНСКИЙ_КРЯЖ"]
    assert plan["relation_text"] == "has resource | ресурс"
    assert next(iter(plan["relation_scores"])) == "HAS_MINERAL"
    assert len(plan["factor_gates"]) == 10
    assert "байкальск" not in reply.answer.lower()


def test_question_text_plans_query_when_llm_omits_query_metadata() -> None:
    agent = Agent(store=build_closed_world_memory())
    degraded_perception = PerceptionResult(
        kind="message",
        candidates=[],
        seed_tokens=[],
    )

    reply = agent.ask(
        "Какие ресурсы есть на Тиманском кряже?",
        perception=degraded_perception,
    )

    plan = reply.full_trace["query_plan"]
    assert plan["anchors"] == ["M_ТИМАНСКИЙ_КРЯЖ"]
    assert plan["relation_text"] == "ресурс"
    assert next(iter(plan["relation_scores"])) == "HAS_MINERAL"
    assert len(plan["factor_gates"]) == 10
    assert "россыпи титановых минералов (Ярега)" in reply.answer
    assert reply.answer != "неизвестно"


def test_question_text_recovers_from_invalid_llm_relation_and_role() -> None:
    agent = Agent(store=build_closed_world_memory())
    invalid_perception = PerceptionResult(
        kind="question",
        candidates=[],
        seed_tokens=[],
        meta={
            "llm_raw": {
                "query": {
                    "relation": "unrelated_attribute",
                    "target_role": "MATERIAL",
                    "cardinality": "many",
                }
            }
        },
    )

    reply = agent.ask(
        "Какие ресурсы есть на Тиманском кряже?",
        perception=invalid_perception,
    )

    plan = reply.full_trace["query_plan"]
    assert next(iter(plan["relation_scores"])) == "HAS_MINERAL"
    assert plan["target_role"] == "OBJECT"
    assert len(plan["factor_gates"]) == 10
    assert reply.answer != "неизвестно"
