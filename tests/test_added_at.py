"""Symbol/fact creation timestamps appear in store metadata and LLM context."""
from __future__ import annotations

import json

from ah_memory.dialogue import DialogueAgent, _format_added_at, _is_recap_request
from ah_memory.agent import Agent
from ah_memory.perception import FactCandidate, JsonLLMPerception, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def test_ensure_m_stamps_added_at_and_tau() -> None:
    store = AHStore()
    store.ah.tau = 7
    m = store.ensure_m("M_IVAN", "Иван")
    assert m.created_tau == 7
    assert m.added_at
    assert not any(p.name in ("added_at", "created_tau") for p in m.Mt)
    added, tau = store.get_added_at("M_IVAN")
    assert added == m.added_at
    assert tau == 7


def test_recap_detection_prevents_storing_summary_as_new_memory() -> None:
    assert _is_recap_request("Подведи итог: что мы решили?")
    assert _is_recap_request("Напомни решения")
    assert not _is_recap_request("Как организовать read-модель?")


def test_abstract_symbol_stamps_added_at() -> None:
    store = AHStore()
    s = store.ensure_abstract("IVAN", {"иван"})
    assert s.added_at
    assert s.created_tau == 0


def test_semantic_factor_metadata_has_added_at() -> None:
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
    factor = store.list_semantic_factors()[0]
    assert factor.metadata.get("added_at")
    assert "created_tau" in factor.metadata
    event = store.list_events()[0]
    assert event.metadata.get("added_at")


def test_llm_context_includes_added_timestamp() -> None:
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
    agent = Agent(store=store)
    dialogue = DialogueAgent(agent)
    factor = store.list_semantic_factors()[0]
    ctx = dialogue._compact_memory_for_llm(
        {
            "activated_nodes": list(factor.variables),
            "events": [],
            "state": {},
        }
    )
    assert "добавлено:" in ctx
    assert "Факты (с временем добавления)" in ctx
    pretty = _format_added_at(str(factor.metadata["added_at"]), int(factor.metadata["created_tau"]))
    assert pretty
    assert pretty.split(",")[0] in ctx or "τ=" in ctx


def test_llm_context_keeps_recent_facts_beside_activated_facts() -> None:
    store = AHStore()
    transform = Transform(store)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="DESIGN",
                    raw_relation="проектируем",
                    canonical_relation="DESIGN",
                    roles={"SUBJECT": "МЫ", "OBJECT": "СИСТЕМА"},
                )
            ],
        )
    )
    old_factor = store.list_semantic_factors()[0]
    store.ah.tau = 10
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="STORE_IN",
                    raw_relation="храним",
                    canonical_relation="STORE_IN",
                    roles={"SUBJECT": "МЫ", "OBJECT": "СОБЫТИЕ", "TOOL": "KAFKA"},
                )
            ],
        )
    )

    dialogue = DialogueAgent(Agent(store=store))
    ctx = dialogue._compact_memory_for_llm(
        {
            "activated_nodes": list(old_factor.variables),
            "events": [],
            "state": {},
        }
    )
    assert "DESIGN" in ctx
    assert "STORE_IN" in ctx
    assert "kafka" in ctx.lower()


def test_decision_status_is_stored_and_shown_in_llm_context() -> None:
    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="CHOOSE",
                    raw_relation="решили использовать",
                    canonical_relation="CHOOSE",
                    roles={"SUBJECT": "МЫ", "OBJECT": "OUTBOX"},
                    statement_type="decision",
                )
            ],
        )
    )
    factor = store.list_semantic_factors()[0]
    event = store.list_events()[0]
    assert factor.metadata["statement_type"] == "decision"
    assert event.metadata["statement_type"] == "decision"

    ctx = DialogueAgent(Agent(store=store))._compact_memory_for_llm(
        {
            "activated_nodes": list(factor.variables),
            "events": [],
            "state": {},
        }
    )
    assert "РЕШЕНИЕ:" in ctx


def test_topic_factor_is_weaker_and_marked_as_nonfactual() -> None:
    store = AHStore()
    agent = Agent(store=store)
    agent.transform.apply(
        PerceptionResult(
            kind="question",
            candidates=[
                FactCandidate(
                    predicate="ASK_ABOUT",
                    raw_relation="обсуждаем",
                    canonical_relation="ASK_ABOUT",
                    roles={"SUBJECT": "ДИАЛОГ", "OBJECT": "САГА"},
                    statement_type="open_question",
                )
            ],
        )
    )
    factor = store.list_semantic_factors()[0]
    assert factor.w < agent.hp.initial_w
    assert factor.metadata["statement_type"] == "open_question"

    ctx = DialogueAgent(agent)._compact_memory_for_llm(
        {
            "activated_nodes": list(factor.variables),
            "events": [],
            "state": {},
        }
    )
    assert "ОТКРЫТЫЙ ВОПРОС:" in ctx


def test_assistant_memory_is_ingested_as_explanation() -> None:
    class FakeClient:
        def chat(self, messages, *, json_mode=True):
            del messages, json_mode
            return json.dumps(
                {
                    "kind": "message",
                    "candidates": [
                        {
                            "predicate": "SUITABLE_FOR",
                            "raw_relation": "подходит для",
                            "roles": {
                                "SUBJECT": "Redis",
                                "OBJECT": "хранение состояния",
                            },
                            "raw_span": "Redis подходит для хранения состояния",
                            "confidence": 0.9,
                            "statement_type": "explanation",
                        }
                    ],
                    "seed_tokens": ["REDIS", "СОСТОЯНИЕ"],
                },
                ensure_ascii=False,
            )

    agent = Agent(store=AHStore())
    dialogue = DialogueAgent(agent, chat_client=FakeClient(), provider="fake")
    perception = dialogue._parse_assistant_memory(
        "Redis подходит для хранения состояния.",
        [],
    )
    assert len(perception.candidates) == 1
    assert perception.candidates[0].statement_type == "explanation"
    assert perception.candidates[0].source == "assistant"

    report = agent.ingest(
        "Redis подходит для хранения состояния.",
        source="assistant",
        perception=perception,
    )
    assert len(report.created_n) == 1
    factor = agent.store.list_semantic_factors()[0]
    assert factor.metadata["source"] == "assistant"


def test_project_anchor_connects_later_factors_without_changing_roles() -> None:
    store = AHStore()
    transform = Transform(store)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="DESIGN",
                    roles={"SUBJECT": "МЫ", "OBJECT": "СИСТЕМА_ЗАКАЗОВ"},
                    canonical_relation="DESIGN",
                )
            ],
        )
    )
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="STORE_IN",
                    roles={"SUBJECT": "СОБЫТИЕ", "LOCATION": "KAFKA"},
                    canonical_relation="STORE_IN",
                    statement_type="decision",
                )
            ],
        )
    )
    later = store.list_semantic_factors()[-1]
    assert "M_СИСТЕМА_ЗАКАЗ" in later.variables
    assert "M_СИСТЕМА_ЗАКАЗ" not in later.roles.values()
    assert later.metadata["context_uid"] == "M_СИСТЕМА_ЗАКАЗ"


def test_recap_retrieval_prioritizes_all_decisions_over_context_noise() -> None:
    store = AHStore()
    transform = Transform(store)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="DESIGN",
                    roles={"SUBJECT": "МЫ", "OBJECT": "СИСТЕМА"},
                    canonical_relation="DESIGN",
                )
            ],
        )
    )
    for predicate, obj in [
        ("CHOOSE", "EVENT_SOURCING"),
        ("USE", "KAFKA"),
        ("STORE_IN", "REDIS"),
    ]:
        transform.apply(
            PerceptionResult(
                kind="fact",
                candidates=[
                    FactCandidate(
                        predicate=predicate,
                        roles={"SUBJECT": "МЫ", "OBJECT": obj},
                        canonical_relation=predicate,
                        statement_type="decision",
                    )
                ],
            )
        )
    transform.apply(
        PerceptionResult(
            kind="message",
            candidates=[
                FactCandidate(
                    predicate="SUGGEST",
                    roles={"SUBJECT": "АССИСТЕНТ", "OBJECT": "ОПТИМИЗАЦИЯ"},
                    canonical_relation="SUGGEST",
                    statement_type="proposal",
                    source="assistant",
                )
            ],
        )
    )

    dialogue = DialogueAgent(Agent(store=store))
    recall = dialogue._memory_context("Напомни итоговую архитектуру")
    assert "EVENT_SOURCING".lower().replace("_", " ") in recall.lower()
    assert "kafka" in recall.lower()
    assert "redis" in recall.lower()

    compact = dialogue._compact_memory_for_llm(
        {
            # Shared context alone must not make every factor semantically relevant.
            "activated_nodes": ["M_СИСТЕМА"],
            "events": [],
            "state": {},
        }
    )
    assert compact.index("РЕШЕНИЕ:") < compact.index("ПРЕДЛОЖЕНИЕ АССИСТЕНТА:")


def test_parser_to_graph_preserves_normalized_decision_roles() -> None:
    def parse_call(prompt: str) -> str:
        request = json.loads(prompt)
        assert "PostgreSQL outbox" in request["text"]
        return json.dumps(
            {
                "kind": "fact",
                "candidates": [
                    {
                        "predicate": "PLACE",
                        "raw_relation": "решили разместить",
                        "raw_span": "write-модель в PostgreSQL outbox",
                        "roles": {
                            "SUBJECT": "write-модель",
                            "LOCATION": "PostgreSQL",
                            "WITH": "outbox",
                        },
                        "confidence": 0.95,
                        "statement_type": "decision",
                    }
                ],
            },
            ensure_ascii=False,
        )

    store = AHStore()
    agent = Agent(store=store, perception=JsonLLMPerception(parse_call))
    report = agent.ingest(
        "Мы решили разместить write-модель в PostgreSQL outbox."
    )

    assert len(report.created_n) == 1
    factor = store.list_semantic_factors()[0]
    assert factor.metadata["statement_type"] == "decision"
    assert factor.roles["LOCATION"] == "M_POSTGRESQL"
    assert factor.roles["WITH"] == "M_OUTBOX"
    assert "M_POSTGRESQL_OUTBOX" not in factor.variables


def test_repeated_recap_does_not_create_assistant_factors() -> None:
    class ReplyOnlyClient:
        def __init__(self) -> None:
            self.json_modes: list[bool] = []

        def chat(self, messages, *, json_mode=True):
            del messages
            self.json_modes.append(json_mode)
            if json_mode:
                raise AssertionError("recap must not invoke assistant memory parser")
            return "Мы решили использовать Kafka."

    store = AHStore()
    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="USE",
                    roles={"SUBJECT": "МЫ", "OBJECT": "KAFKA"},
                    canonical_relation="USE",
                    statement_type="decision",
                )
            ],
        )
    )
    factor_count = len(store.list_semantic_factors())
    client = ReplyOnlyClient()
    dialogue = DialogueAgent(Agent(store=store), chat_client=client, provider="fake")

    dialogue.talk("Подведи итог: что мы решили?")
    dialogue.talk("Напомни итог ещё раз")

    assert client.json_modes == [False, False]
    assert len(store.list_semantic_factors()) == factor_count
    assert not any(
        factor.metadata.get("source") == "assistant"
        for factor in store.list_semantic_factors()
    )
