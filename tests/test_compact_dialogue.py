from __future__ import annotations

import json

from ah_memory.agent import Agent
from ah_memory.dialogue import DialogueAgent
from ah_memory.graph_export import dump_graph
from ah_memory.perception import JsonLLMPerception
from ah_memory.store import AHStore


TURNS = [
    "Мы проектируем систему заказов для маркетплейса.",
    "CRUD в PostgreSQL или event sourcing?",
    "Ок, event sourcing. Как организовать read-модель?",
    (
        "Решили: write-модель в PostgreSQL (outbox), события в Kafka, "
        "топик orders.v1, read-модель в Redis с TTL 5 минут."
    ),
    (
        "Для аудита храним события 90 дней в Kafka, затем архивируем "
        "в S3 Parquet. Напомни итоговую архитектуру."
    ),
]


USER_PARSE = {
    TURNS[0]: {
        "kind": "fact",
        "candidates": [
            {
                "predicate": "DESIGN",
                "raw_span": TURNS[0],
                "roles": {
                    "SUBJECT": "мы",
                    "OBJECT": "система заказов",
                    "PURPOSE": "маркетплейс",
                },
                "statement_type": "assertion",
                "confidence": 0.95,
            }
        ],
    },
    TURNS[1]: {
        "kind": "question",
        "candidates": [
            {
                "predicate": "COMPARE",
                "raw_span": TURNS[1],
                "roles": {
                    "SUBJECT": "диалог",
                    "OBJECT": "CRUD",
                    "WITH": "event sourcing",
                },
                "statement_type": "open_question",
                "confidence": 0.95,
            },
            {
                "predicate": "CONSIDER",
                "raw_span": TURNS[1],
                "roles": {"SUBJECT": "диалог", "OBJECT": "PostgreSQL"},
                "statement_type": "topic",
                "confidence": 0.9,
            },
        ],
    },
    TURNS[2]: {
        "kind": "question",
        "candidates": [
            {
                "predicate": "CHOOSE",
                "raw_span": "Ок, event sourcing.",
                "roles": {"SUBJECT": "мы", "OBJECT": "event sourcing"},
                "statement_type": "decision",
                "confidence": 0.98,
            },
            {
                "predicate": "ORGANIZE",
                "raw_span": "Как организовать read-модель?",
                "roles": {"SUBJECT": "диалог", "OBJECT": "read-модель"},
                "statement_type": "open_question",
                "confidence": 0.95,
            },
        ],
    },
    TURNS[3]: {
        "kind": "fact",
        "candidates": [
            {
                "predicate": "PLACE",
                "raw_span": "write-модель в PostgreSQL (outbox)",
                "roles": {
                    "SUBJECT": "write-модель",
                    "LOCATION": "PostgreSQL",
                    "WITH": "outbox",
                },
                "statement_type": "decision",
                "confidence": 0.98,
            },
            {
                "predicate": "STORE_EVENTS",
                "raw_span": "события в Kafka",
                "roles": {"SUBJECT": "события", "LOCATION": "Kafka"},
                "statement_type": "decision",
                "confidence": 0.98,
            },
            {
                "predicate": "HAS_TOPIC",
                "raw_span": "Kafka, топик orders.v1",
                "roles": {"SUBJECT": "Kafka", "OBJECT": "orders.v1"},
                "statement_type": "decision",
                "confidence": 0.98,
            },
            {
                "predicate": "STORE_IN",
                "raw_span": "read-модель в Redis с TTL 5 минут",
                "roles": {
                    "SUBJECT": "read-модель",
                    "LOCATION": "Redis",
                    "WITH": "TTL",
                    "TIME": "5 минут",
                },
                "statement_type": "decision",
                "confidence": 0.98,
            },
        ],
    },
    TURNS[4]: {
        "kind": "question",
        "candidates": [
            {
                "predicate": "STORE_EVENTS",
                "raw_span": "Для аудита храним события 90 дней в Kafka",
                "roles": {
                    "SUBJECT": "события",
                    "TIME": "90 дней",
                    "LOCATION": "Kafka",
                },
                "statement_type": "assertion",
                "confidence": 0.98,
            },
            {
                "predicate": "ARCHIVE_TO",
                "raw_span": "затем архивируем в S3 Parquet",
                "roles": {"SUBJECT": "события", "LOCATION": "S3 Parquet"},
                "statement_type": "assertion",
                "confidence": 0.98,
            },
        ],
    },
}


class ScriptedChatClient:
    def __init__(self) -> None:
        self.reply_index = 0
        self.final_system_prompt = ""

    def chat(self, messages, *, json_mode=True):
        if json_mode:
            return json.dumps(
                {"kind": "message", "candidates": [], "seed_tokens": []}
            )
        if self.reply_index == len(TURNS) - 1:
            self.final_system_prompt = messages[0]["content"]
        replies = [
            "Для аудита и истории изменений подойдёт event sourcing.",
            "Event sourcing потребует отдельной read-модели.",
            "Read-модель можно обновлять событиями из Kafka.",
            "Архитектурные решения зафиксированы.",
            (
                "Итог: event sourcing, PostgreSQL с outbox, Kafka orders.v1, "
                "Redis с TTL 5 минут и архив S3 Parquet."
            ),
        ]
        reply = replies[self.reply_index]
        self.reply_index += 1
        return reply


def _run_compact_dialogue():
    def parse_user(prompt: str) -> str:
        text = json.loads(prompt)["text"]
        return json.dumps(USER_PARSE[text], ensure_ascii=False)

    store = AHStore()
    client = ScriptedChatClient()
    dialogue = DialogueAgent(
        Agent(store=store, perception=JsonLLMPerception(parse_user)),
        chat_client=client,
        provider="scripted",
    )
    results = [dialogue.talk(turn) for turn in TURNS]
    return store, client, results


def test_compact_architecture_dialogue_builds_inspectable_graph() -> None:
    store, client, results = _run_compact_dialogue()
    factors = store.list_semantic_factors()
    relations = [factor.relation.canonical_label for factor in factors]

    print("COMPACT_DIALOGUE_RELATIONS=" + json.dumps(relations))
    expected_candidates = sum(
        sum(
            candidate["statement_type"] not in {"topic", "open_question"}
            for candidate in payload["candidates"]
        )
        for payload in USER_PARSE.values()
    )
    assert len(factors) == expected_candidates, relations
    anchor_uid = factors[0].roles["OBJECT"]
    assert all(
        factor.metadata.get("context_uid") == anchor_uid
        for factor in factors[1:]
    )
    assert all(anchor_uid not in factor.roles.values() for factor in factors[1:])

    qualified = next(
        factor
        for factor in factors
        if "LOCATION" in factor.roles and "WITH" in factor.roles
    )
    merged_uid = f"{qualified.roles['LOCATION']}_{qualified.roles['WITH']}"
    assert qualified.roles["LOCATION"] != qualified.roles["WITH"]
    assert merged_uid not in qualified.variables

    read_model = next(
        factor
        for factor in factors
        if {"LOCATION", "WITH", "TIME"}.issubset(factor.roles)
    )
    assert "OBJECT" not in read_model.roles

    retention, archive = factors[-2:]
    assert archive.roles["SUBJECT"] == retention.roles["SUBJECT"]
    assert archive.roles["LOCATION"] != retention.roles["LOCATION"]

    prompt = client.final_system_prompt.lower()
    decision_factors = [
        factor
        for factor in factors
        if factor.metadata.get("statement_type") == "decision"
    ]
    assert decision_factors
    for factor in decision_factors:
        labels = [
            dialogue_label
            for role, uid in factor.roles.items()
            if role in {"OBJECT", "LOCATION", "WITH"}
            and (dialogue_label := uid.removeprefix("M_").replace("_", " ").lower())
        ]
        assert any(label in prompt for label in labels)
    assert results[-1].assistant_facts == []
    assert not any(
        factor.metadata.get("source") == "assistant" for factor in factors
    )

    exported = dump_graph(store, mode="hyper")
    exported_factor_ids = {
        node["id"]
        for node in exported["nodes"]
        if node.get("kind") in {"semantic_factor", "hyperedge"}
    }
    assert {factor.uid for factor in factors}.issubset(exported_factor_ids)
    snapshot = {
        "factor_count": len(factors),
        "factors": [
            {
                "relation": factor.relation.canonical_label,
                    "roles": dict(factor.roles),
                    "variables": list(factor.variables),
                "statement_type": factor.metadata.get("statement_type"),
                "context_uid": factor.metadata.get("context_uid"),
            }
            for factor in factors
        ],
        "graph": {
            "nodes": len(exported["nodes"]),
            "edges": len(exported["edges"]),
            "hyperedges": len(exported["hyperedges"]),
        },
    }
    print("COMPACT_DIALOGUE_GRAPH=" + json.dumps(snapshot, ensure_ascii=False))
