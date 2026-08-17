from __future__ import annotations

import json

from ah_memory.morph import lemma, sanitize_roles, slug_uid, uid_too_wide
from ah_memory.perception import (
    FactCandidate,
    JsonLLMPerception,
    SeedPerception,
    candidates_from_llm_json,
    gate_candidates,
    llm_payload_errors,
)


def test_lemma_cases_merge() -> None:
    # Geox / regular inflection via pymorphy (no per-word overrides)
    assert lemma("москве") == "москва"
    assert lemma("брата") == "брат"
    assert lemma("преподавателем") == "преподаватель"
    # Short opaque / surname: general proper-tag + Fixd / low-score Name heuristics
    assert lemma("мифи") == "мифи"
    assert lemma("душкина") == "душкин"
    assert slug_uid("скульптурной лепкой") == "СКУЛЬПТУРНЫЙ_ЛЕПКА" or "ЛЕПКА" in slug_uid(
        "скульптурной лепкой"
    )
    assert slug_uid("НИЯУ МИФИ") == "НИЯУ_МИФИ"
    assert slug_uid("Роман Душкина") == "РОМАН_ДУШКИН"


def test_reject_verb_and_conj_subjects() -> None:
    assert sanitize_roles({"SUBJECT": "ЗАНИМАЮСЬ", "OBJECT": "ЛЕПКА"}) is None
    assert sanitize_roles({"SUBJECT": "ЕСЛИ", "OBJECT": "ПОМОЩЬ"}) is None
    ok = sanitize_roles({"SUBJECT": "Я", "OBJECT": "СКУЛЬПТУРНАЯ_ЛЕПКА"})
    assert ok is not None
    assert ok["SUBJECT"] == "Я"
    assert sanitize_roles({"SUBJECT": "Я", "OBJECT": "СЕРЬЁЗНЫЙ"}) is None
    assert uid_too_wide("Я_СЕРГЕЙ_Я_МЛАДШИЙ_БРАТ_МАКСИМ_ВУЗ_НИЯУ_МИФИ")


def test_seed_perception_no_invented_facts() -> None:
    p = SeedPerception()
    r = p.parse("Я занимаюсь скульптурной лепкой. Если понадобится — обращайся.")
    assert r.candidates == []
    assert "ЕСЛИ" not in r.seed_tokens
    assert "ЗАНИМАЮСЬ" not in r.seed_tokens
    assert not any(s in {"ЗАНИМАЮСЬ", "ЕСЛИ", "ЗАНИМАТЬСЯ", "ПОНАДОБИТЬСЯ"} for s in r.seed_tokens)
    assert any("ЛЕПКА" in s or s == "Я" for s in r.seed_tokens)


def test_seed_perception_no_mega_subjects_from_long_text() -> None:
    p = SeedPerception()
    text = (
        "меня зовут сергей, у меня есть младший брат максим, "
        "я учусь в вузе НИЯУ МИФИ в Москве, "
        "где главным преподавателем на кафедре 22 является Роман Душкин"
    )
    r = p.parse(text)
    assert r.candidates == []
    assert not any("_" in s and s.count("_") > 3 for s in r.seed_tokens)
    assert not any("ПРИВЕТ" == s for s in r.seed_tokens)


def test_gate_rejects_ungrounded_and_bad_pred() -> None:
    text = "Я живу в Москве"
    ok = gate_candidates(
        text,
        [FactCandidate("LIVE_IN", {"SUBJECT": "Я", "LOCATION": "МОСКВА"}, confidence=0.9)],
    )
    assert len(ok) == 1
    # open relations: неизвестный predicate ок, если роли grounded
    fly = gate_candidates(
        text,
        [FactCandidate("FLY", {"SUBJECT": "Я", "OBJECT": "МОСКВА"}, confidence=0.9)],
    )
    assert len(fly) == 1
    bad = gate_candidates(
        text,
        [
            FactCandidate("LIVE_IN", {"SUBJECT": "Я", "LOCATION": "ПАРИЖ"}, confidence=0.9),
            FactCandidate("IS", {"SUBJECT": "ЗАНИМАЮСЬ", "OBJECT": "ЛЕПКА"}, confidence=0.9),
        ],
    )
    assert bad == []


def test_gate_accepts_exact_acronym_surfaces() -> None:
    for text, obj in [
        ("я использую API", "API"),
        ("мы обсуждаем HTTP", "HTTP"),
    ]:
        ok = gate_candidates(
            text,
            [
                FactCandidate(
                    "RELATE",
                    {"SUBJECT": "Я" if text.startswith("я") else "МЫ", "OBJECT": obj},
                    confidence=0.95,
                    canonical_relation="RELATE",
                    raw_relation="упоминаю",
                )
            ],
        )
        assert len(ok) == 1
        assert ok[0].roles["OBJECT"] == obj


def test_json_llm_perception_gates_bad_subjects() -> None:
    def call_fn(prompt: str) -> str:
        data = json.loads(prompt)
        assert "лепк" in data["text"].lower() or "скульптур" in data["text"].lower()
        return json.dumps(
            {
                "kind": "fact",
                "candidates": [
                    {
                        "predicate": "USE",
                        "roles": {"SUBJECT": "Я", "OBJECT": "СКУЛЬПТУРНАЯ_ЛЕПКА"},
                        "confidence": 0.9,
                    },
                    {
                        "predicate": "IS",
                        "roles": {"SUBJECT": "ЕСЛИ", "OBJECT": "ПОМОЩЬ"},
                        "confidence": 0.9,
                    },
                ],
                "seed_tokens": ["Я", "СКУЛЬПТУРНАЯ_ЛЕПКА", "ЕСЛИ"],
            },
            ensure_ascii=False,
        )

    text = "Я занимаюсь скульптурной лепкой. Если понадобится — обращайся."
    r = JsonLLMPerception(call_fn).parse(text)
    assert any(c.predicate == "USE" and c.roles.get("SUBJECT") == "Я" for c in r.candidates)
    assert not any(c.roles.get("SUBJECT") in {"ЗАНИМАЮСЬ", "ЕСЛИ"} for c in r.candidates)
    assert "ЕСЛИ" not in r.seed_tokens


def test_normalize_llm_roles_enforces_closed_protocol() -> None:
    from ah_memory.perception import normalize_llm_roles, roles_from_llm_json

    mapped = normalize_llm_roles(
        {"SUBJECT": "entity a", "LOCATION": "entity b", "UNKNOWN": "entity c"}
    )
    assert mapped == {"SUBJECT": "entity a", "LOCATION": "entity b"}
    assert roles_from_llm_json({"UNKNOWN": "entity c"}) == {}

    invalid = {
        "candidates": [
            {
                "predicate": "RELATE",
                "roles": {"SUBJECT": "entity a", "UNKNOWN": "entity b"},
                "statement_type": "assertion",
            }
        ]
    }
    assert candidates_from_llm_json(invalid) == []
    assert any(
        "unsupported roles" in error for error in llm_payload_errors(invalid)
    )


def test_question_keeps_only_explicitly_typed_assertions() -> None:
    text = "Мы проектируем систему заказов. CRUD или event sourcing?"
    cands = [
        FactCandidate(
            "DESIGN",
            {"SUBJECT": "МЫ", "OBJECT": "СИСТЕМА_ЗАКАЗОВ"},
            confidence=0.9,
            canonical_relation="DESIGN",
            raw_relation="проектируем",
        )
    ]
    gated = gate_candidates(text, cands)
    assert len(gated) == 1

    def call_fn(prompt: str) -> str:
        return json.dumps(
            {
                "kind": "question",
                "candidates": [
                    {
                        "predicate": "DESIGN",
                        "roles": {"SUBJECT": "МЫ", "OBJECT": "СИСТЕМА_ЗАКАЗОВ"},
                        "confidence": 0.9,
                        "raw_relation": "проектируем",
                        "statement_type": "assertion",
                    }
                ],
                "seed_tokens": ["МЫ", "СИСТЕМА_ЗАКАЗОВ"],
            },
            ensure_ascii=False,
        )

    r = JsonLLMPerception(call_fn).parse(text)
    assert r.kind == "question"
    assert len(r.candidates) == 1
    assert r.candidates[0].statement_type == "assertion"

    untyped = {
        "kind": "question",
        "candidates": [
            {
                "predicate": "CHOOSING_BETWEEN",
                "roles": {"SUBJECT": "ПОДХОД", "OBJECT": "CRUD"},
            }
        ],
    }
    assert candidates_from_llm_json(untyped) == []

    typed_topic = candidates_from_llm_json(
        {
            "kind": "question",
            "candidates": [
                {
                    "predicate": "COMPARE",
                    "roles": {
                        "SUBJECT": "ДИАЛОГ",
                        "OBJECT": "CRUD",
                        "WITH": "EVENT_SOURCING",
                    },
                    "raw_relation": "CRUD или event sourcing",
                    "statement_type": "open_question",
                    "confidence": 0.9,
                }
            ],
        }
    )
    gated_topic = gate_candidates(text, typed_topic)
    assert len(gated_topic) == 1
    assert gated_topic[0].statement_type == "open_question"

    pronoun_topic = candidates_from_llm_json(
        {
            "kind": "question",
            "candidates": [
                {
                    "predicate": "ORGANIZE",
                    "roles": {"SUBJECT": "WE", "OBJECT": "READ_МОДЕЛЬ"},
                    "statement_type": "open_question",
                }
            ],
        }
    )
    assert pronoun_topic[0].roles["SUBJECT"] == "WE"

    subject_options = candidates_from_llm_json(
        {
            "kind": "question",
            "candidates": [
                {
                    "predicate": "CHOICE",
                    "roles": {
                        "SUBJECT": "САГА_ИЛИ_ДВУХФАЗНЫЙ_КОМИТ",
                        "OBJECT": "РАСПРЕДЕЛЕННАЯ_ТРАНЗАКЦИЯ",
                    },
                    "statement_type": "open_question",
                }
            ],
        }
    )
    assert subject_options == []

    one_role_question = candidates_from_llm_json(
        {
            "kind": "question",
            "candidates": [
                {
                    "predicate": "POLICY_RETENTION",
                    "roles": {"SUBJECT": "ПОЛИТИКА_RETENTION"},
                    "statement_type": "open_question",
                }
            ],
        }
    )
    assert one_role_question == []


def test_gate_accepts_implicit_first_person_subject() -> None:
    text = (
        "Для аудита заказов храним события 90 дней в Kafka, "
        "затем архив в S3 Parquet."
    )
    candidate = FactCandidate(
        "STORE_EVENTS",
        {
            "SUBJECT": "МЫ",
            "PURPOSE": "АУДИТ_ЗАКАЗ",
            "TIME": "90_ДНЕЙ",
            "TOOL": "KAFKA",
            "LOCATION": "S3_PARQUET",
        },
        confidence=1.0,
    )
    assert len(gate_candidates(text, [candidate])) == 1
    assert gate_candidates(
        "Сервис хранит события в Kafka.",
        [
            FactCandidate(
                "STORE_EVENTS",
                {"SUBJECT": "МЫ", "OBJECT": "СОБЫТИЕ", "TOOL": "KAFKA"},
                confidence=1.0,
            )
        ],
    ) == []

    accepted_decision = gate_candidates(
        "Ок, event sourcing.",
        [
            FactCandidate(
                "CHOOSE",
                {"SUBJECT": "МЫ", "OBJECT": "EVENT_SOURCING"},
                confidence=1.0,
                statement_type="decision",
            )
        ],
    )
    assert len(accepted_decision) == 1


def test_llm_compound_role_values_are_rejected() -> None:
    data = {
        "candidates": [
            {
                "predicate": "AFFECTS",
                "roles": {
                    "SUBJECT": "заказ",
                    "OBJECT": "оплата, склад и доставка",
                },
                "confidence": 1.0,
            }
        ]
    }
    assert candidates_from_llm_json(data) == []

    alternatives = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "COMPARE",
                    "roles": {
                        "SUBJECT": "подход",
                        "OBJECT": "CRUD PostgreSQL event sourcing",
                    },
                }
            ]
        }
    )
    assert alternatives == []

    sequential_clause = {
        "candidates": [
            {
                "predicate": "STORE_EVENTS",
                "roles": {
                    "SUBJECT": "мы",
                    "OBJECT": "события",
                    "LOCATION": "Kafka, затем архив в S3 Parquet",
                    "TIME": "90 дней",
                },
            }
        ]
    }
    assert candidates_from_llm_json(sequential_clause) == []

    proposal_with_list = candidates_from_llm_json(
        {
            "kind": "message",
            "candidates": [
                {
                    "predicate": "STORE_IN",
                    "roles": {
                        "SUBJECT": "состояние",
                        "OBJECT": "PostgreSQL, Redis",
                    },
                    "statement_type": "proposal",
                }
            ],
        }
    )
    assert proposal_with_list == []


def test_llm_normalizes_parenthetical_and_sequential_roles() -> None:
    decision = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "PLACE",
                    "roles": {
                        "SUBJECT": "write-модель",
                        "LOCATION": "PostgreSQL (outbox)",
                    },
                    "statement_type": "decision",
                }
            ]
        }
    )
    assert decision[0].roles["LOCATION"] == "POSTGRESQL"
    assert decision[0].roles["WITH"] == "OUTBOX"

    underscore_decision = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "PLACE",
                    "roles": {
                        "SUBJECT": "write-модель",
                        "LOCATION": "POSTGRESQL_OUTBOX",
                    },
                    "statement_type": "decision",
                }
            ]
        }
    )
    assert underscore_decision[0].roles["LOCATION"] == "POSTGRESQL_OUTBOX"
    assert "WITH" not in underscore_decision[0].roles

    retention = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "STORE_EVENTS",
                    "raw_relation": (
                        "храним события 90 дней в Kafka, "
                        "затем архив в S3 Parquet"
                    ),
                    "roles": {
                        "SUBJECT": "мы",
                        "OBJECT": "события",
                        "TIME": "90 дней",
                        "LOCATION": "Kafka",
                        "TOOL": "S3 Parquet",
                    },
                    "statement_type": "assertion",
                }
            ]
        }
    )
    assert len(retention) == 1
    assert retention[0].roles["LOCATION"] == "KAFKA"
    assert retention[0].roles["TOOL"] == "S3_PARQUET"

    separate_steps = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "STORE_EVENTS",
                    "raw_span": "Храним события 90 дней в Kafka.",
                    "roles": {
                        "SUBJECT": "мы",
                        "TIME": "90 дней",
                        "LOCATION": "Kafka",
                    },
                    "statement_type": "decision",
                },
                {
                    "predicate": "ARCHIVE_TO",
                    "raw_span": "Затем архивируем в S3 Parquet.",
                    "roles": {
                        "SUBJECT": "мы",
                        "LOCATION": "S3 Parquet",
                    },
                    "statement_type": "decision",
                },
            ]
        }
    )
    assert "OBJECT" not in separate_steps[0].roles
    assert separate_steps[1].roles["SUBJECT"] == "МЫ"

    unrelated_steps = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "STORE_EVENTS",
                    "raw_span": "Храним события в Kafka.",
                    "roles": {"SUBJECT": "мы", "LOCATION": "Kafka"},
                    "statement_type": "decision",
                },
                {
                    "predicate": "ARCHIVE_TO",
                    "raw_span": "Архивируем отчёты в S3.",
                    "roles": {"SUBJECT": "мы", "LOCATION": "S3"},
                    "statement_type": "decision",
                },
            ]
        }
    )
    assert unrelated_steps[1].roles["SUBJECT"] == "МЫ"

    ordinary_location = candidates_from_llm_json(
        {
            "candidates": [
                {
                    "predicate": "STORE_IN",
                    "roles": {"SUBJECT": "события", "LOCATION": "S3_PARQUET"},
                    "statement_type": "decision",
                }
            ]
        }
    )
    assert ordinary_location[0].roles["LOCATION"] == "S3_PARQUET"
    assert "WITH" not in ordinary_location[0].roles


def test_payload_validation_requires_explicit_complete_candidate() -> None:
    data = {
        "candidates": [
            {
                "predicate": "RELATE",
                "roles": {"OBJECT": "ENTITY_B"},
                "statement_type": "assertion",
            }
        ]
    }
    assert candidates_from_llm_json(data) == []
    errors = llm_payload_errors(data)
    assert any("requires SUBJECT" in error for error in errors)
    assert any("at least two roles" in error for error in errors)


def test_json_perception_never_retries_invalid_output() -> None:
    calls = 0

    def call_fn(prompt: str) -> str:
        nonlocal calls
        calls += 1
        assert "repair" not in json.loads(prompt)
        return json.dumps(
            {
                "kind": "fact",
                "candidates": [
                    {
                        "predicate": "RELATE",
                        "roles": {"OBJECT": "ENTITY_B"},
                        "statement_type": "assertion",
                    }
                ],
            }
        )

    result = JsonLLMPerception(call_fn).parse("Entity B")
    assert calls == 1
    assert result.candidates == []
    assert result.meta["validation_errors"]


def test_long_llm_relation_label_is_compacted_by_epistemic_type() -> None:
    candidates = candidates_from_llm_json(
        {
            "kind": "message",
            "candidates": [
                {
                    "canonical_relation": (
                        "VERY_LONG_RELATION_LABEL_FROM_MODEL_OUTPUT"
                    ),
                    "roles": {"SUBJECT": "ENTITY_A", "OBJECT": "ENTITY_B"},
                    "statement_type": "explanation",
                }
            ],
        }
    )
    assert candidates[0].predicate == "EXPLAINS"
    assert candidates[0].canonical_relation == "EXPLAINS"
