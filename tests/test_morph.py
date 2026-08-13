from __future__ import annotations

import json

from ah_memory.morph import lemma, sanitize_roles, slug_uid, uid_too_wide
from ah_memory.perception import (
    FactCandidate,
    JsonLLMPerception,
    SeedPerception,
    gate_candidates,
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


def test_json_llm_mock_hobby() -> None:
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
