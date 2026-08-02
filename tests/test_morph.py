from __future__ import annotations

from ah_memory.morph import lemma, sanitize_roles, slug_uid
from ah_memory.perception import RulePerception


def test_lemma_cases_merge() -> None:
    assert lemma("москве") == "москва"
    assert lemma("брата") == "брат"
    assert lemma("преподавателем") == "преподаватель"
    assert slug_uid("скульптурной лепкой") == "СКУЛЬПТУРНЫЙ_ЛЕПКА" or "ЛЕПКА" in slug_uid(
        "скульптурной лепкой"
    )


def test_reject_verb_and_conj_subjects() -> None:
    assert sanitize_roles({"SUBJECT": "ЗАНИМАЮСЬ", "OBJECT": "ЛЕПКА"}) is None
    assert sanitize_roles({"SUBJECT": "ЕСЛИ", "OBJECT": "ПОМОЩЬ"}) is None
    ok = sanitize_roles({"SUBJECT": "Я", "OBJECT": "СКУЛЬПТУРНАЯ_ЛЕПКА"})
    assert ok is not None
    assert ok["SUBJECT"] == "Я"


def test_rule_perception_hobby_and_if() -> None:
    p = RulePerception()
    r = p.parse("Я занимаюсь скульптурной лепкой. Если понадобится — обращайся.")
    preds = {(c.predicate, c.roles.get("SUBJECT")) for c in r.candidates}
    assert ("USE", "Я") in preds or any(c.roles.get("SUBJECT") == "Я" for c in r.candidates)
    assert not any(c.roles.get("SUBJECT") in {"ЗАНИМАЮСЬ", "ЕСЛИ", "ЗАНИМАТЬСЯ"} for c in r.candidates)
    assert "ЕСЛИ" not in r.seed_tokens
    assert "ЗАНИМАЮСЬ" not in r.seed_tokens
