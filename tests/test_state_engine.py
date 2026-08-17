from __future__ import annotations

from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.state_engine import (
    State,
    StateEngine,
    StateOperation,
    TransitionRule,
)
from ah_memory.store import AHStore
from ah_memory.transform import Transform


def _fact(relation: str, object_uid: str) -> FactCandidate:
    return FactCandidate(
        predicate=relation,
        raw_relation=relation.lower(),
        canonical_relation=relation,
        roles={"SUBJECT": "ACTOR", "OBJECT": object_uid},
    )


def test_configured_state_rules_apply_without_domain_defaults() -> None:
    store = AHStore()
    engine = StateEngine(
        (
            TransitionRule(
                "ADD",
                (
                    StateOperation("set", "ACTIVE:{SUBJECT}:{OBJECT}", True),
                    StateOperation("set", "LAST:{SUBJECT}", "{OBJECT}"),
                    StateOperation("append", "HISTORY:{SUBJECT}", "{OBJECT}"),
                ),
            ),
            TransitionRule(
                "REMOVE",
                (StateOperation("set", "ACTIVE:{SUBJECT}:{OBJECT}", False),),
            ),
        )
    )
    transform = Transform(store, state_engine=engine)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                _fact("ADD", "ALPHA"),
                _fact("REMOVE", "ALPHA"),
                _fact("ADD", "BETA"),
            ],
        )
    )

    assert store.state.get("ACTIVE:ACTOR:ALPHA") is False
    assert store.state.get("ACTIVE:ACTOR:BETA") is True
    assert store.state.get("LAST:ACTOR") == "BETA"
    assert list(store.state.history["HISTORY:ACTOR"]) == ["ALPHA", "BETA"]
    assert len(store.state_transitions) == 7


def test_state_transition_rules_are_runtime_configurable() -> None:
    store = AHStore()
    engine = StateEngine(
        (
            TransitionRule(
                "CUSTOM_RELATION",
                (StateOperation("set", "CUSTOM:{SUBJECT}", "{OBJECT}"),),
            ),
        )
    )
    transform = Transform(store, state_engine=engine)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="CUSTOM_RELATION",
                    raw_relation="связал",
                    canonical_relation="CUSTOM_RELATION",
                    roles={"SUBJECT": "A", "OBJECT": "B"},
                )
            ],
        )
    )
    assert store.state.get("CUSTOM:A") == "B"


def test_unknown_state_relation_has_no_implicit_transition() -> None:
    store = AHStore()
    transform = Transform(store, state_engine=StateEngine())
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate(
                    predicate="MET_AT",
                    raw_relation="познакомился с",
                    roles={"SUBJECT": "A", "OBJECT": "B"},
                )
            ],
        )
    )
    assert store.state.values == {}
    assert isinstance(store.state, State)
