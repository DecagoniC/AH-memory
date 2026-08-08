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
        roles={"SUBJECT": "FATHER", "OBJECT": object_uid},
    )


def test_bmw_audi_opel_state_regression() -> None:
    store = AHStore()
    transform = Transform(store)
    transform.apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                _fact("PURCHASE", "BMW"),
                _fact("SELL", "BMW"),
                _fact("PURCHASE", "AUDI"),
                _fact("SELL", "AUDI"),
                _fact("PURCHASE", "OPEL"),
            ],
        )
    )

    assert store.state.owns("FATHER", "BMW") is False
    assert store.state.owns("FATHER", "AUDI") is False
    assert store.state.owns("FATHER", "OPEL") is True
    assert store.state.last_purchase("FATHER") == "OPEL"
    assert store.state.purchase_history("FATHER") == ["BMW", "AUDI", "OPEL"]
    assert len(store.state_transitions) == 11


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
