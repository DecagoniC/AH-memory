"""Configurable deterministic state transitions derived from events."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ah_memory.relations import Event, canonicalize_label


@dataclass(frozen=True)
class State:
    values: Mapping[str, Any] = field(default_factory=dict)
    history: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    applied_events: tuple[str, ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "history": {
                key: list(values) for key, values in self.history.items()
            },
            "applied_events": list(self.applied_events),
        }


@dataclass(frozen=True)
class StateOperation:
    operation: str
    key_template: str
    value_template: str | bool | float | int | None


@dataclass(frozen=True)
class TransitionRule:
    relation: str
    operations: tuple[StateOperation, ...]


@dataclass(frozen=True)
class StateTransition:
    event_uid: str
    relation: str
    operation: str
    key: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_uid": self.event_uid,
            "relation": self.relation,
            "operation": self.operation,
            "key": self.key,
            "before": self.before,
            "after": self.after,
        }


class StateEngine:
    def __init__(self, rules: tuple[TransitionRule, ...] = ()) -> None:
        self._rules: dict[str, tuple[StateOperation, ...]] = {}
        self.last_transitions: list[StateTransition] = []
        for rule in rules:
            self.register_rule(rule)

    def register_rule(self, rule: TransitionRule) -> None:
        self._rules[canonicalize_label(rule.relation)] = tuple(rule.operations)

    def apply(self, state: State, event: Event) -> State:
        values = dict(state.values)
        history = {key: tuple(items) for key, items in state.history.items()}
        transitions: list[StateTransition] = []
        arguments = {
            role.upper(): self._state_uid(reference.uid)
            for role, reference in event.arguments.items()
        }
        relation = canonicalize_label(event.predicate.canonical_label)
        for operation in self._rules.get(relation, ()):
            key = self._render(operation.key_template, arguments)
            value = self._value(operation.value_template, arguments)
            if operation.operation == "set":
                before = values.get(key)
                values[key] = value
                after = value
            elif operation.operation == "delete":
                before = values.get(key)
                values.pop(key, None)
                after = None
            elif operation.operation == "append":
                before = list(history.get(key, ()))
                history[key] = (*history.get(key, ()), value)
                after = list(history[key])
            else:
                raise ValueError(f"unknown state operation: {operation.operation}")
            transitions.append(
                StateTransition(
                    event_uid=event.uid,
                    relation=relation,
                    operation=operation.operation,
                    key=key,
                    before=before,
                    after=after,
                )
            )
        self.last_transitions = transitions
        return replace(
            state,
            values=values,
            history=history,
            applied_events=(*state.applied_events, event.uid),
        )

    @staticmethod
    def _render(template: str, arguments: Mapping[str, str]) -> str:
        rendered = template
        for role, uid in arguments.items():
            rendered = rendered.replace("{" + role + "}", uid)
        if "{" in rendered:
            raise ValueError(f"missing event argument for template: {template}")
        return rendered

    @classmethod
    def _value(
        cls,
        template: str | bool | float | int | None,
        arguments: Mapping[str, str],
    ) -> Any:
        if not isinstance(template, str):
            return template
        return cls._render(template, arguments)

    @staticmethod
    def _state_uid(uid: str) -> str:
        return uid[2:] if uid.startswith("M_") else uid


def default_state_engine() -> StateEngine:
    """Return an empty engine; applications register transition rules explicitly."""
    return StateEngine()
