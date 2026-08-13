"""Factor / event / world-state models for synthetic graphs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


@dataclass(frozen=True)
class SyntheticFactor:
    uid: str
    type: str
    arguments: Mapping[str, str]
    timestamp: int
    properties: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "type": self.type,
            "arguments": dict(self.arguments),
            "timestamp": self.timestamp,
            "properties": dict(self.properties),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class SyntheticEvent:
    uid: str
    factor_uid: str
    timestamp: int
    predicate: str
    arguments: Mapping[str, str]
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "factor_uid": self.factor_uid,
            "timestamp": self.timestamp,
            "predicate": self.predicate,
            "arguments": dict(self.arguments),
            "label": self.label,
        }


@dataclass
class PersonState:
    owns: list[str] = field(default_factory=list)
    works_for: str | None = None
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "owns": list(self.owns),
            "works_for": self.works_for,
            "location": self.location,
        }


@dataclass
class WorldState:
    persons: MutableMapping[str, PersonState] = field(default_factory=dict)

    def ensure(self, person_uid: str) -> PersonState:
        if person_uid not in self.persons:
            self.persons[person_uid] = PersonState()
        return self.persons[person_uid]

    def apply_factor(self, factor: SyntheticFactor) -> None:
        args = factor.arguments
        if factor.type == "PURCHASE":
            buyer = args.get("buyer") or args.get("person")
            obj = args.get("object")
            if buyer and obj:
                state = self.ensure(buyer)
                if obj not in state.owns:
                    state.owns.append(obj)
        elif factor.type == "SELL":
            buyer = args.get("buyer") or args.get("person")
            obj = args.get("object")
            if buyer and obj:
                state = self.ensure(buyer)
                if obj in state.owns:
                    state.owns.remove(obj)
        elif factor.type == "WORKS_FOR":
            person = args.get("person")
            company = args.get("company")
            if person and company:
                self.ensure(person).works_for = company
        elif factor.type == "LIVES_IN":
            person = args.get("person")
            location = args.get("location")
            if person and location:
                self.ensure(person).location = location
        elif factor.type == "MOVE":
            person = args.get("person")
            destination = args.get("to")
            if person and destination:
                self.ensure(person).location = destination
        elif factor.type == "OWNS":
            person = args.get("person")
            obj = args.get("object")
            if person and obj:
                state = self.ensure(person)
                if obj not in state.owns:
                    state.owns.append(obj)

    def to_dict(self) -> dict[str, Any]:
        return {uid: state.to_dict() for uid, state in self.persons.items()}
