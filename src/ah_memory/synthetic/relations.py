"""Extensible factor type schemas for synthetic graphs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class FactorSchema:
    type: str
    roles: tuple[str, ...]
    role_types: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    temporal: bool = False
    state_changing: bool = False
    ah_role_map: Mapping[str, str] = field(default_factory=dict)


# Named synthetic roles -> AH Transform roles
_DEFAULT_AH_MAP: Mapping[str, str] = {
    "subject": "SUBJECT",
    "buyer": "SUBJECT",
    "person": "SUBJECT",
    "child": "SUBJECT",
    "cause": "SUBJECT",
    "previous_event": "SUBJECT",
    "object": "OBJECT",
    "company": "OBJECT",
    "effect": "OBJECT",
    "next_event": "OBJECT",
    "parent": "OBJECT",
    "from": "LOCATION",
    "to": "LOCATION",
    "location": "LOCATION",
    "position": "TIME",
    "time": "TIME",
}


FACTOR_SCHEMAS: dict[str, FactorSchema] = {
    "KNOW": FactorSchema(
        type="KNOW",
        roles=("person", "object"),
        role_types={"person": ("Person",), "object": ("Person", "Company", "Object")},
        ah_role_map={"person": "SUBJECT", "object": "OBJECT"},
    ),
    "WORKS_FOR": FactorSchema(
        type="WORKS_FOR",
        roles=("person", "company", "position", "time"),
        role_types={
            "person": ("Person",),
            "company": ("Company",),
            "position": ("Document",),
            "time": ("Event",),
        },
        temporal=True,
        state_changing=True,
        ah_role_map={
            "person": "SUBJECT",
            "company": "OBJECT",
            "position": "TIME",
            "time": "CAUSE",
        },
    ),
    "LIVES_IN": FactorSchema(
        type="LIVES_IN",
        roles=("person", "location"),
        role_types={"person": ("Person",), "location": ("Place",)},
        state_changing=True,
        ah_role_map={"person": "SUBJECT", "location": "OBJECT"},
    ),
    "LOCATED_IN": FactorSchema(
        type="LOCATED_IN",
        roles=("subject", "location"),
        role_types={
            "subject": ("Company", "Place", "Object"),
            "location": ("Place",),
        },
        ah_role_map={"subject": "SUBJECT", "location": "OBJECT"},
    ),
    "OWNS": FactorSchema(
        type="OWNS",
        roles=("person", "object"),
        role_types={"person": ("Person",), "object": ("Object",)},
        state_changing=True,
        ah_role_map={"person": "SUBJECT", "object": "OBJECT"},
    ),
    "PURCHASE": FactorSchema(
        type="PURCHASE",
        roles=("buyer", "object", "time", "location"),
        role_types={
            "buyer": ("Person",),
            "object": ("Object",),
            "time": ("Event",),
            "location": ("Place",),
        },
        temporal=True,
        state_changing=True,
        ah_role_map={
            "buyer": "SUBJECT",
            "object": "OBJECT",
            "time": "TIME",
            "location": "LOCATION",
        },
    ),
    "SELL": FactorSchema(
        type="SELL",
        roles=("buyer", "object", "time", "location"),
        role_types={
            "buyer": ("Person",),
            "object": ("Object",),
            "time": ("Event",),
            "location": ("Place",),
        },
        temporal=True,
        state_changing=True,
        ah_role_map={
            "buyer": "SUBJECT",
            "object": "OBJECT",
            "time": "TIME",
            "location": "LOCATION",
        },
    ),
    "MOVE": FactorSchema(
        type="MOVE",
        roles=("person", "from", "to", "time"),
        role_types={
            "person": ("Person",),
            "from": ("Place",),
            "to": ("Place",),
            "time": ("Event",),
        },
        temporal=True,
        state_changing=True,
        ah_role_map={
            "person": "SUBJECT",
            "from": "LOCATION",
            "to": "OBJECT",
            "time": "TIME",
        },
    ),
    "VISITS": FactorSchema(
        type="VISITS",
        roles=("person", "location", "time"),
        role_types={
            "person": ("Person",),
            "location": ("Place", "Company"),
            "time": ("Event",),
        },
        temporal=True,
        ah_role_map={
            "person": "SUBJECT",
            "location": "OBJECT",
            "time": "TIME",
        },
    ),
    "USES": FactorSchema(
        type="USES",
        roles=("person", "object"),
        role_types={"person": ("Person",), "object": ("Object",)},
        ah_role_map={"person": "SUBJECT", "object": "OBJECT"},
    ),
    "CREATED": FactorSchema(
        type="CREATED",
        roles=("person", "object", "time"),
        role_types={
            "person": ("Person", "Company"),
            "object": ("Object", "Document"),
            "time": ("Event",),
        },
        temporal=True,
        ah_role_map={
            "person": "SUBJECT",
            "object": "OBJECT",
            "time": "TIME",
        },
    ),
    "CAUSE": FactorSchema(
        type="CAUSE",
        roles=("cause", "effect"),
        role_types={
            "cause": ("Event", "Document"),
            "effect": ("Event",),
        },
        ah_role_map={"cause": "SUBJECT", "effect": "OBJECT"},
    ),
    "FOLLOW": FactorSchema(
        type="FOLLOW",
        roles=("previous_event", "next_event"),
        role_types={
            "previous_event": ("Event",),
            "next_event": ("Event",),
        },
        temporal=True,
        ah_role_map={
            "previous_event": "SUBJECT",
            "next_event": "OBJECT",
        },
    ),
    "PART_OF": FactorSchema(
        type="PART_OF",
        roles=("subject", "object"),
        role_types={"subject": ("Place", "Object"), "object": ("Place",)},
        ah_role_map={"subject": "SUBJECT", "object": "OBJECT"},
    ),
    "IS_A": FactorSchema(
        type="IS_A",
        roles=("child", "parent"),
        role_types={
            "child": ("Object", "Person", "Place"),
            "parent": ("Object", "Person", "Place"),
        },
        ah_role_map={"child": "SUBJECT", "parent": "OBJECT"},
    ),
}


def get_schema(factor_type: str) -> FactorSchema:
    if factor_type not in FACTOR_SCHEMAS:
        raise KeyError(f"unknown factor type: {factor_type}")
    return FACTOR_SCHEMAS[factor_type]


def map_roles_to_ah(
    factor_type: str,
    arguments: Mapping[str, str],
) -> dict[str, str]:
    schema = get_schema(factor_type)
    mapped: dict[str, str] = {}
    for role, value in arguments.items():
        ah_role = schema.ah_role_map.get(role) or _DEFAULT_AH_MAP.get(role)
        if ah_role is None:
            ah_role = role.upper()
        # Keep first assignment for each AH role to satisfy Transform arity.
        if ah_role not in mapped:
            mapped[ah_role] = value
    if "SUBJECT" not in mapped and arguments:
        first = next(iter(arguments.values()))
        mapped["SUBJECT"] = first
    if "OBJECT" not in mapped and len(arguments) >= 2:
        values = list(arguments.values())
        if values[1] != mapped.get("SUBJECT"):
            mapped["OBJECT"] = values[1]
        elif len(values) > 2:
            mapped["OBJECT"] = values[2]
    return mapped


def register_schema(schema: FactorSchema, *, replace: bool = False) -> None:
    if schema.type in FACTOR_SCHEMAS and not replace:
        raise ValueError(f"schema already registered: {schema.type}")
    FACTOR_SCHEMAS[schema.type] = schema
