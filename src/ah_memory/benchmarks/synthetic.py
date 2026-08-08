"""Deterministic synthetic factor graphs for ignition research."""
from __future__ import annotations

from dataclasses import dataclass

from ah_memory.factor_graph import Factor, FactorGraph, FactorKind


@dataclass(frozen=True)
class Scenario:
    name: str
    graph: FactorGraph
    evidence: dict[str, float]
    relevant: frozenset[str]
    irrelevant: frozenset[str] = frozenset()


def chain() -> Scenario:
    variables = ["A", "B", "C", "D"]
    factors = [
        Factor(
            f"F{i}",
            FactorKind.PAIR,
            [variables[i], variables[i + 1]],
            w=1.0,
            potential_key="follow",
        )
        for i in range(3)
    ]
    return Scenario(
        "chain",
        FactorGraph(variables, factors),
        {"A": 3.0},
        frozenset({"B", "C", "D"}),
    )


def branching() -> Scenario:
    return Scenario(
        "branching",
        FactorGraph(
            ["A", "B", "C"],
            [
                Factor("F_AB", FactorKind.PAIR, ["A", "B"], w=1.0, potential_key="assoc"),
                Factor("F_AC", FactorKind.PAIR, ["A", "C"], w=1.0, potential_key="assoc"),
            ],
        ),
        {"A": 3.0},
        frozenset({"B", "C"}),
    )


def is_a() -> Scenario:
    return Scenario(
        "is_a",
        FactorGraph(
            ["DOG", "ANIMAL", "LIVING_THING"],
            [
                Factor(
                    "ISA_DOG_ANIMAL",
                    FactorKind.PAIR,
                    ["DOG", "ANIMAL"],
                    w=1.0,
                    potential_key="is_a",
                ),
                Factor(
                    "ISA_ANIMAL_LIVING",
                    FactorKind.PAIR,
                    ["ANIMAL", "LIVING_THING"],
                    w=1.0,
                    potential_key="is_a",
                ),
            ],
        ),
        {"DOG": 3.0},
        frozenset({"ANIMAL", "LIVING_THING"}),
    )


def competing_concepts() -> Scenario:
    variables = ["ANIMAL", "DOG", "CAT", "HORSE", "BARK"]
    factors = [
        Factor(
            f"ISA_{child}",
            FactorKind.PAIR,
            [child, "ANIMAL"],
            w=1.0,
            potential_key="is_a",
        )
        for child in ("DOG", "CAT", "HORSE")
    ]
    factors.append(
        Factor(
            "ASSOC_BARK_DOG",
            FactorKind.PAIR,
            ["BARK", "DOG"],
            w=1.0,
            potential_key="assoc",
        )
    )
    return Scenario(
        "competing_concepts",
        FactorGraph(variables, factors),
        {"ANIMAL": 2.0, "BARK": 3.0},
        frozenset({"DOG"}),
        frozenset({"CAT", "HORSE"}),
    )


def episodic() -> Scenario:
    variables = ["E1", "E2", "E3", "E4"]
    return Scenario(
        "episodic",
        FactorGraph(
            variables,
            [
                Factor(
                    f"FOLLOW_{i + 1}",
                    FactorKind.PAIR,
                    [variables[i], variables[i + 1]],
                    w=1.0,
                    potential_key="follow",
                )
                for i in range(3)
            ],
        ),
        {"E1": 3.0},
        frozenset({"E2", "E3", "E4"}),
    )


def hypernode(arity: int = 7) -> Scenario:
    variables = [f"ARG_{i}" for i in range(arity)]
    factor = Factor(
        "N_HYPER",
        FactorKind.HYPER,
        variables,
        w=1.0,
        potential_key="hypernode",
        roles={f"ROLE_{i}": uid for i, uid in enumerate(variables)},
    )
    evidence = {uid: 2.5 for uid in variables[:-1]}
    return Scenario(
        "hypernode",
        FactorGraph(variables, [factor]),
        evidence,
        frozenset({variables[-1]}),
    )


def scale_chain(size: int, arity: int = 2) -> Scenario:
    if size < 2:
        raise ValueError("size must be >= 2")
    variables = [f"V{i}" for i in range(size)]
    factors: list[Factor] = []
    if arity <= 2:
        factors = [
            Factor(
                f"F{i}",
                FactorKind.PAIR,
                [variables[i], variables[i + 1]],
                potential_key="assoc",
            )
            for i in range(size - 1)
        ]
    else:
        step = max(1, arity - 1)
        for start in range(0, size - 1, step):
            members = variables[start : min(size, start + arity)]
            if len(members) >= 2:
                factors.append(
                    Factor(
                        f"N{start}",
                        FactorKind.HYPER,
                        members,
                        potential_key="hypernode",
                    )
                )
    return Scenario(
        f"scale_{size}_arity_{arity}",
        FactorGraph(variables, factors),
        {"V0": 3.0},
        frozenset({variables[-1]}),
    )


def all_scenarios() -> tuple[Scenario, ...]:
    return (
        chain(),
        branching(),
        is_a(),
        competing_concepts(),
        episodic(),
        hypernode(),
    )
