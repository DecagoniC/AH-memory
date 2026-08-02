"""Factor-graph view over AH (docs/FACTOR_GRAPH_ACTIVATION.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ah_memory.store import AHStore
from ah_memory.types import (
    Hyperlink,
    LinkId,
    Role,
    SecondOrderSymbol,
)


class FactorKind(str, Enum):
    OBS = "obs"
    PRIOR = "prior"
    PAIR = "pair"
    HYPER = "hyper"


@dataclass
class Factor:
    fid: str
    kind: FactorKind
    variables: list[str]
    # parameters
    w: float = 0.5
    link_id: str = ""
    roles: dict[str, str] = field(default_factory=dict)  # role -> uid
    lambda_obs: float = 0.0
    epsilon: float = 0.05


@dataclass
class FactorGraph:
    variables: list[str]
    factors: list[Factor]
    # adjacency
    var_factors: dict[str, list[str]] = field(default_factory=dict)
    factor_vars: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.var_factors = {v: [] for v in self.variables}
        self.factor_vars = {}
        for f in self.factors:
            self.factor_vars[f.fid] = list(f.variables)
            for v in f.variables:
                if v in self.var_factors:
                    self.var_factors[v].append(f.fid)


def is_variable(store: AHStore, uid: str) -> bool:
    if uid in store.ah.S:
        return True
    try:
        e = store._find_anywhere(uid)  # noqa: SLF001
    except Exception:
        return False
    return isinstance(e, SecondOrderSymbol)


def collect_variables(store: AHStore) -> list[str]:
    vars_: list[str] = list(store.ah.S.keys())
    for e in store.ah.all_hyper().values():
        if isinstance(e, SecondOrderSymbol):
            vars_.append(e.uid)
    return vars_


def build_factor_graph(
    store: AHStore,
    *,
    evidence: dict[str, float] | None = None,
    epsilon: float = 0.05,
    include_prior: bool = True,
) -> FactorGraph:
    """Build FG: variables = S ∪ m; factors = N, L, obs, prior."""
    variables = collect_variables(store)
    var_set = set(variables)
    factors: list[Factor] = []
    evidence = evidence or {}

    if include_prior:
        for v in variables:
            factors.append(
                Factor(
                    fid=f"PRIOR::{v}",
                    kind=FactorKind.PRIOR,
                    variables=[v],
                    epsilon=epsilon,
                )
            )

    for v, lam in evidence.items():
        if v not in var_set or lam <= 0:
            continue
        factors.append(
            Factor(
                fid=f"OBS::{v}",
                kind=FactorKind.OBS,
                variables=[v],
                lambda_obs=lam,
            )
        )

    for link in store.ah.L.values():
        e1, e2 = link.e1.target_uid, link.e2.target_uid
        members = [u for u in (e1, e2) if u in var_set]
        if len(members) < 2:
            # keep order if both vars; if only one, skip pairwise
            continue
        # preserve directed order for IS-A: e1=child, e2=parent
        ordered = [e1, e2] if e1 in var_set and e2 in var_set else members
        factors.append(
            Factor(
                fid=f"L::{link.uid}",
                kind=FactorKind.PAIR,
                variables=ordered,
                w=link.w,
                link_id=link.id,
            )
        )

    for n in store.find_hypernodes():
        roles = {r.value: f.target_uid for r, f in n.fillers.items()}
        members = [uid for uid in roles.values() if uid in var_set]
        # unique preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for u in members:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        if not uniq:
            continue
        factors.append(
            Factor(
                fid=f"N::{n.uid}",
                kind=FactorKind.HYPER,
                variables=uniq,
                w=n.w,
                roles={r: u for r, u in roles.items() if u in seen},
            )
        )

    return FactorGraph(variables=variables, factors=factors)


def role_beta_weight(role: str) -> float:
    if role in {Role.SUBJECT.value, Role.OBJECT.value}:
        return 0.35
    if role in {Role.LOCATION.value, Role.CAUSE.value, Role.TOOL.value}:
        return 0.2
    return 0.1
