"""Immutable factor-graph view over structural AH memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from ah_memory.factor_parameters import FactorParameters
from ah_memory.relations import Relation
from ah_memory.store import AHStore
from ah_memory.types import (
    ElementList,
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


@dataclass(frozen=True)
class Factor:
    fid: str
    kind: FactorKind
    variables: tuple[str, ...] | list[str]
    w: float = 0.5
    link_id: str = ""
    roles: dict[str, str] = field(default_factory=dict)  # role -> uid
    lambda_obs: float = 0.0
    epsilon: float = 0.05
    potential_key: str = ""
    source_uid: str = ""
    relation: Relation | None = None
    parameters: FactorParameters | None = None
    activation: float = 0.0
    confidence: float = 1.0
    embedding: tuple[float, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.embedding is not None:
            object.__setattr__(self, "embedding", tuple(self.embedding))

    @property
    def uid(self) -> str:
        return self.fid

    @property
    def weight(self) -> float:
        return self.w

    def transmission(self, source_uid: str, target_uid: str) -> float:
        """Relation-agnostic asymmetric transmission coefficient."""
        if source_uid == target_uid:
            return 0.0
        if source_uid not in self.variables or target_uid not in self.variables:
            return 0.0
        parameters = self.parameters or FactorParameters()
        relation = self.relation
        if relation is None or relation.properties.symmetric:
            directional = 1.0
        elif not relation.properties.directional:
            directional = 1.0
        else:
            explicit_source = str(self.metadata.get("source_variable") or "")
            role_source = self.roles.get("SUBJECT", "")
            first = explicit_source or role_source or self.variables[0]
            forward = source_uid == first
            directional = (
                0.5 + 0.5 * parameters.directionality
                if forward
                else 0.5 * (1.0 - parameters.directionality)
            )
        semantic_bias = 1.0
        if relation is not None and relation.properties.temporal:
            semantic_bias += parameters.temporal_bias
        if relation is not None and relation.properties.causal:
            semantic_bias += parameters.causal_bias
        return min(
            1.0,
            max(
                0.0,
                parameters.transmission_strength * directional * semantic_bias,
            ),
        )


@dataclass(frozen=True)
class FactorGraph:
    variables: tuple[str, ...] | list[str]
    factors: tuple[Factor, ...] | list[Factor]
    var_factors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    factor_vars: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    factors_by_id: Mapping[str, Factor] = field(default_factory=dict)
    structural_signature: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        variables = tuple(dict.fromkeys(self.variables))
        factors = tuple(self.factors)
        var_factors: dict[str, list[str]] = {v: [] for v in variables}
        factor_vars: dict[str, tuple[str, ...]] = {}
        factors_by_id: dict[str, Factor] = {}
        for f in factors:
            factor_vars[f.fid] = tuple(f.variables)
            factors_by_id[f.fid] = f
            for v in f.variables:
                if v in var_factors:
                    var_factors[v].append(f.fid)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(
            self,
            "var_factors",
            MappingProxyType({k: tuple(v) for k, v in var_factors.items()}),
        )
        object.__setattr__(self, "factor_vars", MappingProxyType(factor_vars))
        object.__setattr__(self, "factors_by_id", MappingProxyType(factors_by_id))
        if not self.structural_signature:
            signature = (
                variables,
                tuple(
                    (
                        f.fid,
                        f.kind.value,
                        f.variables,
                        f.link_id,
                        f.potential_key,
                        round(f.w, 12),
                        f.relation.canonical_label if f.relation else "",
                        tuple(sorted(f.parameters.to_dict().items()))
                        if f.parameters
                        else (),
                    )
                    for f in factors
                    if f.kind not in {FactorKind.OBS, FactorKind.PRIOR}
                ),
            )
            object.__setattr__(self, "structural_signature", signature)


def is_variable(store: AHStore, uid: str) -> bool:
    if uid in store.ah.S:
        return True
    try:
        e = store._find_anywhere(uid)  # noqa: SLF001
    except Exception:
        return False
    return isinstance(e, (SecondOrderSymbol, ElementList))


def collect_variables(store: AHStore) -> list[str]:
    vars_: list[str] = list(store.ah.S.keys())
    for e in store.ah.all_hyper().values():
        if isinstance(e, (SecondOrderSymbol, ElementList)):
            vars_.append(e.uid)
    return list(dict.fromkeys(vars_))


def _link_potential_key(store: AHStore, link_id: str, e1: str, e2: str, uid: str) -> str:
    normalized = link_id.upper().replace("-", "_")
    if normalized == "IS_A":
        return "is_a"
    if normalized == "FOLLOW":
        return "follow"
    if normalized == "BIND":
        return "bind"
    if normalized == "CAUSE":
        return "cause"
    if normalized == "ASSOC":
        one_s = (e1 in store.ah.S) ^ (e2 in store.ah.S)
        if one_s or uid.upper().startswith("L_BIND"):
            return "bind"
        return "assoc"
    return "pair"


def build_factor_graph(
    store: AHStore,
    *,
    evidence: dict[str, float] | None = None,
    epsilon: float = 0.05,
    include_prior: bool = True,
) -> FactorGraph:
    """Build FG: variables = S ∪ m ∪ episode lists; factors = N, L, obs, prior."""
    variables = collect_variables(store)
    var_set = set(variables)
    factors: list[Factor] = []
    evidence = evidence or {}
    semantic_by_source = {
        str(factor.metadata.get("legacy_source_uid")): factor
        for factor in store.list_semantic_factors()
        if factor.metadata.get("legacy_source_uid")
    }

    if include_prior:
        for v in variables:
            factors.append(
                Factor(
                    fid=f"PRIOR::{v}",
                    kind=FactorKind.PRIOR,
                    variables=[v],
                    epsilon=epsilon,
                    potential_key="prior",
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
                potential_key="obs",
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
                potential_key=_link_potential_key(store, link.id, e1, e2, link.uid),
                source_uid=link.uid,
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
        semantic = semantic_by_source.get(n.uid)
        factors.append(
            Factor(
                fid=f"N::{n.uid}",
                kind=FactorKind.HYPER,
                variables=uniq,
                w=n.w,
                roles={r: u for r, u in roles.items() if u in seen},
                potential_key=semantic.potential_key if semantic else "hypernode",
                source_uid=n.uid,
                relation=semantic.relation if semantic else None,
                parameters=semantic.parameters if semantic else None,
                confidence=semantic.confidence if semantic else 1.0,
                embedding=semantic.embedding if semantic else None,
                metadata=dict(semantic.metadata) if semantic else {},
            )
        )

    for semantic in store.list_semantic_factors():
        legacy_source = str(semantic.metadata.get("legacy_source_uid") or "")
        if legacy_source and legacy_source in semantic_by_source:
            continue
        members = [uid for uid in semantic.variables if uid in var_set]
        if len(members) < 2:
            continue
        factors.append(
            Factor(
                fid=semantic.fid,
                kind=semantic.kind,
                variables=members,
                w=semantic.w,
                roles=dict(semantic.roles),
                potential_key=semantic.potential_key,
                source_uid=semantic.source_uid,
                relation=semantic.relation,
                parameters=semantic.parameters,
                activation=semantic.activation,
                confidence=semantic.confidence,
                embedding=semantic.embedding,
                metadata=dict(semantic.metadata),
            )
        )

    return FactorGraph(variables=variables, factors=factors)


def build_structural_factor_graph(store: AHStore) -> FactorGraph:
    """Build reusable topology; evidence is supplied to BPState, not embedded."""
    return build_factor_graph(store, evidence=None, include_prior=True)


def role_beta_weight(role: str) -> float:
    if role in {Role.SUBJECT.value, Role.OBJECT.value}:
        return 0.35
    if role in {Role.LOCATION.value, Role.CAUSE.value, Role.TOOL.value}:
        return 0.2
    return 0.1
