"""Structural invariants for Junior acceptance."""
from __future__ import annotations

from collections import defaultdict

from ah_memory.store import AHStore, iter_follow_edges, iter_is_a_edges
from ah_memory.types import MRef, Property, SRef


class InvariantError(Exception):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def _has_cycle(edges: list[tuple[str, str]]) -> bool:
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for a, b in edges:
        adj[a].append(b)
        nodes.add(a)
        nodes.add(b)
    state: dict[str, int] = {n: 0 for n in nodes}

    def dfs(u: str) -> bool:
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                return True
            if state[v] == 0 and dfs(v):
                return True
        state[u] = 2
        return False

    return any(state[n] == 0 and dfs(n) for n in nodes)


def _property_names_unique(store: AHStore) -> list[str]:
    issues: list[str] = []
    for e in store.ah.all_hyper().values():
        names: list[str] = []
        for attr in ("Pr", "Mt"):
            props = getattr(e, attr, None)
            if props:
                names.extend(p.name for p in props if isinstance(p, Property))
        if len(names) != len(set(names)):
            issues.append(f"duplicate property name in {getattr(e, 'uid', '?')}")
    return issues


def _refs_typed(store: AHStore) -> list[str]:
    issues: list[str] = []

    def check_ref(ref: object, ctx: str) -> None:
        if isinstance(ref, SRef):
            if ref.kind.value != "S":
                issues.append(f"{ctx}: SRef kind != S")
            if ref.target_uid not in store.ah.S:
                issues.append(f"{ctx}: SRef target missing in S: {ref.target_uid}")
        elif isinstance(ref, MRef):
            if ref.kind.value != "M":
                issues.append(f"{ctx}: MRef kind != M")
            if ref.target_uid not in store.ah.all_hyper():
                issues.append(f"{ctx}: MRef target missing: {ref.target_uid}")

    for link in store.ah.L.values():
        check_ref(link.e1, f"link {link.uid}.e1")
        check_ref(link.e2, f"link {link.uid}.e2")
    return issues


def validate(store: AHStore) -> None:
    issues: list[str] = []
    if _has_cycle(list(iter_is_a_edges(store))):
        issues.append("cycle in IS-A hierarchy")
    if _has_cycle(list(iter_follow_edges(store))):
        issues.append("cycle in FOLLOW (H)")
    issues.extend(_property_names_unique(store))
    issues.extend(_refs_typed(store))
    for s in store.ah.S.values():
        if not s.modality_partition_ok():
            issues.append(f"R partition violated for {s.uid}")
    if issues:
        raise InvariantError(issues)
