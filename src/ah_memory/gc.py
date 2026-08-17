"""Garbage collector (monograph §4 p.25)."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from ah_memory.hyperparams import HyperParams
from ah_memory.store import AHStore


@dataclass
class GCReport:
    removed_elements: list[str] = field(default_factory=list)
    removed_links: list[str] = field(default_factory=list)
    protected_ttl: list[str] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.removed_elements) + len(self.removed_links)


def _incident_weights(store: AHStore, uid: str) -> list[float]:
    return [link.w for link in store.find_links(uid)]


def _undirected_edges(store: AHStore) -> list[tuple[str, str]]:
    return [
        (link.e1.target_uid, link.e2.target_uid)
        for link in store.ah.L.values()
    ]


def _connected_to_S(store: AHStore) -> set[str]:
    roots = set(store.ah.S.keys())
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in _undirected_edges(store):
        adj[a].append(b)
        adj[b].append(a)
    seen = set(roots)
    q = deque(roots)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def orphan_uids(store: AHStore) -> set[str]:
    """Return elements matching the challenge definition of a lost node."""
    connected_to_s = _connected_to_S(store)
    elements = set(store.ah.S) | set(store.ah.all_hyper())
    result: set[str] = set()
    for uid in elements:
        weights = _incident_weights(store, uid)
        all_zero = not weights or all(weight == 0.0 for weight in weights)
        detached = uid not in connected_to_s and uid not in store.ah.S
        if all_zero or detached:
            result.add(uid)
    return result


def collect(store: AHStore, hp: HyperParams | None = None) -> GCReport:
    hp = hp or HyperParams()
    tau = store.ah.tau
    report = GCReport()
    linked_to_s = _connected_to_S(store)

    for link in list(store.ah.L.values()):
        if link.w == 0.0:
            store.remove_link(link.uid)
            report.removed_links.append(link.uid)

    candidates: list[str] = []
    for uid in list(store.ah.all_hyper().keys()) + list(store.ah.S.keys()):
        created = 0
        if uid in store.ah.S:
            created = store.ah.S[uid].created_tau
        else:
            e = store.ah.all_hyper()[uid]
            created = int(getattr(e, "created_tau", 0))
        if tau - created < hp.ttl:
            report.protected_ttl.append(uid)
            continue
        weights = _incident_weights(store, uid)
        zero_weights = (not weights) or all(w == 0.0 for w in weights)
        detached = uid not in linked_to_s and uid not in store.ah.S
        if zero_weights or detached:
            if uid in store.ah.S and not zero_weights:
                continue
            candidates.append(uid)

    for uid in candidates:
        if store.remove_element(uid):
            report.removed_elements.append(uid)
        for link in list(store.find_links(uid)):
            if store.remove_link(link.uid):
                report.removed_links.append(link.uid)

    return report
