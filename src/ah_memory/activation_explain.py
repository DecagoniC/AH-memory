"""Human-readable activation chains for debug (seed → factor → node)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ah_memory.factor_graph import Factor, FactorGraph, FactorKind
from ah_memory.store import AHStore


def node_label(store: AHStore, uid: str) -> str:
    bare = uid[2:] if uid.startswith("M_") else uid
    try:
        e = store._find_anywhere(uid)  # noqa: SLF001
        for p in getattr(e, "Pr", []) or []:
            if p.name == "label" and p.value:
                return str(p.value)
    except Exception:
        pass
    if uid in store.ah.S:
        forms = store.ah.S[uid].R.get("TEXT") or set()
        if forms:
            return next(iter(forms))
    return bare.replace("_", " ").lower()


def _factor_via(store: AHStore, f: Factor) -> str:
    if f.kind is FactorKind.HYPER:
        pred = ""
        n_uid = f.fid.split("::", 1)[-1]
        try:
            n = store._find_anywhere(n_uid)  # noqa: SLF001
            tpl = getattr(n, "template", None)
            if tpl is not None:
                tuid = getattr(tpl, "target_uid", "") or ""
                pred = tuid.replace("T_", "") if tuid else ""
        except Exception:
            pass
        if not pred:
            pred = n_uid[2:] if n_uid.startswith("N_") else n_uid
            parts = pred.split("_")
            if parts and parts[-1].isdigit():
                pred = "_".join(parts[:-1]) or pred
        roles = ",".join(sorted(f.roles.keys())) if f.roles else ""
        return f"факт {pred}" + (f" [{roles}]" if roles else "")
    if f.kind is FactorKind.PAIR:
        return f.link_id or "L"
    if f.kind is FactorKind.OBS:
        return "evidence"
    return f.kind.value


@dataclass(frozen=True)
class _Edge:
    frm: str
    to: str
    via: str
    score: float


def _edges(store: AHStore, graph: FactorGraph, beliefs: dict[str, float]) -> list[_Edge]:
    out: list[_Edge] = []
    for f in graph.factors:
        if f.kind not in {FactorKind.HYPER, FactorKind.PAIR}:
            continue
        vs = [v for v in f.variables if v in beliefs]
        if len(vs) < 2:
            continue
        via = _factor_via(store, f)
        bs = [beliefs.get(v, 0.0) for v in vs]
        score = (sum(bs) / len(bs)) * (1.0 + max(0.0, f.w))
        # connect each ordered pair (undirected traversal)
        for i, a in enumerate(vs):
            for b in vs[i + 1 :]:
                out.append(_Edge(a, b, via, score))
                out.append(_Edge(b, a, via, score))
    return out


def build_activation_chains(
    store: AHStore,
    graph: FactorGraph,
    beliefs: dict[str, float],
    *,
    seeds: list[str],
    evidence: dict[str, float],
    threshold: float,
    max_chains: int = 12,
    max_hops: int = 4,
) -> list[str]:
    """
    Approximate causal chains for debug:
    seed/evidence →[factor]→ … → WM node.
    Not exact BP message attribution — shortest high-belief path on the factor graph.
    """
    if not beliefs:
        return []

    seed_set = {u for u in seeds if u in beliefs}
    for u, lam in evidence.items():
        if lam >= 0.2 and u in beliefs:
            seed_set.add(u)
    if not seed_set:
        # strongest evidence-like: top beliefs as pseudo-origins
        seed_set = {
            u
            for u, _ in sorted(beliefs.items(), key=lambda kv: -kv[1])[:3]
        }

    adj: dict[str, list[tuple[str, str, float]]] = {}
    for e in _edges(store, graph, beliefs):
        adj.setdefault(e.frm, []).append((e.to, e.via, e.score))
    for v in adj:
        adj[v].sort(key=lambda t: -t[2])

    # BFS multi-source: parent[v] = (prev, via); depth from nearest seed
    parent: dict[str, tuple[str | None, str]] = {s: (None, "seed") for s in seed_set}
    depth: dict[str, int] = {s: 0 for s in seed_set}
    q: deque[str] = deque(seed_set)
    while q:
        u = q.popleft()
        if depth[u] >= max_hops:
            continue
        for v, via, _sc in adj.get(u, []):
            if v in parent:
                continue
            if beliefs.get(v, 0.0) < min(0.15, threshold * 0.4):
                continue
            parent[v] = (u, via)
            depth[v] = depth[u] + 1
            q.append(v)

    targets = sorted(
        (u for u, b in beliefs.items() if b > threshold),
        key=lambda u: -beliefs[u],
    )
    if not targets:
        targets = [u for u, _ in sorted(beliefs.items(), key=lambda kv: -kv[1])[:8]]

    lines: list[str] = []
    seen_paths: set[tuple[str, ...]] = set()

    def fmt_node(uid: str, *, origin: bool = False) -> str:
        lab = node_label(store, uid)
        b = beliefs.get(uid, 0.0)
        lam = evidence.get(uid)
        bits = [f"b={b:.2f}"]
        if origin and uid in seed_set:
            bits.insert(0, "seed")
        if lam is not None and lam >= 0.05:
            bits.append(f"λ={lam:.2f}")
        return f"«{lab}» {uid} ({', '.join(bits)})"

    for tgt in targets:
        if tgt not in parent and tgt not in seed_set:
            continue
        # reconstruct
        nodes: list[str] = []
        vias: list[str] = []
        cur: str | None = tgt
        guard = 0
        while cur is not None and guard < max_hops + 2:
            nodes.append(cur)
            prev, via = parent.get(cur, (None, "seed"))
            if prev is None:
                break
            vias.append(via)
            cur = prev
            guard += 1
        nodes.reverse()
        vias.reverse()
        key = tuple(nodes)
        if key in seen_paths:
            continue
        seen_paths.add(key)

        if len(nodes) == 1:
            lines.append(fmt_node(nodes[0], origin=True))
        else:
            parts = [fmt_node(nodes[0], origin=True)]
            for i, via in enumerate(vias):
                parts.append(f"—[{via}]→ {fmt_node(nodes[i + 1])}")
            lines.append(" ".join(parts))
        if len(lines) >= max_chains:
            break

    # orphan seeds that never reached others
    for s in sorted(seed_set, key=lambda u: -beliefs.get(u, 0.0)):
        if any(s in p for p in seen_paths):
            continue
        lines.append(fmt_node(s, origin=True) + " (без распространения)")
        if len(lines) >= max_chains:
            break

    return lines
