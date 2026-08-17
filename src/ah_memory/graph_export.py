"""Serialize AH as hypergraph: vertices S/M + semantic factors + binary L."""
from __future__ import annotations

from typing import Any, Mapping

from ah_memory.store import AHStore
from ah_memory.types import SecondOrderSymbol

_LAYER_X = {
    "S": 0,
    "template": 260,
    "vertex": 560,
    "hyperedge": 920,
    "episode": 1220,
}

_ROLE_COLOR = {
    "SUBJECT": "#e76f51",
    "OBJECT": "#2a9d8f",
    "LOCATION": "#457b9d",
    "TIME": "#9b5de5",
    "CAUSE": "#f4a261",
    "TOOL": "#e9c46a",
    "MATERIAL": "#90be6d",
    "PURPOSE": "#577590",
    "HOW-TO": "#ef476f",
}


def dump_graph(
    store: AHStore,
    *,
    limit_nodes: int | None = None,
    mode: str = "hyper",
    activation: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    mode:
      hyper — hyperedges as diamond hubs + role spokes; binary L dimmed
      all   — include templates as nodes
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    hyperedges: list[dict[str, Any]] = []
    layer_counters: dict[str, int] = {}

    def _act(uid: str, fallback: float = 0.0) -> float:
        if activation is not None:
            return float(activation.get(uid, fallback))
        return fallback

    def _place(layer: str) -> tuple[float, float]:
        i = layer_counters.get(layer, 0)
        layer_counters[layer] = i + 1
        col = _LAYER_X.get(layer, 600)
        row = i % 20
        stack = i // 20
        return float(col + stack * 36), float(row * 64 - 580)

    # --- vertices: S + m/g/k (not N, not T unless mode=all) ---
    for uid, s in store.ah.S.items():
        if store.get_relation(uid) is not None and mode != "all":
            continue
        px, py = _place("S")
        nodes.append(
            {
                "id": uid,
                "label": next(iter(s.R.get("TEXT", (uid,))), uid),
                "group": "S",
                "kind": "vertex",
                "activation": round(_act(uid, s.x), 4),
                "x": px,
                "y": py,
                "title": f"S {uid}\nR={{{_fmt_R(s.R)}}}\nact={_act(uid, s.x):.3f}",
            }
        )

    # --- vertices: S + M ---
    for section_name, bucket in (("C", store.ah.C), ("P", store.ah.P), ("H", store.ah.H)):
        for uid, e in bucket.items():
            if not isinstance(e, SecondOrderSymbol):
                continue
            act = _act(uid, float(e.x))
            label = uid
            for p in e.Pr:
                if p.name == "label":
                    label = p.value
            px, py = _place("vertex")
            nodes.append(
                {
                    "id": uid,
                    "label": label,
                    "group": f"{section_name}_m",
                    "kind": "vertex",
                    "activation": round(act, 4),
                    "x": px,
                    "y": py,
                    "title": f"{section_name} m {uid}\nact={act:.3f}",
                }
            )

    # --- semantic factors (open relations) ---
    co_member_pairs: set[tuple[str, str]] = set()
    for factor in store.list_semantic_factors():
        roles = dict(factor.roles)
        members = list(dict.fromkeys(factor.variables))
        relation = factor.relation
        canonical = (
            relation.canonical_label if relation is not None else "RELATED_TO"
        )
        raw_relation = str(factor.metadata.get("raw_relation") or canonical)
        role_lines = "\n".join(
            f"{role} → {_short(uid)}" for role, uid in roles.items()
        )
        label = (
            f"⟦{raw_relation} / {canonical}⟧\n{role_lines}"
            if roles
            else f"⟦{raw_relation} / {canonical}⟧"
        )
        px, py = _place("hyperedge")
        factor_activation = (
            sum(_act(uid) for uid in members) / len(members)
            if members
            else 0.0
        )
        nodes.append(
            {
                "id": factor.uid,
                "label": label,
                "group": "hyperedge",
                "kind": "semantic_factor",
                "predicate": canonical,
                "raw_relation": raw_relation,
                "activation": round(factor_activation, 4),
                "x": px,
                "y": py,
                "title": (
                    f"SEMANTIC FACTOR {factor.uid}\n"
                    f"raw={raw_relation}\ncanonical={canonical}\n"
                    f"w={factor.weight:.3f} confidence={factor.confidence:.3f}\n"
                    f"parameters={factor.parameters.to_dict() if factor.parameters else {}}"
                ),
            }
        )
        hyperedges.append(
            {
                "id": factor.uid,
                "predicate": canonical,
                "raw_relation": raw_relation,
                "w": factor.weight,
                "confidence": factor.confidence,
                "roles": roles,
                "members": members,
                "hub": factor.uid,
                "parameters": (
                    factor.parameters.to_dict()
                    if factor.parameters is not None
                    else {}
                ),
            }
        )
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                co_member_pairs.add(tuple(sorted((a, b))))
        directional = bool(
            relation is not None
            and relation.properties.directional
            and not relation.properties.symmetric
        )
        source = (
            str(factor.metadata.get("source_variable") or "")
            or roles.get("SUBJECT", "")
            or (members[0] if members else "")
        )
        for role, target in roles.items():
            source_spoke = directional and target == source
            edges.append(
                {
                    "id": f"{factor.uid}__{role}",
                    "from": target if source_spoke else factor.uid,
                    "to": factor.uid if source_spoke else target,
                    "label": role,
                    "kind": "semantic_incidence",
                    "role": role,
                    "relation": canonical,
                    "w": round(float(factor.weight), 4),
                    "confidence": round(float(factor.confidence), 4),
                    "arrows": "to" if directional else "",
                    "dashes": False,
                    "width": 2.5,
                    "color": {
                        "color": _ROLE_COLOR.get(role, "#e9c46a"),
                        "highlight": "#fff",
                    },
                    "title": (
                        f"{raw_relation} / {canonical}: {role} = {target} · "
                        f"w={factor.weight:.3f} c={factor.confidence:.3f}"
                    ),
                }
            )

    # --- binary associative links L ---
    for link in store.ah.L.values():
        pair = tuple(sorted((link.e1.target_uid, link.e2.target_uid)))
        if link.id == "ASSOC" and pair in co_member_pairs:
            continue
        # ASSOC симметрична; IS-A / FOLLOW направлены e1→e2
        directed = link.id in {"IS-A", "FOLLOW"}
        edges.append(
            {
                "uid": link.uid,
                "id": link.uid,
                "from": link.e1.target_uid,
                "to": link.e2.target_uid,
                "label": link.id,
                "kind": "binary",
                "w": round(float(link.w), 4),
                "arrows": "to" if directed else "",
                "dashes": link.id != "IS-A",
                "width": 1 if link.id != "IS-A" else 2,
                "color": {
                    "color": "#98c1d9" if link.id == "IS-A" else "#4a5568",
                    "opacity": 0.45,
                },
                "title": (
                    f"binary {link.id} w={link.w:.3f}"
                    + ("" if directed else " (undirected)")
                ),
            }
        )

    if limit_nodes is not None and len(nodes) > limit_nodes:
        # prefer keeping hyperedges + their members
        he_ids = {h["id"] for h in hyperedges}
        member_ids = {m for h in hyperedges for m in h["members"]}
        priority = he_ids | member_ids
        rest = [n for n in nodes if n["id"] not in priority]
        rest_sorted = sorted(rest, key=lambda n: n.get("activation", 0), reverse=True)
        keep_nodes = [n for n in nodes if n["id"] in priority] + rest_sorted
        keep_nodes = keep_nodes[: max(limit_nodes, len(priority))]
        keep = {n["id"] for n in keep_nodes}
        nodes = keep_nodes
        edges = [e for e in edges if e["from"] in keep and e["to"] in keep]
        hyperedges = [
            h for h in hyperedges if h["id"] in keep and all(m in keep for m in h["members"])
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
        "mode": mode,
        "stats": {
            "S": len(store.ah.S),
            "C": len(store.ah.C),
            "P": len(store.ah.P),
            "H": len(store.ah.H),
            "L": len(store.ah.L),
            "hyperedges": len(store.semantic_factors),
            "semantic_factors": len(store.semantic_factors),
            "relations": len(store.list_relations()),
            "events": len(store.events),
            "graph_size": store.graph_size(),
            "tau": store.ah.tau,
        },
    }


def dump_ah_json(store: AHStore) -> dict[str, Any]:
    g = dump_graph(store, mode="hyper")
    return {
        "tau": store.ah.tau,
        "S": {
            uid: {
                "R": {m: sorted(forms) for m, forms in s.R.items()},
                "activation": s.x,
                "created_tau": s.created_tau,
                "added_at": getattr(s, "added_at", "") or "",
            }
            for uid, s in store.ah.S.items()
        },
        "stats": g["stats"],
        "hyperedges": g["hyperedges"],
        "links": [
            {
                "uid": l.uid,
                "id": l.id,
                "w": l.w,
                "e1": l.e1.target_uid,
                "e2": l.e2.target_uid,
            }
            for l in store.ah.L.values()
        ],
        "relations": store.relations.to_dict(),
        "semantic_factors": [
            {
                "uid": factor.uid,
                "relation": (
                    factor.relation.canonical_label
                    if factor.relation is not None
                    else None
                ),
                "variables": list(factor.variables),
                "roles": dict(factor.roles),
                "weight": factor.weight,
                "confidence": factor.confidence,
                "parameters": (
                    factor.parameters.to_dict()
                    if factor.parameters is not None
                    else None
                ),
                "metadata": dict(factor.metadata),
            }
            for factor in store.list_semantic_factors()
        ],
        "events": [event.to_dict() for event in store.list_events()],
        "state": store.state.to_dict(),
        "state_transitions": list(store.state_transitions),
    }


def _short(uid: str) -> str:
    if uid.startswith("M_"):
        return uid[2:]
    if uid.startswith("G_"):
        return uid[2:]
    if uid.startswith("K_"):
        return uid[2:]
    return uid


def _fmt_R(R: dict[str, set[str]]) -> str:
    parts = []
    for m, forms in R.items():
        parts.append(f"{m}:{'|'.join(sorted(forms)[:5])}")
    return ", ".join(parts)
