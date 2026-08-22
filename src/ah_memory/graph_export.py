"""Serialize AH as hypergraph: vertices + hyperedges N + binary L."""
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
    del mode  # templates removed; kept for API compatibility
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

    for uid, s in store.ah.S.items():
        if store.get_relation(uid) is not None:
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

    for section_name, bucket in (("C", store.ah.C), ("P", store.ah.P), ("H", store.ah.H)):
        for uid, e in bucket.items():
            if not isinstance(e, SecondOrderSymbol):
                continue
            act = _act(uid, float(getattr(e, "x", 0.0)))
            label = uid
            for p in e.Pr:
                if p.name == "label":
                    label = p.value
            kind_meta = next((p.value for p in e.Mt if p.name == "kind"), "")
            layer = "episode" if kind_meta == "Episode" or section_name == "H" else "vertex"
            group = f"{section_name}_k" if kind_meta == "Episode" else f"{section_name}_m"
            px, py = _place(layer)
            nodes.append(
                {
                    "id": uid,
                    "label": label,
                    "group": group,
                    "kind": "vertex",
                    "activation": round(act, 4),
                    "x": px,
                    "y": py,
                    "title": f"{section_name} m {uid}\nact={act:.3f}",
                }
            )

    # Semantic factors drawn as N-hyperedges (pre-merge visual: yellow hub + roles + mesh).
    for factor in store.list_semantic_factors():
        roles = dict(factor.roles)
        members = list(dict.fromkeys([*roles.values(), *factor.variables]))
        relation = factor.relation
        pred = (
            relation.canonical_label if relation is not None else "RELATED_TO"
        )
        raw_relation = str(factor.metadata.get("raw_relation") or pred)
        role_lines = "\n".join(f"{r} → {_short(v)}" for r, v in roles.items())
        label = f"⟦{pred}⟧\n{role_lines}" if roles else f"⟦{pred}⟧"
        px, py = _place("hyperedge")
        n_activation = _act(
            factor.uid,
            (
                sum(_act(uid) for uid in members) / len(members)
                if activation is not None and members
                else 0.0
            ),
        )
        nodes.append(
            {
                "id": factor.uid,
                "label": label,
                "group": "hyperedge",
                "kind": "hyperedge",
                "predicate": pred,
                "raw_relation": raw_relation,
                "activation": round(n_activation, 4),
                "x": px,
                "y": py,
                "title": (
                    f"HYPEREDGE {factor.uid}\npredicate={pred}\nw={factor.weight:.3f}\n"
                    + "\n".join(f"{r}: {v}" for r, v in roles.items())
                ),
            }
        )
        hyperedges.append(
            {
                "id": factor.uid,
                "predicate": pred,
                "w": factor.weight,
                "roles": roles,
                "members": members,
                "hub": factor.uid,
            }
        )
        for role, target in roles.items():
            edges.append(
                {
                    "id": f"{factor.uid}__{role}",
                    "from": factor.uid,
                    "to": target,
                    "label": role,
                    "kind": "hyper_incidence",
                    "role": role,
                    "w": round(float(factor.weight), 4),
                    "arrows": "",
                    "dashes": False,
                    "width": 2.5,
                    "color": {
                        "color": _ROLE_COLOR.get(role, "#e9c46a"),
                        "highlight": "#fff",
                    },
                    "title": f"hyperedge {pred}: {role} = {target} · w={factor.weight:.3f}",
                }
            )
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                edges.append(
                    {
                        "id": f"{factor.uid}__mesh__{a}__{b}",
                        "from": a,
                        "to": b,
                        "label": "",
                        "kind": "hyper_mesh",
                        "hyperedge": factor.uid,
                        "predicate": pred,
                        "w": round(float(factor.weight), 4),
                        "arrows": "",
                        "dashes": True,
                        "width": 2,
                        "color": {
                            "color": "#ffe566",
                            "opacity": 0.75,
                            "highlight": "#fff",
                        },
                        "title": (
                            f"mesh ⟦{pred}⟧: {a} ↔ {b} (n={len(members)}) · "
                            f"w={factor.weight:.3f}"
                        ),
                    }
                )

    for link in store.ah.L.values():
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
            h
            for h in hyperedges
            if h["id"] in keep and all(m in keep for m in h["members"])
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
        "mode": "hyper",
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
    from ah_memory.store_codec import snapshot_store

    g = dump_graph(store, mode="hyper")
    payload = snapshot_store(store)
    payload["stats"] = g["stats"]
    payload["hyperedges"] = g["hyperedges"]
    payload["links"] = [
        {
            "uid": l.uid,
            "id": l.id,
            "w": l.w,
            "e1": l.e1.target_uid,
            "e2": l.e2.target_uid,
        }
        for l in store.ah.L.values()
    ]
    return payload


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
