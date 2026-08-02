"""Serialize AH as hypergraph: vertices + hyperedges N + binary L."""
from __future__ import annotations

from typing import Any

from ah_memory.perception import PREDICATES
from ah_memory.store import AHStore
from ah_memory.types import (
    ElementList,
    FunctionalSymbol,
    Hyperlink,
    SecondOrderSymbol,
    Template,
)

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

    def _place(layer: str) -> tuple[float, float]:
        i = layer_counters.get(layer, 0)
        layer_counters[layer] = i + 1
        col = _LAYER_X.get(layer, 600)
        row = i % 20
        stack = i // 20
        return float(col + stack * 36), float(row * 64 - 580)

    # --- vertices: S + m/g/k (not N, not T unless mode=all) ---
    for uid, s in store.ah.S.items():
        if uid in PREDICATES and mode != "all":
            continue
        px, py = _place("S")
        nodes.append(
            {
                "id": uid,
                "label": next(iter(s.R.get("TEXT", (uid,))), uid),
                "group": "S",
                "kind": "vertex",
                "activation": round(s.x, 4),
                "x": px,
                "y": py,
                "title": f"S {uid}\nR={{{_fmt_R(s.R)}}}\nact={s.x:.3f}",
            }
        )

    for section_name, bucket in (("C", store.ah.C), ("P", store.ah.P), ("H", store.ah.H)):
        for uid, e in bucket.items():
            if isinstance(e, Hyperlink):
                continue
            if isinstance(e, Template) and mode != "all":
                continue

            act = float(getattr(e, "x", 0.0))
            if isinstance(e, SecondOrderSymbol):
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
            elif isinstance(e, FunctionalSymbol):
                px, py = _place("vertex")
                nodes.append(
                    {
                        "id": uid,
                        "label": f"⟨{e.id}⟩",
                        "group": f"{section_name}_g",
                        "kind": "vertex",
                        "activation": round(act, 4),
                        "x": px,
                        "y": py,
                        "title": f"functional {e.id} {uid}",
                    }
                )
            elif isinstance(e, ElementList):
                label = uid
                for p in e.Pr:
                    if p.name == "label":
                        label = p.value
                px, py = _place("episode" if section_name == "H" else "vertex")
                nodes.append(
                    {
                        "id": uid,
                        "label": label,
                        "group": f"{section_name}_k",
                        "kind": "vertex",
                        "activation": round(act, 4),
                        "x": px,
                        "y": py,
                        "title": f"list {uid}",
                    }
                )
            elif isinstance(e, Template):
                px, py = _place("template")
                pred = e.predicate.target_uid
                nodes.append(
                    {
                        "id": uid,
                        "label": f"T:{pred}",
                        "group": f"{section_name}_T",
                        "kind": "template",
                        "activation": round(act, 4),
                        "x": px,
                        "y": py,
                        "title": f"template {uid} → {pred}",
                    }
                )

    # --- hyperedges N ---
    for n in store.find_hypernodes():
        pred = "?"
        try:
            tpl = store.get_template(n.template.target_uid)
            pred = tpl.predicate.target_uid
        except Exception:
            pred = n.template.target_uid

        roles = {r.value: f.target_uid for r, f in n.fillers.items()}
        members = list(dict.fromkeys(roles.values()))
        role_lines = "\n".join(f"{r} → {_short(v)}" for r, v in roles.items())
        label = f"⟦{pred}⟧\n{role_lines}" if roles else f"⟦{pred}⟧"

        px, py = _place("hyperedge")
        nodes.append(
            {
                "id": n.uid,
                "label": label,
                "group": "hyperedge",
                "kind": "hyperedge",
                "predicate": pred,
                "activation": round(n.x, 4),
                "x": px,
                "y": py,
                "title": (
                    f"HYPEREDGE {n.uid}\npredicate={pred}\nw={n.w:.3f}\n"
                    + "\n".join(f"{r}: {v}" for r, v in roles.items())
                ),
            }
        )

        hyperedges.append(
            {
                "id": n.uid,
                "predicate": pred,
                "w": n.w,
                "roles": roles,
                "members": members,
                "hub": n.uid,
            }
        )

        # incidence spokes: hyperedge hub —role→ actant (undirected visually)
        for role, target in roles.items():
            edges.append(
                {
                    "id": f"{n.uid}__{role}",
                    "from": n.uid,
                    "to": target,
                    "label": role,
                    "kind": "hyper_incidence",
                    "role": role,
                    "arrows": "",
                    "dashes": False,
                    "width": 2.5,
                    "color": {"color": _ROLE_COLOR.get(role, "#e9c46a"), "highlight": "#fff"},
                    "title": f"hyperedge {pred}: {role} = {target}",
                }
            )

        if mode == "all":
            edges.append(
                {
                    "id": f"{n.uid}__tpl",
                    "from": n.uid,
                    "to": n.template.target_uid,
                    "label": "T",
                    "kind": "template_ref",
                    "arrows": "to",
                    "dashes": True,
                    "width": 1,
                    "color": {"color": "#667788"},
                }
            )

    # --- binary associative links L ---
    for link in store.ah.L.values():
        edges.append(
            {
                "id": link.uid,
                "from": link.e1.target_uid,
                "to": link.e2.target_uid,
                "label": link.id,
                "kind": "binary",
                "arrows": "to",
                "dashes": link.id != "IS-A",
                "width": 1 if link.id != "IS-A" else 2,
                "color": {
                    "color": "#98c1d9" if link.id == "IS-A" else "#4a5568",
                    "opacity": 0.45,
                },
                "title": f"binary {link.id} w={link.w:.3f}",
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
            "hyperedges": len(store.find_hypernodes()),
            "graph_size": store.graph_size(),
            "tau": store.ah.tau,
        },
    }


def dump_ah_json(store: AHStore) -> dict[str, Any]:
    g = dump_graph(store, mode="hyper")
    return {
        "tau": store.ah.tau,
        "S": {
            uid: {"R": {m: sorted(forms) for m, forms in s.R.items()}, "activation": s.x}
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
