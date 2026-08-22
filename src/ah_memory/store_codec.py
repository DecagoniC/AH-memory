"""Round-trip JSON snapshot of AHStore (graph + open-semantics)."""
from __future__ import annotations

import re
from typing import Any, Mapping

from ah_memory.factor_graph import Factor, FactorKind
from ah_memory.factor_parameters import FactorParameters
from ah_memory.relation_registry import RelationRegistry
from ah_memory.relations import Event, NodeRef, Relation, RelationProperties
from ah_memory.state_engine import State
from ah_memory.store import AHStore
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    HyperElement,
    MRef,
    Property,
    SRef,
    SecondOrderSymbol,
    Section,
)

SNAPSHOT_FORMAT = "ah-store-v1"
_UID_SEQ_RE = re.compile(r"_(\d+)$")


def snapshot_store(store: AHStore) -> dict[str, Any]:
    """Serialize the live store so it can be restored later."""
    return {
        "format": SNAPSHOT_FORMAT,
        "tau": store.ah.tau,
        "revision": store.ah.revision,
        "uid_seq": store._uid_seq,
        "S": {uid: _symbol_to_dict(s) for uid, s in store.ah.S.items()},
        "C": {uid: _element_to_dict(e) for uid, e in store.ah.C.items()},
        "P": {uid: _element_to_dict(e) for uid, e in store.ah.P.items()},
        "H": {uid: _element_to_dict(e) for uid, e in store.ah.H.items()},
        "L": {uid: _link_to_dict(link) for uid, link in store.ah.L.items()},
        "relations": store.relations.to_dict(),
        "semantic_factors": [_factor_to_dict(f) for f in store.list_semantic_factors()],
        "events": [event.to_dict() for event in store.list_events()],
        "state": store.state.to_dict(),
        "state_transitions": list(store.state_transitions),
    }


def restore_store(payload: Mapping[str, Any]) -> AHStore:
    """Rebuild an AHStore from snapshot_store() (or a library envelope)."""
    body = _unwrap(payload)
    store = AHStore()
    store.ah.tau = int(body.get("tau") or 0)
    store.ah.revision = int(body.get("revision") or 0)
    store._uid_seq = int(body.get("uid_seq") or 0)

    for uid, raw in dict(body.get("S") or {}).items():
        store.ah.S[str(uid)] = _symbol_from_dict(str(uid), raw)

    for section, key in ((Section.C, "C"), (Section.P, "P"), (Section.H, "H")):
        bucket = store.ah.section(section)
        for uid, raw in dict(body.get(key) or {}).items():
            bucket[str(uid)] = _element_from_dict(str(uid), raw)

    relations = RelationRegistry()
    for raw in dict(body.get("relations") or {}).values():
        if isinstance(raw, Mapping):
            relations.register_relation(_relation_from_dict(raw), replace=True)
    store.relations = relations

    events: dict[str, Event] = {}
    for raw in list(body.get("events") or []):
        if not isinstance(raw, Mapping):
            continue
        event = _event_from_dict(raw)
        events[event.uid] = event
    store.events = events

    factors: dict[str, Factor] = {}
    for raw in list(body.get("semantic_factors") or []):
        if not isinstance(raw, Mapping):
            continue
        factor = _factor_from_dict(raw, store)
        factors[factor.uid] = factor
    store.semantic_factors = factors

    for uid, raw in dict(body.get("L") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        store.ah.L[str(uid)] = _link_from_dict(str(uid), raw, store)

    store.state = _state_from_dict(body.get("state") or {})
    store.state_transitions = [
        dict(item) for item in list(body.get("state_transitions") or []) if isinstance(item, Mapping)
    ]
    if store._uid_seq <= 0:
        store._uid_seq = _infer_uid_seq(store)
    return store


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("store"), Mapping):
        return payload["store"]
    return payload


def _symbol_to_dict(s: AbstractSymbol) -> dict[str, Any]:
    return {
        "R": {m: sorted(forms) for m, forms in s.R.items()},
        "x": s.x,
        "created_tau": s.created_tau,
        "added_at": getattr(s, "added_at", "") or "",
    }


def _symbol_from_dict(uid: str, raw: Mapping[str, Any]) -> AbstractSymbol:
    r_raw = dict(raw.get("R") or {})
    r = {str(mod): set(str(f) for f in forms) for mod, forms in r_raw.items()}
    return AbstractSymbol(
        uid=uid,
        R=r,
        x=float(raw.get("x") or raw.get("activation") or 0.0),
        created_tau=int(raw.get("created_tau") or 0),
        added_at=str(raw.get("added_at") or ""),
    )


def _props_to_dict(props: list[Property]) -> list[dict[str, str]]:
    return [
        {"name": p.name, "value": p.value, "type": p.type, "unit": p.unit}
        for p in props
    ]


def _props_from_dict(raw: Any) -> list[Property]:
    out: list[Property] = []
    for item in list(raw or []):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        value = str(item.get("value") or "")
        if not name or not value:
            continue
        out.append(
            Property(
                name=name,
                value=value,
                type=str(item.get("type") or ""),
                unit=str(item.get("unit") or ""),
            )
        )
    return out


def _element_to_dict(e: HyperElement) -> dict[str, Any]:
    if isinstance(e, SecondOrderSymbol):
        return {
            "type": "M",
            "uid": e.uid,
            "Pr": _props_to_dict(e.Pr),
            "Mt": _props_to_dict(e.Mt),
            "x": e.x,
            "created_tau": e.created_tau,
            "added_at": getattr(e, "added_at", "") or "",
        }
    if isinstance(e, SRef):
        return {"type": "SRef", "target_uid": e.target_uid}
    if isinstance(e, MRef):
        return {"type": "MRef", "target_uid": e.target_uid}
    uid = getattr(e, "uid", None)
    return {"type": "M", "uid": str(uid or ""), "Pr": [], "Mt": []}


def _element_from_dict(uid: str, raw: Mapping[str, Any]) -> HyperElement:
    kind = str(raw.get("type") or "M")
    if kind == "SRef":
        return SRef(target_uid=str(raw.get("target_uid") or uid))
    if kind == "MRef":
        return MRef(target_uid=str(raw.get("target_uid") or uid))
    return SecondOrderSymbol(
        uid=str(raw.get("uid") or uid),
        Pr=_props_from_dict(raw.get("Pr")),
        Mt=_props_from_dict(raw.get("Mt")),
        x=float(raw.get("x") or 0.0),
        created_tau=int(raw.get("created_tau") or 0),
        added_at=str(raw.get("added_at") or ""),
    )


def _link_to_dict(link: AssocLink) -> dict[str, Any]:
    return {
        "uid": link.uid,
        "id": link.id,
        "w": link.w,
        "e1": {"kind": link.e1.kind.value, "target_uid": link.e1.target_uid},
        "e2": {"kind": link.e2.kind.value, "target_uid": link.e2.target_uid},
        "created_tau": link.created_tau,
        "added_at": getattr(link, "added_at", "") or "",
    }


def _ref_from_dict(raw: Mapping[str, Any], store: AHStore) -> SRef | MRef:
    target = str(raw.get("target_uid") or "")
    kind = str(raw.get("kind") or "M").upper()
    if kind == "S":
        return store.s_ref(target)
    return store.m_ref(target)


def _link_from_dict(uid: str, raw: Mapping[str, Any], store: AHStore) -> AssocLink:
    e1_raw = raw.get("e1")
    e2_raw = raw.get("e2")
    if not isinstance(e1_raw, Mapping):
        e1_raw = {"kind": "M", "target_uid": str(raw.get("e1") or "")}
    if not isinstance(e2_raw, Mapping):
        e2_raw = {"kind": "M", "target_uid": str(raw.get("e2") or "")}
    return AssocLink(
        uid=str(raw.get("uid") or uid),
        id=str(raw.get("id") or "ASSOC"),
        w=float(raw.get("w") or 0.0),
        e1=_ref_from_dict(e1_raw, store),
        e2=_ref_from_dict(e2_raw, store),
        created_tau=int(raw.get("created_tau") or 0),
        added_at=str(raw.get("added_at") or ""),
    )


def _relation_from_dict(raw: Mapping[str, Any]) -> Relation:
    props_raw = raw.get("properties") or {}
    props = RelationProperties(
        **{
            key: bool(props_raw.get(key, getattr(RelationProperties(), key)))
            for key in RelationProperties().to_dict()
        }
    ) if isinstance(props_raw, Mapping) else RelationProperties()
    embedding = raw.get("embedding")
    return Relation(
        uid=str(raw.get("uid") or ""),
        raw_label=str(raw.get("raw_label") or raw.get("canonical_label") or "RELATED_TO"),
        canonical_label=str(raw.get("canonical_label") or "RELATED_TO"),
        arity=max(1, int(raw.get("arity") or 2)),
        properties=props,
        embedding=tuple(float(x) for x in embedding) if embedding else None,
        metadata=dict(raw.get("metadata") or {}),
    )


def _event_from_dict(raw: Mapping[str, Any]) -> Event:
    pred_raw = raw.get("predicate")
    if isinstance(pred_raw, Mapping):
        predicate = _relation_from_dict(pred_raw)
    else:
        label = str(pred_raw or "RELATED_TO")
        predicate = Relation(uid=f"REL_{label}", raw_label=label, canonical_label=label)
    arguments: dict[str, NodeRef] = {}
    for role, node in dict(raw.get("arguments") or {}).items():
        if isinstance(node, Mapping):
            arguments[str(role)] = NodeRef(
                uid=str(node.get("uid") or ""),
                role=str(node.get("role") or role),
            )
        else:
            arguments[str(role)] = NodeRef(uid=str(node), role=str(role))
    return Event(
        uid=str(raw.get("uid") or ""),
        predicate=predicate,
        arguments=arguments,
        timestamp=raw.get("timestamp"),
        confidence=float(raw.get("confidence") or 1.0),
        raw_span=raw.get("raw_span"),
        metadata=dict(raw.get("metadata") or {}),
    )


def _factor_to_dict(factor: Factor) -> dict[str, Any]:
    return {
        "uid": factor.uid,
        "fid": factor.fid,
        "kind": factor.kind.value,
        "relation": factor.relation.to_dict() if factor.relation is not None else None,
        "variables": list(factor.variables),
        "roles": dict(factor.roles),
        "weight": factor.weight,
        "w": factor.w,
        "confidence": factor.confidence,
        "activation": factor.activation,
        "link_id": factor.link_id,
        "lambda_obs": factor.lambda_obs,
        "epsilon": factor.epsilon,
        "potential_key": factor.potential_key,
        "source_uid": factor.source_uid,
        "parameters": factor.parameters.to_dict() if factor.parameters is not None else None,
        "embedding": list(factor.embedding) if factor.embedding is not None else None,
        "metadata": dict(factor.metadata),
    }


def _factor_from_dict(raw: Mapping[str, Any], store: AHStore) -> Factor:
    rel_raw = raw.get("relation")
    relation: Relation | None = None
    if isinstance(rel_raw, Mapping):
        relation = _relation_from_dict(rel_raw)
        existing = store.get_relation(relation.canonical_label)
        relation = existing or store.register_relation(relation)
    elif rel_raw:
        existing = store.get_relation(str(rel_raw))
        if existing is not None:
            relation = existing
    params_raw = raw.get("parameters")
    parameters = None
    if isinstance(params_raw, Mapping):
        allowed = FactorParameters().to_dict()
        parameters = FactorParameters(
            **{
                key: float(params_raw.get(key, default))
                for key, default in allowed.items()
            }
        )
    kind_raw = str(raw.get("kind") or "hyper")
    try:
        kind = FactorKind(kind_raw)
    except ValueError:
        kind = FactorKind.HYPER
    embedding = raw.get("embedding")
    fid = str(raw.get("fid") or raw.get("uid") or "")
    return Factor(
        fid=fid,
        kind=kind,
        variables=list(raw.get("variables") or []),
        w=float(raw.get("w") if raw.get("w") is not None else raw.get("weight") or 0.5),
        link_id=str(raw.get("link_id") or ""),
        roles=dict(raw.get("roles") or {}),
        lambda_obs=float(raw.get("lambda_obs") or 0.0),
        epsilon=float(raw.get("epsilon") or 0.05),
        potential_key=str(raw.get("potential_key") or ""),
        source_uid=str(raw.get("source_uid") or ""),
        relation=relation,
        parameters=parameters,
        activation=float(raw.get("activation") or 0.0),
        confidence=float(raw.get("confidence") or 1.0),
        embedding=tuple(float(x) for x in embedding) if embedding else None,
        metadata=dict(raw.get("metadata") or {}),
    )


def _state_from_dict(raw: Mapping[str, Any]) -> State:
    history_raw = dict(raw.get("history") or {})
    return State(
        values=dict(raw.get("values") or {}),
        history={str(k): tuple(v) for k, v in history_raw.items()},
        applied_events=tuple(str(x) for x in list(raw.get("applied_events") or [])),
    )


def _infer_uid_seq(store: AHStore) -> int:
    seq = 0
    uids = [
        *store.ah.S,
        *store.ah.C,
        *store.ah.P,
        *store.ah.H,
        *store.ah.L,
        *store.semantic_factors,
        *store.events,
    ]
    for uid in uids:
        match = _UID_SEQ_RE.search(str(uid))
        if match:
            seq = max(seq, int(match.group(1)))
    return seq
