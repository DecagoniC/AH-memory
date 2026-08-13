"""Transform: FactCandidate → open relations (Event + semantic Factor).

Только open-path: Event + semantic Factor, без legacy T/N.
Identity (если передан) мержит mention в существующий bare UID до ensure_*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from ah_memory.factor_graph import Factor, FactorKind
from ah_memory.factor_parameters import (
    FactorParameterGenerator,
    RuleBasedParameterGenerator,
)
from ah_memory.hyperparams import HyperParams
from ah_memory.perception import PREDICATES, FactCandidate, PerceptionResult, slug_uid
from ah_memory.relation_normalizer import ExactNormalizer, RelationNormalizer
from ah_memory.relations import (
    Event,
    NodeRef,
    NormalizedRelation,
    RelationContext,
    RelationProperties,
    canonicalize_label,
)
from ah_memory.state_engine import StateEngine, default_state_engine
from ah_memory.store import AHStore
from ah_memory.types import AssocLink, LinkId, Section


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class IngestReport:
    """Что получилось после apply: созданные факторы, seeds, ошибки."""

    created_n: list[str] = field(default_factory=list)
    seed_uids: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    perception: dict = field(default_factory=dict)


def _label_from_uid(uid: str) -> str:
    bare = uid[2:] if uid.startswith("M_") else uid
    return bare.replace("_", " ").lower()


class Transform:
    def __init__(
        self,
        store: AHStore,
        hp: HyperParams | None = None,
        *,
        relation_normalizer: RelationNormalizer | None = None,
        parameter_generator: FactorParameterGenerator | None = None,
        state_engine: StateEngine | None = None,
        identity=None,
    ) -> None:
        self.store = store
        self.hp = hp or HyperParams()
        self.relation_normalizer = relation_normalizer or RelationNormalizer(
            store.relations,
            (ExactNormalizer(),),
        )
        self.parameter_generator = (
            parameter_generator or RuleBasedParameterGenerator()
        )
        self.state_engine = state_engine or default_state_engine()
        self.identity = identity

    def apply(self, perception: PerceptionResult, section: Section = Section.C) -> IngestReport:
        del section  # open events не кладутся в C/P/H
        report = IngestReport()
        compound_heads = self._compound_role_heads(perception)
        for tok in perception.seed_tokens:
            tok_slug = slug_uid(tok)
            if tok_slug in compound_heads:
                continue
            uid = self._resolve_bare(tok)
            if uid in PREDICATES:
                continue
            m_uid = self._m_uid(uid)
            surface = tok.lower().replace("ё", "е")
            self.store.ensure_abstract(uid, {surface})
            self.store.ensure_m(m_uid, _label_from_uid(uid))
            if self.identity is not None:
                self.identity.attach_alias(uid, surface)
            report.seed_uids.append(uid)
            report.seed_uids.append(m_uid)

        for cand in perception.candidates:
            try:
                normalized = self._normalize_candidate(cand)
                factor_uid = self._record_semantic(cand, normalized)
                report.created_n.append(factor_uid)
            except Exception as exc:  # noqa: BLE001
                report.skipped.append(f"{cand.predicate}:{exc}")

        report.perception = perception.to_graph_json()
        return report

    def _normalize_candidate(self, candidate: FactCandidate) -> NormalizedRelation:
        raw_relation = candidate.raw_relation or candidate.predicate
        canonical_hint = candidate.canonical_relation or candidate.predicate
        known = self.store.get_relation(canonical_hint)
        if known is not None:
            return NormalizedRelation(
                raw_label=raw_relation,
                canonical_label=known.canonical_label,
                confidence=candidate.confidence,
                embedding=known.embedding,
                properties=known.properties,
                strategy="parser_hint",
            )
        if candidate.canonical_relation:
            return NormalizedRelation(
                raw_label=raw_relation,
                canonical_label=canonicalize_label(candidate.canonical_relation),
                confidence=candidate.confidence,
                properties=RelationProperties(directional=True),
                strategy="parser_new_relation",
                created=True,
            )
        context = RelationContext(
            text=candidate.raw_span or "",
            subject_uid=candidate.roles.get("SUBJECT"),
            object_uid=candidate.roles.get("OBJECT"),
            roles=dict(candidate.roles),
        )
        return self.relation_normalizer.normalize(raw_relation, context)

    def _record_semantic(
        self,
        candidate: FactCandidate,
        normalized: NormalizedRelation,
    ) -> str:
        relation = self.store.get_relation(normalized.canonical_label)
        if relation is None:
            relation = self.store.register_relation(
                normalized.to_relation(arity=max(1, len(candidate.roles)))
            )
        arguments = {
            role.upper(): NodeRef(
                uid=self._resolve_value(
                    value,
                    context=candidate.raw_span or candidate.raw_relation or "",
                ),
                role=role.upper(),
            )
            for role, value in candidate.roles.items()
        }
        if len(arguments) < 2:
            raise ValueError("open relation factor requires at least two arguments")
        event_uid = self.store.new_uid("E")
        added_at = _now_iso()
        created_tau = self.store.ah.tau
        event = Event(
            uid=event_uid,
            predicate=relation,
            arguments=arguments,
            timestamp=arguments.get("TIME").uid if "TIME" in arguments else None,
            confidence=candidate.confidence,
            raw_span=candidate.raw_span,
            metadata={
                "raw_relation": candidate.raw_relation or candidate.predicate,
                "normalization": normalized.strategy,
                "normalization_confidence": normalized.confidence,
                "added_at": added_at,
                "created_tau": created_tau,
            },
        )
        self.store.add_event(event)
        variables = list(
            dict.fromkeys(reference.uid for reference in arguments.values())
        )
        parameters = self.parameter_generator.generate(relation)
        factor_uid = f"SF::{event_uid}"
        factor = Factor(
            fid=factor_uid,
            kind=FactorKind.HYPER,
            variables=variables,
            w=self.hp.initial_w,
            roles={role: reference.uid for role, reference in arguments.items()},
            potential_key="semantic",
            source_uid=event_uid,
            relation=relation,
            parameters=parameters,
            confidence=candidate.confidence,
            embedding=relation.embedding,
            metadata={
                "event_uid": event_uid,
                "raw_relation": candidate.raw_relation or candidate.predicate,
                "canonical_relation": relation.canonical_label,
                "source_variable": arguments.get(
                    "SUBJECT",
                    next(iter(arguments.values())),
                ).uid,
                "added_at": added_at,
                "created_tau": created_tau,
            },
        )
        self.store.add_semantic_factor(factor)
        self._weave_assoc_mesh(variables, w=self.hp.initial_w)
        self._bind_subject_surface(candidate.roles.get("SUBJECT"))
        next_state = self.state_engine.apply(self.store.state, event)
        self.store.state = next_state
        self.store.state_transitions.extend(
            transition.to_dict()
            for transition in self.state_engine.last_transitions
        )
        return factor_uid

    def _bind_subject_surface(self, subj: str | None) -> None:
        if not subj:
            return
        s_uid = slug_uid(subj[2:] if subj.startswith("M_") else subj)
        m_uid = self._m_uid(s_uid)
        if s_uid not in self.store.ah.S:
            return
        if any(
            l.id in {LinkId.ASSOC.value, LinkId.BIND.value}
            and {l.e1.target_uid, l.e2.target_uid} == {m_uid, s_uid}
            for l in self.store.ah.L.values()
        ):
            return
        self.store.add_link(
            AssocLink(
                uid=self.store.new_uid("L_BIND"),
                id=LinkId.BIND.value,
                w=1.0,
                e1=self.store.m_ref(m_uid),
                e2=self.store.s_ref(s_uid),
            )
        )

    def _weave_assoc_mesh(self, m_uids: list[str], *, w: float) -> None:
        """ASSOC между актантами одного факта (микротема для ignition)."""
        uniq: list[str] = []
        seen: set[str] = set()
        for u in m_uids:
            if not u.startswith("M_") or u in seen:
                continue
            if u not in self.store.ah.all_hyper():
                continue
            seen.add(u)
            uniq.append(u)
        if len(uniq) < 2:
            return
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                ends = {a, b}
                if any(
                    l.id == LinkId.ASSOC.value
                    and {l.e1.target_uid, l.e2.target_uid} == ends
                    for l in self.store.ah.L.values()
                ):
                    continue
                self.store.add_link(
                    AssocLink(
                        uid=self.store.new_uid("L_MESH"),
                        id=LinkId.ASSOC.value,
                        w=w,
                        e1=self.store.m_ref(a),
                        e2=self.store.m_ref(b),
                    )
                )

    def _compound_role_heads(self, perception: PerceptionResult) -> set[str]:
        """Head token of multi-word role fillers — не материализовать отдельный seed."""
        heads: set[str] = set()
        for cand in perception.candidates:
            for val in cand.roles.values():
                raw = val[2:] if str(val).startswith("M_") else str(val)
                norm = raw.lower().replace("ё", "е").strip()
                parts = [p for p in re.split(r"[\s_]+", norm) if p]
                if len(parts) >= 2:
                    heads.add(slug_uid(parts[0]))
        return heads

    def _resolve_bare(self, value: str, context: str | None = None) -> str:
        raw = value[2:] if str(value).startswith("M_") else str(value)
        if self.identity is not None and self.identity.enabled:
            hit = self.identity.resolve_bare_uid(raw, context=context)
            if hit:
                return hit if not hit.startswith("M_") else hit[2:]
        return slug_uid(raw)

    def _resolve_value(self, value: str, context: str | None = None) -> str:
        raw = value[2:] if str(value).startswith("M_") else str(value)
        uid = self._resolve_bare(raw, context=context)
        m_uid = self._m_uid(uid)
        surface = raw.lower().replace("ё", "е")
        self.store.ensure_abstract(uid, {surface})
        self.store.ensure_m(m_uid, _label_from_uid(uid))
        if self.identity is not None:
            self.identity.attach_alias(uid, surface)
        return m_uid

    @staticmethod
    def _m_uid(token: str) -> str:
        t = slug_uid(token)
        if t.startswith("M_"):
            return t
        return f"M_{t}"
