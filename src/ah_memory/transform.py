"""Deterministic Transform: FactCandidate → AH ops. No domain hardcode."""
from __future__ import annotations

from dataclasses import dataclass, field

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
from ah_memory.templates import ensure_template
from ah_memory.types import AssocLink, Hyperlink, LinkId, Role, Section


PRED_TO_TEMPLATE = {
    "CREATE": "T_CREATE",
    "IS": "T_IS",
    "LIVE_IN": "T_LIVE_IN",
    "BE_BORN": "T_BE_BORN",
    "HAVE": "T_HAVE",
    "RUN": "T_RUN",
    "BE_COLORED": "T_COLOR",
    "CAUSE_EVENT": "T_CAUSE",
    "USE": "T_USE",
    "MOVE": "T_MOVE",
}


@dataclass
class IngestReport:
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

    def apply(self, perception: PerceptionResult, section: Section = Section.C) -> IngestReport:
        report = IngestReport()
        for tok in perception.seed_tokens:
            uid = slug_uid(tok)
            if uid in PREDICATES or uid in PRED_TO_TEMPLATE:
                continue
            m_uid = self._m_uid(uid)
            self.store.ensure_abstract(uid, {tok.lower() if tok == tok.lower() else uid.lower()})
            s = self.store.ah.S.get(uid)
            if s is not None:
                forms = s.R.setdefault("TEXT", set())
                forms.add(tok.lower())
            self.store.ensure_m(m_uid, _label_from_uid(uid))
            report.seed_uids.append(uid)
            report.seed_uids.append(m_uid)

        for cand in perception.candidates:
            try:
                normalized = self._normalize_candidate(cand)
                legacy_predicate = (
                    cand.predicate.upper()
                    if cand.predicate.upper() in PRED_TO_TEMPLATE
                    else normalized.canonical_label
                )
                if legacy_predicate in PRED_TO_TEMPLATE:
                    legacy_candidate = FactCandidate(
                        predicate=legacy_predicate,
                        roles=dict(cand.roles),
                        raw_span=cand.raw_span,
                        confidence=cand.confidence,
                        raw_relation=cand.raw_relation or cand.predicate,
                        canonical_relation=normalized.canonical_label,
                    )
                    n_uid = self._ingest_candidate(legacy_candidate, section)
                    self._record_semantic(
                        legacy_candidate,
                        normalized,
                        legacy_source_uid=n_uid,
                    )
                else:
                    n_uid = self._ingest_open_candidate(cand, normalized)
                report.created_n.append(n_uid)
            except Exception as exc:  # noqa: BLE001
                report.skipped.append(f"{cand.predicate}:{exc}")

        # mesh only within each N (see _ingest_candidate), not across bag-of-seeds
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

    def _ingest_open_candidate(
        self,
        candidate: FactCandidate,
        normalized: NormalizedRelation,
    ) -> str:
        return self._record_semantic(candidate, normalized)

    def _record_semantic(
        self,
        candidate: FactCandidate,
        normalized: NormalizedRelation,
        *,
        legacy_source_uid: str | None = None,
    ) -> str:
        relation = self.store.get_relation(normalized.canonical_label)
        if relation is None:
            relation = self.store.register_relation(
                normalized.to_relation(arity=max(1, len(candidate.roles)))
            )
        arguments = {
            role.upper(): NodeRef(
                uid=self._resolve_value(value),
                role=role.upper(),
            )
            for role, value in candidate.roles.items()
        }
        if len(arguments) < 2:
            raise ValueError("open relation factor requires at least two arguments")
        event_uid = self.store.new_uid("E")
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
                "legacy_source_uid": legacy_source_uid or "",
                "source_variable": arguments.get(
                    "SUBJECT",
                    next(iter(arguments.values())),
                ).uid,
            },
        )
        self.store.add_semantic_factor(factor)
        next_state = self.state_engine.apply(self.store.state, event)
        self.store.state = next_state
        self.store.state_transitions.extend(
            transition.to_dict()
            for transition in self.state_engine.last_transitions
        )
        return legacy_source_uid or factor_uid

    def _ingest_candidate(self, cand: FactCandidate, section: Section) -> str:
        pred = cand.predicate.upper()
        if pred not in PRED_TO_TEMPLATE:
            raise ValueError(f"unknown predicate {pred}")
        tpl_uid = ensure_template(self.store, pred)

        fillers: dict[Role, object] = {}
        for role_name, value in cand.roles.items():
            role = Role(role_name)
            target = self._resolve_value(value)
            fillers[role] = self.store.m_ref(target)

        tpl = self.store.get_template(tpl_uid)
        slot_roles = {a.role for a in tpl.actants}
        if Role.SUBJECT in slot_roles and Role.SUBJECT not in fillers:
            raise ValueError("SUBJECT required")
        if Role.OBJECT in slot_roles and Role.OBJECT not in fillers and pred not in {
            "LIVE_IN",
            "BE_BORN",
            "RUN",
            "MOVE",
        }:
            if Role.LOCATION not in fillers and Role.HOW_TO not in fillers:
                raise ValueError("OBJECT/LOCATION required")

        for existing in self.store.find_hypernodes():
            if existing.template.target_uid != tpl_uid:
                continue
            if {r.value: f.target_uid for r, f in existing.fillers.items()} == {
                r.value: f.target_uid for r, f in fillers.items()  # type: ignore[attr-defined]
            }:
                # still ensure mesh exists for older nodes
                self._weave_assoc_mesh(
                    [f.target_uid for f in existing.fillers.values()],
                    w=self.hp.initial_w,
                )
                return existing.uid

        n_uid = self.store.new_uid(f"N_{pred}")
        self.store.add_element(
            section,
            Hyperlink(
                uid=n_uid,
                w=self.hp.initial_w,
                template=self.store.m_ref(tpl_uid),
                fillers=fillers,  # type: ignore[arg-type]
            ),
        )
        actant_ms = [f.target_uid for f in fillers.values()]  # type: ignore[attr-defined]
        self._weave_assoc_mesh(actant_ms, w=self.hp.initial_w)

        subj = cand.roles.get("SUBJECT")
        if subj:
            s_uid = slug_uid(subj[2:] if subj.startswith("M_") else subj)
            m_uid = self._m_uid(s_uid)
            if s_uid in self.store.ah.S and not any(
                l.id in {LinkId.ASSOC.value, LinkId.BIND.value}
                and {l.e1.target_uid, l.e2.target_uid} == {m_uid, s_uid}
                for l in self.store.ah.L.values()
            ):
                self.store.add_link(
                    AssocLink(
                        uid=self.store.new_uid("L_BIND"),
                        id=LinkId.BIND.value,
                        w=1.0,
                        e1=self.store.m_ref(m_uid),
                        e2=self.store.s_ref(s_uid),
                    )
                )
        return n_uid

    def _weave_assoc_mesh(self, m_uids: list[str], *, w: float) -> None:
        """Complete ASSOC graph among distinct m-actants (shared micro-theme)."""
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

    def _resolve_value(self, value: str) -> str:
        raw = value[2:] if str(value).startswith("M_") else str(value)
        uid = slug_uid(raw)
        m_uid = self._m_uid(uid)
        self.store.ensure_abstract(uid, {raw.lower()})
        self.store.ensure_m(m_uid, _label_from_uid(uid))
        return m_uid

    @staticmethod
    def _m_uid(token: str) -> str:
        t = slug_uid(token)
        if t.startswith("M_"):
            return t
        return f"M_{t}"
