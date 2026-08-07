"""Deterministic Transform: FactCandidate → AH ops. No domain hardcode."""
from __future__ import annotations

from dataclasses import dataclass, field

from ah_memory.hyperparams import HyperParams
from ah_memory.perception import PREDICATES, FactCandidate, PerceptionResult, slug_uid
from ah_memory.store import AHStore
from ah_memory.templates import ensure_template
from ah_memory.types import AssocLink, Hyperlink, LinkId, Role, Section


PRED_TO_TEMPLATE = {
    "CREATE": "T_CREATE",
    "IS": "T_IS",
    "LIVE_IN": "T_LIVE_IN",
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
    def __init__(self, store: AHStore, hp: HyperParams | None = None) -> None:
        self.store = store
        self.hp = hp or HyperParams()

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
                n_uid = self._ingest_candidate(cand, section)
                report.created_n.append(n_uid)
            except Exception as exc:  # noqa: BLE001
                report.skipped.append(f"{cand.predicate}:{exc}")

        # mesh only within each N (see _ingest_candidate), not across bag-of-seeds
        report.perception = perception.to_graph_json()
        return report

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
                l.id == LinkId.ASSOC.value
                and {l.e1.target_uid, l.e2.target_uid} == {m_uid, s_uid}
                for l in self.store.ah.L.values()
            ):
                self.store.add_link(
                    AssocLink(
                        uid=self.store.new_uid("L_BIND"),
                        id=LinkId.ASSOC.value,
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
