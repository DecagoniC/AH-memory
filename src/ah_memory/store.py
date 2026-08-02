"""Full AHStore operations (monograph §4 table 3)."""
from __future__ import annotations

from typing import Any, Iterable

from ah_memory.types import (
    AH,
    AbstractSymbol,
    AssocLink,
    ElementList,
    ElementRef,
    FunctionalSymbol,
    HyperElement,
    Hyperlink,
    MRef,
    Property,
    Role,
    SRef,
    SecondOrderSymbol,
    Section,
    Template,
)


class AHError(Exception):
    pass


class AHStore:
    def __init__(self, ah: AH | None = None) -> None:
        self.ah = ah or AH()
        self._ref_seq = 0
        self._uid_seq = 0

    def clear(self) -> None:
        """Wipe all AH sets (in-place)."""
        self.ah = AH()
        self._ref_seq = 0
        self._uid_seq = 0

    def new_uid(self, prefix: str) -> str:
        self._uid_seq += 1
        return f"{prefix}_{self._uid_seq:05d}"

    def _next_ref(self, prefix: str = "REF") -> str:
        self._ref_seq += 1
        return f"{prefix}{self._ref_seq:06d}"

    def s_ref(self, target_uid: str, ref_uid: str | None = None) -> SRef:
        return SRef(target_uid=target_uid, ref_uid=ref_uid or self._next_ref("S"))

    def m_ref(self, target_uid: str, ref_uid: str | None = None) -> MRef:
        return MRef(target_uid=target_uid, ref_uid=ref_uid or self._next_ref("M"))

    def add_abstract_symbol(self, s: AbstractSymbol) -> AbstractSymbol:
        if s.uid in self.ah.S:
            raise AHError(f"abstract symbol already exists: {s.uid}")
        if not s.modality_partition_ok():
            raise AHError(f"R modalities intersect for {s.uid}")
        s.created_tau = self.ah.tau
        self.ah.S[s.uid] = s
        return s

    def edit_abstract_symbol(self, uid: str, s: AbstractSymbol) -> AbstractSymbol:
        if uid not in self.ah.S:
            raise AHError(f"abstract symbol not found: {uid}")
        if s.uid != uid:
            raise AHError("editAbstractSymbol: UID mismatch")
        if not s.modality_partition_ok():
            raise AHError(f"R modalities intersect for {s.uid}")
        self.ah.S[uid] = s
        return s

    def add_element(self, section: Section, e: HyperElement) -> HyperElement:
        bucket = self.ah.section(section)
        uid = getattr(e, "uid", None)
        if uid is None:
            raise AHError("element must have uid")
        if uid in bucket:
            raise AHError(f"element already exists in {section.value}: {uid}")
        if hasattr(e, "created_tau"):
            e.created_tau = self.ah.tau  # type: ignore[attr-defined]
        self._check_property_unique(e)
        bucket[uid] = e
        return e

    def edit_element(self, section: Section, uid: str, e: HyperElement) -> HyperElement:
        bucket = self.ah.section(section)
        if uid not in bucket:
            raise AHError(f"element not found in {section.value}: {uid}")
        if getattr(e, "uid", None) != uid:
            raise AHError("editElement: UID mismatch")
        self._check_property_unique(e)
        bucket[uid] = e
        return e

    def add_property(self, uid: str, prop: Property, *, meta: bool = False) -> HyperElement:
        e = self._find_anywhere(uid)
        attrs = getattr(e, "Mt" if meta else "Pr", None)
        if attrs is None:
            raise AHError(f"element has no properties: {uid}")
        names = {p.name for p in attrs}
        if prop.name in names:
            raise AHError(f"property exists: {uid}.{prop.name}")
        attrs.append(prop)
        self._check_property_unique(e)
        return e

    def edit_property(self, uid: str, prop: Property, *, meta: bool = False) -> HyperElement:
        e = self._find_anywhere(uid)
        attrs = getattr(e, "Mt" if meta else "Pr", None)
        if attrs is None:
            raise AHError(f"element has no properties: {uid}")
        for i, p in enumerate(attrs):
            if p.name == prop.name:
                attrs[i] = prop
                return e
        raise AHError(f"property not found: {uid}.{prop.name}")

    def add_link(self, link: AssocLink) -> AssocLink:
        if link.uid in self.ah.L:
            raise AHError(f"link already exists: {link.uid}")
        self.ah.L[link.uid] = link
        return link

    def get_abstract_symbol(self, uid: str) -> AbstractSymbol:
        try:
            return self.ah.S[uid]
        except KeyError as exc:
            raise AHError(f"abstract symbol not found: {uid}") from exc

    def find_abstract_symbols(self, query: str = "") -> list[AbstractSymbol]:
        q = query.lower()
        out: list[AbstractSymbol] = []
        for s in self.ah.S.values():
            if not q or q in s.uid.lower() or any(q in f for forms in s.R.values() for f in forms):
                out.append(s)
        return out

    def get_s_reference(self, ref_uid: str) -> SRef:
        for e in self.ah.all_hyper().values():
            for ref in self._iter_refs(e):
                if isinstance(ref, SRef) and ref.ref_uid == ref_uid:
                    return ref
        for link in self.ah.L.values():
            for ref in (link.e1, link.e2):
                if isinstance(ref, SRef) and ref.ref_uid == ref_uid:
                    return ref
        raise AHError(f"SRef not found: {ref_uid}")

    def find_s_references(self, target_uid: str) -> list[SRef]:
        found: list[SRef] = []
        for e in self.ah.all_hyper().values():
            for ref in self._iter_refs(e):
                if isinstance(ref, SRef) and ref.target_uid == target_uid:
                    found.append(ref)
        for link in self.ah.L.values():
            for ref in (link.e1, link.e2):
                if isinstance(ref, SRef) and ref.target_uid == target_uid:
                    found.append(ref)
        return found

    def get_m_reference(self, ref_uid: str) -> MRef:
        for e in self.ah.all_hyper().values():
            for ref in self._iter_refs(e):
                if isinstance(ref, MRef) and ref.ref_uid == ref_uid:
                    return ref
        for link in self.ah.L.values():
            for ref in (link.e1, link.e2):
                if isinstance(ref, MRef) and ref.ref_uid == ref_uid:
                    return ref
        raise AHError(f"MRef not found: {ref_uid}")

    def find_m_references(self, target_uid: str) -> list[MRef]:
        found: list[MRef] = []
        for e in self.ah.all_hyper().values():
            for ref in self._iter_refs(e):
                if isinstance(ref, MRef) and ref.target_uid == target_uid:
                    found.append(ref)
        for link in self.ah.L.values():
            for ref in (link.e1, link.e2):
                if isinstance(ref, MRef) and ref.target_uid == target_uid:
                    found.append(ref)
        return found

    def get_symbol(self, uid: str) -> SecondOrderSymbol:
        e = self._find_anywhere(uid)
        if not isinstance(e, SecondOrderSymbol):
            raise AHError(f"not a second-order symbol: {uid}")
        return e

    def find_symbols(self, query: str = "") -> list[SecondOrderSymbol]:
        q = query.lower()
        out: list[SecondOrderSymbol] = []
        for e in self.ah.all_hyper().values():
            if not isinstance(e, SecondOrderSymbol):
                continue
            labels = " ".join(p.value for p in e.Pr if p.name == "label").lower()
            if not q or q in e.uid.lower() or q in labels:
                out.append(e)
        return out

    def get_list(self, uid: str) -> ElementList:
        e = self._find_anywhere(uid)
        if not isinstance(e, ElementList):
            raise AHError(f"not a list: {uid}")
        return e

    def find_lists(self, kind: str | None = None) -> list[ElementList]:
        out: list[ElementList] = []
        for e in self.ah.all_hyper().values():
            if not isinstance(e, ElementList):
                continue
            if kind is None:
                out.append(e)
                continue
            if any(p.name == "kind" and p.value == kind for p in e.Mt):
                out.append(e)
        return out

    def get_template(self, uid: str) -> Template:
        e = self._find_anywhere(uid)
        if not isinstance(e, Template):
            raise AHError(f"not a template: {uid}")
        return e

    def find_templates(self) -> list[Template]:
        return [e for e in self.ah.all_hyper().values() if isinstance(e, Template)]

    def get_hypernode(self, uid: str) -> Hyperlink:
        e = self._find_anywhere(uid)
        if not isinstance(e, Hyperlink):
            raise AHError(f"not a hyperlink: {uid}")
        return e

    def find_hypernodes(self) -> list[Hyperlink]:
        return [e for e in self.ah.all_hyper().values() if isinstance(e, Hyperlink)]

    def find_roles(self, role: str | Role, value: str) -> list[Hyperlink]:
        role_e = Role(role) if not isinstance(role, Role) else role
        found: list[Hyperlink] = []
        for e in self.ah.all_hyper().values():
            if not isinstance(e, Hyperlink):
                continue
            filler = e.fillers.get(role_e)
            if filler is not None and filler.target_uid == value:
                found.append(e)
        return found

    def get_link(self, uid: str) -> AssocLink:
        try:
            return self.ah.L[uid]
        except KeyError as exc:
            raise AHError(f"link not found: {uid}") from exc

    def find_links(self, e: ElementRef | str) -> list[AssocLink]:
        target = e.target_uid if isinstance(e, (SRef, MRef)) else e
        out: list[AssocLink] = []
        for link in self.ah.L.values():
            ends = {link.e1.target_uid, link.e2.target_uid}
            if target in ends:
                out.append(link)
        return out

    def ensure_abstract(self, uid: str, forms: set[str] | None = None) -> AbstractSymbol:
        if uid in self.ah.S:
            return self.ah.S[uid]
        return self.add_abstract_symbol(
            AbstractSymbol(uid=uid, R={"TEXT": forms or {uid.lower()}})
        )

    def ensure_m(self, uid: str, label: str | None = None, section: Section = Section.C) -> SecondOrderSymbol:
        try:
            e = self._find_anywhere(uid)
            if isinstance(e, SecondOrderSymbol):
                return e
        except AHError:
            pass
        return self.add_element(  # type: ignore[return-value]
            section,
            SecondOrderSymbol(
                uid=uid,
                Pr=[Property(name="label", value=label or uid)],
            ),
        )

    def remove_element(self, uid: str) -> bool:
        for bucket in (self.ah.C, self.ah.P, self.ah.H):
            if uid in bucket:
                del bucket[uid]
                return True
        if uid in self.ah.S:
            del self.ah.S[uid]
            return True
        return False

    def remove_link(self, uid: str) -> bool:
        if uid in self.ah.L:
            del self.ah.L[uid]
            return True
        return False

    def graph_size(self) -> int:
        return len(self.ah.C) + len(self.ah.P) + len(self.ah.H) + len(self.ah.L)

    def get_x(self, uid: str) -> float:
        if uid in self.ah.S:
            return self.ah.S[uid].x
        e = self._find_anywhere(uid)
        return float(getattr(e, "x", 0.0))

    def set_x(self, uid: str, x: float) -> None:
        if uid in self.ah.S:
            self.ah.S[uid].x = x
            return
        e = self._find_anywhere(uid)
        if hasattr(e, "x"):
            e.x = x  # type: ignore[attr-defined]

    def all_activatable_uids(self) -> list[str]:
        uids = list(self.ah.S.keys())
        for e in self.ah.all_hyper().values():
            uid = getattr(e, "uid", None)
            if uid is not None and hasattr(e, "x"):
                uids.append(uid)
        return uids

    def _find_anywhere(self, uid: str) -> HyperElement:
        for bucket in (self.ah.C, self.ah.P, self.ah.H):
            if uid in bucket:
                return bucket[uid]
        raise AHError(f"element not found: {uid}")

    def _find_section(self, uid: str) -> Section | None:
        for sec in Section:
            if uid in self.ah.section(sec):
                return sec
        return None

    @staticmethod
    def _iter_refs(e: HyperElement) -> Iterable[ElementRef]:
        if isinstance(e, Hyperlink):
            yield e.template
            yield from e.fillers.values()
        elif isinstance(e, Template):
            yield e.predicate
        elif isinstance(e, FunctionalSymbol):
            yield from e.operands
        elif isinstance(e, ElementList):
            yield from e.items

    @staticmethod
    def _check_property_unique(e: HyperElement) -> None:
        names: list[str] = []
        for attr in ("Pr", "Mt"):
            props = getattr(e, attr, None)
            if props:
                names.extend(p.name for p in props)
        if len(names) != len(set(names)):
            raise AHError(f"duplicate property name in {getattr(e, 'uid', '?')}")


def iter_is_a_edges(store: AHStore) -> Iterable[tuple[str, str]]:
    from ah_memory.types import LinkId

    for link in store.ah.L.values():
        if link.id == LinkId.IS_A.value:
            yield link.e1.target_uid, link.e2.target_uid


def iter_follow_edges(store: AHStore) -> Iterable[tuple[str, str]]:
    from ah_memory.types import LinkId

    for link in store.ah.L.values():
        if link.id == LinkId.FOLLOW.value:
            yield link.e1.target_uid, link.e2.target_uid
