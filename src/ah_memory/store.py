"""AHStore — CRUD над AH (монография §4, таблица 3).

Читать после types.py. Здесь нет perception/LLM: только создать/найти/связать.
Поверх AH лежат open-semantics слои: RelationRegistry, events, semantic_factors, State.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ah_memory.relation_registry import (
    RelationRegistry,
    default_relation_registry,
)
from ah_memory.relations import Event, Relation
from ah_memory.state_engine import State
from ah_memory.types import (
    AH,
    AbstractSymbol,
    AssocLink,
    ElementRef,
    HyperElement,
    MRef,
    Property,
    SRef,
    SecondOrderSymbol,
    Section,
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class AHError(Exception):
    pass


class AHStore:
    # ── Состояние хранилища ──────────────────────────────────────────────────
    # Зачем: ah — классический граф S/C/P/H/L; relations/events/factors —
    # параллельный слой open relations (то, что пишет Transform для новых предикатов).

    def __init__(self, ah: AH | None = None) -> None:
        self.ah = ah or AH()
        self._uid_seq = 0
        self._s_refs: dict[str, SRef] = {}
        self._m_refs: dict[str, MRef] = {}
        self.relations: RelationRegistry = default_relation_registry()
        self.semantic_factors: dict[str, Any] = {}
        self.events: dict[str, Event] = {}
        self.state = State()
        self.state_transitions: list[dict[str, Any]] = []

    def clear(self) -> None:
        """Полный сброс графа и open-semantics (in-place)."""
        self.ah = AH()
        self._uid_seq = 0
        self._s_refs = {}
        self._m_refs = {}
        self.relations = default_relation_registry()
        self.semantic_factors = {}
        self.events = {}
        self.state = State()
        self.state_transitions = []

    def replace_from(self, other: "AHStore") -> None:
        """Adopt another store's contents while keeping this object identity."""
        self.ah = other.ah
        self._uid_seq = other._uid_seq
        self._s_refs = other._s_refs
        self._m_refs = other._m_refs
        self.relations = other.relations
        self.semantic_factors = other.semantic_factors
        self.events = other.events
        self.state = other.state
        self.state_transitions = other.state_transitions

    def _touch_structure(self) -> None:
        # Зачем: UI/кэши могут инвалидироваться по revision без полного diff.
        self.ah.revision += 1

    def _stamp_creation(self, obj: Any) -> None:
        """Проставить τ (тик AH) и wall-clock added_at на новый символ."""
        iso = _now_iso()
        if hasattr(obj, "created_tau"):
            obj.created_tau = self.ah.tau
        if hasattr(obj, "added_at") and not getattr(obj, "added_at", ""):
            obj.added_at = iso

    def get_added_at(self, uid: str) -> tuple[str, int | None]:
        """Return (added_at ISO or '', created_tau or None) for a symbol."""
        bare = uid[2:] if str(uid).startswith("M_") else str(uid)
        if bare in self.ah.S:
            s = self.ah.S[bare]
            return (getattr(s, "added_at", "") or "", int(s.created_tau))
        try:
            e = self._find_anywhere(uid if str(uid).startswith("M_") else f"M_{bare}")
        except AHError:
            try:
                e = self._find_anywhere(bare)
            except AHError:
                return ("", None)
        added = getattr(e, "added_at", "") or ""
        tau = getattr(e, "created_tau", None)
        return (added, int(tau) if tau is not None else None)

    # ── UID / ссылки ─────────────────────────────────────────────────────────
    # Зачем: new_uid — для автогенерируемых N/E/L; s_ref/m_ref — один экземпляр
    # ссылки на target (все рёбра на один UID разделяют один SRef/MRef).

    def new_uid(self, prefix: str) -> str:
        self._uid_seq += 1
        return f"{prefix}_{self._uid_seq:05d}"

    def s_ref(self, target_uid: str) -> SRef:
        ref = self._s_refs.get(target_uid)
        if ref is None:
            ref = SRef(target_uid=target_uid)
            self._s_refs[target_uid] = ref
        return ref

    def m_ref(self, target_uid: str) -> MRef:
        ref = self._m_refs.get(target_uid)
        if ref is None:
            ref = MRef(target_uid=target_uid)
            self._m_refs[target_uid] = ref
        return ref

    def _canon_ref(self, ref: ElementRef) -> ElementRef:
        if isinstance(ref, SRef):
            return self.s_ref(ref.target_uid)
        if isinstance(ref, MRef):
            return self.m_ref(ref.target_uid)
        return ref

    # ── Базовый CRUD (таблица операций §4) ───────────────────────────────────

    def add_abstract_symbol(self, s: AbstractSymbol) -> AbstractSymbol:
        if s.uid in self.ah.S:
            raise AHError(f"abstract symbol already exists: {s.uid}")
        if not s.modality_partition_ok():
            raise AHError(f"R modalities intersect for {s.uid}")
        self._stamp_creation(s)
        self.ah.S[s.uid] = s
        self._touch_structure()
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
        self._stamp_creation(e)
        self._check_property_unique(e)
        bucket[uid] = e
        self._touch_structure()
        return e

    def edit_element(self, section: Section, uid: str, e: HyperElement) -> HyperElement:
        bucket = self.ah.section(section)
        if uid not in bucket:
            raise AHError(f"element not found in {section.value}: {uid}")
        if getattr(e, "uid", None) != uid:
            raise AHError("editElement: UID mismatch")
        self._check_property_unique(e)
        bucket[uid] = e
        self._touch_structure()
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
        link.e1 = self._canon_ref(link.e1)  # type: ignore[assignment]
        link.e2 = self._canon_ref(link.e2)  # type: ignore[assignment]
        self._stamp_creation(link)
        self.ah.L[link.uid] = link
        self._touch_structure()
        return link

    # ── Open semantics (параллельно классическому N/T) ───────────────────────
    # Зачем: LLM может выдать любой predicate; registry хранит канон + свойства,
    # Event — конкретное событие, semantic_factor — узел для belief propagation.

    def register_relation(self, relation: Relation) -> Relation:
        return self.relations.register_relation(relation)

    def get_relation(self, canonical_label: str) -> Relation | None:
        return self.relations.get_relation(canonical_label)

    def list_relations(self) -> tuple[Relation, ...]:
        return self.relations.list_relations()

    def find_similar_relations(
        self,
        embedding: Iterable[float],
        *,
        limit: int = 5,
        min_similarity: float = -1.0,
    ) -> list[tuple[Relation, float]]:
        return self.relations.find_similar_relations(
            tuple(embedding),
            limit=limit,
            min_similarity=min_similarity,
        )

    def add_semantic_factor(self, factor: Any) -> Any:
        uid = str(getattr(factor, "uid", getattr(factor, "fid", "")))
        if not uid:
            raise AHError("semantic factor must have uid")
        if uid in self.semantic_factors:
            return self.semantic_factors[uid]
        self.semantic_factors[uid] = factor
        self._touch_structure()
        return factor

    def list_semantic_factors(self) -> tuple[Any, ...]:
        return tuple(self.semantic_factors.values())

    def add_event(self, event: Event) -> Event:
        if event.uid in self.events:
            return self.events[event.uid]
        self.events[event.uid] = event
        return event

    def list_events(self) -> tuple[Event, ...]:
        return tuple(self.events.values())

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

    def find_symbols_by_kind(self, kind: str | None = None) -> list[SecondOrderSymbol]:
        """M с Mt kind=… (например Episode)."""
        out: list[SecondOrderSymbol] = []
        for e in self.ah.all_hyper().values():
            if not isinstance(e, SecondOrderSymbol):
                continue
            if kind is None:
                out.append(e)
                continue
            if any(p.name == "kind" and p.value == kind for p in e.Mt):
                out.append(e)
        return out

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

    # ── Ingest-friendly helpers ──────────────────────────────────────────────
    # Зачем: Transform зовёт именно их. ensure_abstract мержит новые словоформы
    # в TEXT (важно для identity); ensure_m создаёт сущность с label, если нет.

    def ensure_abstract(self, uid: str, forms: set[str] | None = None) -> AbstractSymbol:
        if uid in self.ah.S:
            if forms:
                self.ah.S[uid].R.setdefault("TEXT", set()).update(
                    f.lower().replace("ё", "е") for f in forms if f
                )
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
                self._touch_structure()
                return True
        if uid in self.ah.S:
            del self.ah.S[uid]
            self._touch_structure()
            return True
        return False

    def remove_link(self, uid: str) -> bool:
        if uid in self.ah.L:
            del self.ah.L[uid]
            self._touch_structure()
            return True
        return False

    def graph_size(self) -> int:
        return (
            len(self.ah.C)
            + len(self.ah.P)
            + len(self.ah.H)
            + len(self.ah.L)
            + len(self.semantic_factors)
        )

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
