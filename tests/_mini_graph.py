"""Minimal open-relation store for tests (no dog/rabbit demos)."""
from __future__ import annotations

from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform
from ah_memory.types import AssocLink, LinkId, Property, SecondOrderSymbol, Section


def build_mini_open_store() -> AHStore:
    store = AHStore()
    for uid, label in {
        "ENTITY": "сущность",
        "KIND": "вид",
        "PLACE": "место",
    }.items():
        store.ensure_abstract(uid, {label, uid.lower()})
        store.ensure_m(f"M_{uid}", label)

    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=[
                FactCandidate("IS", {"SUBJECT": "ENTITY", "OBJECT": "KIND"}),
                FactCandidate("LIVE_IN", {"SUBJECT": "ENTITY", "LOCATION": "PLACE"}),
            ],
            seed_tokens=["ENTITY", "KIND", "PLACE"],
        )
    )

    store.add_element(
        Section.H,
        SecondOrderSymbol(
            uid="EP_1",
            Pr=[Property(name="label", value="episode 1")],
            Mt=[Property(name="kind", value="Episode")],
        ),
    )
    store.add_element(
        Section.H,
        SecondOrderSymbol(
            uid="EP_2",
            Pr=[Property(name="label", value="episode 2")],
            Mt=[Property(name="kind", value="Episode")],
        ),
    )
    store.add_link(
        AssocLink(
            uid=store.new_uid("L_FOLLOW"),
            id=LinkId.FOLLOW.value,
            w=0.8,
            e1=store.m_ref("EP_1"),
            e2=store.m_ref("EP_2"),
        )
        )
    from ah_memory.graph_library import remember_fixture

    remember_fixture(
        "mini-open",
        store,
        source_text="Сущность — это вид. Сущность обитает в месте.",
    )
    return store
