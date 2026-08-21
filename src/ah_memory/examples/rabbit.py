"""Эталон «заяц» для M4-бенчмарка: явные FactCandidate, без шаблонов T/N."""
from __future__ import annotations

from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform
from ah_memory.types import AssocLink, LinkId, Property, SecondOrderSymbol, Section


RABBIT_TEXT = (
    "Заяц — маленький дикий зверёк, который обитает на лугу или в лесу. "
    "У него сильные задние лапы, поэтому бегает он очень быстро. "
    "Уши зайца длинные, а хвост — круглый и пушистый. "
    "Летом шерсть зайца коричневого цвета, а зимой — белого."
)

EXPECTED_FACT_KEYS = [
    "IS:HARE:BEAST",
    "LIVE_IN:HARE",
    "HAVE:HARE:HIND_LEG",
    "RUN:HARE",
    "HAVE:HARE:EAR",
    "HAVE:HARE:TAIL",
    "BE_COLORED:FUR:BROWN",
    "BE_COLORED:FUR:WHITE",
]

_RABBIT_CANDIDATES = [
    FactCandidate("IS", {"SUBJECT": "HARE", "OBJECT": "BEAST"}),
    FactCandidate("IS", {"SUBJECT": "HARE", "OBJECT": "SMALL"}),
    FactCandidate("LIVE_IN", {"SUBJECT": "HARE", "LOCATION": "MEADOW"}),
    FactCandidate("LIVE_IN", {"SUBJECT": "HARE", "LOCATION": "FOREST"}),
    FactCandidate("HAVE", {"SUBJECT": "HARE", "OBJECT": "HIND_LEG"}),
    FactCandidate("RUN", {"SUBJECT": "HARE", "HOW-TO": "VERY_FAST"}),
    FactCandidate("HAVE", {"SUBJECT": "HARE", "OBJECT": "EAR"}),
    FactCandidate("HAVE", {"SUBJECT": "HARE", "OBJECT": "TAIL"}),
    FactCandidate("BE_COLORED", {"SUBJECT": "FUR", "OBJECT": "BROWN", "TIME": "SUMMER"}),
    FactCandidate("BE_COLORED", {"SUBJECT": "FUR", "OBJECT": "WHITE", "TIME": "WINTER"}),
]

_LABELS = {
    "HARE": "Заяц",
    "BEAST": "зверёк",
    "SMALL": "маленький",
    "MEADOW": "луг",
    "FOREST": "лес",
    "HIND_LEG": "задние лапы",
    "VERY_FAST": "очень быстро",
    "EAR": "уши",
    "TAIL": "хвост",
    "FUR": "шерсть",
    "BROWN": "коричневый",
    "WHITE": "белый",
    "SUMMER": "лето",
    "WINTER": "зима",
    "ANIMAL": "Животное",
}


def build_rabbit_memory() -> AHStore:
    store = AHStore()
    for uid, label in _LABELS.items():
        store.ensure_abstract(uid, {label.lower(), uid.lower()})
        store.ensure_m(f"M_{uid}", label)

    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=list(_RABBIT_CANDIDATES),
            seed_tokens=list(_LABELS.keys()),
        ),
        section=Section.C,
    )

    if not any(l.id == LinkId.IS_A.value and l.e1.target_uid == "M_HARE" for l in store.ah.L.values()):
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_ISA"),
                id=LinkId.IS_A.value,
                w=0.9,
                e1=store.m_ref("M_HARE"),
                e2=store.m_ref("M_BEAST"),
            )
        )
    if not any(l.id == LinkId.IS_A.value and l.e1.target_uid == "M_BEAST" for l in store.ah.L.values()):
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_ISA"),
                id=LinkId.IS_A.value,
                w=0.9,
                e1=store.m_ref("M_BEAST"),
                e2=store.m_ref("M_ANIMAL"),
            )
        )

    if "EP_LESSON_1" not in store.ah.H:
        store.add_element(
            Section.H,
            SecondOrderSymbol(
                uid="EP_LESSON_1",
                Pr=[Property(name="label", value="Урок: кто такой заяц")],
                Mt=[Property(name="kind", value="Episode")],
            ),
        )
        store.add_element(
            Section.H,
            SecondOrderSymbol(
                uid="EP_LESSON_2",
                Pr=[Property(name="label", value="Урок: линька")],
                Mt=[Property(name="kind", value="Episode")],
            ),
        )
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_FOLLOW"),
                id=LinkId.FOLLOW.value,
                w=0.8,
                e1=store.m_ref("EP_LESSON_1"),
                e2=store.m_ref("EP_LESSON_2"),
            )
        )
    return store


def extracted_fact_keys(store: AHStore) -> set[str]:
    keys: set[str] = set()
    for factor in store.list_semantic_factors():
        pred = (
            factor.relation.canonical_label.upper()
            if factor.relation is not None
            else ""
        )
        subj = (factor.roles.get("SUBJECT") or "").replace("M_", "")
        obj = (factor.roles.get("OBJECT") or "").replace("M_", "")
        if pred in {"IS", "IS_A"} and subj and obj:
            keys.add(f"IS:{subj}:{obj}")
        if pred in {"LIVE_IN", "LIVEIN"} and subj:
            keys.add(f"LIVE_IN:{subj}")
        if pred == "HAVE" and subj and obj:
            keys.add(f"HAVE:{subj}:{obj}")
        if pred == "RUN" and subj:
            keys.add(f"RUN:{subj}")
        if pred in {"BE_COLORED", "COLORED", "HAS_COLOR"} and subj and obj:
            keys.add(f"BE_COLORED:{subj}:{obj}")
    return keys


def rabbit_auto_score(store: AHStore) -> tuple[int, int]:
    got = extracted_fact_keys(store)
    hit = sum(1 for k in EXPECTED_FACT_KEYS if k in got)
    return hit, len(EXPECTED_FACT_KEYS)


def syntactic_answer_who_is_hare(store: AHStore) -> str:
    from ah_memory.dsl import DSLInterpreter

    return str(DSLInterpreter(store).execute("answer_who(M_HARE)").value)
