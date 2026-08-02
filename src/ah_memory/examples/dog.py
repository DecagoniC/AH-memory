"""Ignition demo: dog case (monograph §8)."""
from __future__ import annotations

from ah_memory.ignition import ActivationSeed, IgnitionEngine
from ah_memory.store import AHStore
from ah_memory.templates import seed_templates
from ah_memory.types import (
    AssocLink,
    Hyperlink,
    LinkId,
    Property,
    Role,
    SecondOrderSymbol,
    Section,
)


def build_dog_memory() -> AHStore:
    store = AHStore()
    seed_templates(store)

    for uid, forms in {
        "DOG": {"собака", "собаки", "пёс"},
        "REX": {"рекс", "рекса"},
        "SCRUFF": {"загривок"},
        "PET": {"потрепать", "ласка"},
        "ANIMAL": {"животное"},
    }.items():
        store.ensure_abstract(uid, forms)

    store.ensure_m("M_DOG", "Собака")
    store.ensure_m("M_REX", "Рекс")
    store.ensure_m("M_SCRUFF", "Загривок")
    store.ensure_m("M_PET_SCRUFF", "Потрепать за загривок")
    store.ensure_m("M_ANIMAL", "Животное")

    # C: general dog knowledge; P: private model of Rex
    store.add_element(
        Section.C,
        Hyperlink(
            uid="N_DOG_IS_ANIMAL",
            w=0.8,
            template=store.m_ref("T_IS"),
            fillers={Role.SUBJECT: store.m_ref("M_DOG"), Role.OBJECT: store.m_ref("M_ANIMAL")},
        ),
    )
    store.add_element(
        Section.P,
        SecondOrderSymbol(
            uid="M_REX_MODEL",
            Pr=[Property(name="label", value="Модель Рекса")],
            Mt=[Property(name="likes", value="PET_SCRUFF")],
        ),
    )
    store.add_element(
        Section.P,
        Hyperlink(
            uid="N_REX_LIKES_PET",
            w=0.85,
            template=store.m_ref("T_USE"),
            fillers={
                Role.SUBJECT: store.m_ref("M_REX"),
                Role.OBJECT: store.m_ref("M_SCRUFF"),
                Role.TOOL: store.m_ref("M_PET_SCRUFF"),
                Role.PURPOSE: store.m_ref("M_REX_MODEL"),
            },
        ),
    )

    store.add_link(
        AssocLink(
            uid="L_ISA_REX_DOG",
            id=LinkId.IS_A.value,
            w=0.95,
            e1=store.m_ref("M_REX"),
            e2=store.m_ref("M_DOG"),
        )
    )
    store.add_link(
        AssocLink(
            uid="L_ISA_DOG_ANIMAL",
            id=LinkId.IS_A.value,
            w=0.9,
            e1=store.m_ref("M_DOG"),
            e2=store.m_ref("M_ANIMAL"),
        )
    )
    for m, s in [("M_DOG", "DOG"), ("M_REX", "REX"), ("M_SCRUFF", "SCRUFF")]:
        store.add_link(
            AssocLink(
                uid=f"L_BIND_{s}",
                id=LinkId.ASSOC.value,
                w=1.0,
                e1=store.m_ref(m),
                e2=store.s_ref(s),
            )
        )
    return store


def run_dog_ignition(ticks: int = 8) -> list[str]:
    """Seed DOG/REX sensory symbols and return WM evolution labels."""
    store = build_dog_memory()
    eng = IgnitionEngine(store)
    eng.seed(
        [
            ActivationSeed("DOG", 0.9),
            ActivationSeed("REX", 0.9),
            ActivationSeed("M_DOG", 0.7),
            ActivationSeed("M_REX", 0.7),
        ]
    )
    wm_hist: list[str] = []
    for _ in range(ticks):
        tr = eng.tick()
        wm_hist.append(",".join(tr.wm))
    return wm_hist
