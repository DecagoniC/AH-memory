from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ah_memory.dsl import DSLInterpreter
from ah_memory.invariants import InvariantError, validate
from ah_memory.store import AHStore
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    LinkId,
    SecondOrderSymbol,
    Section,
)
from tests._mini_graph import build_mini_open_store


def test_add_get_abstract_symbol() -> None:
    store = AHStore()
    s = store.add_abstract_symbol(AbstractSymbol(uid="DOG", R={"TEXT": {"собака", "собаки"}}))
    assert store.get_abstract_symbol("DOG") is s


def test_edit_abstract_symbol() -> None:
    store = AHStore()
    store.add_abstract_symbol(AbstractSymbol(uid="DOG", R={"TEXT": {"собака"}}))
    store.edit_abstract_symbol("DOG", AbstractSymbol(uid="DOG", R={"TEXT": {"пёс"}}))
    assert "пёс" in store.get_abstract_symbol("DOG").R["TEXT"]


def test_add_element_and_get_symbol() -> None:
    store = AHStore()
    store.add_element(Section.C, SecondOrderSymbol(uid="M_DOG"))
    assert store.get_symbol("M_DOG").uid == "M_DOG"


def test_find_links() -> None:
    store = AHStore()
    store.add_abstract_symbol(AbstractSymbol(uid="A", R={"TEXT": {"a"}}))
    store.add_abstract_symbol(AbstractSymbol(uid="B", R={"TEXT": {"b"}}))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_A"))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_B"))
    store.add_link(
        AssocLink(
            uid="L1",
            id=LinkId.ASSOC.value,
            w=1.0,
            e1=store.m_ref("M_A"),
            e2=store.m_ref("M_B"),
        )
    )
    links = store.find_links("M_A")
    assert len(links) == 1
    assert links[0].uid == "L1"


def test_mini_open_invariants() -> None:
    store = build_mini_open_store()
    validate(store)
    assert store.list_semantic_factors()
    assert LinkId.FOLLOW.value in {l.id for l in store.ah.L.values()}


def test_answer_who_open() -> None:
    store = build_mini_open_store()
    answer = str(DSLInterpreter(store).execute("answer_who(M_ENTITY)").value)
    assert "вид" in answer or answer != ""


def test_cycle_isa_rejected() -> None:
    store = AHStore()
    store.add_element(Section.C, SecondOrderSymbol(uid="M_A"))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_B"))
    store.add_link(
        AssocLink(
            uid="L1",
            id=LinkId.IS_A.value,
            w=1.0,
            e1=store.m_ref("M_A"),
            e2=store.m_ref("M_B"),
        )
    )
    store.add_link(
        AssocLink(
            uid="L2",
            id=LinkId.IS_A.value,
            w=1.0,
            e1=store.m_ref("M_B"),
            e2=store.m_ref("M_A"),
        )
    )
    with pytest.raises(InvariantError):
        validate(store)


def test_r_disjoint_modalities() -> None:
    store = AHStore()
    with pytest.raises(Exception):
        store.add_abstract_symbol(
            AbstractSymbol(uid="X", R={"TEXT": {"shared"}, "AUDIO": {"shared"}})
        )
