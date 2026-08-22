from __future__ import annotations

from ah_memory.agent import Agent
from ah_memory.graph_library import GraphLibrary, make_graph_id
from ah_memory.store_codec import restore_store, snapshot_store
from tests._mini_graph import build_mini_open_store


def test_store_snapshot_roundtrip() -> None:
    original = build_mini_open_store()
    restored = restore_store(snapshot_store(original))
    assert restored.graph_size() == original.graph_size()
    assert set(restored.ah.S) == set(original.ah.S)
    assert set(restored.ah.C) == set(original.ah.C)
    assert set(restored.ah.L) == set(original.ah.L)
    assert set(restored.semantic_factors) == set(original.semantic_factors)
    orig_roles = {
        uid: dict(factor.roles) for uid, factor in original.semantic_factors.items()
    }
    rest_roles = {
        uid: dict(factor.roles) for uid, factor in restored.semantic_factors.items()
    }
    assert rest_roles == orig_roles
    assert {rel.canonical_label for rel in restored.list_relations()} == {
        rel.canonical_label for rel in original.list_relations()
    }


def test_graph_library_save_load_delete(tmp_path) -> None:
    lib = GraphLibrary(tmp_path)
    store = build_mini_open_store()
    meta = lib.save(store, name="Demo Graph", source_text="source document only")
    assert meta["id"] == make_graph_id("Demo Graph")
    ids = {item["id"] for item in lib.list()}
    assert meta["id"] in ids
    loaded, loaded_meta = lib.load(meta["id"])
    assert loaded_meta["name"] == "Demo Graph"
    assert loaded.graph_size() == store.graph_size()
    assert loaded_meta.get("source_text") == "source document only"
    assert lib.delete(meta["id"]) is True
    assert lib.list() == []


def test_remember_fixture_writes_named_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AH_GRAPH_LIBRARY", str(tmp_path))
    store = build_mini_open_store()
    lib = GraphLibrary(tmp_path)
    names = {item["name"] for item in lib.list()}
    assert "mini-open" in names
    loaded, meta = lib.load("mini-open")
    assert meta["kind"] == "fixture"
    assert "обитает" in (meta.get("source_text") or "")
    assert loaded.graph_size() == store.graph_size()


def test_agent_adopt_store_restores_askable_graph() -> None:
    agent = Agent()
    assert agent.store.graph_size() == 0
    agent.adopt_store(build_mini_open_store())
    assert agent.store.graph_size() > 0
    assert agent.store is agent.ignition.store
    assert agent.store is agent.dsl.store
