"""On-disk catalog of named AH graph snapshots."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ah_memory.graph_export import dump_graph
from ah_memory.store import AHStore
from ah_memory.store_codec import restore_store, snapshot_store

_ID_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё_-]+")
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "graph_library"


def default_library_dir() -> Path:
    override = os.environ.get("AH_GRAPH_LIBRARY")
    if override:
        return Path(override)
    return _DEFAULT_DIR


def library_enabled() -> bool:
    flag = os.environ.get("AH_GRAPH_LIBRARY_DISABLE", "").strip().lower()
    return flag not in {"1", "true", "yes", "on"}


class GraphLibrary:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_library_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            items.append(_meta(payload, path.stem))
        items.sort(key=lambda row: (row.get("kind") != "fixture", str(row.get("name") or "")))
        return items

    def save(
        self,
        store: AHStore,
        *,
        name: str,
        kind: str = "user",
        graph_id: str | None = None,
        source_text: str = "",
    ) -> dict[str, Any]:
        graph_id = graph_id or make_graph_id(name)
        path = self._path(graph_id)
        now = _now_iso()
        created_at = now
        previous_source = ""
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                created_at = str(previous.get("created_at") or now)
                previous_source = str(previous.get("source_text") or "")
            except (OSError, json.JSONDecodeError):
                created_at = now
        stats = dump_graph(store)["stats"]
        record = {
            "id": graph_id,
            "name": name.strip() or graph_id,
            "kind": kind,
            "created_at": created_at,
            "updated_at": now,
            "stats": stats,
            "source_text": (source_text or previous_source).strip(),
            "store": snapshot_store(store),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return _meta(record, graph_id)

    def load(self, graph_id: str) -> tuple[AHStore, dict[str, Any]]:
        path = self._path(graph_id)
        if not path.is_file():
            raise FileNotFoundError(graph_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = restore_store(payload)
        meta = _meta(payload, graph_id)
        meta["source_text"] = str(payload.get("source_text") or "")
        return store, meta

    def delete(self, graph_id: str) -> bool:
        path = self._path(graph_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path(self, graph_id: str) -> Path:
        safe = make_graph_id(graph_id)
        return self.root / f"{safe}.json"


def make_graph_id(name: str) -> str:
    slug = _ID_RE.sub("-", (name or "").strip().lower().replace(" ", "-")).strip("-._")
    return slug[:80] or f"graph-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def remember_fixture(
    name: str,
    store: AHStore,
    *,
    source_text: str = "",
) -> dict[str, Any] | None:
    """Persist a named test/fixture graph and its independent RAG source text."""
    if not library_enabled():
        return None
    try:
        return GraphLibrary().save(
            store,
            name=name,
            kind="fixture",
            graph_id=make_graph_id(name),
            source_text=source_text,
        )
    except OSError:
        return None


def _meta(payload: Mapping[str, Any], fallback_id: str) -> dict[str, Any]:
    return {
        "id": str(payload.get("id") or fallback_id),
        "name": str(payload.get("name") or fallback_id),
        "kind": str(payload.get("kind") or "user"),
        "created_at": str(payload.get("created_at") or ""),
        "updated_at": str(payload.get("updated_at") or ""),
        "stats": dict(payload.get("stats") or {}),
        "source_chars": len(str(payload.get("source_text") or "")),
    }


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
