"""Serialize synthetic datasets to JSON / JSONL / GraphML / zip."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ah_memory.synthetic.ground_truth import SyntheticWorld


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_dataset(world: SyntheticWorld, directory: str | Path) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        **world.to_summary(),
        "seed": world.config.random_seed,
        "format": "ah_memory.synthetic.v1",
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(root / "entities.jsonl", [entity.to_dict() for entity in world.entities])
    _write_jsonl(root / "factors.jsonl", [factor.to_dict() for factor in world.factors])
    _write_jsonl(root / "events.jsonl", [event.to_dict() for event in world.events])
    _write_jsonl(
        root / "documents.jsonl",
        [document.to_dict() for document in world.documents],
    )
    _write_jsonl(root / "queries.jsonl", [query.to_dict() for query in world.queries])
    (root / "ground_truth.json").write_text(
        json.dumps(world.to_ground_truth_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "graph.graphml").write_text(to_graphml(world), encoding="utf-8")
    return root


def to_graphml(world: SyntheticWorld) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '<key id="label" for="node" attr.name="label" attr.type="string"/>',
        '<key id="type" for="node" attr.name="type" attr.type="string"/>',
        '<key id="rel" for="edge" attr.name="relation" attr.type="string"/>',
        '<graph id="G" edgedefault="directed">',
    ]
    for entity in world.entities:
        lines.append(
            f'<node id="{escape(entity.uid)}">'
            f'<data key="label">{escape(entity.name)}</data>'
            f'<data key="type">{escape(entity.type)}</data>'
            f"</node>"
        )
    for factor in world.factors:
        lines.append(
            f'<node id="{escape(factor.uid)}">'
            f'<data key="label">{escape(factor.type)}</data>'
            f'<data key="type">factor</data>'
            f"</node>"
        )
        for role, target in factor.arguments.items():
            edge_id = f"{factor.uid}_{role}"
            lines.append(
                f'<edge id="{escape(edge_id)}" source="{escape(factor.uid)}" '
                f'target="{escape(target)}">'
                f'<data key="rel">{escape(role)}</data>'
                f"</edge>"
            )
    lines.extend(["</graph>", "</graphml>"])
    return "\n".join(lines)


def export_zip(world: SyntheticWorld, zip_path: str | Path) -> Path:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = zip_path.with_suffix("")
    export_dataset(world, tmp_dir)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(tmp_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(tmp_dir).as_posix())
    return zip_path


def world_to_json(world: SyntheticWorld) -> dict[str, Any]:
    return {
        "metadata": world.to_summary(),
        "entities": [entity.to_dict() for entity in world.entities],
        "factors": [factor.to_dict() for factor in world.factors],
        "events": [event.to_dict() for event in world.events],
        "documents": [document.to_dict() for document in world.documents],
        "queries": [query.to_dict() for query in world.queries],
        "ground_truth": world.to_ground_truth_dict(),
    }
