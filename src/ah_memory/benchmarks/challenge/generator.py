"""Deterministic generator for the committed challenge JSONL fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from ah_memory.benchmarks.challenge.schema import (
    CHALLENGE_DATA_DIR,
    QAItem,
    RoleCorpusItem,
    SourceDocument,
    SourceFact,
)


_ROLE_DOMAINS: tuple[
    tuple[str, tuple[tuple[str, str, str], ...]], ...
] = (
    (
        "logistics",
        (
            ("Mara", "sealed crate", "north depot"),
            ("Ivo", "blue parcel", "river warehouse"),
            ("Nia", "tool case", "central hub"),
            ("Oren", "sample box", "east terminal"),
            ("Pia", "spare battery", "hill station"),
        ),
    ),
    (
        "healthcare",
        (
            ("Dr. Vale", "sterile kit", "ward seven"),
            ("Nurse Kian", "scan record", "imaging room"),
            ("Medic Rhea", "oxygen pack", "triage bay"),
            ("Dr. Sol", "test vial", "lab annex"),
            ("Nurse Uma", "care chart", "recovery unit"),
        ),
    ),
    (
        "software",
        (
            ("Ari", "release bundle", "staging cluster"),
            ("Bo", "audit log", "secure archive"),
            ("Cyra", "model snapshot", "compute node"),
            ("Dev", "patch set", "test sandbox"),
            ("Eli", "backup image", "cold storage"),
        ),
    ),
    (
        "field_research",
        (
            ("Fara", "soil sample", "coastal lab"),
            ("Gio", "sensor array", "forest station"),
            ("Hana", "survey map", "base camp"),
            ("Jules", "water sample", "mobile lab"),
            ("Kora", "specimen tray", "island outpost"),
        ),
    ),
    (
        "arts",
        (
            ("Lio", "bronze mask", "west gallery"),
            ("Mina", "score manuscript", "music archive"),
            ("Noa", "costume trunk", "main theater"),
            ("Pavel", "film reel", "screening room"),
            ("Quin", "canvas study", "restoration studio"),
        ),
    ),
)

_VARIANTS = ("clean", "typo", "inversion", "ellipsis")


def generate_role_items() -> list[RoleCorpusItem]:
    """Generate 25 examples for each of four extraction variants."""
    items: list[RoleCorpusItem] = []
    sequence = 1
    for variant in _VARIANTS:
        for domain, examples in _ROLE_DOMAINS:
            for subject, object_, location in examples:
                if variant == "clean":
                    text = f"{subject} delivered the {object_} to the {location}."
                elif variant == "typo":
                    text = f"{subject} delviered the {object_} to the {location}."
                elif variant == "inversion":
                    text = (
                        f"To the {location}, the {object_} was delivered by {subject}."
                    )
                else:
                    text = f"{subject}: {object_} — {location}."
                items.append(
                    RoleCorpusItem(
                        item_id=f"M1_{sequence:03d}",
                        text=text,
                        variant=variant,
                        domain=domain,
                        expected_roles={
                            "SUBJECT": subject,
                            "OBJECT": object_,
                            "LOCATION": location,
                        },
                    )
                )
                sequence += 1
    return items


_CHAIN_LABELS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "FOLLOW": (
        ("scout", "river trail", "ridge trail", "valley trail", "forest trail", "coast trail", "harbor trail"),
        ("probe", "beacon alpha", "beacon beta", "beacon gamma", "beacon delta", "beacon epsilon", "home beacon"),
        ("caravan", "north marker", "east marker", "south marker", "west marker", "gate marker", "market marker"),
    ),
    "IS-A": (
        ("wren", "songbird", "bird", "vertebrate", "animal", "organism", "life form"),
        ("opal", "mineraloid", "geologic material", "natural material", "physical substance", "matter", "physical entity"),
        ("kayak", "paddle craft", "watercraft", "vehicle", "artifact", "object", "entity"),
    ),
    "CAUSE": (
        ("voltage surge", "fuse trip", "circuit isolation", "controller stop", "pump stop", "pressure drop", "alarm"),
        ("heavy rain", "soil saturation", "slope movement", "road blockage", "traffic diversion", "arrival delay", "schedule change"),
        ("low temperature", "ice formation", "valve restriction", "flow reduction", "sensor alert", "maintenance call", "inspection"),
    ),
}


def _statement(relation: str, subject: str, object_: str) -> str:
    if relation == "FOLLOW":
        return f"{subject.title()} follows {object_}."
    if relation == "IS-A":
        return f"{subject.title()} is a {object_}."
    return f"{subject.title()} causes {object_}."


def _question(relation: str, start: str, depth: int) -> str:
    if relation == "FOLLOW":
        return f"After {depth} link(s), what does {start} follow?"
    if relation == "IS-A":
        return f"After {depth} classification link(s), what is {start}?"
    return f"After {depth} causal link(s), what ultimately results from {start}?"


def generate_qa_items() -> list[QAItem]:
    """Generate the fixed 20-question, depth-stratified M2/M4 corpus."""
    depths = (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 6)
    relations = ("FOLLOW", "IS-A", "CAUSE")
    items: list[QAItem] = []
    for index, depth in enumerate(depths, start=1):
        relation = relations[(index - 1) % len(relations)]
        chain_set = _CHAIN_LABELS[relation][((index - 1) // 3) % 3]
        nodes = tuple(f"{label} {index:02d}" for label in chain_set[: depth + 1])
        item_id = f"QA_{index:03d}"
        facts: list[SourceFact] = []
        documents: list[SourceDocument] = []
        for step, (subject, object_) in enumerate(zip(nodes, nodes[1:]), start=1):
            fact_uid = f"{item_id}_F{step:02d}"
            statement = _statement(relation, subject, object_)
            facts.append(
                SourceFact(
                    uid=fact_uid,
                    subject=subject,
                    relation=relation,
                    object=object_,
                )
            )
            documents.append(
                SourceDocument(
                    uid=f"{item_id}_D{step:02d}",
                    text=statement,
                    fact_uids=(fact_uid,),
                )
            )
        items.append(
            QAItem(
                item_id=item_id,
                question=_question(relation, nodes[0], depth),
                answer=nodes[-1],
                depth=depth,
                relation_type=relation,
                proof_path=tuple(fact.uid for fact in facts),
                source_facts=tuple(facts),
                source_documents=tuple(documents),
            )
        )
    return items


def render_jsonl(items: Iterable[RoleCorpusItem | QAItem]) -> bytes:
    """Return canonical UTF-8 JSONL bytes with stable field ordering."""
    lines = (
        json.dumps(
            item.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in items
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def corpus_bytes() -> dict[str, bytes]:
    """Build both committed files without filesystem or network input."""
    return {
        "m1_roles.jsonl": render_jsonl(generate_role_items()),
        "m2_m4_qa.jsonl": render_jsonl(generate_qa_items()),
    }


def write_corpora(output_dir: str | Path = CHALLENGE_DATA_DIR) -> None:
    """Write both deterministic corpora to an existing or new directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, content in corpus_bytes().items():
        (destination / filename).write_bytes(content)


if __name__ == "__main__":
    write_corpora()
