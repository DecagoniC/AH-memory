"""Distractor path generation for activation benchmarks."""
from __future__ import annotations

import random
from typing import Sequence

from ah_memory.synthetic.entities import Entity
from ah_memory.synthetic.events import SyntheticFactor


def generate_distractor_factors(
    rng: random.Random,
    *,
    proof_factors: Sequence[SyntheticFactor],
    entities_by_type: dict[str, list[Entity]],
    factor_seq_start: int,
    timestamp_base: int,
    count: int,
) -> list[SyntheticFactor]:
    """Create semantically similar but incorrect paths near a proof chain."""
    if count <= 0 or not proof_factors:
        return []
    persons = entities_by_type.get("Person", [])
    companies = entities_by_type.get("Company", [])
    places = entities_by_type.get("Place", [])
    objects = entities_by_type.get("Object", [])
    if not persons or not (companies or places or objects):
        return []

    distractors: list[SyntheticFactor] = []
    seq = factor_seq_start
    for index in range(count):
        kind = index % 3
        person = rng.choice(persons)
        ts = timestamp_base + index + 1
        if kind == 0 and companies and places:
            company = rng.choice(companies)
            place = rng.choice(places)
            other = rng.choice(persons)
            distractors.append(
                SyntheticFactor(
                    uid=f"factor_{seq:06d}",
                    type="KNOW",
                    arguments={"person": person.uid, "object": other.uid},
                    timestamp=ts,
                    properties={"distractor": True, "of_proof": proof_factors[0].uid},
                    weight=0.55,
                )
            )
            seq += 1
            distractors.append(
                SyntheticFactor(
                    uid=f"factor_{seq:06d}",
                    type="WORKS_FOR",
                    arguments={"person": other.uid, "company": company.uid},
                    timestamp=ts + 1,
                    properties={"distractor": True, "of_proof": proof_factors[0].uid},
                    weight=0.55,
                )
            )
            seq += 1
            distractors.append(
                SyntheticFactor(
                    uid=f"factor_{seq:06d}",
                    type="LOCATED_IN",
                    arguments={"subject": company.uid, "location": place.uid},
                    timestamp=ts + 2,
                    properties={"distractor": True, "of_proof": proof_factors[0].uid},
                    weight=0.55,
                )
            )
            seq += 1
        elif kind == 1 and places:
            place = rng.choice(places)
            parent = rng.choice(places)
            if parent.uid == place.uid and len(places) > 1:
                parent = places[(places.index(place) + 1) % len(places)]
            distractors.append(
                SyntheticFactor(
                    uid=f"factor_{seq:06d}",
                    type="LIVES_IN",
                    arguments={"person": person.uid, "location": place.uid},
                    timestamp=ts,
                    properties={"distractor": True, "of_proof": proof_factors[0].uid},
                    weight=0.5,
                )
            )
            seq += 1
            distractors.append(
                SyntheticFactor(
                    uid=f"factor_{seq:06d}",
                    type="PART_OF",
                    arguments={"subject": place.uid, "object": parent.uid},
                    timestamp=ts + 1,
                    properties={"distractor": True, "of_proof": proof_factors[0].uid},
                    weight=0.5,
                )
            )
            seq += 1
        elif objects:
            obj = rng.choice(objects)
            distractors.append(
                SyntheticFactor(
                    uid=f"factor_{seq:06d}",
                    type="USES",
                    arguments={"person": person.uid, "object": obj.uid},
                    timestamp=ts,
                    properties={"distractor": True, "of_proof": proof_factors[0].uid},
                    weight=0.5,
                )
            )
            seq += 1
        if len(distractors) >= count * 2:
            break
    return distractors[: max(count, 1) * 3]
