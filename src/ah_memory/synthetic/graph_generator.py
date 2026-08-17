"""Deterministic synthetic world generator with known ground truth."""
from __future__ import annotations

import time
from random import Random
from typing import Any

from ah_memory.synthetic.config import SyntheticGraphConfig
from ah_memory.synthetic.distractors import generate_distractor_factors
from ah_memory.synthetic.entities import (
    Entity,
    make_entity_uid,
    pick_name,
)
from ah_memory.synthetic.events import (
    SyntheticEvent,
    SyntheticFactor,
    WorldState,
)
from ah_memory.synthetic.ground_truth import SyntheticWorld
from ah_memory.synthetic.query_generator import QueryGenerator
from ah_memory.synthetic.text_generator import TextGenerator


# (child, parent) — directed IS_A, acyclic by construction
OBJECT_TAXONOMY: tuple[tuple[str, str], ...] = (
    ("Vehicle", "Object"),
    ("Car", "Vehicle"),
    ("Sedan", "Car"),
    ("SUV", "Car"),
    ("Motorcycle", "Vehicle"),
    ("Device", "Object"),
    ("Computer", "Device"),
    ("Phone", "Device"),
)

PERSON_TAXONOMY: tuple[tuple[str, str], ...] = (
    ("Employee", "Person"),
    ("Manager", "Person"),
    ("Engineer", "Person"),
)


class SyntheticGraphGenerator:
    def __init__(
        self,
        config: SyntheticGraphConfig,
        *,
        text_generator: TextGenerator | None = None,
        query_generator: QueryGenerator | None = None,
    ) -> None:
        self.config = config
        self.rng = Random(config.random_seed)
        self.text_generator = text_generator or TextGenerator()
        self.query_generator = query_generator or QueryGenerator(self.rng)

    def generate(self) -> SyntheticWorld:
        started = time.perf_counter()
        entities_by_type = self._create_entities()
        entities = [entity for group in entities_by_type.values() for entity in group]

        factors: list[SyntheticFactor] = []
        events: list[SyntheticEvent] = []
        world_state = WorldState()
        proof_chains: list[dict[str, Any]] = []
        factor_seq = 1
        event_seq = 1
        time_seq = 1_700_000_000

        # Hierarchy concept entities (as Place/Object-like labels stored as Object).
        hierarchy_entities, hierarchy_factors, factor_seq = self._build_hierarchies(
            factor_seq, time_seq
        )
        entities.extend(hierarchy_entities)
        for entity in hierarchy_entities:
            entities_by_type.setdefault(entity.type, []).append(entity)
        factors.extend(hierarchy_factors)

        # Multi-hop chains with known proof paths.
        hop_factors, hop_chains, factor_seq, time_seq = self._build_multihop_chains(
            entities_by_type, factor_seq, time_seq
        )
        factors.extend(hop_factors)
        proof_chains.extend(hop_chains)

        # Temporal ownership stories + FOLLOW edges + controlled re-purchase.
        temporal_factors, temporal_events, temporal_chains, factor_seq, event_seq, time_seq = (
            self._build_temporal_stories(
                entities_by_type, factor_seq, event_seq, time_seq
            )
        )
        factors.extend(temporal_factors)
        events.extend(temporal_events)
        proof_chains.extend(temporal_chains)

        # Causal chains.
        causal_factors, causal_chains, factor_seq, event_seq, time_seq = (
            self._build_causal_chains(
                entities_by_type, factor_seq, event_seq, time_seq
            )
        )
        factors.extend(causal_factors)
        proof_chains.extend(causal_chains)

        # Fill remaining factor budget with clustered local facts.
        factors, factor_seq, time_seq = self._fill_local_clusters(
            entities_by_type, factors, factor_seq, time_seq
        )

        # Distractors attached to proof chains.
        distractor_budget = int(len(proof_chains) * self.config.distractor_ratio) + 1
        distractors: list[SyntheticFactor] = []
        for chain in proof_chains:
            proof_uids = chain.get("factor_uids") or []
            proof_fs = [f for f in factors if f.uid in proof_uids]
            created = generate_distractor_factors(
                self.rng,
                proof_factors=proof_fs,
                entities_by_type=entities_by_type,
                factor_seq_start=factor_seq,
                timestamp_base=time_seq,
                count=max(1, distractor_budget // max(1, len(proof_chains))),
            )
            distractors.extend(created)
            factor_seq += len(created)
            time_seq += len(created) + 1
            chain["distractor_factor_uids"] = [item.uid for item in created]
        factors.extend(distractors)

        # Trim / pad to target factor count while keeping proof factors.
        factors = self._fit_factor_budget(factors, entities_by_type, factor_seq, time_seq)

        # Apply world state in timestamp order (non-distractors first by time).
        ordered = sorted(
            [f for f in factors if not f.properties.get("distractor")],
            key=lambda item: (item.timestamp, item.uid),
        )
        for factor in ordered:
            world_state.apply_factor(factor)

        # Ensure events cover temporal factors.
        if len(events) < self.config.num_events:
            events, event_seq = self._materialize_events(
                factors, events, event_seq
            )

        entity_map = {entity.uid: entity for entity in entities}
        documents = self.text_generator.build_documents(
            [f for f in factors if not f.properties.get("distractor")],
            entity_map,
            max_documents=min(len(factors), max(20, self.config.num_documents or 50)),
        )
        queries = self.query_generator.generate(
            entities=entity_map,
            factors=factors,
            world_state=world_state,
            proof_chains=proof_chains,
            num_queries=self.config.num_queries,
            max_hop_depth=self.config.max_hop_depth,
        )

        elapsed = time.perf_counter() - started
        return SyntheticWorld(
            config=self.config,
            entities=entities,
            factors=factors,
            events=events[: self.config.num_events]
            if len(events) > self.config.num_events
            else events,
            documents=documents,
            queries=queries,
            world_state=world_state,
            generation_time_sec=elapsed,
            proof_chains=proof_chains,
            metadata={
                "seed": self.config.random_seed,
                "preset": self.config.preset,
                "relation_types": list(self.config.relation_types),
            },
        )

    def _create_entities(self) -> dict[str, list[Entity]]:
        counts = self.config.resolved_counts()
        by_type: dict[str, list[Entity]] = {}
        for entity_type, count in counts.items():
            group: list[Entity] = []
            for index in range(1, count + 1):
                uid = make_entity_uid(entity_type, index)
                group.append(
                    Entity(
                        uid=uid,
                        type=entity_type,
                        name=pick_name(entity_type, index - 1),
                        attributes={"index": index},
                    )
                )
            by_type[entity_type] = group
        # Dedicated event entities for temporal markers.
        event_count = max(self.config.num_events // 4, 10)
        events: list[Entity] = []
        for index in range(1, event_count + 1):
            uid = make_entity_uid("Event", index)
            events.append(
                Entity(
                    uid=uid,
                    type="Event",
                    name=pick_name("Event", index - 1),
                    attributes={"index": index},
                )
            )
        by_type["Event"] = events
        return by_type

    def _build_hierarchies(
        self,
        factor_seq: int,
        timestamp: int,
    ) -> tuple[list[Entity], list[SyntheticFactor], int]:
        entities: list[Entity] = []
        factors: list[SyntheticFactor] = []
        concept_index = 1
        uid_by_label: dict[str, str] = {}

        def ensure_concept(label: str) -> str:
            nonlocal concept_index
            if label in uid_by_label:
                return uid_by_label[label]
            uid = make_entity_uid("Object", 900000 + concept_index)
            concept_index += 1
            uid_by_label[label] = uid
            entities.append(
                Entity(
                    uid=uid,
                    type="Object",
                    name=label,
                    attributes={"concept": True, "label": label},
                )
            )
            return uid

        for child_label, parent_label in OBJECT_TAXONOMY + PERSON_TAXONOMY:
            child_uid = ensure_concept(child_label)
            parent_uid = ensure_concept(parent_label)
            factors.append(
                SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type="IS_A",
                    arguments={"child": child_uid, "parent": parent_uid},
                    timestamp=timestamp,
                    properties={"hierarchy": True},
                    weight=1.0,
                )
            )
            factor_seq += 1
            timestamp += 1
        return entities, factors, factor_seq

    def _build_multihop_chains(
        self,
        entities_by_type: dict[str, list[Entity]],
        factor_seq: int,
        time_seq: int,
    ) -> tuple[list[SyntheticFactor], list[dict[str, Any]], int, int]:
        persons = entities_by_type["Person"]
        companies = entities_by_type["Company"]
        places = entities_by_type["Place"]
        if not persons or not companies or len(places) < 2:
            return [], [], factor_seq, time_seq

        factors: list[SyntheticFactor] = []
        chains: list[dict[str, Any]] = []
        chain_count = max(5, min(len(persons), self.config.num_queries // 2, 40))
        for index in range(chain_count):
            depth = 1 + (index % self.config.max_hop_depth)
            person = persons[index % len(persons)]
            company = companies[index % len(companies)]
            city = places[index % len(places)]
            country = places[(index + 1) % len(places)]
            region = places[(index + 2) % len(places)]
            continent = places[(index + 3) % len(places)]

            path_nodes = [person.uid]
            path_factors: list[str] = []

            f1 = SyntheticFactor(
                uid=f"factor_{factor_seq:06d}",
                type="WORKS_FOR",
                arguments={"person": person.uid, "company": company.uid},
                timestamp=time_seq,
                properties={"chain": True, "depth": depth},
                weight=0.9,
            )
            factor_seq += 1
            time_seq += 1
            factors.append(f1)
            path_factors.append(f1.uid)
            path_nodes.append(company.uid)

            if depth >= 2:
                f2 = SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type="LOCATED_IN",
                    arguments={"subject": company.uid, "location": city.uid},
                    timestamp=time_seq,
                    properties={"chain": True, "depth": depth},
                    weight=0.9,
                )
                factor_seq += 1
                time_seq += 1
                factors.append(f2)
                path_factors.append(f2.uid)
                path_nodes.append(city.uid)

            if depth >= 3:
                f3 = SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type="PART_OF",
                    arguments={"subject": city.uid, "object": country.uid},
                    timestamp=time_seq,
                    properties={"chain": True, "depth": depth},
                    weight=0.9,
                )
                factor_seq += 1
                time_seq += 1
                factors.append(f3)
                path_factors.append(f3.uid)
                path_nodes.append(country.uid)

            if depth >= 4:
                f4 = SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type="PART_OF",
                    arguments={"subject": country.uid, "object": region.uid},
                    timestamp=time_seq,
                    properties={"chain": True, "depth": depth},
                    weight=0.85,
                )
                factor_seq += 1
                time_seq += 1
                factors.append(f4)
                path_factors.append(f4.uid)
                path_nodes.append(region.uid)

            if depth >= 5:
                f5 = SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type="PART_OF",
                    arguments={"subject": region.uid, "object": continent.uid},
                    timestamp=time_seq,
                    properties={"chain": True, "depth": depth},
                    weight=0.8,
                )
                factor_seq += 1
                time_seq += 1
                factors.append(f5)
                path_factors.append(f5.uid)
                path_nodes.append(continent.uid)

            # Pad deeper hops with KNOW bridges if needed.
            while len(path_factors) < depth and len(persons) > 1:
                other = persons[(index + len(path_factors)) % len(persons)]
                bridge = SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type="KNOW",
                    arguments={"person": path_nodes[0], "object": other.uid},
                    timestamp=time_seq,
                    properties={"chain": True, "depth": depth, "pad": True},
                    weight=0.6,
                )
                factor_seq += 1
                time_seq += 1
                factors.append(bridge)
                path_factors.append(bridge.uid)
                path_nodes.append(other.uid)

            chains.append(
                {
                    "kind": "multi_hop",
                    "depth": depth,
                    "factor_uids": path_factors,
                    "nodes": path_nodes,
                    "answer_uid": path_nodes[-1],
                    "seed_uid": person.uid,
                }
            )
        return factors, chains, factor_seq, time_seq

    def _build_temporal_stories(
        self,
        entities_by_type: dict[str, list[Entity]],
        factor_seq: int,
        event_seq: int,
        time_seq: int,
    ) -> tuple[
        list[SyntheticFactor],
        list[SyntheticEvent],
        list[dict[str, Any]],
        int,
        int,
        int,
    ]:
        persons = entities_by_type["Person"]
        objects = entities_by_type["Object"]
        places = entities_by_type["Place"]
        event_ents = entities_by_type["Event"]
        if not persons or len(objects) < 3 or not places:
            return [], [], [], factor_seq, event_seq, time_seq

        factors: list[SyntheticFactor] = []
        events: list[SyntheticEvent] = []
        chains: list[dict[str, Any]] = []
        story_count = max(3, min(len(persons) // 2, 20))

        for index in range(story_count):
            person = persons[index % len(persons)]
            cars = [
                objects[(index * 3) % len(objects)],
                objects[(index * 3 + 1) % len(objects)],
                objects[(index * 3 + 2) % len(objects)],
            ]
            place = places[index % len(places)]
            story_factor_uids: list[str] = []
            story_event_uids: list[str] = []
            sequence = [
                ("PURCHASE", cars[0]),
                ("SELL", cars[0]),
                ("PURCHASE", cars[1]),
                ("SELL", cars[1]),
                ("PURCHASE", cars[2]),
            ]
            # Controlled contradiction / state change: repurchase first car later.
            if index % 2 == 0:
                sequence.append(("PURCHASE", cars[0]))

            prev_event_uid: str | None = None
            for step, (predicate, obj) in enumerate(sequence):
                marker = event_ents[(event_seq - 1) % len(event_ents)]
                factor = SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type=predicate,
                    arguments={
                        "buyer": person.uid,
                        "object": obj.uid,
                        "time": marker.uid,
                        "location": place.uid,
                    },
                    timestamp=time_seq,
                    properties={
                        "temporal_story": True,
                        "step": step,
                        "contradiction": predicate == "PURCHASE"
                        and obj.uid == cars[0].uid
                        and step > 0,
                    },
                    weight=0.95,
                )
                factor_seq += 1
                time_seq += 10
                factors.append(factor)
                story_factor_uids.append(factor.uid)

                event_uid = f"event_{event_seq:06d}"
                event_seq += 1
                events.append(
                    SyntheticEvent(
                        uid=event_uid,
                        factor_uid=factor.uid,
                        timestamp=factor.timestamp,
                        predicate=predicate,
                        arguments=dict(factor.arguments),
                        label=f"t{step + 1}",
                    )
                )
                story_event_uids.append(event_uid)

                if prev_event_uid is not None:
                    prev_marker = event_ents[(event_seq - 3) % len(event_ents)]
                    follow = SyntheticFactor(
                        uid=f"factor_{factor_seq:06d}",
                        type="FOLLOW",
                        arguments={
                            "previous_event": prev_marker.uid,
                            "next_event": marker.uid,
                        },
                        timestamp=time_seq,
                        properties={
                            "temporal_story": True,
                            "story_events": [prev_event_uid, event_uid],
                        },
                        weight=1.0,
                    )
                    factor_seq += 1
                    time_seq += 1
                    factors.append(follow)
                    story_factor_uids.append(follow.uid)
                prev_event_uid = event_uid

            chains.append(
                {
                    "kind": "temporal",
                    "depth": len(sequence),
                    "factor_uids": story_factor_uids,
                    "nodes": [person.uid, cars[2].uid],
                    "answer_uid": cars[2].uid,
                    "seed_uid": person.uid,
                    "person_uid": person.uid,
                    "final_owns": [cars[2].uid]
                    + ([cars[0].uid] if index % 2 == 0 else []),
                }
            )
        return factors, events, chains, factor_seq, event_seq, time_seq

    def _build_causal_chains(
        self,
        entities_by_type: dict[str, list[Entity]],
        factor_seq: int,
        event_seq: int,
        time_seq: int,
    ) -> tuple[list[SyntheticFactor], list[dict[str, Any]], int, int, int]:
        events = entities_by_type.get("Event", [])
        documents = entities_by_type.get("Document", [])
        if len(events) < 2:
            return [], [], factor_seq, event_seq, time_seq
        factors: list[SyntheticFactor] = []
        chains: list[dict[str, Any]] = []
        count = max(3, min(len(events) // 2, 15))
        for index in range(count):
            cause = documents[index % len(documents)] if documents else events[index]
            effect = events[(index + 1) % len(events)]
            factor = SyntheticFactor(
                uid=f"factor_{factor_seq:06d}",
                type="CAUSE",
                arguments={"cause": cause.uid, "effect": effect.uid},
                timestamp=time_seq,
                properties={"causal": True},
                weight=0.9,
            )
            factor_seq += 1
            time_seq += 1
            factors.append(factor)
            chains.append(
                {
                    "kind": "causal",
                    "depth": 1,
                    "factor_uids": [factor.uid],
                    "nodes": [cause.uid, effect.uid],
                    "answer_uid": cause.uid,
                    "seed_uid": effect.uid,
                }
            )
        return factors, chains, factor_seq, event_seq, time_seq

    def _fill_local_clusters(
        self,
        entities_by_type: dict[str, list[Entity]],
        factors: list[SyntheticFactor],
        factor_seq: int,
        time_seq: int,
    ) -> tuple[list[SyntheticFactor], int, int]:
        persons = entities_by_type["Person"]
        companies = entities_by_type["Company"]
        places = entities_by_type["Place"]
        objects = entities_by_type["Object"]
        allowed = set(self.config.relation_types)
        target = self.config.num_factors
        attempts = 0
        while len(factors) < target and attempts < target * 4:
            attempts += 1
            choice = self.rng.choice(
                [
                    name
                    for name in (
                        "LIVES_IN",
                        "WORKS_FOR",
                        "USES",
                        "VISITS",
                        "KNOW",
                        "CREATED",
                        "OWNS",
                        "LOCATED_IN",
                    )
                    if name in allowed
                ]
                or ["KNOW"]
            )
            if choice == "LIVES_IN" and persons and places:
                person = self.rng.choice(persons)
                place = self.rng.choice(places)
                args = {"person": person.uid, "location": place.uid}
            elif choice == "WORKS_FOR" and persons and companies:
                person = self.rng.choice(persons)
                company = self.rng.choice(companies)
                args = {"person": person.uid, "company": company.uid}
            elif choice == "USES" and persons and objects:
                person = self.rng.choice(persons)
                obj = self.rng.choice(objects)
                args = {"person": person.uid, "object": obj.uid}
            elif choice == "VISITS" and persons and places:
                person = self.rng.choice(persons)
                place = self.rng.choice(places)
                marker = self.rng.choice(entities_by_type["Event"])
                args = {
                    "person": person.uid,
                    "location": place.uid,
                    "time": marker.uid,
                }
            elif choice == "CREATED" and persons and objects:
                person = self.rng.choice(persons)
                obj = self.rng.choice(objects)
                marker = self.rng.choice(entities_by_type["Event"])
                args = {
                    "person": person.uid,
                    "object": obj.uid,
                    "time": marker.uid,
                }
            elif choice == "OWNS" and persons and objects:
                person = self.rng.choice(persons)
                obj = self.rng.choice(objects)
                args = {"person": person.uid, "object": obj.uid}
            elif choice == "LOCATED_IN" and companies and places:
                company = self.rng.choice(companies)
                place = self.rng.choice(places)
                args = {"subject": company.uid, "location": place.uid}
            elif persons:
                a = self.rng.choice(persons)
                b = self.rng.choice(persons)
                if a.uid == b.uid:
                    continue
                choice = "KNOW"
                args = {"person": a.uid, "object": b.uid}
            else:
                continue
            factors.append(
                SyntheticFactor(
                    uid=f"factor_{factor_seq:06d}",
                    type=choice,
                    arguments=args,
                    timestamp=time_seq,
                    properties={"cluster": True},
                    weight=round(self.rng.uniform(0.45, 0.85), 3),
                )
            )
            factor_seq += 1
            time_seq += 1
        return factors, factor_seq, time_seq

    def _fit_factor_budget(
        self,
        factors: list[SyntheticFactor],
        entities_by_type: dict[str, list[Entity]],
        factor_seq: int,
        time_seq: int,
    ) -> list[SyntheticFactor]:
        target = self.config.num_factors
        if len(factors) == target:
            return factors
        if len(factors) > target:
            essential = [
                factor
                for factor in factors
                if factor.properties.get("chain")
                or factor.properties.get("temporal_story")
                or factor.properties.get("hierarchy")
                or factor.properties.get("causal")
            ]
            others = [factor for factor in factors if factor not in essential]
            keep_others = max(0, target - len(essential))
            return essential + others[:keep_others]
        # pad
        padded = list(factors)
        padded, _, _ = self._fill_local_clusters(
            entities_by_type, padded, factor_seq, time_seq
        )
        return padded[:target] if len(padded) > target else padded

    def _materialize_events(
        self,
        factors: list[SyntheticFactor],
        events: list[SyntheticEvent],
        event_seq: int,
    ) -> tuple[list[SyntheticEvent], int]:
        existing = {event.factor_uid for event in events}
        out = list(events)
        for factor in factors:
            if factor.uid in existing:
                continue
            if factor.type not in {"PURCHASE", "SELL", "MOVE", "VISITS", "CREATED", "FOLLOW"}:
                continue
            out.append(
                SyntheticEvent(
                    uid=f"event_{event_seq:06d}",
                    factor_uid=factor.uid,
                    timestamp=factor.timestamp,
                    predicate=factor.type,
                    arguments=dict(factor.arguments),
                    label=f"t{event_seq}",
                )
            )
            event_seq += 1
            if len(out) >= self.config.num_events:
                break
        return out, event_seq
