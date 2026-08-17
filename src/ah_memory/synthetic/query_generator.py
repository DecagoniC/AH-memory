"""Benchmark question generation with ground-truth proof paths."""
from __future__ import annotations

from random import Random
from typing import Any, Mapping, Sequence

from ah_memory.synthetic.entities import Entity
from ah_memory.synthetic.events import SyntheticFactor, WorldState
from ah_memory.synthetic.ground_truth import SyntheticQuery


class QueryGenerator:
    def __init__(self, rng: Random | None = None) -> None:
        self.rng = rng or Random(0)

    def generate(
        self,
        *,
        entities: Mapping[str, Entity],
        factors: Sequence[SyntheticFactor],
        world_state: WorldState,
        proof_chains: Sequence[Mapping[str, Any]],
        num_queries: int,
        max_hop_depth: int,
    ) -> list[SyntheticQuery]:
        queries: list[SyntheticQuery] = []
        factor_map = {factor.uid: factor for factor in factors}
        qid = 1
        # Reserve capacity so temporal/causal/hierarchy are not starved.
        buckets: dict[str, list[SyntheticQuery]] = {
            "multi_hop": [],
            "temporal": [],
            "causal": [],
            "direct": [],
            "distractor": [],
            "hierarchy": [],
            "aggregation": [],
            "contradiction": [],
        }

        def add(query: SyntheticQuery) -> None:
            nonlocal qid
            query = SyntheticQuery(
                query_id=f"q_{qid:06d}",
                question=query.question,
                answer=query.answer,
                answer_uid=query.answer_uid,
                answer_type=query.answer_type,
                category=query.category,
                required_depth=query.required_depth,
                proof_path=query.proof_path,
                required_nodes=query.required_nodes,
                distractor_factor_uids=query.distractor_factor_uids,
                seed_uids=query.seed_uids,
            )
            qid += 1
            buckets.setdefault(query.category, []).append(query)

        # Multi-hop / distractor from proof chains.
        for chain in proof_chains:
            kind = str(chain.get("kind", "multi_hop"))
            nodes = list(chain.get("nodes") or [])
            factor_uids = tuple(chain.get("factor_uids") or ())
            distractors = tuple(chain.get("distractor_factor_uids") or ())
            seed = str(chain.get("seed_uid") or (nodes[0] if nodes else ""))
            answer_uid = str(chain.get("answer_uid") or (nodes[-1] if nodes else ""))
            if not seed or not answer_uid or seed not in entities or answer_uid not in entities:
                continue
            seed_ent = entities[seed]
            answer_ent = entities[answer_uid]
            depth = int(chain.get("depth") or max(1, len(factor_uids)))

            if kind == "multi_hop":
                add(
                    SyntheticQuery(
                        query_id="",
                        question=(
                            f"В какой сущности заканчивается цепочка от {seed_ent.name} "
                            f"длины {depth}?"
                            if depth > 2
                            else f"Где находится компания, в которой работает {seed_ent.name}?"
                            if depth == 2
                            else f"В какой компании работает {seed_ent.name}?"
                        ),
                        answer=answer_ent.name,
                        answer_uid=answer_uid,
                        answer_type="entity",
                        category="multi_hop" if depth > 1 else "direct",
                        required_depth=min(depth, max_hop_depth),
                        proof_path=factor_uids,
                        required_nodes=tuple(nodes),
                        distractor_factor_uids=distractors,
                        seed_uids=(seed,),
                    )
                )
                if distractors:
                    add(
                        SyntheticQuery(
                            query_id="",
                            question=f"В какой компании работает {seed_ent.name}?",
                            answer=entities[nodes[1]].name if len(nodes) > 1 else answer_ent.name,
                            answer_uid=nodes[1] if len(nodes) > 1 else answer_uid,
                            answer_type="entity",
                            category="distractor",
                            required_depth=1,
                            proof_path=factor_uids[:1],
                            required_nodes=tuple(nodes[:2]),
                            distractor_factor_uids=distractors,
                            seed_uids=(seed,),
                        )
                    )
            elif kind == "temporal":
                owns = list(chain.get("final_owns") or [answer_uid])
                final_uid = owns[-1]
                if final_uid not in entities:
                    continue
                add(
                    SyntheticQuery(
                        query_id="",
                        question=f"Какой объект сейчас принадлежит {seed_ent.name}?",
                        answer=entities[final_uid].name,
                        answer_uid=final_uid,
                        answer_type="entity",
                        category="temporal",
                        required_depth=depth,
                        proof_path=factor_uids,
                        required_nodes=(seed, final_uid),
                        distractor_factor_uids=distractors,
                        seed_uids=(seed,),
                    )
                )
                if any(
                    factor_map[uid].properties.get("contradiction")
                    for uid in factor_uids
                    if uid in factor_map
                ):
                    add(
                        SyntheticQuery(
                            query_id="",
                            question=(
                                f"Учитывая порядок сделок, чем владеет {seed_ent.name} "
                                "после всех операций?"
                            ),
                            answer=entities[final_uid].name,
                            answer_uid=final_uid,
                            answer_type="entity",
                            category="contradiction",
                            required_depth=depth,
                            proof_path=factor_uids,
                            required_nodes=(seed, final_uid),
                            distractor_factor_uids=distractors,
                            seed_uids=(seed,),
                        )
                    )
            elif kind == "causal":
                add(
                    SyntheticQuery(
                        query_id="",
                        question=f"Почему произошёл {entities[seed].name}?",
                        answer=answer_ent.name,
                        answer_uid=answer_uid,
                        answer_type="entity",
                        category="causal",
                        required_depth=1,
                        proof_path=factor_uids,
                        required_nodes=(seed, answer_uid),
                        distractor_factor_uids=distractors,
                        seed_uids=(seed,),
                    )
                )

        # Direct LIVES_IN
        for factor in factors:
            if factor.type != "LIVES_IN" or factor.properties.get("distractor"):
                continue
            person = factor.arguments.get("person")
            location = factor.arguments.get("location")
            if not person or not location:
                continue
            if person not in entities or location not in entities:
                continue
            add(
                SyntheticQuery(
                    query_id="",
                    question=f"Где живёт {entities[person].name}?",
                    answer=entities[location].name,
                    answer_uid=location,
                    answer_type="entity",
                    category="direct",
                    required_depth=1,
                    proof_path=(factor.uid,),
                    required_nodes=(person, location),
                    seed_uids=(person,),
                )
            )

        # Hierarchy IS_A
        for factor in factors:
            if factor.type != "IS_A":
                continue
            child = factor.arguments.get("child")
            parent = factor.arguments.get("parent")
            if not child or not parent:
                continue
            if child not in entities or parent not in entities:
                continue
            add(
                SyntheticQuery(
                    query_id="",
                    question=f"Является ли {entities[child].name} разновидностью {entities[parent].name}?",
                    answer="да",
                    answer_uid=parent,
                    answer_type="boolean",
                    category="hierarchy",
                    required_depth=1,
                    proof_path=(factor.uid,),
                    required_nodes=(child, parent),
                    seed_uids=(child,),
                )
            )

        # Aggregation from world state.
        for person_uid, state in world_state.persons.items():
            if not state.owns or person_uid not in entities:
                continue
            owned = [uid for uid in state.owns if uid in entities]
            if not owned:
                continue
            answer_uid = owned[-1]
            proof = [
                factor.uid
                for factor in factors
                if factor.type in {"PURCHASE", "SELL", "OWNS"}
                and (
                    factor.arguments.get("buyer") == person_uid
                    or factor.arguments.get("person") == person_uid
                )
                and not factor.properties.get("distractor")
            ]
            add(
                SyntheticQuery(
                    query_id="",
                    question=f"Какие объекты принадлежат {entities[person_uid].name}?",
                    answer=entities[answer_uid].name,
                    answer_uid=answer_uid,
                    answer_type="entity",
                    category="aggregation",
                    required_depth=2,
                    proof_path=tuple(proof[-6:]),
                    required_nodes=(person_uid, *owned),
                    seed_uids=(person_uid,),
                )
            )

        # Pad with KNOW directs.
        for factor in factors:
            if factor.type != "KNOW" or factor.properties.get("distractor"):
                continue
            person = factor.arguments.get("person")
            other = factor.arguments.get("object")
            if not person or not other:
                continue
            if person not in entities or other not in entities:
                continue
            add(
                SyntheticQuery(
                    query_id="",
                    question=f"Кого знает {entities[person].name}?",
                    answer=entities[other].name,
                    answer_uid=other,
                    answer_type="entity",
                    category="direct",
                    required_depth=1,
                    proof_path=(factor.uid,),
                    required_nodes=(person, other),
                    seed_uids=(person,),
                )
            )

        order = (
            "temporal",
            "multi_hop",
            "hierarchy",
            "causal",
            "aggregation",
            "contradiction",
            "distractor",
            "direct",
        )
        # Round-robin pull so each category appears when available.
        indices = {key: 0 for key in order}
        while len(queries) < num_queries:
            progressed = False
            for key in order:
                idx = indices[key]
                bucket = buckets.get(key) or []
                if idx < len(bucket):
                    queries.append(bucket[idx])
                    indices[key] = idx + 1
                    progressed = True
                    if len(queries) >= num_queries:
                        break
            if not progressed:
                break
        # Renumber for stable ids after mix.
        renumbered: list[SyntheticQuery] = []
        for index, query in enumerate(queries, start=1):
            renumbered.append(
                SyntheticQuery(
                    query_id=f"q_{index:06d}",
                    question=query.question,
                    answer=query.answer,
                    answer_uid=query.answer_uid,
                    answer_type=query.answer_type,
                    category=query.category,
                    required_depth=query.required_depth,
                    proof_path=query.proof_path,
                    required_nodes=query.required_nodes,
                    distractor_factor_uids=query.distractor_factor_uids,
                    seed_uids=query.seed_uids,
                )
            )
        return renumbered
