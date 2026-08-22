"""Query planning and semantic entry-point selection for the AH graph."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Mapping, Sequence

from ah_memory.factor_graph import Factor
from ah_memory.morph import STOP, lemma
from ah_memory.perception import PerceptionResult
from ah_memory.relation_normalizer import deterministic_embedding
from ah_memory.relation_registry import cosine_similarity
from ah_memory.store import AHStore

EmbedFn = Callable[[str], Sequence[float]]
_ROLES = frozenset(
    {
        "SUBJECT",
        "OBJECT",
        "LOCATION",
        "TIME",
        "CAUSE",
        "TOOL",
        "MATERIAL",
        "PURPOSE",
        "HOW-TO",
        "WITH",
    }
)
_NONFACTUAL = frozenset({"topic", "open_question", "proposal", "explanation"})


@dataclass(frozen=True)
class QueryPlan:
    anchors: tuple[str, ...] = ()
    relation_text: str = ""
    target_role: str = "OBJECT"
    cardinality: str = "one"
    factor_scores: Mapping[str, float] = field(default_factory=dict)
    relation_scores: Mapping[str, float] = field(default_factory=dict)

    @property
    def factor_uids(self) -> tuple[str, ...]:
        return tuple(self.factor_scores)


class GraphQueryPlanner:
    """Resolve entity anchors plus a semantic relation/role intent."""

    def __init__(
        self,
        store: AHStore,
        *,
        embed: EmbedFn | None = None,
        dimensions: int = 128,
        min_similarity: float = 0.12,
        min_margin: float = 0.025,
        fallback_min_similarity: float = 0.18,
        fallback_min_margin: float = 0.035,
    ) -> None:
        self.store = store
        self.embed = embed or (
            lambda text: deterministic_embedding(text, dimensions)
        )
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.fallback_min_similarity = fallback_min_similarity
        self.fallback_min_margin = fallback_min_margin
        self._revision = -1
        self._relation_docs: dict[str, str] = {}
        self._relation_vectors: dict[str, tuple[float, ...]] = {}
        self._relation_label_vectors: dict[str, tuple[float, ...]] = {}

    def bind_store(self, store: AHStore) -> None:
        self.store = store
        self._revision = -1
        self._relation_docs.clear()
        self._relation_vectors.clear()
        self._relation_label_vectors.clear()

    def plan(
        self,
        question: str,
        perception: PerceptionResult,
        anchors: Sequence[str],
    ) -> QueryPlan:
        anchor_set = {
            uid for uid in anchors if uid in self.store.ah.all_hyper()
        }
        query_meta = self._query_metadata(perception)
        target_role = self._target_role(question, perception, query_meta)
        relation_text = self._relation_text(
            question,
            perception,
            query_meta,
            anchor_set,
        )
        cardinality = str(query_meta.get("cardinality") or "one").lower()
        if cardinality not in {"one", "many"}:
            cardinality = "one"
        if cardinality == "one":
            tokens = set(question.lower().replace("ё", "е").split())
            if tokens.intersection(
                {"какие", "каковы", "перечисли", "перечислите", "все"}
            ):
                cardinality = "many"
        explicit_relation_intent = (
            bool(query_meta.get("relation"))
            or any(
                candidate.statement_type in {"topic", "open_question"}
                and self._candidate_relation(candidate)
                for candidate in perception.candidates
            )
        )
        inferred_relation_intent = (
            bool(anchor_set)
            and (
                perception.kind == "question"
                or question.rstrip().endswith("?")
            )
            and bool(self._semantic_query_terms(question, anchor_set))
        )
        has_relation_intent = (
            explicit_relation_intent or inferred_relation_intent
        )
        if not has_relation_intent:
            return QueryPlan(
                anchors=tuple(anchor_set),
                relation_text=relation_text,
                target_role=target_role,
                cardinality=cardinality,
            )

        grouped = self._eligible_factors(anchor_set)
        if not grouped or not relation_text:
            return QueryPlan(
                anchors=tuple(anchor_set),
                relation_text=relation_text,
                target_role=target_role,
                cardinality=cardinality,
            )
        if not any(
            target_role in factor.roles
            for factors in grouped.values()
            for factor in factors
        ):
            recovered_role = self._infer_target_role(
                [
                    factor
                    for factors in grouped.values()
                    for factor in factors
                ],
                anchor_set,
            )
            if recovered_role is not None:
                target_role = recovered_role

        exact_labels = {
            self._candidate_relation(candidate)
            for candidate in perception.candidates
            if candidate.statement_type in {"topic", "open_question"}
        }
        exact_labels.discard("")
        relation_scores: dict[str, float] = {}
        if exact_labels.intersection(grouped):
            for label in grouped:
                relation_scores[label] = 1.0 if label in exact_labels else 0.0
        else:
            self._refresh_relation_index()
            relation_scores = self._score_relations(
                grouped,
                relation_text,
                target_role,
                cardinality,
            )

        ranked = sorted(
            relation_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked:
            return QueryPlan(
                anchors=tuple(anchor_set),
                relation_text=relation_text,
                target_role=target_role,
                cardinality=cardinality,
            )
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        exact = best_score >= 0.999
        min_similarity = self.min_similarity
        min_margin = self.min_margin
        if inferred_relation_intent and not exact:
            min_similarity = max(
                min_similarity,
                self.fallback_min_similarity,
            )
            min_margin = max(min_margin, self.fallback_min_margin)
        failed_threshold = not exact and (
            best_score < min_similarity
            or best_score - second_score < min_margin
        )
        if (
            failed_threshold
            and explicit_relation_intent
            and inferred_relation_intent
        ):
            fallback_text = " ".join(
                self._semantic_query_terms(question, anchor_set)
            )
            fallback_scores = self._score_relations(
                grouped,
                fallback_text,
                target_role,
                cardinality,
            )
            fallback_ranked = sorted(
                fallback_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if fallback_ranked:
                fallback_best = fallback_ranked[0][1]
                fallback_second = (
                    fallback_ranked[1][1]
                    if len(fallback_ranked) > 1
                    else -1.0
                )
                if (
                    fallback_best >= self.fallback_min_similarity
                    and fallback_best - fallback_second
                    >= self.fallback_min_margin
                ):
                    ranked = fallback_ranked
                    best_label, best_score = ranked[0]
                    failed_threshold = False
        if failed_threshold:
            return QueryPlan(
                anchors=tuple(anchor_set),
                relation_text=relation_text,
                target_role=target_role,
                cardinality=cardinality,
                relation_scores=dict(ranked),
            )

        selected = grouped[best_label]
        factor_scores = {
            factor.uid: 1.0
            for factor in selected
            if target_role in factor.roles
        }
        if not factor_scores:
            recovered_role = self._infer_target_role(selected, anchor_set)
            if recovered_role is not None:
                target_role = recovered_role
                factor_scores = {
                    factor.uid: 1.0
                    for factor in selected
                    if target_role in factor.roles
                }
        return QueryPlan(
            anchors=tuple(anchor_set),
            relation_text=relation_text,
            target_role=target_role,
            cardinality=cardinality,
            factor_scores=factor_scores,
            relation_scores=dict(ranked),
        )

    def _score_relations(
        self,
        grouped: Mapping[str, Sequence[Factor]],
        relation_text: str,
        target_role: str,
        cardinality: str,
    ) -> dict[str, float]:
        if not relation_text:
            return {}
        query_vector = tuple(float(x) for x in self.embed(relation_text))
        scores: dict[str, float] = {}
        for label, factors in grouped.items():
            vector = self._relation_vectors.get(label)
            label_vector = self._relation_label_vectors.get(label)
            if vector is None or label_vector is None:
                continue
            score = (
                0.7 * cosine_similarity(query_vector, label_vector)
                + 0.3 * cosine_similarity(query_vector, vector)
            )
            if cardinality == "many":
                distinct_values = {
                    factor.roles[target_role]
                    for factor in factors
                    if target_role in factor.roles
                }
                if distinct_values:
                    score += 0.08 * (
                        1.0 - 1.0 / len(distinct_values)
                    )
            scores[label] = score
        return scores

    @staticmethod
    def _infer_target_role(
        factors: Sequence[Factor],
        anchors: set[str],
    ) -> str | None:
        role_values: dict[str, set[str]] = {}
        role_coverage: dict[str, int] = {}
        for factor in factors:
            for role, uid in factor.roles.items():
                if role == "SUBJECT" or uid in anchors:
                    continue
                role_values.setdefault(role, set()).add(uid)
                role_coverage[role] = role_coverage.get(role, 0) + 1
        if not role_values:
            return None
        return max(
            role_values,
            key=lambda role: (
                role_coverage[role],
                len(role_values[role]),
                role == "OBJECT",
            ),
        )

    def decode(self, plan: QueryPlan) -> tuple[str, list[str]]:
        records: list[tuple[Factor, str, str, str]] = []
        support: list[str] = []
        seen: set[str] = set()
        events = {event.uid: event for event in self.store.list_events()}
        for factor_uid in plan.factor_uids:
            factor = self.store.semantic_factors.get(factor_uid)
            if factor is None:
                continue
            value_uid = factor.roles.get(plan.target_role)
            if not value_uid or value_uid in seen:
                continue
            seen.add(value_uid)
            event = events.get(str(factor.metadata.get("event_uid") or ""))
            surface = (
                (event.raw_span or "").strip()
                if event is not None
                else ""
            ) or self._label(value_uid)
            if not surface:
                continue
            raw_relation = str(
                factor.metadata.get("raw_relation")
                or (
                    factor.relation.raw_label
                    if factor.relation is not None
                    else ""
                )
            ).strip()
            records.append((factor, value_uid, surface, raw_relation))
        if not records:
            return "неизвестно", []

        relation_members: dict[str, list[str]] = {}
        for _, value_uid, surface, raw_relation in records:
            if not raw_relation or ":" in raw_relation:
                continue
            relation_tokens = set(self._text_lemmas(raw_relation))
            if not relation_tokens:
                continue
            members = [
                other_uid
                for _, other_uid, other_surface, _ in records
                if (
                    tokens := set(self._text_lemmas(other_surface))
                )
                and len(tokens.intersection(relation_tokens)) / len(tokens) >= 0.6
            ]
            if len(members) >= 2:
                relation_members.setdefault(raw_relation, members)

        outputs: list[str] = []
        consumed: set[str] = set()
        for factor, value_uid, surface, raw_relation in records:
            if value_uid in consumed:
                continue
            group = next(
                (
                    (text, members)
                    for text, members in relation_members.items()
                    if value_uid in members
                ),
                None,
            )
            if group is not None:
                text, members = group
                outputs.append(text)
                consumed.update(members)
                for member_uid in members:
                    member = next(
                        item for item in records if item[1] == member_uid
                    )
                    support.extend([member[0].uid, member_uid])
                continue

            rendered, related_support = self._enrich_surface(
                factor,
                value_uid,
                surface,
                raw_relation,
                events,
                records,
            )
            outputs.append(rendered)
            consumed.add(value_uid)
            support.extend([factor.uid, value_uid, *related_support])

        outputs = list(dict.fromkeys(output for output in outputs if output))
        if plan.cardinality == "one":
            outputs = outputs[:1]
        return "; ".join(outputs), list(dict.fromkeys(support))

    def _enrich_surface(
        self,
        factor: Factor,
        value_uid: str,
        surface: str,
        raw_relation: str,
        events: Mapping[str, object],
        selected: list[tuple[Factor, str, str, str]],
    ) -> tuple[str, list[str]]:
        rendered = surface
        base_tokens = set(self._text_lemmas(surface))
        selected_uids = {item[0].uid for item in selected}
        relation_tokens = set(self._text_lemmas(raw_relation))
        relation_mentions = sum(
            bool(
                tokens := set(self._text_lemmas(other_surface))
            )
            and len(tokens.intersection(relation_tokens)) / len(tokens) >= 0.6
            for _, _, other_surface, _ in selected
        )
        if (
            raw_relation
            and ":" not in raw_relation
            and len(raw_relation) > len(rendered)
            and base_tokens
            and len(base_tokens.intersection(relation_tokens)) / len(base_tokens)
            >= 0.6
            and relation_mentions <= 1
        ):
            rendered = raw_relation

        adjuncts: list[str] = []
        related_support: list[str] = []
        for related in self.store.list_semantic_factors():
            if (
                related.uid == factor.uid
                or related.uid in selected_uids
                or related.relation is None
                or related.roles.get("SUBJECT") != value_uid
                or str(related.metadata.get("statement_type") or "assertion")
                in _NONFACTUAL
            ):
                continue
            event = events.get(str(related.metadata.get("event_uid") or ""))
            raw_span = str(getattr(event, "raw_span", "") or "").strip()
            if not raw_span:
                continue
            raw_tokens = set(self._text_lemmas(raw_span))
            if (
                base_tokens
                and len(base_tokens.intersection(raw_tokens)) / len(base_tokens)
                >= 0.6
                and len(raw_span) > len(rendered)
            ):
                rendered = raw_span
            elif raw_span not in rendered and raw_span not in adjuncts:
                adjuncts.append(raw_span)
            related_support.append(related.uid)
            related_support.extend(
                uid
                for role, uid in related.roles.items()
                if role != "SUBJECT"
            )
        if adjuncts:
            rendered = f"{rendered} ({', '.join(adjuncts)})"
        return rendered, related_support

    def _eligible_factors(
        self,
        anchors: set[str],
    ) -> dict[str, list[Factor]]:
        grouped: dict[str, list[Factor]] = {}
        for factor in self.store.list_semantic_factors():
            if factor.relation is None:
                continue
            statement_type = str(
                factor.metadata.get("statement_type") or "assertion"
            )
            if statement_type in _NONFACTUAL:
                continue
            role_values = set(factor.roles.values())
            if anchors and not anchors.intersection(role_values):
                continue
            grouped.setdefault(
                factor.relation.canonical_label.upper(),
                [],
            ).append(factor)
        return grouped

    def _refresh_relation_index(self) -> None:
        revision = self.store.ah.revision
        if revision == self._revision:
            return
        grouped = self._eligible_factors(set())
        events = {event.uid: event for event in self.store.list_events()}
        docs: dict[str, str] = {}
        for label, factors in grouped.items():
            parts = [label.replace("_", " ")]
            for factor in factors:
                if factor.relation is not None:
                    parts.extend(
                        [
                            factor.relation.raw_label,
                            factor.relation.canonical_label.replace("_", " "),
                        ]
                    )
                parts.append(str(factor.metadata.get("raw_relation") or ""))
                event = events.get(str(factor.metadata.get("event_uid") or ""))
                if event is not None:
                    parts.append(event.raw_span or "")
                parts.extend(self._label(uid) for uid in factor.roles.values())
            docs[label] = " | ".join(
                dict.fromkeys(part.strip() for part in parts if part.strip())
            )
        self._relation_docs = docs
        self._relation_vectors = {
            label: tuple(float(x) for x in self.embed(document))
            for label, document in docs.items()
        }
        self._relation_label_vectors = {
            label: tuple(
                float(x)
                for x in self.embed(label.lower().replace("_", " "))
            )
            for label in docs
        }
        self._revision = revision

    @staticmethod
    def _candidate_relation(candidate) -> str:
        return str(
            candidate.canonical_relation
            or candidate.predicate
            or ""
        ).upper()

    @staticmethod
    def _query_metadata(perception: PerceptionResult) -> dict:
        raw = perception.meta.get("llm_raw")
        if not isinstance(raw, dict):
            return {}
        query = raw.get("query")
        return dict(query) if isinstance(query, dict) else {}

    def _relation_text(
        self,
        question: str,
        perception: PerceptionResult,
        query_meta: Mapping,
        anchors: set[str],
    ) -> str:
        parts: list[str] = []
        relation = query_meta.get("relation")
        if relation:
            parts.append(str(relation).replace("_", " "))
        for candidate in perception.candidates:
            if candidate.statement_type not in {"topic", "open_question"}:
                continue
            parts.extend(
                [
                    (
                        candidate.canonical_relation
                        or candidate.predicate
                    ).replace("_", " "),
                    candidate.raw_relation or "",
                ]
            )
        query_terms = self._semantic_query_terms(question, anchors)
        if query_terms:
            parts.append(" ".join(query_terms))
        elif not parts:
            parts.append(question.replace("_", " "))
        return " | ".join(
            dict.fromkeys(
                part.lower().strip()
                for part in parts
                if part and part.strip()
            )
        )

    def _semantic_query_terms(
        self,
        question: str,
        anchors: set[str],
    ) -> list[str]:
        anchor_terms: set[str] = set()
        for uid in anchors:
            anchor_terms.update(self._text_lemmas(self._label(uid)))
            bare = uid.removeprefix("M_")
            abstract = self.store.ah.S.get(bare)
            if abstract is not None:
                for form in abstract.R.get("TEXT") or set():
                    anchor_terms.update(self._text_lemmas(form))
        return [
            token
            for token in self._text_lemmas(question)
            if token not in anchor_terms and token not in STOP
        ]

    @staticmethod
    def _text_lemmas(text: str) -> list[str]:
        return [
            normalized
            for token in re.findall(r"[a-zа-яё0-9]+", text.lower())
            if len(token) > 1
            and (normalized := lemma(token))
            and normalized not in STOP
        ]

    def _target_role(
        self,
        question: str,
        perception: PerceptionResult,
        query_meta: Mapping,
    ) -> str:
        explicit = str(query_meta.get("target_role") or "").upper()
        if explicit in _ROLES:
            return explicit
        for candidate in perception.candidates:
            if candidate.statement_type not in {"topic", "open_question"}:
                continue
            unresolved = [
                role
                for role, uid in candidate.roles.items()
                if role != "SUBJECT"
                and f"M_{uid.removeprefix('M_')}" not in self.store.ah.all_hyper()
            ]
            if unresolved:
                return unresolved[0]
            for role in ("OBJECT", "LOCATION", "TIME", "CAUSE"):
                if role in candidate.roles:
                    return role
        low = question.lower().replace("ё", "е")
        protocol_cues = (
            (("где", "куда", "откуда"), "LOCATION"),
            (("когда",), "TIME"),
            (("почему", "зачем"), "CAUSE"),
            (("чем",), "TOOL"),
        )
        for cues, role in protocol_cues:
            if any(cue in low.split() for cue in cues):
                return role
        return "OBJECT"

    def _label(self, uid: str) -> str:
        try:
            symbol = self.store.get_symbol(uid)
        except Exception:
            symbol = None
        if symbol is not None:
            for prop in symbol.Pr:
                if prop.name == "label" and prop.value:
                    return prop.value
        bare = uid.removeprefix("M_")
        abstract = self.store.ah.S.get(bare)
        if abstract is not None:
            forms = abstract.R.get("TEXT") or set()
            if forms:
                return sorted(forms)[0]
        return bare.replace("_", " ").lower()
