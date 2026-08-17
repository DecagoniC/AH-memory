"""Strategy-based raw-to-canonical relation normalization."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from ah_memory.relation_registry import RelationRegistry, cosine_similarity
from ah_memory.relations import (
    NormalizedRelation,
    RelationContext,
    RelationProperties,
    Vector,
    canonicalize_label,
)

EmbeddingFunction = Callable[[str], Sequence[float]]


def deterministic_embedding(text: str, dimensions: int = 32) -> Vector:
    """Dependency-free character-ngram embedding for deterministic baselines."""
    normalized = f"  {text.strip().lower().replace('ё', 'е')}  "
    vector = [0.0] * dimensions
    for size in (2, 3, 4):
        for index in range(max(0, len(normalized) - size + 1)):
            gram = normalized[index : index + size].encode("utf-8")
            digest = hashlib.blake2b(gram, digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


class NormalizationStrategy(Protocol):
    name: str

    def normalize(
        self,
        raw_relation: str,
        registry: RelationRegistry,
        context: RelationContext | None = None,
    ) -> NormalizedRelation | None: ...


class ExactNormalizer:
    name = "exact"

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        source = aliases or {}
        self.aliases = {
            self._key(raw): canonicalize_label(canonical)
            for raw, canonical in source.items()
        }

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(value.strip().lower().replace("ё", "е").split())

    def normalize(
        self,
        raw_relation: str,
        registry: RelationRegistry,
        context: RelationContext | None = None,
    ) -> NormalizedRelation | None:
        canonical = self.aliases.get(self._key(raw_relation))
        if canonical is None:
            direct = registry.get_relation(raw_relation)
            canonical = direct.canonical_label if direct is not None else None
        if canonical is None:
            return None
        relation = registry.get_relation(canonical)
        properties = (
            relation.properties if relation is not None else RelationProperties()
        )
        return NormalizedRelation(
            raw_label=raw_relation,
            canonical_label=canonical,
            confidence=1.0,
            embedding=relation.embedding if relation is not None else None,
            properties=properties,
            strategy=self.name,
        )


class EmbeddingNormalizer:
    name = "embedding"

    def __init__(
        self,
        embedder: EmbeddingFunction = deterministic_embedding,
        *,
        similarity_threshold: float = 0.55,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def normalize(
        self,
        raw_relation: str,
        registry: RelationRegistry,
        context: RelationContext | None = None,
    ) -> NormalizedRelation | None:
        raw_embedding = tuple(float(value) for value in self.embedder(raw_relation))
        best = None
        for relation in registry.list_relations():
            relation_embedding = relation.embedding or tuple(
                float(value) for value in self.embedder(relation.canonical_label)
            )
            similarity = cosine_similarity(raw_embedding, relation_embedding)
            if best is None or similarity > best[1]:
                best = (relation, similarity)
        if best is None or best[1] < self.similarity_threshold:
            return None
        relation, similarity = best
        return NormalizedRelation(
            raw_label=raw_relation,
            canonical_label=relation.canonical_label,
            confidence=min(1.0, max(0.0, similarity)),
            embedding=raw_embedding,
            properties=relation.properties,
            strategy=self.name,
        )


class LLMNormalizer:
    name = "llm"

    def __init__(self, call_fn: Callable[[str], str | Mapping]) -> None:
        self.call_fn = call_fn

    def normalize(
        self,
        raw_relation: str,
        registry: RelationRegistry,
        context: RelationContext | None = None,
    ) -> NormalizedRelation | None:
        prompt = {
            "task": "normalize_relation",
            "raw_relation": raw_relation,
            "known_canonical_relations": [
                relation.canonical_label for relation in registry.list_relations()
            ],
            "context": context.to_dict() if context is not None else None,
            "response_schema": {
                "canonical_label": "UPPER_SNAKE_CASE",
                "confidence": "float 0..1",
                "create_new": "bool",
                "properties": {
                    "directional": "bool",
                    "symmetric": "bool",
                    "transitive": "bool",
                    "temporal": "bool",
                    "causal": "bool",
                    "state_changing": "bool",
                },
            },
        }
        raw = self.call_fn(json.dumps(prompt, ensure_ascii=False))
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        canonical = canonicalize_label(str(data.get("canonical_label") or raw_relation))
        known = registry.get_relation(canonical)
        create_new = bool(data.get("create_new", known is None))
        properties_raw = dict(data.get("properties") or {})
        properties = (
            known.properties
            if known is not None and not properties_raw
            else RelationProperties(
                directional=bool(properties_raw.get("directional", True)),
                symmetric=bool(properties_raw.get("symmetric", False)),
                transitive=bool(properties_raw.get("transitive", False)),
                temporal=bool(properties_raw.get("temporal", False)),
                causal=bool(properties_raw.get("causal", False)),
                state_changing=bool(properties_raw.get("state_changing", False)),
            )
        )
        return NormalizedRelation(
            raw_label=raw_relation,
            canonical_label=canonical,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            properties=properties,
            strategy=self.name,
            created=create_new,
        )


class RelationNormalizer:
    def __init__(
        self,
        registry: RelationRegistry,
        strategies: Sequence[NormalizationStrategy] | None = None,
        *,
        create_unknown: bool = True,
    ) -> None:
        self.registry = registry
        self.strategies = tuple(strategies or (ExactNormalizer(),))
        self.create_unknown = create_unknown

    def normalize(
        self,
        raw_relation: str,
        context: RelationContext | None = None,
    ) -> NormalizedRelation:
        raw_relation = raw_relation.strip()
        if not raw_relation:
            raise ValueError("raw_relation must not be empty")
        for strategy in self.strategies:
            normalized = strategy.normalize(raw_relation, self.registry, context)
            if normalized is None:
                continue
            self._register(normalized, context)
            return normalized
        if not self.create_unknown:
            raise LookupError(raw_relation)
        normalized = NormalizedRelation(
            raw_label=raw_relation,
            canonical_label=canonicalize_label(raw_relation),
            confidence=0.5,
            embedding=deterministic_embedding(raw_relation),
            properties=RelationProperties(directional=True),
            strategy="new_relation",
            created=True,
        )
        self._register(normalized, context)
        return normalized

    def _register(
        self,
        normalized: NormalizedRelation,
        context: RelationContext | None,
    ) -> None:
        existing = self.registry.get_relation(normalized.canonical_label)
        if existing is not None:
            if existing.embedding is None and normalized.embedding is not None:
                self.registry.register_relation(
                    replace(existing, embedding=normalized.embedding),
                    replace=True,
                )
            return
        arity = max(1, len(context.roles)) if context is not None else 2
        self.registry.register_relation(normalized.to_relation(arity=arity))
