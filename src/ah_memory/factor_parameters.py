"""Serializable semantic factor parameters and generators."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence

from ah_memory.relation_normalizer import deterministic_embedding
from ah_memory.relations import Relation, canonicalize_label


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class FactorParameters:
    transmission_strength: float = 0.5
    decay: float = 0.1
    directionality: float = 0.5
    selectivity: float = 0.5
    temporal_bias: float = 0.0
    causal_bias: float = 0.0
    persistence: float = 0.5

    def clipped(self) -> "FactorParameters":
        return FactorParameters(
            **{
                key: _clip01(value)
                for key, value in asdict(self).items()
            }
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "FactorParameters":
        padded = list(values[:7]) + [0.5] * max(0, 7 - len(values))
        return cls(*(_clip01(value) for value in padded[:7]))


class FactorParameterGenerator(Protocol):
    def generate(self, relation: Relation) -> FactorParameters: ...


class FixedParameterGenerator:
    def __init__(self, parameters: FactorParameters | None = None) -> None:
        self.parameters = parameters or FactorParameters()

    def generate(self, relation: Relation) -> FactorParameters:
        return self.parameters


class RuleBasedParameterGenerator:
    """Maps semantic properties to parameters without relation-label branches."""

    def __init__(
        self,
        base: FactorParameters | None = None,
        *,
        overrides: Mapping[str, FactorParameters] | None = None,
    ) -> None:
        self.base = base or FactorParameters()
        self.overrides = {
            canonicalize_label(label): parameters
            for label, parameters in (overrides or {}).items()
        }

    def generate(self, relation: Relation) -> FactorParameters:
        override = self.overrides.get(relation.canonical_label)
        if override is not None:
            return override.clipped()
        properties = relation.properties
        directionality = (
            0.9 if properties.directional and not properties.symmetric else 0.5
        )
        transmission = self.base.transmission_strength
        if properties.causal:
            transmission += 0.2
        if properties.state_changing:
            transmission += 0.1
        return FactorParameters(
            transmission_strength=transmission,
            decay=0.05 if properties.temporal else self.base.decay,
            directionality=directionality,
            selectivity=0.75 if properties.causal else self.base.selectivity,
            temporal_bias=0.8 if properties.temporal else 0.0,
            causal_bias=0.8 if properties.causal else 0.0,
            persistence=0.8 if properties.state_changing else self.base.persistence,
        ).clipped()


class EmbeddingParameterGenerator:
    """Deterministic linear projection θ = sigmoid(M e + b)."""

    def __init__(
        self,
        *,
        embedding_dimensions: int = 32,
        seed: int = 42,
        matrix: Sequence[Sequence[float]] | None = None,
        bias: Sequence[float] | None = None,
    ) -> None:
        self.embedding_dimensions = embedding_dimensions
        rng = random.Random(seed)
        self.matrix = [
            list(row)
            for row in (
                matrix
                or [
                    [rng.uniform(-0.35, 0.35) for _ in range(embedding_dimensions)]
                    for _ in range(7)
                ]
            )
        ]
        self.bias = list(bias or [0.0] * 7)
        if len(self.matrix) != 7 or len(self.bias) != 7:
            raise ValueError("projection must produce seven factor parameters")
        if any(len(row) != embedding_dimensions for row in self.matrix):
            raise ValueError("projection matrix dimension mismatch")

    def generate(self, relation: Relation) -> FactorParameters:
        embedding = relation.embedding or deterministic_embedding(
            relation.canonical_label,
            self.embedding_dimensions,
        )
        if len(embedding) != self.embedding_dimensions:
            raise ValueError("relation embedding dimension mismatch")
        projected = [
            self._sigmoid(
                sum(weight * value for weight, value in zip(row, embedding))
                + offset
            )
            for row, offset in zip(self.matrix, self.bias)
        ]
        return FactorParameters.from_vector(projected)

    def set_projection(
        self,
        matrix: Sequence[Sequence[float]],
        bias: Sequence[float],
    ) -> None:
        candidate = EmbeddingParameterGenerator(
            embedding_dimensions=self.embedding_dimensions,
            matrix=matrix,
            bias=bias,
        )
        self.matrix = candidate.matrix
        self.bias = candidate.bias

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            exp = pow(2.718281828459045, -value)
            return 1.0 / (1.0 + exp)
        exp = pow(2.718281828459045, value)
        return exp / (1.0 + exp)
