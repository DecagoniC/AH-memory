"""Optional competition and inhibition over continuous activation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from ah_memory.activation import clip01
from ah_memory.factor_graph import FactorGraph


@dataclass(frozen=True)
class CompetitionParameters:
    enabled: bool = False
    strength: float = 0.0
    top_k: int = 10


class CompetitionFunction(Protocol):
    def apply(
        self,
        activation: Mapping[str, float],
        graph: FactorGraph,
        parameters: CompetitionParameters,
    ) -> dict[str, float]: ...


class NoCompetition:
    def apply(
        self,
        activation: Mapping[str, float],
        graph: FactorGraph,
        parameters: CompetitionParameters,
    ) -> dict[str, float]:
        return dict(activation)


class GlobalInhibition:
    def apply(
        self,
        activation: Mapping[str, float],
        graph: FactorGraph,
        parameters: CompetitionParameters,
    ) -> dict[str, float]:
        if not parameters.enabled or parameters.strength <= 0.0:
            return dict(activation)
        total = sum(activation.values())
        count = max(1, len(activation) - 1)
        return {
            uid: clip01(value - parameters.strength * (total - value) / count)
            for uid, value in activation.items()
        }


class LocalCompetition:
    def apply(
        self,
        activation: Mapping[str, float],
        graph: FactorGraph,
        parameters: CompetitionParameters,
    ) -> dict[str, float]:
        if not parameters.enabled or parameters.strength <= 0.0:
            return dict(activation)
        out: dict[str, float] = {}
        factors = graph.factors_by_id
        for uid, value in activation.items():
            neighbours: set[str] = set()
            for fid in graph.var_factors.get(uid, ()):
                neighbours.update(factors[fid].variables)
            neighbours.discard(uid)
            inhibition = (
                sum(activation.get(other, 0.0) for other in neighbours)
                / max(1, len(neighbours))
            )
            out[uid] = clip01(value - parameters.strength * inhibition)
        return out


class TopKNormalization:
    def apply(
        self,
        activation: Mapping[str, float],
        graph: FactorGraph,
        parameters: CompetitionParameters,
    ) -> dict[str, float]:
        if not parameters.enabled:
            return dict(activation)
        keep = {
            uid
            for uid, _ in sorted(
                activation.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: max(1, parameters.top_k)]
        }
        return {uid: value if uid in keep else 0.0 for uid, value in activation.items()}


def competition_by_name(name: str) -> CompetitionFunction:
    normalized = name.strip().lower()
    if normalized in {"none", "off", "disabled"}:
        return NoCompetition()
    if normalized in {"global", "global_inhibition"}:
        return GlobalInhibition()
    if normalized in {"local", "local_competition"}:
        return LocalCompetition()
    if normalized in {"top_k", "topk"}:
        return TopKNormalization()
    raise ValueError(f"unknown competition function: {name}")
