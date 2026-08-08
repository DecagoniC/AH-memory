"""Relation-agnostic parameterized activation over semantic factors."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from ah_memory.factor_graph import Factor, FactorGraph
from ah_memory.factor_parameters import (
    FactorParameterGenerator,
    FactorParameters,
    RuleBasedParameterGenerator,
)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class Message:
    activation: float
    source_uid: str
    target_uid: str
    factor_uid: str
    timestep: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation": self.activation,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "factor_uid": self.factor_uid,
            "timestep": self.timestep,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActivationContext:
    timestep: int
    transmission: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ActivationFunction(Protocol):
    def compute(
        self,
        source_activation: float,
        target_activation: float,
        factor: Factor,
        parameters: FactorParameters,
        context: ActivationContext,
    ) -> float: ...


class LinearActivation:
    def compute(
        self,
        source_activation: float,
        target_activation: float,
        factor: Factor,
        parameters: FactorParameters,
        context: ActivationContext,
    ) -> float:
        signal = (
            source_activation
            * factor.weight
            * factor.confidence
            * context.transmission
        )
        return _clip01(signal)


class SigmoidActivation:
    def compute(
        self,
        source_activation: float,
        target_activation: float,
        factor: Factor,
        parameters: FactorParameters,
        context: ActivationContext,
    ) -> float:
        linear = (
            source_activation
            * factor.weight
            * factor.confidence
            * context.transmission
        )
        centered = 8.0 * (linear - parameters.selectivity)
        sigmoid = 1.0 / (1.0 + math.exp(-centered))
        return _clip01(sigmoid)


class SaturatingReLUActivation:
    def compute(
        self,
        source_activation: float,
        target_activation: float,
        factor: Factor,
        parameters: FactorParameters,
        context: ActivationContext,
    ) -> float:
        signal = (
            source_activation
            * factor.weight
            * factor.confidence
            * context.transmission
            - parameters.selectivity * 0.25
        )
        return _clip01(max(0.0, signal))


class DecayActivation:
    def compute(
        self,
        source_activation: float,
        target_activation: float,
        factor: Factor,
        parameters: FactorParameters,
        context: ActivationContext,
    ) -> float:
        retained = target_activation * (1.0 - parameters.decay)
        incoming = (
            source_activation
            * factor.weight
            * factor.confidence
            * context.transmission
        )
        return _clip01(retained + incoming)


@dataclass
class PropagationTrace:
    activated_nodes: list[str] = field(default_factory=list)
    activated_factors: list[str] = field(default_factory=list)
    timesteps: list[dict[str, float]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    state_transitions: list[dict[str, Any]] = field(default_factory=list)
    final_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivationEngine:
    def __init__(
        self,
        activation_function: ActivationFunction | None = None,
        parameter_generator: FactorParameterGenerator | None = None,
    ) -> None:
        self.activation_function = activation_function or LinearActivation()
        self.parameter_generator = (
            parameter_generator or RuleBasedParameterGenerator()
        )

    def propagate(
        self,
        message: Message,
        factor: Factor,
        source: Any,
        target: Any,
    ) -> Message:
        source_uid = self._uid(source)
        target_uid = self._uid(target)
        parameters = factor.parameters or self._parameters(factor)
        transmission = factor.transmission(source_uid, target_uid)
        target_activation = float(
            message.metadata.get("target_activation", 0.0)
        )
        output = self.activation_function.compute(
            message.activation,
            target_activation,
            factor,
            parameters,
            ActivationContext(
                timestep=message.timestep,
                transmission=transmission,
                metadata=message.metadata,
            ),
        )
        relation_label = (
            factor.relation.canonical_label
            if factor.relation is not None
            else factor.potential_key or factor.link_id or "RELATED_TO"
        )
        return Message(
            activation=output,
            source_uid=source_uid,
            target_uid=target_uid,
            factor_uid=factor.uid,
            timestep=message.timestep + 1,
            metadata={
                **dict(message.metadata),
                "relation": relation_label,
                "weight": factor.weight,
                "confidence": factor.confidence,
                "transmission": transmission,
                "activation_before": target_activation,
                "activation_after": output,
                "parameters": parameters.to_dict(),
            },
        )

    def run(
        self,
        graph: FactorGraph,
        evidence: Mapping[str, float],
        *,
        timesteps: int = 3,
        threshold: float = 0.1,
        trace: bool = False,
    ) -> tuple[dict[str, float], PropagationTrace]:
        activation = {
            uid: _clip01(float(evidence.get(uid, 0.0)))
            for uid in graph.variables
        }
        result_trace = PropagationTrace(timesteps=[dict(activation)])
        for timestep in range(max(0, timesteps)):
            incoming: dict[str, float] = {uid: 0.0 for uid in graph.variables}
            for factor in graph.factors:
                if len(factor.variables) < 2:
                    continue
                for source_uid in factor.variables:
                    if activation.get(source_uid, 0.0) <= 0.0:
                        continue
                    for target_uid in factor.variables:
                        if target_uid == source_uid:
                            continue
                        outgoing = self.propagate(
                            Message(
                                activation=activation[source_uid],
                                source_uid=source_uid,
                                target_uid=target_uid,
                                factor_uid=factor.uid,
                                timestep=timestep,
                                metadata={
                                    "target_activation": activation.get(
                                        target_uid,
                                        0.0,
                                    )
                                },
                            ),
                            factor,
                            source_uid,
                            target_uid,
                        )
                        incoming[target_uid] += outgoing.activation
                        if trace:
                            result_trace.messages.append(outgoing.to_dict())
                if any(incoming.get(uid, 0.0) > 0.0 for uid in factor.variables):
                    if factor.uid not in result_trace.activated_factors:
                        result_trace.activated_factors.append(factor.uid)
                    if factor.relation is not None:
                        relation_data = factor.relation.to_dict()
                        if relation_data not in result_trace.relations:
                            result_trace.relations.append(relation_data)
            activation = {
                uid: _clip01(
                    activation.get(uid, 0.0)
                    * (
                        1.0
                        - self._mean_decay(graph, uid)
                    )
                    + incoming.get(uid, 0.0)
                )
                for uid in graph.variables
            }
            result_trace.timesteps.append(dict(activation))
        result_trace.activated_nodes = [
            uid for uid, value in activation.items() if value >= threshold
        ]
        result_trace.final_evidence = sorted(
            result_trace.activated_nodes,
            key=lambda uid: activation[uid],
            reverse=True,
        )
        return activation, result_trace

    def _parameters(self, factor: Factor) -> FactorParameters:
        if factor.relation is None:
            return FactorParameters()
        return self.parameter_generator.generate(factor.relation)

    def _mean_decay(self, graph: FactorGraph, uid: str) -> float:
        decays = []
        for fid in graph.var_factors.get(uid, ()):
            factor = graph.factors_by_id[fid]
            decays.append(
                (
                    factor.parameters or self._parameters(factor)
                ).decay
            )
        return sum(decays) / len(decays) if decays else 0.1

    @staticmethod
    def _uid(node: Any) -> str:
        return str(getattr(node, "uid", node))
