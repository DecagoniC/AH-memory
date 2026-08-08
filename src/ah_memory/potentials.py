"""Pluggable factor potentials for persistent message passing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ah_memory.factor_graph import Factor

Message = tuple[float, float]


def normalize(message: Message) -> Message:
    total = message[0] + message[1]
    if total <= 0.0 or not math.isfinite(total):
        return (0.5, 0.5)
    return (message[0] / total, message[1] / total)


@dataclass(frozen=True)
class PotentialParameters:
    association_strength: float = 0.4
    bind_strength: float = 0.8
    is_a_up_weight: float = 0.9
    is_a_down_weight: float = 0.2
    follow_forward_weight: float = 0.8
    follow_backward_weight: float = 0.1
    cause_forward_weight: float = 0.8
    cause_backward_weight: float = 0.15
    hypernode_strength: float = 0.6
    hypernode_mode: str = "soft_and"
    hypernode_lambda: float = 0.25
    factor_evaluation: str = "auto"
    exact_max_arity: int = 12


class FactorPotential(Protocol):
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message: ...


def _weighted_binary(source: Message, weight: float) -> Message:
    """Soft excitatory message; neutral at zero strength."""
    p = source[1]
    gain = max(0.0, weight)
    score = 0.5 + (p - 0.5) * min(1.0, gain)
    return normalize((1.0 - score, score))


class PriorPotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        return normalize((1.0, max(1e-12, factor.epsilon)))


class ObservationPotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        return normalize((1.0, math.exp(min(50.0, factor.lambda_obs))))


class AssociativePotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        source = _other_message(factor, target_variable, incoming_messages)
        return _weighted_binary(
            source,
            factor.w * parameters.association_strength,
        )


class BindPotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        source = _other_message(factor, target_variable, incoming_messages)
        return _weighted_binary(source, factor.w * parameters.bind_strength)


class IsAPotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        source = _other_message(factor, target_variable, incoming_messages)
        # variables[0]=child, variables[1]=parent
        weight = (
            parameters.is_a_up_weight
            if len(factor.variables) > 1 and target_variable == factor.variables[1]
            else parameters.is_a_down_weight
        )
        return _weighted_binary(source, factor.w * weight)


class FollowPotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        source = _other_message(factor, target_variable, incoming_messages)
        # variables[0]=earlier, variables[1]=later
        weight = (
            parameters.follow_forward_weight
            if len(factor.variables) > 1 and target_variable == factor.variables[1]
            else parameters.follow_backward_weight
        )
        return _weighted_binary(source, factor.w * weight)


class CausePotential:
    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        others = [
            incoming_messages.get(uid, (0.5, 0.5))[1]
            for uid in factor.variables
            if uid != target_variable
        ]
        probability = sum(others) / len(others) if others else 0.5
        source = (1.0 - probability, probability)
        object_uid = factor.roles.get("OBJECT")
        weight = (
            parameters.cause_forward_weight
            if object_uid is not None and target_variable == object_uid
            else parameters.cause_backward_weight
        )
        return _weighted_binary(source, factor.w * weight)


class SemanticPotential:
    """Generic n-ary potential parameterized by Relation and FactorParameters."""

    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        weighted: list[tuple[float, float]] = []
        for source_uid in factor.variables:
            if source_uid == target_variable:
                continue
            coefficient = factor.transmission(source_uid, target_variable)
            probability = incoming_messages.get(source_uid, (0.5, 0.5))[1]
            weighted.append((probability, coefficient))
        total_weight = sum(weight for _, weight in weighted)
        if total_weight <= 0.0:
            return (0.5, 0.5)
        source_probability = sum(
            probability * weight for probability, weight in weighted
        ) / total_weight
        strength = min(1.0, max(0.0, factor.w * factor.confidence))
        score = 0.5 + (source_probability - 0.5) * strength
        return normalize((1.0 - score, score))


class HypernodePotential:
    """N-ary factor with explicit exact/approximate evaluation."""

    def message_to(
        self,
        factor: "Factor",
        target_variable: str,
        incoming_messages: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        others = [u for u in factor.variables if u != target_variable]
        mode = parameters.factor_evaluation.lower()
        if mode not in {"exact", "approximate", "auto"}:
            raise ValueError(f"unknown factor_evaluation: {mode}")
        use_exact = mode == "exact" or (
            mode == "auto" and len(factor.variables) <= parameters.exact_max_arity
        )
        if use_exact:
            return self._exact(factor, target_variable, others, incoming_messages, parameters)
        return self._approximate(factor, others, incoming_messages, parameters)

    def _exact(
        self,
        factor: "Factor",
        target: str,
        others: list[str],
        incoming: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        acc = [0.0, 0.0]
        for configuration in product((0, 1), repeat=len(others)):
            assignment = dict(zip(others, configuration))
            incoming_product = 1.0
            for uid, value in assignment.items():
                message = incoming.get(uid, (0.5, 0.5))
                incoming_product *= message[value]
            for target_value in (0, 1):
                assignment[target] = target_value
                acc[target_value] += (
                    self._potential_value(factor, assignment, parameters)
                    * incoming_product
                )
        return normalize((acc[0], acc[1]))

    def _approximate(
        self,
        factor: "Factor",
        others: list[str],
        incoming: Mapping[str, Message],
        parameters: PotentialParameters,
    ) -> Message:
        if not others:
            return (0.5, 0.5)
        probabilities = [incoming.get(u, (0.5, 0.5))[1] for u in others]
        mode = parameters.hypernode_mode.lower()
        if mode == "and":
            signal = math.prod(probabilities)
        elif mode == "pairwise":
            pairs = [
                probabilities[i] * probabilities[j]
                for i in range(len(probabilities))
                for j in range(i + 1, len(probabilities))
            ]
            signal = sum(pairs) / len(pairs) if pairs else probabilities[0]
        elif mode == "soft_and":
            signal = (
                sum(probabilities) / len(probabilities)
                + parameters.hypernode_lambda * math.prod(probabilities)
            ) / (1.0 + parameters.hypernode_lambda)
        else:
            raise ValueError(f"unknown hypernode_mode: {mode}")
        return _weighted_binary(
            (1.0 - signal, signal),
            factor.w * parameters.hypernode_strength,
        )

    @staticmethod
    def _potential_value(
        factor: "Factor",
        assignment: Mapping[str, int],
        parameters: PotentialParameters,
    ) -> float:
        values = [assignment.get(uid, 0) for uid in factor.variables]
        mode = parameters.hypernode_mode.lower()
        if mode == "and":
            semantic = float(math.prod(values))
        elif mode == "pairwise":
            pairs = [
                values[i] * values[j]
                for i in range(len(values))
                for j in range(i + 1, len(values))
            ]
            semantic = sum(pairs) / len(pairs) if pairs else float(values[0])
        elif mode == "soft_and":
            semantic = (
                sum(values) / max(1, len(values))
                + parameters.hypernode_lambda * math.prod(values)
            )
        else:
            raise ValueError(f"unknown hypernode_mode: {mode}")
        strength = max(0.0, factor.w * parameters.hypernode_strength)
        return math.exp(min(50.0, strength * semantic))


def evaluation_mode_for(
    factor: "Factor",
    parameters: PotentialParameters,
) -> str:
    if (factor.potential_key or factor.kind.value) != "hypernode":
        return "closed_form"
    configured = parameters.factor_evaluation.lower()
    if configured == "auto":
        return (
            "exact"
            if len(factor.variables) <= parameters.exact_max_arity
            else "approximate"
        )
    return configured


def _other_message(
    factor: "Factor",
    target: str,
    incoming: Mapping[str, Message],
) -> Message:
    for uid in factor.variables:
        if uid != target:
            return incoming.get(uid, (0.5, 0.5))
    return (0.5, 0.5)


def default_potential_registry() -> dict[str, FactorPotential]:
    return {
        "prior": PriorPotential(),
        "obs": ObservationPotential(),
        "assoc": AssociativePotential(),
        "bind": BindPotential(),
        "is_a": IsAPotential(),
        "follow": FollowPotential(),
        "cause": CausePotential(),
        "semantic": SemanticPotential(),
        "hypernode": HypernodePotential(),
        "pair": AssociativePotential(),
    }
