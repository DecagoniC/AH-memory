"""Continuous activation dynamics independent of factor semantics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class ActivationParameters:
    decay: float = 0.1
    eta: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


class ActivationFunction(Protocol):
    def __call__(
        self,
        previous_activation: float,
        incoming_signal: float,
        evidence: float,
        parameters: ActivationParameters,
    ) -> float: ...


class LinearDecayActivation:
    def __call__(
        self,
        previous_activation: float,
        incoming_signal: float,
        evidence: float,
        parameters: ActivationParameters,
    ) -> float:
        return clip01(
            (1.0 - parameters.decay) * previous_activation
            + parameters.eta * incoming_signal
            + evidence
        )


class SigmoidActivation:
    def __call__(
        self,
        previous_activation: float,
        incoming_signal: float,
        evidence: float,
        parameters: ActivationParameters,
    ) -> float:
        value = (
            parameters.alpha * previous_activation
            + parameters.beta * incoming_signal
            + parameters.gamma * evidence
        )
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)


class SaturatedReLUActivation:
    def __call__(
        self,
        previous_activation: float,
        incoming_signal: float,
        evidence: float,
        parameters: ActivationParameters,
    ) -> float:
        return clip01(
            (1.0 - parameters.decay) * previous_activation
            + parameters.eta * max(0.0, incoming_signal + evidence)
        )


def activation_by_name(name: str) -> ActivationFunction:
    normalized = name.strip().lower()
    if normalized in {"linear", "linear_decay"}:
        return LinearDecayActivation()
    if normalized in {"sigmoid", "logistic"}:
        return SigmoidActivation()
    if normalized in {"relu", "leaky", "saturated_relu"}:
        return SaturatedReLUActivation()
    raise ValueError(f"unknown activation function: {name}")
