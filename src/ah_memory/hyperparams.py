"""Ignition / factor-graph hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HyperParams:
    ttl: int = 32
    lambda_decay: float = 0.15  # legacy; prior ε replaces g in FG mode
    threshold_t: float = 0.55
    eta: float = 0.05
    pacemaker_period: int = 8
    seed_delta: float = 0.8
    initial_w: float = 0.5
    # Factor-graph BP (docs/FACTOR_GRAPH_ACTIVATION.md §12)
    fg_kappa: float = 2.0
    fg_lambda: float = 3.0
    fg_epsilon: float = 0.05
    fg_damp: float = 0.4
    fg_rounds: int = 2
    fg_trace_eps: float = 1e-3
    fg_hebb_tau: float = 0.15

    def g(self, x: float, dt: float = 1.0) -> float:
        return x * math.exp(-self.lambda_decay * dt)

    def h(self, w: float, x_out: float, x_in: float) -> float:
        return min(1.0, max(0.0, w + self.eta * x_out * x_in))

    @staticmethod
    def f_sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))
