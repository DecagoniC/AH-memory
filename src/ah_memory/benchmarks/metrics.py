"""Metrics for activation trajectories."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def calculate_metrics(
    history: Sequence[Mapping[str, float]],
    *,
    threshold: float,
    relevant: set[str] | None = None,
    irrelevant: set[str] | None = None,
) -> dict[str, float]:
    relevant = relevant or set()
    irrelevant = irrelevant or set()
    if not history:
        return {
            "propagation_latency": math.inf,
            "peak_activation": 0.0,
            "activation_half_life": math.inf,
            "spread": 0.0,
            "selectivity": 0.0,
            "stability": 1.0,
            "oscillation": 0.0,
            "convergence": 0.0,
        }

    nodes = set().union(*(snapshot.keys() for snapshot in history))
    target_nodes = relevant or nodes
    latency = math.inf
    for tick, snapshot in enumerate(history):
        if any(snapshot.get(uid, 0.0) >= threshold for uid in target_nodes):
            latency = float(tick)
            break

    peak = max(
        (value for snapshot in history for value in snapshot.values()),
        default=0.0,
    )
    final = history[-1]
    spread = float(sum(value >= threshold for value in final.values()))
    comparison = irrelevant or (nodes - relevant)
    relevant_mean = (
        sum(final.get(uid, 0.0) for uid in relevant) / len(relevant)
        if relevant
        else 0.0
    )
    irrelevant_mean = (
        sum(final.get(uid, 0.0) for uid in comparison) / len(comparison)
        if comparison
        else 0.0
    )
    selectivity = relevant_mean / max(1e-12, irrelevant_mean)
    all_active_ratio = (
        sum(value >= threshold for value in final.values()) / max(1, len(final))
    )
    stability = 1.0 - all_active_ratio

    convergence = 0.0
    if len(history) >= 2:
        convergence = max(
            abs(history[-1].get(uid, 0.0) - history[-2].get(uid, 0.0))
            for uid in nodes
        )

    oscillation = _oscillation_score(history, nodes)
    half_life = _half_life(history, target_nodes)
    return {
        "propagation_latency": latency,
        "peak_activation": peak,
        "activation_half_life": half_life,
        "spread": spread,
        "selectivity": selectivity,
        "stability": stability,
        "oscillation": oscillation,
        "convergence": convergence,
    }


def _half_life(
    history: Sequence[Mapping[str, float]],
    nodes: set[str],
) -> float:
    if not nodes:
        return math.inf
    series = [max(snapshot.get(uid, 0.0) for uid in nodes) for snapshot in history]
    peak = max(series, default=0.0)
    if peak <= 0.0:
        return math.inf
    peak_tick = series.index(peak)
    for tick in range(peak_tick + 1, len(series)):
        if series[tick] <= peak / 2.0:
            return float(tick - peak_tick)
    return math.inf


def _oscillation_score(
    history: Sequence[Mapping[str, float]],
    nodes: set[str],
) -> float:
    if len(history) < 4 or not nodes:
        return 0.0
    changes = []
    for uid in nodes:
        values = [snapshot.get(uid, 0.0) for snapshot in history]
        signs = [
            1 if b - a > 1e-6 else -1 if b - a < -1e-6 else 0
            for a, b in zip(values, values[1:])
        ]
        alternations = sum(
            left != 0 and right != 0 and left != right
            for left, right in zip(signs, signs[1:])
        )
        changes.append(alternations / max(1, len(signs) - 1))
    return sum(changes) / len(changes)
