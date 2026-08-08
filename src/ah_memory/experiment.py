"""Reproducible configuration and runner for ignition experiments."""
from __future__ import annotations

import csv
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ah_memory.activation import ActivationParameters, activation_by_name
from ah_memory.belief_propagation import BPState, BeliefPropagation
from ah_memory.competition import CompetitionParameters, competition_by_name
from ah_memory.factor_graph import FactorGraph
from ah_memory.potentials import PotentialParameters


@dataclass(frozen=True)
class InferenceConfig:
    max_ticks: int = 20
    threshold: float = 0.7
    convergence_epsilon: float = 0.001
    damping: float = 0.4
    evidence_decay: float = 0.5
    factor_evaluation: str = "auto"
    exact_max_arity: int = 12
    contribution_mode: str = "delta"


@dataclass(frozen=True)
class ActivationConfig:
    type: str = "linear"
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    decay: float = 0.1
    eta: float = 0.5


@dataclass(frozen=True)
class FactorConfig:
    is_a_up: float = 0.9
    is_a_down: float = 0.2
    follow_forward: float = 0.8
    follow_backward: float = 0.1
    association: float = 0.4
    bind: float = 0.8
    cause_forward: float = 0.8
    cause_backward: float = 0.15
    hypernode_strength: float = 0.6
    hypernode_mode: str = "soft_and"
    hypernode_lambda: float = 0.25


@dataclass(frozen=True)
class CompetitionConfig:
    enabled: bool = False
    type: str = "none"
    strength: float = 0.0
    top_k: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    random_seed: int = 42
    history_retention: int = 100

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ExperimentConfig":
        raw = raw or {}
        return cls(
            activation=ActivationConfig(**dict(raw.get("activation") or {})),
            factors=FactorConfig(**dict(raw.get("factors") or {})),
            inference=InferenceConfig(**dict(raw.get("inference") or {})),
            competition=CompetitionConfig(**dict(raw.get("competition") or {})),
            random_seed=int(raw.get("random_seed", 42)),
            history_retention=int(raw.get("history_retention", 100)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    state: BPState
    metrics: dict[str, float]
    graph_construction_ms: float
    total_run_ms: float
    config: ExperimentConfig

    def summary(self) -> dict[str, Any]:
        mode_counts: dict[str, int] = {}
        for mode in self.state.factor_evaluation_modes.values():
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        return {
            "name": self.name,
            **self.metrics,
            "graph_construction_ms": self.graph_construction_ms,
            "bp_step_ms": self.state.timings_ms.get("bp_step", 0.0),
            "activation_update_ms": self.state.timings_ms.get(
                "activation_update",
                0.0,
            ),
            "total_tick_ms": self.state.timings_ms.get("total_tick", 0.0),
            "total_run_ms": self.total_run_ms,
            "ticks": self.state.tick,
            "exact_factors": mode_counts.get("exact", 0),
            "approximate_factors": mode_counts.get("approximate", 0),
        }


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig.from_mapping(raw.get("experiment", raw))


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        random.seed(self.config.random_seed)

    def make_bp(self) -> BeliefPropagation:
        cfg = self.config
        return BeliefPropagation(
            damp=cfg.inference.damping,
            rounds=1,
            potential_parameters=PotentialParameters(
                association_strength=cfg.factors.association,
                bind_strength=cfg.factors.bind,
                is_a_up_weight=cfg.factors.is_a_up,
                is_a_down_weight=cfg.factors.is_a_down,
                follow_forward_weight=cfg.factors.follow_forward,
                follow_backward_weight=cfg.factors.follow_backward,
                cause_forward_weight=cfg.factors.cause_forward,
                cause_backward_weight=cfg.factors.cause_backward,
                hypernode_strength=cfg.factors.hypernode_strength,
                hypernode_mode=cfg.factors.hypernode_mode,
                hypernode_lambda=cfg.factors.hypernode_lambda,
                factor_evaluation=cfg.inference.factor_evaluation,
                exact_max_arity=cfg.inference.exact_max_arity,
            ),
            activation_function=activation_by_name(cfg.activation.type),
            activation_parameters=ActivationParameters(
                decay=cfg.activation.decay,
                eta=cfg.activation.eta,
                alpha=cfg.activation.alpha,
                beta=cfg.activation.beta,
                gamma=cfg.activation.gamma,
            ),
            competition=competition_by_name(cfg.competition.type),
            competition_parameters=CompetitionParameters(
                enabled=cfg.competition.enabled,
                strength=cfg.competition.strength,
                top_k=cfg.competition.top_k,
            ),
            working_memory_threshold=cfg.inference.threshold,
            history_retention=cfg.history_retention,
            trace_retention=cfg.history_retention,
            contribution_mode=cfg.inference.contribution_mode,
        )

    def run(
        self,
        name: str,
        graph: FactorGraph,
        evidence: Mapping[str, float],
        *,
        relevant: Iterable[str] = (),
        irrelevant: Iterable[str] = (),
    ) -> ExperimentResult:
        from ah_memory.benchmarks.metrics import calculate_metrics

        started = time.perf_counter()
        bp = self.make_bp()
        graph_ms = (time.perf_counter() - started) * 1000.0
        state = bp.initialize(graph, evidence)
        for _ in range(self.config.inference.max_ticks):
            previous = dict(state.activation)
            decayed_evidence = {
                uid: value * self.config.inference.evidence_decay
                for uid, value in state.evidence.items()
                if value * self.config.inference.evidence_decay >= 1e-6
            }
            state = bp.step(graph, state, decayed_evidence)
            delta = max(
                (
                    abs(state.activation.get(uid, 0.0) - previous.get(uid, 0.0))
                    for uid in graph.variables
                ),
                default=0.0,
            )
            if delta <= self.config.inference.convergence_epsilon and state.tick > 1:
                break
        total_ms = (time.perf_counter() - started) * 1000.0
        retention = max(1, self.config.history_retention)
        state.activation_history = state.activation_history[-retention:]
        state.belief_history = state.belief_history[-retention:]
        state.message_history = state.message_history[-retention:]
        state.working_memory_history = state.working_memory_history[-retention:]
        state.trace = state.trace[-retention:]
        metrics = calculate_metrics(
            state.activation_history,
            threshold=self.config.inference.threshold,
            relevant=set(relevant),
            irrelevant=set(irrelevant),
        )
        return ExperimentResult(
            name=name,
            state=state,
            metrics=metrics,
            graph_construction_ms=graph_ms,
            total_run_ms=total_ms,
            config=self.config,
        )


def grid_search(
    scenario_factory,
    output_csv: str | Path,
    *,
    base: ExperimentConfig | None = None,
    activation_types: Iterable[str] = ("linear", "sigmoid", "relu"),
    decays: Iterable[float] = (0.01, 0.05, 0.1),
    thresholds: Iterable[float] = (0.5, 0.7, 0.9),
    factor_strengths: Iterable[float] = (0.2, 0.5, 0.8),
) -> list[dict[str, Any]]:
    base = base or ExperimentConfig()
    rows: list[dict[str, Any]] = []
    for activation_type, decay, threshold, strength in itertools.product(
        activation_types,
        decays,
        thresholds,
        factor_strengths,
    ):
        config = replace(
            base,
            activation=replace(
                base.activation,
                type=activation_type,
                decay=float(decay),
            ),
            inference=replace(base.inference, threshold=float(threshold)),
            factors=replace(
                base.factors,
                association=float(strength),
                hypernode_strength=float(strength),
            ),
        )
        scenario = scenario_factory()
        result = ExperimentRunner(config).run(
            scenario.name,
            scenario.graph,
            scenario.evidence,
            relevant=scenario.relevant,
            irrelevant=scenario.irrelevant,
        )
        rows.append(
            {
                "activation": activation_type,
                "decay": decay,
                "threshold": threshold,
                "factor_strength": strength,
                **result.summary(),
            }
        )
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    path.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows
