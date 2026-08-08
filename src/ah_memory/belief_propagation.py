"""Persistent damped message passing on an AH factor graph."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping

from ah_memory.activation import (
    ActivationFunction,
    ActivationParameters,
    LinearDecayActivation,
)
from ah_memory.competition import (
    CompetitionFunction,
    CompetitionParameters,
    NoCompetition,
)
from ah_memory.factor_graph import FactorGraph, FactorKind
from ah_memory.potentials import (
    FactorPotential,
    Message,
    PotentialParameters,
    default_potential_registry,
    evaluation_mode_for,
    normalize,
)


def _norm2(m0: float, m1: float) -> tuple[float, float]:
    s = m0 + m1
    if s <= 0 or not math.isfinite(s):
        return 0.5, 0.5
    return m0 / s, m1 / s


def _damp(
    old: tuple[float, float], new: tuple[float, float], delta: float
) -> tuple[float, float]:
    m0 = (1 - delta) * new[0] + delta * old[0]
    m1 = (1 - delta) * new[1] + delta * old[1]
    return _norm2(m0, m1)


def _logit(value: float) -> float:
    bounded = min(1.0 - 1e-12, max(1e-12, value))
    return math.log(bounded / (1.0 - bounded))


@dataclass
class BPResult:
    beliefs: dict[str, float]  # b_v(1)
    trace_factors: list[str]
    rounds: int
    max_belief: float = 0.0
    state: "BPState | None" = None


@dataclass(frozen=True)
class ActivationEvent:
    tick: int
    source_uid: str
    factor_uid: str
    target_uid: str
    message_value: float
    contribution: float
    evaluation_mode: str = "closed_form"


@dataclass
class BPState:
    tick: int
    variable_to_factor: dict[tuple[str, str], Message]
    factor_to_variable: dict[tuple[str, str], Message]
    beliefs: dict[str, float]
    activation: dict[str, float]
    evidence: dict[str, float]
    working_memory: dict[str, dict] = field(default_factory=dict)
    trace: list[ActivationEvent] = field(default_factory=list)
    activation_history: list[dict[str, float]] = field(default_factory=list)
    belief_history: list[dict[str, float]] = field(default_factory=list)
    message_history: list[dict[tuple[str, str], Message]] = field(default_factory=list)
    working_memory_history: list[dict[str, dict]] = field(default_factory=list)
    graph_signature: tuple = field(default_factory=tuple)
    timings_ms: dict[str, float] = field(default_factory=dict)
    factor_evaluation_modes: dict[str, str] = field(default_factory=dict)


@dataclass
class BeliefPropagation:
    kappa: float = 2.0
    gamma_isa: float = 2.0
    gamma_assoc: float = 1.5
    gamma_follow: float = 1.0
    gamma_default: float = 1.0
    damp: float = 0.4
    rounds: int = 2
    trace_eps: float = 1e-3
    max_arity: int | None = None  # compatibility only; never truncates
    potential_parameters: PotentialParameters = field(default_factory=PotentialParameters)
    activation_function: ActivationFunction = field(default_factory=LinearDecayActivation)
    activation_parameters: ActivationParameters = field(default_factory=ActivationParameters)
    competition: CompetitionFunction = field(default_factory=NoCompetition)
    competition_parameters: CompetitionParameters = field(default_factory=CompetitionParameters)
    potentials: Mapping[str, FactorPotential] = field(
        default_factory=default_potential_registry
    )
    working_memory_threshold: float = 0.7
    history_retention: int = 100
    trace_retention: int = 1000
    contribution_mode: str = "delta"

    def run(self, graph: FactorGraph) -> BPResult:
        """Compatibility wrapper: initialize once and execute configured rounds."""
        if not graph.variables:
            return BPResult(beliefs={}, trace_factors=[], rounds=0)
        state = self.initialize(graph)
        for _ in range(max(1, self.rounds)):
            state = self.step(graph, state)
        trace = self._trace_from_events(state.trace)
        return BPResult(
            beliefs=state.beliefs,
            trace_factors=trace,
            rounds=self.rounds,
            max_belief=max(state.beliefs.values()) if state.beliefs else 0.0,
            state=state,
        )

    def initialize(
        self,
        graph: FactorGraph,
        evidence: Mapping[str, float] | None = None,
    ) -> BPState:
        v2f: dict[tuple[str, str], Message] = {}
        f2v: dict[tuple[str, str], Message] = {}
        for factor in graph.factors:
            for variable in factor.variables:
                v2f[(variable, factor.fid)] = (0.5, 0.5)
                f2v[(factor.fid, variable)] = (0.5, 0.5)
        beliefs = {uid: 0.5 for uid in graph.variables}
        activation = {uid: 0.0 for uid in graph.variables}
        return BPState(
            tick=0,
            variable_to_factor=v2f,
            factor_to_variable=f2v,
            beliefs=beliefs,
            activation=activation,
            evidence={
                uid: max(0.0, float(value))
                for uid, value in (evidence or {}).items()
                if uid in beliefs
            },
            activation_history=[dict(activation)],
            belief_history=[dict(beliefs)],
            message_history=[dict(f2v)],
            working_memory_history=[{}],
            graph_signature=graph.structural_signature,
        )

    def step(
        self,
        graph: FactorGraph,
        state: BPState,
        evidence: Mapping[str, float] | None = None,
        parameters: PotentialParameters | None = None,
    ) -> BPState:
        """Advance one synchronous round using messages from the previous state."""
        started = time.perf_counter()
        state = self._reconcile(graph, state)
        evidence_now = dict(state.evidence)
        if evidence is not None:
            evidence_now = {
                uid: max(0.0, float(value))
                for uid, value in evidence.items()
                if uid in graph.var_factors
            }
        potential_parameters = parameters or self.potential_parameters

        new_v2f: dict[tuple[str, str], Message] = {}
        for variable in graph.variables:
            local = self._evidence_likelihood(evidence_now.get(variable, 0.0))
            for fid in graph.var_factors.get(variable, ()):
                m0, m1 = local
                for other_fid in graph.var_factors.get(variable, ()):
                    if other_fid == fid:
                        continue
                    incoming = state.factor_to_variable[(other_fid, variable)]
                    m0 *= incoming[0]
                    m1 *= incoming[1]
                proposed = normalize((m0, m1))
                new_v2f[(variable, fid)] = _damp(
                    state.variable_to_factor[(variable, fid)],
                    proposed,
                    self.damp,
                )

        new_f2v: dict[tuple[str, str], Message] = {}
        events: list[ActivationEvent] = []
        evaluation_modes: dict[str, str] = {}
        next_tick = state.tick + 1
        for factor in graph.factors:
            evaluation_modes[factor.fid] = evaluation_mode_for(
                factor,
                potential_parameters,
            )
            potential = self.potentials.get(
                factor.potential_key or factor.kind.value,
                self.potentials["pair"],
            )
            incoming = {
                uid: new_v2f[(uid, factor.fid)]
                for uid in factor.variables
            }
            for target in factor.variables:
                proposed = potential.message_to(
                    factor,
                    target,
                    incoming,
                    potential_parameters,
                )
                old = state.factor_to_variable[(factor.fid, target)]
                message = _damp(old, proposed, self.damp)
                new_f2v[(factor.fid, target)] = message
                contribution = self._contribution(message[1], old[1])
                if abs(contribution) > self.trace_eps:
                    events.append(
                        ActivationEvent(
                            tick=next_tick,
                            source_uid=self._event_source(factor.variables, target, incoming),
                            factor_uid=factor.fid,
                            target_uid=target,
                            message_value=message[1],
                            contribution=contribution,
                            evaluation_mode=evaluation_modes[factor.fid],
                        )
                    )

        bp_finished = time.perf_counter()
        beliefs = self._beliefs_from_messages(graph, new_f2v, evidence_now)
        signals = self._incoming_signals(graph, new_f2v)
        activation = {
            uid: self.activation_function(
                state.activation.get(uid, 0.0),
                signals.get(uid, 0.0),
                1.0 - math.exp(-evidence_now.get(uid, 0.0)),
                self.activation_parameters,
            )
            for uid in graph.variables
        }
        activation = self.competition.apply(
            activation,
            graph,
            self.competition_parameters,
        )
        support: dict[str, list[str]] = {}
        for event in events:
            if event.contribution > 0.0:
                support.setdefault(event.target_uid, []).append(event.factor_uid)
        working_memory = {
            uid: {
                "uid": uid,
                "activation": value,
                "entered_at": state.working_memory.get(uid, {}).get(
                    "entered_at",
                    next_tick,
                ),
                "support": support.get(uid, []),
            }
            for uid, value in activation.items()
            if value >= self.working_memory_threshold
        }
        activation_finished = time.perf_counter()
        return BPState(
            tick=next_tick,
            variable_to_factor=new_v2f,
            factor_to_variable=new_f2v,
            beliefs=beliefs,
            activation=activation,
            evidence=evidence_now,
            working_memory=working_memory,
            trace=(state.trace + events)[-max(1, self.trace_retention) :],
            activation_history=(
                state.activation_history + [dict(activation)]
            )[-max(1, self.history_retention) :],
            belief_history=(
                state.belief_history + [dict(beliefs)]
            )[-max(1, self.history_retention) :],
            message_history=(
                state.message_history + [dict(new_f2v)]
            )[-max(1, self.history_retention) :],
            working_memory_history=(
                state.working_memory_history
                + [{uid: dict(entry) for uid, entry in working_memory.items()}]
            )[-max(1, self.history_retention) :],
            graph_signature=graph.structural_signature,
            timings_ms={
                "bp_step": (bp_finished - started) * 1000.0,
                "activation_update": (activation_finished - bp_finished) * 1000.0,
                "total_tick": (activation_finished - started) * 1000.0,
            },
            factor_evaluation_modes=evaluation_modes,
        )

    @staticmethod
    def _evidence_likelihood(value: float) -> Message:
        return normalize((1.0, math.exp(min(50.0, max(0.0, value)))))

    def _beliefs_from_messages(
        self,
        graph: FactorGraph,
        factor_to_variable: Mapping[tuple[str, str], Message],
        evidence: Mapping[str, float],
    ) -> dict[str, float]:
        beliefs: dict[str, float] = {}
        for variable in graph.variables:
            m0, m1 = self._evidence_likelihood(evidence.get(variable, 0.0))
            for fid in graph.var_factors.get(variable, ()):
                incoming = factor_to_variable[(fid, variable)]
                m0 *= incoming[0]
                m1 *= incoming[1]
            beliefs[variable] = normalize((m0, m1))[1]
        return beliefs

    @staticmethod
    def _incoming_signals(
        graph: FactorGraph,
        factor_to_variable: Mapping[tuple[str, str], Message],
    ) -> dict[str, float]:
        signals: dict[str, float] = {}
        for variable in graph.variables:
            positive = [
                max(0.0, 2.0 * factor_to_variable[(fid, variable)][1] - 1.0)
                for fid in graph.var_factors.get(variable, ())
                if graph.factors_by_id[fid].kind
                not in {FactorKind.PRIOR, FactorKind.OBS}
            ]
            signals[variable] = sum(positive)
        return signals

    def _reconcile(self, graph: FactorGraph, state: BPState) -> BPState:
        if state.graph_signature == graph.structural_signature:
            return state
        fresh = self.initialize(graph, state.evidence)
        for uid in graph.variables:
            fresh.beliefs[uid] = state.beliefs.get(uid, fresh.beliefs[uid])
            fresh.activation[uid] = state.activation.get(uid, 0.0)
        for key in fresh.variable_to_factor:
            fresh.variable_to_factor[key] = state.variable_to_factor.get(
                key,
                fresh.variable_to_factor[key],
            )
        for key in fresh.factor_to_variable:
            fresh.factor_to_variable[key] = state.factor_to_variable.get(
                key,
                fresh.factor_to_variable[key],
            )
        fresh.tick = state.tick
        fresh.trace = list(state.trace)
        fresh.activation_history = list(state.activation_history)
        fresh.belief_history = list(state.belief_history)
        fresh.message_history = list(state.message_history)
        fresh.working_memory = {
            uid: dict(entry)
            for uid, entry in state.working_memory.items()
            if uid in graph.var_factors
        }
        fresh.working_memory_history = list(state.working_memory_history)
        fresh.factor_evaluation_modes = dict(state.factor_evaluation_modes)
        return fresh

    @staticmethod
    def _event_source(
        variables: tuple[str, ...] | list[str],
        target: str,
        incoming: Mapping[str, Message],
    ) -> str:
        others = [uid for uid in variables if uid != target]
        if not others:
            return target
        return max(others, key=lambda uid: incoming.get(uid, (0.5, 0.5))[1])

    @staticmethod
    def _trace_from_events(events: list[ActivationEvent]) -> list[str]:
        scores: dict[str, float] = {}
        for event in events:
            scores[event.factor_uid] = scores.get(event.factor_uid, 0.0) + abs(
                event.contribution
            )
        return [
            uid
            for uid, _ in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:40]
        ]

    def _contribution(self, current: float, previous: float) -> float:
        if self.contribution_mode == "delta":
            return current - previous
        if self.contribution_mode == "counterfactual_logit":
            return _logit(current) - _logit(previous)
        raise ValueError(f"unknown contribution_mode: {self.contribution_mode}")
