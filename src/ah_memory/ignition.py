"""Ignition via factor-graph belief propagation (docs/FACTOR_GRAPH_ACTIVATION.md)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

from ah_memory.activation_explain import build_activation_chains
from ah_memory.activation import ActivationParameters, activation_by_name
from ah_memory.belief_propagation import (
    ActivationEvent,
    BPState,
    BeliefPropagation,
)
from ah_memory.competition import CompetitionParameters, competition_by_name
from ah_memory.factor_graph import FactorGraph, build_structural_factor_graph
from ah_memory.hyperparams import HyperParams
from ah_memory.potentials import PotentialParameters


@dataclass(frozen=True)
class ActivationSeed:
    uid: str
    delta_x: float


@dataclass
class TickTrace:
    tau: int
    activated: list[str]
    wm: list[str]
    weight_updates: int
    z_stats: dict[str, float] = field(default_factory=dict)
    trace_factors: list[str] = field(default_factory=list)
    beliefs_top: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, float] = field(default_factory=dict)
    seeds_applied: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    activation_top: dict[str, float] = field(default_factory=dict)
    events: list[ActivationEvent] = field(default_factory=list)
    convergence: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class WorkingMemoryEntry:
    uid: str
    activation: float
    entered_at: int
    support: list[str] = field(default_factory=list)


class WorkingMemory:
    def __init__(self) -> None:
        self._entries: dict[str, WorkingMemoryEntry] = {}

    def sync(
        self,
        activation: Mapping[str, float] | set[str],
        *,
        tick: int = 0,
        threshold: float = 0.0,
        support: Mapping[str, list[str]] | None = None,
    ) -> None:
        if isinstance(activation, set):
            activation = {uid: 1.0 for uid in activation}
        previous = self._entries
        self._entries = {
            uid: WorkingMemoryEntry(
                uid=uid,
                activation=float(value),
                entered_at=previous.get(
                    uid,
                    WorkingMemoryEntry(uid, float(value), tick),
                ).entered_at,
                support=list((support or {}).get(uid, ())),
            )
            for uid, value in activation.items()
            if value >= threshold
        }

    def contents(self) -> frozenset[str]:
        return frozenset(self._entries)

    def entries(self) -> tuple[WorkingMemoryEntry, ...]:
        return tuple(
            sorted(
                self._entries.values(),
                key=lambda entry: entry.activation,
                reverse=True,
            )
        )

    def snapshot(self) -> list[dict]:
        return [
            {
                "uid": entry.uid,
                "activation": entry.activation,
                "entered_at": entry.entered_at,
                "support": list(entry.support),
            }
            for entry in self.entries()
        ]


class IgnitionEngine:
    """Persistent factor messages + independent continuous activation state."""

    def __init__(
        self,
        store=None,
        hp: HyperParams | None = None,
        *,
        graph: FactorGraph | None = None,
        activation_function=None,
        activation=None,
        parameters: PotentialParameters | None = None,
    ) -> None:
        if store is None and graph is None:
            raise ValueError("store or graph is required")
        self.store = store
        self.hp = hp or HyperParams()
        self.wm = WorkingMemory()
        self.traces: list[TickTrace] = []
        self._pending_seeds: list[ActivationSeed] = []
        self._evidence: dict[str, float] = {}
        self.graph = graph or build_structural_factor_graph(store)
        self._structural_revision = (
            getattr(store.ah, "revision", 0) if store is not None else None
        )
        potential_parameters = parameters or PotentialParameters(
            association_strength=self.hp.association_strength,
            bind_strength=self.hp.bind_strength,
            is_a_up_weight=self.hp.is_a_up_weight,
            is_a_down_weight=self.hp.is_a_down_weight,
            follow_forward_weight=self.hp.follow_forward_weight,
            follow_backward_weight=self.hp.follow_backward_weight,
            hypernode_mode=self.hp.hypernode_mode,
            factor_evaluation=self.hp.factor_evaluation,
            exact_max_arity=self.hp.exact_max_arity,
        )
        selected_activation = (
            activation_function
            or activation
            or activation_by_name(self.hp.activation_type)
        )
        self.bp = BeliefPropagation(
            kappa=self.hp.fg_kappa,
            damp=self.hp.fg_damp,
            rounds=self.hp.fg_rounds,
            trace_eps=self.hp.fg_trace_eps,
            potential_parameters=potential_parameters,
            activation_function=selected_activation,
            activation_parameters=ActivationParameters(
                decay=self.hp.lambda_decay,
                eta=self.hp.activation_eta,
                alpha=self.hp.activation_alpha,
                beta=self.hp.activation_beta,
                gamma=self.hp.activation_gamma,
            ),
            competition=competition_by_name(self.hp.competition_type),
            competition_parameters=CompetitionParameters(
                enabled=self.hp.competition_enabled,
                strength=self.hp.competition_strength,
                top_k=self.hp.competition_top_k,
            ),
            working_memory_threshold=self.hp.threshold_t,
            history_retention=self.hp.history_retention,
            trace_retention=self.hp.trace_retention,
            contribution_mode=self.hp.contribution_mode,
        )
        self.state = self.bp.initialize(self.graph)

    def initialize(
        self,
        evidence: Mapping[str, float] | None = None,
    ) -> BPState:
        self._ensure_graph()
        self._evidence = {
            uid: max(0.0, float(value))
            for uid, value in (evidence or {}).items()
            if uid in self.graph.var_factors
        }
        self.state = self.bp.initialize(self.graph, self._evidence)
        return self.state

    def seed(self, seeds: list[ActivationSeed]) -> None:
        self._pending_seeds.extend(seeds)

    def tick(
        self,
        state: BPState | None = None,
        evidence: Mapping[str, float] | None = None,
    ) -> TickTrace | BPState:
        """With an explicit state return BPState; without it preserve legacy TickTrace."""
        self._ensure_graph()
        if state is not None:
            return self.bp.step(self.graph, state, evidence)
        return self._legacy_tick(evidence)

    def run(
        self,
        evidence_or_ticks: Mapping[str, float] | int | None = None,
        *,
        ticks: int | None = None,
        evidence: Mapping[str, float] | None = None,
    ) -> list[TickTrace] | BPState:
        """Compatibility `run(6)` or simulation `run(evidence, ticks=20)`."""
        if isinstance(evidence_or_ticks, int) and ticks is None and evidence is None:
            traces: list[TickTrace] = []
            for _ in range(evidence_or_ticks):
                result = self.tick()
                assert isinstance(result, TickTrace)
                traces.append(result)
            return traces
        evidence_map = evidence or (
            evidence_or_ticks if isinstance(evidence_or_ticks, Mapping) else {}
        )
        state = self.initialize(evidence_map)
        for _ in range(ticks or 20):
            state = self.bp.step(self.graph, state, state.evidence)
        self.state = state
        return state

    def _legacy_tick(
        self,
        evidence: Mapping[str, float] | None = None,
    ) -> TickTrace:
        started = time.perf_counter()
        hp = self.hp
        seeds_applied = self._prepare_evidence(evidence)
        before_activation = dict(self.state.activation)
        bp_started = time.perf_counter()
        self.state = self.bp.step(self.graph, self.state, self._evidence)
        bp_ms = (time.perf_counter() - bp_started) * 1000.0
        new_events = [
            event for event in self.state.trace if event.tick == self.state.tick
        ]

        support: dict[str, list[str]] = {}
        for event in new_events:
            if event.contribution > 0:
                support.setdefault(event.target_uid, []).append(event.factor_uid)
        self.wm.sync(
            self.state.activation,
            tick=self.state.tick,
            threshold=hp.threshold_t,
            support=support,
        )
        activated = sorted(self.wm.contents())
        weight_updates = self._hebb_update() if hp.fg_hebb_enabled else 0

        if self.store is not None:
            # Compatibility mirror only; BPState remains authoritative.
            for uid, value in self.state.activation.items():
                try:
                    self.store.set_x(uid, float(value))
                except Exception:
                    continue
            self.store.ah.tau += 1
            tau = self.store.ah.tau
        else:
            tau = self.state.tick

        beliefs_top = dict(
            sorted(
                self.state.beliefs.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        )
        activation_top = dict(
            sorted(
                self.state.activation.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        )
        evidence_snap = {
            uid: round(value, 4)
            for uid, value in sorted(
                self._evidence.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:24]
        }
        chains = (
            build_activation_chains(
                self.store,
                self.graph,
                self.state.beliefs,
                seeds=seeds_applied,
                evidence=evidence_snap,
                threshold=hp.threshold_t,
            )
            if self.store is not None
            else []
        )
        trace_factors = self.bp._trace_from_events(new_events)[:24]
        convergence = max(
            (
                abs(self.state.activation.get(uid, 0.0) - previous)
                for uid, previous in before_activation.items()
            ),
            default=0.0,
        )
        trace = TickTrace(
            tau=tau,
            activated=activated,
            wm=activated,
            weight_updates=weight_updates,
            z_stats={
                "max_belief": max(self.state.beliefs.values(), default=0.0),
                "max_activation": max(self.state.activation.values(), default=0.0),
                "n_vars": float(len(self.graph.variables)),
                "n_factors": float(len(self.graph.factors)),
                "bp_rounds": 1.0,
            },
            trace_factors=trace_factors,
            beliefs_top={uid: round(value, 4) for uid, value in beliefs_top.items()},
            activation_top={
                uid: round(value, 4) for uid, value in activation_top.items()
            },
            evidence=evidence_snap,
            seeds_applied=seeds_applied,
            chains=chains,
            events=list(new_events),
            convergence=convergence,
            timings_ms={
                "bp_step": self.state.timings_ms.get("bp_step", bp_ms),
                "activation_update": self.state.timings_ms.get(
                    "activation_update",
                    0.0,
                ),
                "total_tick": (time.perf_counter() - started) * 1000.0,
            },
        )
        self.traces.append(trace)
        return trace

    def _ensure_graph(self) -> None:
        if self.store is None:
            return
        revision = getattr(self.store.ah, "revision", 0)
        if revision == self._structural_revision:
            return
        self.graph = build_structural_factor_graph(self.store)
        self._structural_revision = revision
        self.state = self.bp._reconcile(self.graph, self.state)

    def _prepare_evidence(
        self,
        explicit: Mapping[str, float] | None,
    ) -> list[str]:
        if explicit is not None:
            self._evidence = {
                uid: max(0.0, float(value))
                for uid, value in explicit.items()
                if uid in self.graph.var_factors
            }
        else:
            for uid in list(self._evidence):
                self._evidence[uid] *= self.hp.evidence_decay
                if self._evidence[uid] < 0.05:
                    del self._evidence[uid]

        seeds_applied: list[str] = []
        for seed in self._pending_seeds:
            lam = self.hp.fg_lambda * max(0.1, min(1.0, seed.delta_x))
            candidates = [seed.uid]
            m_uid = seed.uid if seed.uid.startswith("M_") else f"M_{seed.uid}"
            if m_uid not in candidates:
                candidates.append(m_uid)
            for index, uid in enumerate(candidates):
                if uid not in self.graph.var_factors:
                    continue
                scale = 1.0 if index == 0 else 0.8
                self._evidence[uid] = self._evidence.get(uid, 0.0) + lam * scale
                if uid not in seeds_applied:
                    seeds_applied.append(uid)
        self._pending_seeds.clear()

        if (
            self.store is not None
            and self.hp.pacemaker_period > 0
            and self.store.ah.tau % self.hp.pacemaker_period == 0
        ):
            targets = list(self.wm.contents())[:5] or list(self.store.ah.S)[:3]
            for uid in targets:
                if uid in self.graph.var_factors:
                    self._evidence[uid] = self._evidence.get(uid, 0.0) + 0.4
        return seeds_applied

    def _hebb_update(self) -> int:
        if self.store is None:
            return 0
        updates = 0
        for link in self.store.ah.L.values():
            first = self.state.activation.get(link.e1.target_uid, 0.0)
            second = self.state.activation.get(link.e2.target_uid, 0.0)
            if first * second > self.hp.fg_hebb_tau:
                link.w = self.hp.h(link.w, first, second)
                updates += 1
        return updates
