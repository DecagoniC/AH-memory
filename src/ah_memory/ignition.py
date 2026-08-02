"""Ignition via factor-graph belief propagation (docs/FACTOR_GRAPH_ACTIVATION.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ah_memory.belief_propagation import BeliefPropagation
from ah_memory.factor_graph import build_factor_graph, is_variable
from ah_memory.hyperparams import HyperParams
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


class WorkingMemory:
    def __init__(self) -> None:
        self._uids: set[str] = set()

    def sync(self, excited: set[str]) -> None:
        self._uids = set(excited)

    def contents(self) -> frozenset[str]:
        return frozenset(self._uids)


class IgnitionEngine:
    """Focus of activation = damped loopy BP; x_v := b_v(1)."""

    def __init__(self, store, hp: HyperParams | None = None) -> None:
        self.store = store
        self.hp = hp or HyperParams()
        self.wm = WorkingMemory()
        self.traces: list[TickTrace] = []
        self._pending_seeds: list[ActivationSeed] = []
        self._evidence: dict[str, float] = {}
        self.bp = BeliefPropagation(
            kappa=self.hp.fg_kappa,
            damp=self.hp.fg_damp,
            rounds=self.hp.fg_rounds,
            trace_eps=self.hp.fg_trace_eps,
        )

    def seed(self, seeds: list[ActivationSeed]) -> None:
        self._pending_seeds.extend(seeds)

    def tick(self) -> TickTrace:
        store = self.store
        hp = self.hp
        ah = store.ah

        # evidence from seeds (λ), keep soft carry of previous evidence
        for v in list(self._evidence.keys()):
            self._evidence[v] *= 0.5
            if self._evidence[v] < 0.05:
                del self._evidence[v]

        seeds_applied: list[str] = []
        for s in self._pending_seeds:
            # map delta_x ≈ 0.8 → λ ≈ seed_lambda * delta
            lam = hp.fg_lambda * max(0.1, min(1.0, s.delta_x))
            if is_variable(store, s.uid):
                self._evidence[s.uid] = self._evidence.get(s.uid, 0.0) + lam
                seeds_applied.append(s.uid)
            # also try M_ form
            m_uid = s.uid if s.uid.startswith("M_") else f"M_{s.uid}"
            if is_variable(store, m_uid):
                self._evidence[m_uid] = self._evidence.get(m_uid, 0.0) + lam * 0.8
                if m_uid not in seeds_applied:
                    seeds_applied.append(m_uid)
        self._pending_seeds.clear()

        # pacemaker ν → weak evidence on WM / S
        if hp.pacemaker_period > 0 and ah.tau % hp.pacemaker_period == 0:
            targets = list(self.wm.contents())[:5] or list(store.ah.S.keys())[:3]
            for uid in targets:
                if is_variable(store, uid):
                    self._evidence[uid] = self._evidence.get(uid, 0.0) + 0.4

        graph = build_factor_graph(
            store,
            evidence=self._evidence,
            epsilon=hp.fg_epsilon,
            include_prior=True,
        )
        result = self.bp.run(graph)

        # write beliefs → x
        for uid, b1 in result.beliefs.items():
            try:
                store.set_x(uid, float(b1))
            except Exception:
                continue

        # hypernodes: x = mean belief of variable actants
        for n in store.find_hypernodes():
            bs = [
                result.beliefs[f.target_uid]
                for f in n.fillers.values()
                if f.target_uid in result.beliefs
            ]
            if bs:
                try:
                    store.set_x(n.uid, sum(bs) / len(bs))
                except Exception:
                    pass

        activated = [u for u, b in result.beliefs.items() if b > hp.threshold_t]
        self.wm.sync(set(activated))

        # Hebb on beliefs (spec §7)
        weight_updates = 0
        tau_h = hp.fg_hebb_tau
        for link in ah.L.values():
            b1 = result.beliefs.get(link.e1.target_uid, 0.0)
            b2 = result.beliefs.get(link.e2.target_uid, 0.0)
            if b1 * b2 > tau_h:
                link.w = hp.h(link.w, b1, b2)
                weight_updates += 1
        for n in store.find_hypernodes():
            bs = [
                result.beliefs.get(f.target_uid, 0.0) for f in n.fillers.values()
            ]
            if bs and math_prod(bs) > tau_h:
                mean_b = sum(bs) / len(bs)
                n.w = hp.h(n.w, mean_b, mean_b)
                weight_updates += 1

        ah.tau += 1
        top = dict(
            sorted(result.beliefs.items(), key=lambda kv: kv[1], reverse=True)[:12]
        )
        evidence_snap = {
            k: round(v, 4)
            for k, v in sorted(self._evidence.items(), key=lambda kv: -kv[1])[:24]
        }
        trace = TickTrace(
            tau=ah.tau,
            activated=sorted(activated),
            wm=sorted(self.wm.contents()),
            weight_updates=weight_updates,
            z_stats={
                "max_belief": result.max_belief,
                "n_vars": float(len(graph.variables)),
                "n_factors": float(len(graph.factors)),
                "bp_rounds": float(result.rounds),
            },
            trace_factors=result.trace_factors[:24],
            beliefs_top={k: round(v, 4) for k, v in top.items()},
            evidence=evidence_snap,
            seeds_applied=seeds_applied,
        )
        self.traces.append(trace)
        return trace

    def run(self, n_ticks: int) -> list[TickTrace]:
        return [self.tick() for _ in range(n_ticks)]


def math_prod(xs: list[float]) -> float:
    p = 1.0
    for x in xs:
        p *= x
    return p
