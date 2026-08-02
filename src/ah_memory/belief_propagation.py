"""Damped loopy belief propagation on AH factor graph."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from ah_memory.factor_graph import (
    Factor,
    FactorGraph,
    FactorKind,
    role_beta_weight,
)
from ah_memory.types import LinkId


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


@dataclass
class BPResult:
    beliefs: dict[str, float]  # b_v(1)
    trace_factors: list[str]
    rounds: int
    max_belief: float = 0.0


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
    max_arity: int = 6

    def run(self, graph: FactorGraph) -> BPResult:
        if not graph.variables:
            return BPResult(beliefs={}, trace_factors=[], rounds=0)

        # messages: (src, dst) where one is factor fid, other is var
        # store as key ("v2f", v, fid) and ("f2v", fid, v)
        v2f: dict[tuple[str, str], tuple[float, float]] = {}
        f2v: dict[tuple[str, str], tuple[float, float]] = {}

        for f in graph.factors:
            for v in f.variables:
                v2f[(v, f.fid)] = (0.5, 0.5)
                f2v[(f.fid, v)] = (0.5, 0.5)

        factors_by_id = {f.fid: f for f in graph.factors}
        prev_beliefs = {v: 0.5 for v in graph.variables}

        for _ in range(max(1, self.rounds)):
            # variable -> factor
            new_v2f: dict[tuple[str, str], tuple[float, float]] = {}
            for v in graph.variables:
                for fid in graph.var_factors.get(v, []):
                    m0 = m1 = 1.0
                    for fid2 in graph.var_factors.get(v, []):
                        if fid2 == fid:
                            continue
                        a0, a1 = f2v[(fid2, v)]
                        m0 *= a0
                        m1 *= a1
                    new_v2f[(v, fid)] = _damp(v2f[(v, fid)], _norm2(m0, m1), self.damp)
            v2f = new_v2f

            # factor -> variable
            new_f2v: dict[tuple[str, str], tuple[float, float]] = {}
            for f in graph.factors:
                for v in f.variables:
                    msg = self._factor_to_var(f, v, v2f)
                    new_f2v[(f.fid, v)] = _damp(f2v[(f.fid, v)], msg, self.damp)
            f2v = new_f2v

            beliefs = self._beliefs(graph, f2v)
            prev_beliefs = beliefs

        beliefs = self._beliefs(graph, f2v)
        trace = self._trace_factors(graph, beliefs, factors_by_id)
        return BPResult(
            beliefs=beliefs,
            trace_factors=trace,
            rounds=self.rounds,
            max_belief=max(beliefs.values()) if beliefs else 0.0,
        )

    def _beliefs(
        self, graph: FactorGraph, f2v: dict[tuple[str, str], tuple[float, float]]
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for v in graph.variables:
            m0 = m1 = 1.0
            for fid in graph.var_factors.get(v, []):
                a0, a1 = f2v[(fid, v)]
                m0 *= a0
                m1 *= a1
            _, p1 = _norm2(m0, m1)
            out[v] = p1
        return out

    def _factor_to_var(
        self,
        f: Factor,
        v: str,
        v2f: dict[tuple[str, str], tuple[float, float]],
    ) -> tuple[float, float]:
        others = [u for u in f.variables if u != v]
        if len(others) > self.max_arity - 1:
            others = others[: self.max_arity - 1]

        acc0 = acc1 = 0.0
        # configurations of others
        if not others:
            # unary
            for xv in (0, 1):
                pot = self._potential(f, {v: xv})
                if xv == 0:
                    acc0 += pot
                else:
                    acc1 += pot
            return _norm2(acc0, acc1)

        for conf in product((0, 1), repeat=len(others)):
            assign = {u: conf[i] for i, u in enumerate(others)}
            # product of incoming messages
            msg_prod = 1.0
            for u in others:
                m0, m1 = v2f[(u, f.fid)]
                msg_prod *= m0 if assign[u] == 0 else m1
            for xv in (0, 1):
                assign[v] = xv
                pot = self._potential(f, assign)
                if xv == 0:
                    acc0 += pot * msg_prod
                else:
                    acc1 += pot * msg_prod
        return _norm2(acc0, acc1)

    def _potential(self, f: Factor, assign: dict[str, int]) -> float:
        if f.kind is FactorKind.OBS:
            x = assign.get(f.variables[0], 0)
            return math.exp(f.lambda_obs) if x == 1 else 1.0

        if f.kind is FactorKind.PRIOR:
            x = assign.get(f.variables[0], 0)
            return f.epsilon if x == 1 else 1.0

        if f.kind is FactorKind.PAIR:
            if len(f.variables) < 2:
                return 1.0
            e1, e2 = f.variables[0], f.variables[1]
            x1, x2 = assign.get(e1, 0), assign.get(e2, 0)
            return self._pair_potential(f.link_id, f.w, x1, x2)

        if f.kind is FactorKind.HYPER:
            xs = [assign.get(u, 0) for u in f.variables]
            alpha = self.kappa * max(0.0, min(1.0, f.w))
            prod = 1
            for x in xs:
                prod *= x
            # role-aware betas
            beta_term = 0.0
            if f.roles:
                inv = {u: r for r, u in f.roles.items()}
                for u, x in zip(f.variables, xs):
                    role = inv.get(u, "")
                    beta_term += role_beta_weight(role) * alpha * x
            else:
                beta_term = 0.15 * alpha * sum(xs)
            return math.exp(alpha * prod + beta_term)

        return 1.0

    def _pair_potential(self, link_id: str, w: float, x1: int, x2: int) -> float:
        w = max(0.0, min(1.0, w))
        if link_id == LinkId.IS_A.value:
            g = self.gamma_isa * w
            # e1=child, e2=parent; penalize child=1,parent=0
            if x1 == 1 and x2 == 0:
                return math.exp(-g)
            if x1 == 1 and x2 == 1:
                return math.exp(g)
            return 1.0
        if link_id == LinkId.FOLLOW.value:
            g = self.gamma_follow * w
            return math.exp(g * x1 * x2)
        if link_id == LinkId.ASSOC.value:
            g = self.gamma_assoc * w
            return math.exp(g * x1 * x2)
        g = self.gamma_default * w
        return math.exp(g * x1 * x2)

    def _trace_factors(
        self,
        graph: FactorGraph,
        beliefs: dict[str, float],
        factors_by_id: dict[str, Factor],
    ) -> list[str]:
        """Factors whose member beliefs suggest contribution to WM coalition."""
        scored: list[tuple[float, str]] = []
        for f in graph.factors:
            if f.kind in {FactorKind.PRIOR, FactorKind.OBS}:
                continue
            if not f.variables:
                continue
            bs = [beliefs.get(v, 0.0) for v in f.variables]
            score = sum(bs) / len(bs)
            if f.kind is FactorKind.HYPER:
                score *= 1.0 + f.w
            if score > self.trace_eps:
                scored.append((score, f.fid))
        scored.sort(reverse=True)
        return [fid for _, fid in scored[:40]]
