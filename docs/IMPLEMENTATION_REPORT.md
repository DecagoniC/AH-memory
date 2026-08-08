# Persistent Ignition Platform — implementation report

## Result

The prototype is split into three independently testable levels:

1. AH structural memory (`AHStore`);
2. immutable/cacheable factor topology (`FactorGraph`);
3. persistent query state (`BPState`).

Legacy `BeliefPropagation.run(graph)` and `IgnitionEngine(store, hp)`
remain available. New simulations use `initialize(evidence)` and
`tick(state)`.

## Baseline and verification

- Baseline before refactoring: 31 tests, 0.92 s.
- Final suite: 54 tests, 0.60 s on the same workstation.
- No old tests were removed.
- `git diff --check`: clean.
- IDE diagnostics: no errors.
- Public simulation smoke test: persistent state, contribution events
  and structured working memory are returned after one tick.

## Implemented model

Messages are retained between ticks:

```text
m(v→f)^{t+1} = normalize(e_v × Π m(f'→v)^t)
m(f→v)^{t+1} = Potential_f({m(u→f)^{t+1}}, θ_f)
z_v^{t+1} = Σ positive_contribution(m(f→v)^{t+1})
x_v^{t+1} = G(x_v^t, z_v^{t+1}, e_v^t, θ)
```

Activation and normalized BP belief are separate state vectors.
Competition is applied after activation. Working-memory entries retain
activation, first-entry tick and supporting factor UIDs.

Runtime contribution is message delta. Optional
`counterfactual_logit` stores the change in message log-odds.

## Factor semantics

- New bindings use `LinkId.BIND`; legacy `ASSOC(S↔M)` is recognized as
  BIND without migration.
- ASSOC remains symmetric M↔M.
- IS_A uses independent upward/downward weights.
- FOLLOW and CAUSE use independent forward/backward weights.
- Episode `ElementList` objects are stable factor-graph variables, so
  existing FOLLOW links are no longer dropped.
- Hypernodes are true n-ary factors with `and`, `soft_and` and
  `pairwise` semantics.
- Exact mode enumerates all `2^(arity-1)` assignments. No arity is
  silently truncated.
- Approximate mode performs documented continuous aggregation.
- Auto mode records the selected exact/approximate mode in state and
  trace.

## Experiments

Artifacts:

- `results/synthetic.json` — six deterministic scenarios;
- `results/grid.csv` and `results/grid.json` — 81 parameter runs;
- `results/chain_history.json` — activation trajectory;
- `results/performance.csv` — sizes and arities.

At the strict default threshold 0.7, none of the six baseline
scenarios propagated to a relevant non-seed node. This is a valid
negative experimental result, not hidden by tuning.

The grid search produced finite propagation latency in 24 of 81 runs.
The earliest propagation was tick 1 for sigmoid activation with decay
0.01, threshold 0.5 and factor strength 0.2. That configuration also
activated all four chain nodes (`stability=0`), demonstrating the
expected propagation/selectivity trade-off rather than establishing it
as the preferred model.

On the recorded workstation, one approximate tick at 10,000 variables:

- arity 2: construction 58.04 ms, BP 95.08 ms, activation 43.58 ms,
  total 138.66 ms;
- arity 4: construction 26.51 ms, BP 130.77 ms, activation 37.30 ms,
  total 168.07 ms;
- arity 7: construction 17.78 ms, BP 124.45 ms, activation 36.16 ms,
  total 160.61 ms.

These are local performance measurements and are intentionally outside
the normal test suite.

## Main files

Core:

- `factor_graph.py` — immutable topology, incidence and structural
  signature;
- `belief_propagation.py` — BPState, persistent messages, events and
  histories;
- `potentials.py` — semantic potential registry;
- `activation.py` and `competition.py` — pluggable dynamics;
- `ignition.py` — simulation API and compatibility adapter.

Experiments:

- `experiment.py` — typed configuration, deterministic runner and grid
  search;
- `benchmarks/synthetic.py` — six scenarios and scalable graphs;
- `benchmarks/metrics.py` — latency, peak, half-life, spread,
  selectivity, stability, oscillation and convergence;
- `scripts/run_experiments.py`, `scripts/plot_experiment.py`,
  `scripts/bench_perf.py`.

Regression coverage:

- `test_persistent_inference.py`;
- `test_factor_potentials.py`;
- `test_activation_dynamics.py`;
- `test_experiments.py`.

## Reproduction

```bash
python -m pytest -q
python scripts/run_experiments.py --output results/synthetic.json
python scripts/run_experiments.py --grid --output results/grid.csv
python scripts/plot_experiment.py --scenario chain
python scripts/bench_perf.py --output results/performance.csv
```

## Known limitations

- Loopy BP has no global convergence guarantee.
- Exact n-ary evaluation is exponential; large factors require explicit
  approximate or auto mode.
- Approximate hypernode aggregation is an experimental heuristic.
- Legacy AH `x` fields are still mirrored for compatibility, but new
  inference does not read them as authoritative state.
- Hebbian structural-weight learning is disabled by default during a
  persistent simulation; changing topology/weights should start a new
  controlled run.
- Current default parameters are deliberately uncalibrated. Benchmark
  results show both under-propagation and over-spread regimes.
