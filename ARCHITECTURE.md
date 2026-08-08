# AH-memory architecture

## Current compatible core

The original path remains operational:

```text
Perception → FactCandidate → Transform → AHStore (S/C/P/H/L)
                                      ↓
                           immutable FactorGraph
                                      ↓
                          persistent BPState
                                      ↓
                     activation / competition / WM
```

Legacy predicates, templates, Hyperlink, `BeliefPropagation.run()` and
`IgnitionEngine(store, hp)` are preserved. Legacy object `x` fields are
still mirrored for the UI.

## Open semantic layer

The new path is additive:

```text
LLM raw_relation
    ↓
RelationNormalizer (Exact | Embedding | LLM)
    ↓
RelationRegistry + RelationProperties
    ↓
Event + n-ary semantic Factor
    ↓
FactorParameterGenerator (fixed | rules | embedding projection)
    ↓
relation-agnostic ActivationEngine
    ↓
Working memory / deterministic StateEngine / full trace
```

`embedding` is semantic metadata or an input to the parameter
projection. It is not the memory store and does not replace factor-graph
retrieval.

## Integration points

- `FactCandidate` retains `predicate` for compatibility and additionally
  stores `raw_relation` plus `canonical_relation`.
- `Transform` normalizes relations, records `Event`, creates a semantic
  `Factor`, generates parameters and applies configured state rules.
- Known legacy predicates still create their old Hyperlink. Its
  factor-graph representation is enriched from the semantic factor
  rather than duplicated.
- Unknown relations create semantic factors directly and require no
  source-code change.
- `build_factor_graph()` includes semantic factors. The `semantic`
  potential uses only relation properties and generated parameters.
- `AgentReply`, dialogue API and `/api/trace` expose nodes, factors,
  timesteps, relations, events, transitions and final evidence.

## Compatibility boundary

The old relation-specific BP potentials remain as a baseline backend.
The new `semantic_activation.ActivationEngine` contains no branches by
relation label. Semantic differences enter through:

```text
RelationProperties + FactorParameters + factor.transmission(source,target)
```

`StateEngine` relation-to-transition mappings are runtime configuration,
not LLM reasoning and not activation logic.

## Experiments

`benchmark.py` compares:

- `fixed`: fixed relation vocabulary and identical parameters;
- `normalized`: open normalization and rule-generated parameters;
- `learned`: normalization, relation embedding and deterministic linear
  parameter projection.

The ten fixtures in `benchmarks/memory_aggregation` measure relation and
event accuracy, deterministic state, factor-graph retrieval, activation,
path accuracy and latency.
