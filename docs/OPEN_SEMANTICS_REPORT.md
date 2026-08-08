# Open relation semantics — implementation report

## Audit

Reusable components:

- `FactCandidate` and LLM perception already provided a fact boundary;
- `Transform` was the single graph mutation point;
- `AHStore` already owned structural revision and entity resolution;
- `Factor` was immutable, n-ary and integrated with cached topology;
- persistent BP, continuous activation, competition, WM and message
  histories were already available;
- Agent/API already exposed per-tick traces.

Primary constraints found:

- perception used a closed predicate whitelist;
- Transform required a static predicate-to-template mapping;
- raw relation text was discarded;
- legacy potentials selected behavior by relation kind;
- there was no deterministic event-to-state layer;
- aggregation benchmarks did not cover relation normalization.

The implementation adds a semantic layer above those components. No
legacy backend or public wrapper was removed.

## New components

- `relations.py`: Relation, properties, context, normalized result,
  NodeRef and Event.
- `relation_registry.py`: dynamic register/get/list/similarity API.
- `relation_normalizer.py`: Exact, Embedding and LLM strategies.
- `factor_parameters.py`: serializable parameters and fixed/rule/
  embedding generators.
- `semantic_activation.py`: relation-agnostic message passing and full
  propagation trace.
- `state_engine.py`: configurable deterministic transition rules.
- `benchmarks/aggregation.py`: extraction baseline, metrics and three
  comparison modes.

Known legacy facts create their original Hyperlink and enrich its graph
factor with relation semantics. Unknown relations create semantic
factors directly. Both paths use the same entity nodes and cached
factor graph.

## Parameterized activation

The baseline semantic message is:

```text
message(i→j) =
    source_activation
    × factor_weight
    × factor_confidence
    × factor.transmission(i,j)
```

`factor.transmission` uses only RelationProperties and generated
FactorParameters. The activation engine contains no branches for
PURCHASE, SELL, IS_A, FOLLOW or any other label.

Supported activation functions:

- LinearActivation;
- SigmoidActivation;
- SaturatingReLUActivation;
- DecayActivation.

Embedding parameters use deterministic:

```text
theta_relation = sigmoid(M × embedding_relation + bias)
```

The projection is replaceable through `set_projection` for later
training.

## Deterministic state

StateEngine applies runtime TransitionRule data. Default rules:

```text
PURCHASE(subject, object)
  → OWNS(subject, object) = true
  → LAST_PURCHASE(subject) = object
  → append PURCHASE_HISTORY(subject, object)

SELL(subject, object)
  → OWNS(subject, object) = false
```

The BMW/Audi/Opel regression verifies:

- BMW ownership false;
- Audi ownership false;
- Opel ownership true;
- last purchase Opel;
- purchase history BMW, Audi, Opel.

## Benchmark

Ten JSON scenarios cover simple purchase, buy/sell, multiple assets,
temporal order, parallel ownership, conflicts, synonyms, coreference,
noise and a long chain.

Final averages:

- Fixed:
  - relation normalization 0.90;
  - event extraction 0.90;
  - state accuracy 0.9333;
  - activation precision 0.2647;
  - activation recall 0.80;
  - path accuracy 0.8667;
  - Recall@K 0.40; MRR 0.5533;
  - mean propagation latency 0.8 tick (1 tick where a current-owned target exists).
- Normalized + rule parameters:
  - relation normalization 1.00;
  - event extraction 1.00;
  - state accuracy 1.00;
  - activation precision 0.3380;
  - activation recall 0.80;
  - path accuracy 0.9667;
  - Recall@K 0.40; MRR 0.5533;
  - mean propagation latency 0.8 tick (1 tick where a current-owned target exists).
- Normalized + embedding parameters:
  - relation normalization 1.00;
  - event extraction 1.00;
  - state accuracy 1.00;
  - activation precision 0.3079;
  - activation recall 0.80;
  - path accuracy 0.9667;
  - Recall@K 0.40; MRR 0.5533;
  - mean propagation latency 0.8 tick (1 tick where a current-owned target exists).

These results support the infrastructure hypothesis: one activation
formula yields measurably different activation precision when factor
parameters change. They do not establish the untrained embedding
projection as superior; rule parameters performed better on this small
suite.

## Trace and API

Agent/API answers now expose:

```text
answer
activated_nodes
activated_factors
timesteps
relations
events
state
state_transitions
final_evidence
```

Debug mode additionally prints source, target, factor, relation,
weight, confidence, activation before/after and all factor parameters
for each message.

## Reproduction

```bash
python -m pytest -q
python benchmark.py --mode fixed
python benchmark.py --mode normalized
python benchmark.py --mode learned
python benchmark.py --mode learned --trace
```

Results are stored in:

- `results/memory_aggregation.json`;
- `results/memory_aggregation_trace.json`.

## Limitations

- Benchmark extraction is deliberately deterministic and narrow; the
  production LLM parser remains responsible for open text extraction.
- The dependency-free embedding is a reproducible baseline, not a
  linguistic model.
- The embedding projection is initialized, not trained.
- Recall@K is unchanged across current modes; parameterization affected
  activation precision but not ranking on this small graph set.
- Legacy relation-specific BP potentials remain for compatibility.
  Open semantic factors use the generic semantic potential and the new
  relation-agnostic engine.
