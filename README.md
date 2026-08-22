# AH-memory

Прототип ассоциативно-гетерархической памяти до уровня **Senior («Воспламенение»)**.

## Уровни

| Уровень | Статус |
|---------|--------|
| Junior — скелет AH | готово |
| Middle — парсинг + шаблоны T | готово |
| Senior — Ignition / GC / DSL | готово |

Документы: [DOMAIN](docs/DOMAIN.md) · [INTERFACES](docs/INTERFACES.md) · [HYPERPARAMS](docs/HYPERPARAMS.md)

## Сравнение АГ vs БЯМ+RAG (M4)

Один и тот же корпус `benchmarks/m4/closed_world.txt` (статья Википедии «Тиманский кряж») идёт в чанки RAG и в граф AH после батч-ingest.

Ключ DeepSeek — в `.env` (`DEEPSEEK_API_KEY=...`).

```bash
# полный gold-бенчмарк + LLM RAG
python scripts/compare_ah_vs_rag.py --m4

# один вопрос
python scripts/compare_ah_vs_rag.py -q "Что такое Тиманский кряж?"
```

В веб-UI: кнопки **Сравнить** и **M4 бенчмарк** (`POST /api/compare`, `/api/compare/m4`).


## Веб-интерфейс

1. Ключи в `.env` (не коммить): `DEEPSEEK_API_KEY=...` и/или `GIGACHAT_CREDENTIALS=...`
   (алиас `GIGACHAT_API_KEY`; scope: `GIGACHAT_SCOPE=GIGACHAT_API_PERS`).
2. Установка и запуск:

```bash
pip install -e ".[dev]"
ah-web
```

Открой http://127.0.0.1:8000 — чат слева, граф справа, кнопка «Скачать JSON» для дампа.

### Entity Resolution Benchmark

Отдельный benchmark разрешения упоминаний на символы графа
(морфология / синонимы / negative false-merge / contextual), без изменения synthetic aggregation.

```bash
python -m benchmark.entity_resolution
# или
python -m ah_memory.benchmarks.entity_resolution
```

Результаты: `results/entity_resolution/{summary,threshold_sweep,cases,activation_traces}.json`.
В UI — вкладка **Entity Resolution**.

### Synthetic Graph

Кнопка **«Синтезировать граф»** создаёт контролируемый synthetic world с ground truth
(сущности, факторы, события, документы, вопросы, proof paths), загружает его в АГ-память
и позволяет запустить benchmark активации.

CLI (Small, seed=42):

```bash
python -c "
from ah_memory.synthetic import get_preset, SyntheticGraphGenerator, ingest_world, run_benchmark, export_dataset
world = SyntheticGraphGenerator(get_preset('small')).generate()
print(world.stats(), round(world.generation_time_sec, 3))
ingest = ingest_world(world)
report = run_benchmark(ingest.store, world, ingest, limit=20)
print(report.aggregate)
export_dataset(world, 'results/synthetic_small_42')
"
```

## Модули

- `perception` / `transform` — текст → open relations (Event / Factor)
- `ignition` — такт активации + WM
- `gc` — сборка мусора с TTL
- `dsl` — интерпретатор запросов
- `agent` — цикл ingest/ask/step_message
- `baselines.vanilla_rag` — БЯМ + FAISS RAG (контрольный агент M4)
- `compare` — side-by-side АГ vs RAG + прогон M4
- `synthetic` — контролируемый synthetic world + benchmark активации
- `benchmarks.entity_resolution` / `benchmarks.challenge` — ER и challenge-тесты

## Architecture

```text
AH Memory (S, C, P, H, L)
    ↓ structural adapter
Immutable FactorGraph
    ↓ initialize / step
Persistent BPState
    ↓ ActivationFunction + CompetitionFunction
Continuous activation x ∈ [0,1]
    ↓ threshold
Structured Working Memory
    ↓ message attribution
Contribution Trace
```

`AHStore` содержит структуру памяти. Один `FactorGraph` можно использовать
для нескольких независимых запросов (`BPState`) без копирования AH. Поля
`x` в старых типах пока зеркалируются `IgnitionEngine` для совместимости
с UI, но источником истины является `BPState.activation`.

Топология фактор-графа кешируется между тактами и перестраивается только
после структурного изменения AH. Evidence не является частью топологии.

## Mathematical Model

Обозначения:

- `x_v^t ∈ [0,1]` — continuous activation variable `v`;
- `m_{v→f}^t` — сообщение variable → factor;
- `m_{f→v}^t` — сообщение factor → variable;
- `e_v^t` — внешнее evidence;
- `θ` — параметры potential/dynamics;
- `z_v^t = Σ_f contribution(m_{f→v}^t)` — входной сигнал.

Один такт:

```text
m(v→f)^{t+1} = F_V(e_v^t, x_v^t, {m(f'→v)^t}, θ)
m(f→v)^{t+1} = Potential_f({m(u→f)^{t+1}}, θ_f)
x_v^{t+1} = Activation(x_v^t, z_v^t, e_v^t, θ)
WM^{t+1} = {v | x_v^{t+1} ≥ threshold}
```

Доступны `LinearDecayActivation`, `SigmoidActivation`,
`SaturatedReLUActivation`; `NoCompetition`, local/global inhibition и
top-k; potentials BIND, ASSOC, IS_A, FOLLOW, CAUSE и Hypernode.

IS_A и FOLLOW направлены и имеют независимые forward/backward веса.
Hypernode работает как настоящий n-арный фактор в режимах `and`,
`soft_and`, `pairwise`. Exact inference никогда не обрезает арность;
`auto` явно переключается на approximate выше `exact_max_arity`.

## Simulation API

```python
from ah_memory.activation import SigmoidActivation
from ah_memory.benchmarks.synthetic import competing_concepts
from ah_memory.ignition import IgnitionEngine

scenario = competing_concepts()
engine = IgnitionEngine(
    graph=scenario.graph,
    activation=SigmoidActivation(),
)
state = engine.initialize({"DOG": 1.0, "BARK": 0.8})

for _ in range(20):
    state = engine.tick(state)

print(state.activation)
print(state.trace)
print(state.activation_history)
```

Смена activation function или registry factor potentials не изменяет AH
и `FactorGraph`.

## Experiments

Шесть synthetic-сценариев: chain, branching, IS_A, competing concepts,
episodic FOLLOW и 7-арный hypernode.

```bash
python scripts/run_experiments.py --output results/synthetic.json
python scripts/run_experiments.py --grid --output results/grid.csv
python scripts/plot_experiment.py --scenario chain
python scripts/bench_perf.py --output results/performance.csv
```

Grid search перебирает activation/decay/threshold/factor strength и
сохраняет CSV+JSON. Метрики: propagation latency, peak, half-life,
spread, selectivity, stability, oscillation, convergence. Performance
runner отдельно измеряет construction, BP step, activation update и
total tick для N=100…10000.

Конфигурация находится в секции `experiment` файла `config.yaml`.
Платформа не предполагает, что одна модель заранее верна: отсутствие
распространения, excessive spread и instability являются допустимыми
результатами эксперимента.

Итоговые измерения и ограничения: [docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md).

## Open relation semantics

Relations are runtime data rather than a closed enum:

```python
from ah_memory.relation_normalizer import ExactNormalizer, RelationNormalizer
from ah_memory.relation_registry import default_relation_registry

registry = default_relation_registry()
normalizer = RelationNormalizer(registry, [ExactNormalizer()])
relation = normalizer.normalize("приобрёл")

assert relation.raw_label == "приобрёл"
assert relation.canonical_label == "PURCHASE"
```

`RelationRegistry.register_relation()` adds a new canonical relation
without changing source code. Exact, embedding and LLM strategies are
independent from parsing, graph storage and activation. The LLM can
suggest or create a canonical relation, but only `Transform` mutates
memory.

Semantic factors remain n-ary and receive serializable parameters from
`FixedParameterGenerator`, `RuleBasedParameterGenerator` or
`EmbeddingParameterGenerator`. The new activation engine has one
relation-agnostic formula; direction, temporal/causal bias, selectivity
and persistence come from factor parameters.

State transitions are deterministic runtime rules. The default rules
implement PURCHASE/SELL ownership, last purchase and purchase history.

Architecture details: [ARCHITECTURE.md](ARCHITECTURE.md).

### Memory aggregation experiment

```bash
python benchmark.py --mode fixed
python benchmark.py --mode normalized
python benchmark.py --mode learned
python benchmark.py --mode learned --trace
```

The fixtures in `benchmarks/memory_aggregation` cover purchase/sell,
multiple assets, temporal order, parallel ownership, conflicts,
synonyms, coreference, noise and a long chain. Results are written to
`results/memory_aggregation.json`.

Audit, formulas and measured comparison:
[docs/OPEN_SEMANTICS_REPORT.md](docs/OPEN_SEMANTICS_REPORT.md).
