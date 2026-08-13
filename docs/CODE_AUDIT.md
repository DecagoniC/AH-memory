# AH-memory — аудит кода

Документ описывает назначение модулей и ключевых сущностей репозитория **ah-memory** (прототип ассоциативно-гетерархической памяти для LLM-агентов).

Версия пакета: `0.4.0` · корень исходников: `src/ah_memory/`

---

## 1. Карта репозитория

```
AH-memory/
├── src/ah_memory/          # ядро библиотеки
│   ├── …                   # runtime: store, perception, ignition, identity…
│   ├── benchmarks/         # synthetic ignition + aggregation + entity resolution
│   ├── synthetic/          # генератор synthetic world + GT + ingest
│   └── examples/           # (пусто; демо dog/rabbit удалены)
├── web/                    # FastAPI + SPA UI
├── benchmark/              # CLI-shim: python -m benchmark.entity_resolution
├── tests/                  # pytest
├── scripts/                # утилиты запуска/экспорта
├── config.yaml             # конфиг (LLM, embedding, identity, experiment)
├── docs/CODE_AUDIT.md      # этот файл
└── results/                # артефакты benchmark (не код)
```

---

## 2. Главный пайплайн runtime

> В модулях первого круга (`types`, `store`, `perception`, `transform`,
> `identity`, `agent`, `dialogue`) у блоков стоят комментарии «Зачем: …».

```text
текст пользователя
      │
      ▼
 perception.parse  ──► FactCandidate / seed_tokens
      │
      ▼
 transform.apply   ──► identity.resolve? ──► AHStore (S / M / N / events)
      │
      ▼
 ignition / ActivationEngine / BeliefPropagation
      │
      ▼
 dialogue / Agent.ask  ──► ответ + ingest ответов
```

| Слой | Модули | Роль |
|------|--------|------|
| Ввод | `perception`, `morph`, `gigachat_llm` / `deepseek` | текст → факты/семена |
| Идентичность | `identity`, ER-resolvers | mention → существующий символ |
| Материализация | `transform`, `store` | факты → Event / semantic Factor |
| Семантика связей | `relations`, `relation_registry`, `relation_normalizer` | open relations |
| Активация | `factor_graph`, `potentials`, `belief_propagation`, `ignition`, `semantic_activation` | возбуждение |
| Состояние | `state_engine` | детерминированные переходы OWNERSHIP и т.п. |
| Диалог | `agent`, `dialogue` | цикл ingest/ask |
| UI | `web/app.py`, `web/static/index.html` | чат, граф, synthetic/ER |

---

## 3. Ядро `src/ah_memory/`

### 3.1 `__init__.py`
Публичный фасад пакета: реэкспорт `Agent`, `AHStore`, `IgnitionEngine`, `ActivationEngine`, типов отношений и т.д. Константа `__version__`.

### 3.2 `types.py` — модель данных AH
Типы монографии §3.

| Сущность | Назначение |
|----------|------------|
| `Section` | секции графа: C / P / H |
| `AbstractSymbol` (S) | лексика 1-го порядка, `R["TEXT"]` = формы |
| `SecondOrderSymbol` (M) | смыслы/сущности 2-го порядка |
| `Hyperlink` (N) | n-арный факт с ролями |
| `AssocLink` (L) | бинарные/ассоциативные связи |
| `Template` (T) | шаблон предиката (`predicate: str`, не узел в S) |
| `AH` | контейнер S/C/P/H/L |
| `Property` | свойства на M (`label`, `catalog_uid`, …) |

### 3.3 `store.py` — хранилище
`AHStore` — in-memory CRUD по таблице операций.

| Метод / символ | Назначение |
|----------------|------------|
| `ensure_abstract` / `ensure_m` | create-if-missing S/M; формы мержатся в `TEXT` |
| `add_link` / `add_element` | рёбра и узлы |
| `find_abstract_symbols` / `find_symbols` | поиск по подстроке (не ingest-identity) |
| `semantic_factors` / `events` | open-semantics слой |
| `graph_size` | размер для UI/метрик |
| `AHError` | ошибки доступа к UID |

### 3.4 `config.py` — конфигурация
Загрузка `config.yaml` + `config.local.yaml` + `.env`.

| Класс | Поля (ключевые) |
|-------|-----------------|
| `GigaChatConfig` | `credentials`, `scope`, `base_url`, `model`, `verify_ssl` |
| `DeepSeekConfig` | `api_key`, `base_url`, `model` |
| `AgentConfig` | `use_llm`, `ticks`, `preload`, `llm_provider` |
| `OpenSemanticsConfig` | `normalization_mode`, `embedding_similarity_threshold` |
| `EmbeddingConfig` | `model` (`EmbeddingsGigaR` / ngram), `dimensions` |
| `IdentityConfig` | `enabled`, `use_embeddings`, `safety_threshold`, `margin` |
| `AppConfig` | агрегат всех секций |
| `load_config()` | точка входа |

Константы: `ROOT`, `DEFAULT_CONFIG`, `LOCAL_CONFIG`.

### 3.5 `hyperparams.py`
`HyperParams` — пороги ignition/BP: `seed_delta`, decay, damping, competition, TTL для GC.

### 3.6 `morph.py` — морфология RU
| Функция / константа | Назначение |
|---------------------|------------|
| `lemma` | лемма pymorphy (с предпочтением имён) |
| `slug_uid` | surface → `UPPER_SNAKE` UID |
| `is_entity_token` / `is_nounish` | фильтры POS |
| `sanitize_roles` / `seeds_from_roles` | очистка ролей LLM |
| `STOP` | стоп-слова |
| `MAX_UID_PARTS` | лимит частей UID |

### 3.7 `perception.py` — восприятие
| Сущность | Назначение |
|----------|------------|
| `FactCandidate` | сырой факт: predicate, roles, confidence, raw_relation |
| `PerceptionResult` | `kind` + candidates + seed_tokens |
| `SeedPerception` | offline: только семена из текста |
| `JsonLLMPerception` | JSON от LLM → кандидаты |
| `gate_candidates` | морфологический gate |
| `PREDICATES` | reserved labels (не entity UID); ingest всегда open |

### 3.8 `transform.py` — материализация
`Transform.apply(perception)` пишет в store.

| Сущность | Назначение |
|----------|------------|
| `IngestReport` | созданные N, seed_uids, skipped |
| `_resolve_bare` / `_resolve_value` | mention → bare/M UID (+ identity) |
| `_ingest_candidate` | legacy N-факты по шаблонам |
| `_record_semantic` | open semantic events/factors |
| `identity: SymbolIdentityService \| None` | resolve-before-create |

### 3.9 `identity.py` — разрешение символов (runtime)
Пайплайн: **exact → morphology → alias → gated embedding**. Cosine ≠ merge.

| Сущность | Назначение |
|----------|------------|
| `DEFAULT_SYNONYMS` | лексикон mention→канон |
| `DEFAULT_ANTI_MERGE` | пары, которые нельзя склеивать |
| `COMPATIBLE_KINDS` | type-gate (person/place/brand/concept) |
| `IdentityPolicy` | `safety_threshold`, `margin`, anti_merge, synonyms |
| `AliasResolver` | только лексикон |
| `IdentityGate` | порог + margin + type + anti-merge |
| `GatedEmbeddingResolver` | embedding → candidates → gate |
| `SafeHybridResolver` | полный пайплайн |
| `SymbolIdentityService` | `resolve_bare_uid`, `attach_alias` над store |
| `catalog_from_store` | SymbolSpec[] из AH.S |
| `build_identity_service` | фабрика для Agent/web |

### 3.10 `templates.py`
Удалён: ingest только open relations (Event + Factor).

### 3.11 Open semantics: отношения
| Файл | Назначение |
|------|------------|
| `relations.py` | `Relation`, `NormalizedRelation`, `Event`, `NodeRef`, `canonicalize_label` |
| `relation_registry.py` | `RelationRegistry`, `cosine_similarity`, `default_relation_registry` |
| `relation_normalizer.py` | `ExactNormalizer`, `EmbeddingNormalizer`, `RelationNormalizer`, `deterministic_embedding` |

Ключевые поля `NormalizedRelation`: `canonical_label`, `raw_relation`, `embedding`, `confidence`.

### 3.12 Фактор-граф и активация
| Файл | Ключевое |
|------|----------|
| `factor_graph.py` | `Factor`, `FactorGraph`, `build_structural_factor_graph`, `FactorKind` |
| `factor_parameters.py` | `FactorParameters`, `RuleBased` / `Embedding` / `Fixed` generators |
| `potentials.py` | `IsAPotential`, `FollowPotential`, `SemanticPotential`, `HypernodePotential`, … |
| `belief_propagation.py` | `BeliefPropagation`, `BPState`, `BPResult`, evidence injection |
| `activation.py` | непрерывные динамики (`LinearDecay`, `Sigmoid`, …) |
| `competition.py` | `GlobalInhibition`, `LocalCompetition`, `TopKNormalization` |
| `semantic_activation.py` | `ActivationEngine`, `PropagationTrace`, Linear/Sigmoid/ReLU/Decay |
| `ignition.py` | `IgnitionEngine`, `WorkingMemory`, `ActivationSeed`, `TickTrace` |
| `activation_explain.py` | `build_activation_chains`, `node_label` — объяснения для UI/LLM |

### 3.13 `state_engine.py`
Детерминированные состояния (например OWNERSHIP после PURCHASE/SELL): `State`, `TransitionRule`, `StateEngine`, `default_state_engine`.

### 3.14 `gc.py` / `invariants.py` / `dsl.py`
| Файл | Назначение |
|------|------------|
| `gc.py` | `collect(store, hp)` → `GCReport` |
| `invariants.py` | `validate`, `InvariantError` |
| `dsl.py` | `DSLInterpreter` — мини-язык запросов к store |

### 3.15 `graph_export.py`
`dump_graph` (узлы/рёбра для vis-network), `dump_ah_json` (полный дамп).

### 3.16 `agent.py` / `dialogue.py`
| Класс | Назначение |
|-------|------------|
| `Agent` | `ingest`, `ask`, `step_message`; держит store, perception, transform, ignition |
| `AgentReply` | ответ, trace_uids, seeds, activation extras |
| `DialogueAgent` | история чата, LLM-ответ, компактный memory-контекст |
| `TurnResult` | результат одного хода диалога |

### 3.17 LLM-клиенты
| Файл | Ключевое |
|------|----------|
| `gigachat_llm.py` | `GigaChatClient` (OAuth, chat, **embeddings**), `GigaChatEmbedder`, `HybridPerception`, `SYSTEM_PROMPT`, `DEFAULT_GIGAR_INSTRUCTION` |
| `deepseek.py` | `DeepSeekClient`, `DeepSeekHybridPerception` |

### 3.18 `corpus.py` / `experiment.py`
| Файл | Назначение |
|------|------------|
| `experiment.py` | `ExperimentConfig`, `ExperimentRunner`, `grid_search`, секции activation/factors/competition |

### 3.19 `examples/`
| Файл | Назначение |
|------|------------|
| *(удалены)* | dog/rabbit demos сняты; фикстуры тестов — `tests/_mini_graph.py` |

---

## 4. `benchmarks/` — исследовательские бенчмарки

### 4.1 Ignition / aggregation
| Файл | Назначение |
|------|------------|
| `benchmarks/synthetic.py` | сценарии: `chain`, `branching`, `is_a`, `competing_concepts`, … |
| `benchmarks/metrics.py` | `calculate_metrics` по траекториям активации |
| `benchmarks/aggregation.py` | suite агрегации памяти: `run_suite`, `run_scenario` |

### 4.2 Entity Resolution (`benchmarks/entity_resolution/`)

```text
mention → Exact → Morphology → Alias → Embedding(+IdentityGate)
                → ActivationEngine probe → metrics / threshold sweep
```

| Файл | Назначение |
|------|------------|
| `cases.py` | `CaseType`, `ExpectedRelation`, `SymbolSpec` (+ `context_cues`), `EntityResolutionCase`, `ResolutionResult` |
| `dataset.py` | `control_symbols/cases/facts`, `DATASET_VERSION` |
| `dataset_extended.py` | AMBIGUOUS / SURFACE / HOLD_OUT / hard-neg; `HOLD_OUT_POSITIVES` |
| `resolvers.py` | `ExactResolver`, `MorphologyResolver`, `EmbeddingResolver`, `HybridResolver`, `forms_match`, `disambiguate_by_context`, `make_embed_fn` |
| `generator.py` | `build_resolution_store` — catalog → AHStore + uid_map |
| `evaluator.py` | `evaluate_case`, `run_activation_probe`, cosine к target |
| `metrics.py` | `CaseOutcome`, `MetricBundle`, `find_optimal_threshold`, `DEFAULT_THRESHOLDS` |
| `runner.py` | `EntityResolutionBenchmark`, CLI `main`, JSON в `results/entity_resolution/` |
| `__main__.py` | `python -m ah_memory.benchmarks.entity_resolution` |

**Типы кейсов (`CaseType`):**  
`MORPHOLOGY`, `SYNONYM`, `NEGATIVE`, `SEMANTIC_NEAR`, `CONTEXTUAL`, `AMBIGUOUS`, `SURFACE`, `HOLD_OUT`.

---

## 5. `synthetic/` — синтетический мир

| Файл | Назначение |
|------|------------|
| `config.py` | `SyntheticGraphConfig`, `DEFAULT_RELATION_TYPES`, `merge_config` |
| `presets.py` | `PRESETS` (`tiny`/`small`/`medium`/`large`/`stress`), `get_preset` |
| `entities.py` | `Entity`, пулы имён PERSON/PLACE/OBJECT/… |
| `relations.py` | `FACTOR_SCHEMAS`, `map_roles_to_ah` |
| `events.py` | `SyntheticFactor`, `SyntheticEvent`, `WorldState` |
| `graph_generator.py` | `SyntheticGraphGenerator.generate()` |
| `query_generator.py` | вопросы + proof paths |
| `text_generator.py` | NL-шаблоны для факторов |
| `distractors.py` | `generate_distractor_factors` |
| `ground_truth.py` | `SyntheticWorld`, `SyntheticQuery`, `SyntheticDocument` |
| `ingest.py` | `ingest_world` (+ optional `identity`) → `IngestResult` |
| `benchmark.py` | `run_benchmark`, `evaluate_query`, `proof_view` |
| `serializer.py` | `export_dataset`, `export_zip`, GraphML/JSONL |

---

## 6. Web UI

### `web/app.py`
FastAPI:

| Эндпоинт / хелпер | Назначение |
|-------------------|------------|
| `/api/chat` | диалог |
| `/api/graph`, `/api/dump`, `/api/trace` | визуализация / дамп |
| `/api/synthetic/*` | generate, benchmark, proof, download |
| `/api/entity-resolution/*` | ER benchmark + status |
| `_build_core` / `_build_identity` | Agent + SymbolIdentityService |
| `main` | uvicorn + освобождение порта |

### `web/static/index.html`
SPA: чат, трасса активации, граф (vis-network), вкладки Synthetic / Entity Resolution, переключатель LLM.

---

## 7. Тесты, скрипты, shim

### `tests/`
| Файл | Что проверяет |
|------|----------------|
| `test_entity_resolution.py` | базовый ER + activation after resolve |
| `test_er_corpus_extended.py` | ambiguous/surface/hold-out/hard-neg/ingest |
| `test_identity_ingest.py` | синонимы при ingest, Audi≠BMW |
| `test_identity_paraphrase.py` | paraphrased queries + tiny synthetic |
| `test_synthetic_graph.py` | generator/ingest/benchmark |
| `test_morph.py` | морфология/UID |
| `test_open_relations.py` | нормализация отношений |
| `test_semantic_activation.py` / `test_factor_*` / `test_persistent_inference.py` | активация/BP |
| `test_state_engine.py` | состояния |
| `test_memory_aggregation_benchmark.py` | агрегация |
| `test_junior.py` / `test_middle_senior.py` | acceptance / scale |
| `test_added_at.py` | timestamps |
| `test_experiments.py` | experiment runner |

### `scripts/`
| Файл | Назначение |
|------|------------|
| `export_er_test_catalog.py` | → `results/entity_resolution/test_catalog.md\|json` |
| `run_experiments.py` | ignition experiments |
| `bench_perf.py` | perf vs size |
| `plot_experiment.py` | визуализация activation |
| `_dump_live.py` | дамп с живого сервера |

### `benchmark/`
Shim: `python -m benchmark.entity_resolution` → runner ER.

---

## 8. Конфигурация (`config.yaml`) — смысловые секции

| Секция | Зачем |
|--------|-------|
| `gigachat` / `deepseek` | ключи и модели LLM |
| `agent` | ticks, preload, provider |
| `open_semantics` | режим нормализации отношений |
| `embedding` | бэкенд эмбеддингов для ER/identity |
| `identity` | resolve-before-create + safety threshold |
| `experiment` | activation/BP/competition knobs |

---

## 9. Где что менять (шпаргалка)

| Задача | Куда смотреть |
|--------|----------------|
| Новое правило синонимов | `identity.DEFAULT_SYNONYMS` |
| Запрет false merge | `identity.DEFAULT_ANTI_MERGE` |
| Поведение ingest UID | `transform._resolve_bare`, `identity.SymbolIdentityService` |
| Новый ER-кейс | `dataset.py` / `dataset_extended.py` |
| Другая embedding-модель | `config.yaml` → `embedding.model`, `resolvers.make_embed_fn` |
| Новый тип фактора synthetic | `synthetic/relations.py` + generators |
| UI вкладка / API | `web/app.py` + `web/static/index.html` |
| Параметры активации | `hyperparams.py` / `config.yaml` → `experiment` |

---

## 10. Зависимости между пакетами (упрощённо)

```text
web.app ──► agent, dialogue, synthetic, ER runner, config, identity
agent   ──► perception, transform(+identity), ignition, store
transform ──► store, relation_normalizer, state_engine, identity
identity  ──► morph, ER resolvers/cases (типы), store
ER runner ──► dataset, resolvers, evaluator, metrics, generator, gigachat embeddings
synthetic ──► Transform/store (ingest), ActivationEngine (benchmark)
```

Циклический импорт `transform ↔ ER package` разорван: `benchmarks/entity_resolution/__init__.py` лениво подгружает `runner`.

---

*Документ сгенерирован как аудит для навигации по коду. При крупных рефакторингах обновляйте соответствующие секции.*
