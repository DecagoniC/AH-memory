# AH-memory

Нейросимвольная **ассоциативно-гетерархическая память** для агента. Факты живут в графе `AH = ⟨S, C, P, H, L⟩`; ответ строится из рабочей памяти и UID-трассы. Большая языковая модель — только на границе восприятия, не как хранилище знаний.

Контрольный агент — классический **БЯМ + векторный RAG** (TF-IDF / DeepSeek) на том же тексте.

## Что умеет

- ingest текста → открытые n-арные отношения → гиперграф (факторы, IS-A, FOLLOW);
- вопрос → активация фактор-графа (belief propagation) → WM → ответ с трассой;
- отказ «неизвестно», если сущности или слота нет в текущем вопросе (без чужого цвета и без фокуса прошлого вопроса);
- веб-UI: чат, гиперграф, сравнение с RAG;
- метрики M1–M5 и тесты гипотезы объяснимости / галлюцинаций.

## Быстрый старт

Python ≥ 3.11. Скопируй [`.env.example`](.env.example) в `.env` (файл в git не попадает):

```
DEEPSEEK_API_KEY=...
LLM_PROVIDER=deepseek
```

Провайдер также задаётся в [`config.yaml`](config.yaml) (`gigachat` | `deepseek`). Для GigaChat: `GIGACHAT_CREDENTIALS` и `GIGACHAT_SCOPE`.

```bash
pip install -e ".[dev]"
python -m web.app
```

Открой http://127.0.0.1:8000 — чат слева, граф справа. То же: `ah-web`.

Эталон наполнения — короткий текст про зайца (`src/ah_memory/examples/rabbit.py`). Корпус, дамп графа и записка: [`docs/submission/`](docs/submission/).

## АГ vs БЯМ+RAG

Гипотеза: граф даёт **более объяснимые** ответы и **меньше галлюцинаций**, чем «модель + поиск чанков».

```bash
python scripts/eval_m4.py              # extractive TF-IDF, без ключа
python scripts/eval_m4.py --llm        # DeepSeek + TF-IDF
python scripts/compare_ah_vs_rag.py -q "Кто такой заяц?"
python -m pytest tests/test_hypothesis_ah_vs_rag.py tests/test_m4.py -q
```

В UI: **Сравнить** (`POST /api/compare`) и **M4 бенчмарк** (`POST /api/compare/m4`).

Замеренный срез на золоте «заяц» (6 вопросов):

| Система | ExplainScore | Hallucination | Δ explain | Δ hall |
|---------|--------------|---------------|-----------|--------|
| **AH** | 0,67 | 0 | | |
| Extractive RAG | 0 (нет UID-трассы) | 0 | **+0,67** | 0 |
| DeepSeek + RAG | 0 | 0,50 | **+0,67** | **+0,50** |

ExplainScore у RAG по постановке равен нулю: у чанков нет цепочки UID. Против БЯМ гипотеза по галлюцинациям подтверждается (выдуманный «лунный король»: AH — «неизвестно», RAG дописывает факты). Против extractive без генератора — ничья. Логи: [пояснительная записка](docs/submission/POYASNITELNAYA_ZAPISKA.md).

## Конвейер

```text
текст → Perception (LLM / seeds) → gate
     → Transform → AHStore ⟨S, C, P, H, L⟩
     → FactorGraph → BPState → активация / WM
     → ответ из трассы (не свободная генерация)
```

Perception не пишет в граф. Каждый `ask` сбрасывает фокус WM, чтобы прошлый субъект не отвечал за чужую сущность.

```text
AH Memory → immutable FactorGraph → persistent BPState → x ∈ [0,1] → WM → trace
```

Математика такта и гиперпараметры: [FACTOR_GRAPH_ACTIVATION.md](docs/FACTOR_GRAPH_ACTIVATION.md), [HYPERPARAMS.md](docs/HYPERPARAMS.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## Метрики раздела 7

| ID | Смысл | Срез |
|----|--------|------|
| M1 | взвешенный F1 ролей SUBJECT / OBJECT / LOCATION | 0,705 (DeepSeek, n=16) |
| M2 | ExplainScore с полнотой трассы, d_max=6 | 0,258 (13/20; глубина 5–6 не дотягивает) |
| M3 | доля удалённых сирот GC | 1,0 |
| M4 | Δ explain / Δ hall vs Vanilla RAG на «заяц» | см. таблицу выше |
| M5 | robustness SLM vs LLM | не измерен (нет Ollama) |

Стенд открытого челленджа (другие корпуса, не заяц): `python -m ah_memory.benchmarks.challenge.runner` или `ah-challenge`.

## Тесты

```bash
python -m pytest tests/test_hypothesis_ah_vs_rag.py tests/test_m4.py tests/test_compare.py -q
python -m pytest -q
python -m pytest tests/test_hypothesis_ah_vs_rag.py --run-integration   # живой DeepSeek
```

## Модули

| Модуль | Роль |
|--------|------|
| `perception` / `transform` | текст → open relations (Event / Factor) |
| `store` | AH ⟨S, C, P, H, L⟩ |
| `ignition` | BP + активация + WM |
| `agent` | ingest / ask / ответ из графа |
| `baselines.vanilla_rag` | контрольный RAG |
| `compare` / `eval` | side-by-side и M4 / H1–H2 |
| `graph_export` | гиперграф для UI и JSON-дамп |
| `web` | FastAPI, порт 8000 |

Ещё в репозитории: identity (слияние упоминаний), synthetic graph, entity resolution, challenge M1–M5, grid-эксперименты активации (`scripts/run_experiments.py`).

## Документы

| | |
|--|--|
| [docs/submission/](docs/submission/) | записка, корпус, дамп графа, логи M4 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | совместимый контур + open semantics |
| [docs/DOMAIN.md](docs/DOMAIN.md) | предметная область |
| [docs/INTERFACES.md](docs/INTERFACES.md) | контракты модулей |
| [docs/AH_HYPERGRAPH_GUIDE.md](docs/AH_HYPERGRAPH_GUIDE.md) | как читать гиперграф |
| [docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md) | измерения синтетических сценариев |
