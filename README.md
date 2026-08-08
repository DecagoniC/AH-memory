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

Ключ DeepSeek — в `.env` (`DEEPSEEK_API_KEY=...`).

```bash
# полный gold-бенчмарк + LLM RAG
python scripts/compare_ah_vs_rag.py --m4

# один вопрос
python scripts/compare_ah_vs_rag.py -q "Кто такой заяц?"
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
Демо:

```bash
python -c "from ah_memory.examples.rabbit import build_rabbit_memory, syntactic_answer_who_is_hare, rabbit_auto_score; s=build_rabbit_memory(); print(rabbit_auto_score(s), syntactic_answer_who_is_hare(s))"
python -c "from ah_memory.examples.dog import run_dog_ignition; print(run_dog_ignition(6)[-1])"
python -c "from ah_memory.corpus import build_encyclopedia; st,c=build_encyclopedia(); print(len(st.ah.S), st.graph_size(), len(c.split()))"
```

## Модули

- `perception` / `transform` — текст → факты → `N`
- `templates` — ≥8 шаблонов, CREATE×7
- `ignition` — 8-шаговый такт + WM + pacemaker
- `gc` — сборка мусора с TTL
- `dsl` — интерпретатор запросов
- `agent` — цикл ingest/ask/step_message
- `baselines.vanilla_rag` — БЯМ + TF-IDF RAG (контрольный агент M4)
- `compare` — side-by-side АГ vs RAG + прогон M4
- `corpus` — энциклопедия N≥1000, \|S\|≥150, ≥15k слов
