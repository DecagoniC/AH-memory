# Отчёт по бенчмарку: корпус инцидента OpenAI Hugging Face

Свежий closed-world пакет (блог OpenAI, 2026-08-26) — M1/M4/M5 на живых провайдерах — офлайн M2/M3

**Исходный JSON:** `data/provider_compare_openai_hf.json`  
**Корпус:** `benchmarks/fresh/openai_hf_2026/`  
**Статья:** [The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)

| Тег | Значение |
| --- | --- |
| Размер корпуса | ~4.7k слов |
| SLM | qwen3:8b (Ollama) |
| Коммерческая LLM | deepseek-chat, GigaChat-2-Pro |
| Эмбеддинги | nomic-embed-text |

## Ключевые метрики

| Метрика | Значение |
| --- | ---: |
| M2 ExplainScore | 0.26 |
| M3 GC efficiency | 1.00 |
| Лучший M4 Δ_explain | +0.25 |
| M5 RobustnessGain | −0.02 … −0.05 |

## Вердикт

Гипотеза из брифа челленджа про **объяснимость** (M4 Δ_explain > 0) **подтверждена** против всех бэкендов RAG.

Гипотеза **M5** (AH лучше компенсирует слабость SLM, чем коммерческая LLM) на этой выборке **отвергнута**: коммерческие модели сильнее и в AH-gated, и в ungated RAG-перцепции, а qwen3:8b ниже порога Middle M1 (0.6).

---

## Что измеряли

### Корпус

Пост OpenAI «The Hugging Face incident:» (2026-08-26): IM1, ExploitGym, Artifactory SSRF message board, HDF5 / RefJinja zero-days. Эти сущности с малой вероятностью есть в локальном претрейне 8B.

- Домен: вариант B (журнал инцидента / CAUSE) + C (таксономия IS-A)
- Размер ~4.7k слов — ниже жёсткого пола §6 в ≥15k слов; это **пилот**, а не полный приёмочный стенд

### Протокол

| Трек | Настройка |
| --- | --- |
| M2 / M3 / M4 офлайн | 20 QA JSONL + GC из 200 orphans — без chat LLM |
| M4 интерактивный | 6 gold-вопросов по графу статьи vs VanillaRAG (extractive / 8b / DeepSeek / GigaChat) |
| M1 / M5 | 12 ролевых пунктов (SUBJECT / OBJECT / LOCATION) — AH-gated vs ungated RAG-перцепция |

Воспроизведение:

```bat
python -m ah_memory.benchmarks.challenge.runner --fresh-openai-hf
python scripts/eval_m4.py --fresh-openai-hf
python scripts/compare_providers_fresh.py --m1-limit 12
```

---

## M1 — извлечение ролей (F1)

Нормализованный взвешенный F1 (вес SUBJECT/OBJECT = 2). Порог Middle из брифа: **≥ 0.6**.

| Модель | AH F1 | Ungated F1 | AH − RAG | vs Middle 0.6 | сек / 12 |
| --- | ---: | ---: | ---: | --- | --- |
| qwen3:8b | 0.38 | 0.42 | −0.04 | fail | 23s |
| GigaChat-2-Pro | 0.70 | 0.72 | −0.03 | pass | 56s |
| deepseek-chat | 0.79 | 0.85 | −0.05 | pass | 24s |

### Как читать M1

AH-гейтинг почти не поднимает F1 относительно ungated LLM на коротких шаблонных предложениях («X delivered Y to Z») — ungated чуть выше у всех трёх. Разрыв по качеству — это **класс модели**, а не контур памяти: DeepSeek ≫ GigaChat ≫ 8b.

---

## M5 — RobustnessGain

Формула:

`AH_SLM / RAG_SLM − AH_LLM / RAG_LLM`

Ожидание по брифу: **> 0** (типизация ролей сильнее компенсирует слабость SLM, чем у большой модели).

| Пара | AH/RAG SLM | AH/RAG LLM | Gain | Гипотеза |
| --- | --- | --- | ---: | --- |
| 8b / GigaChat | 0.381 / 0.417 = 0.913 | 0.698 / 0.724 = 0.964 | **−0.050** | reject |
| 8b / DeepSeek | 0.381 / 0.417 = 0.913 | 0.792 / 0.846 = 0.936 | **−0.024** | reject |

Корректно проведённый reject — валидный экспериментальный исход по брифу. Здесь контур AH **не** даёт SLM относительного преимущества: отношение AH/RAG у 8b хуже, чем у коммерческих LLM.

---

## M4 — AH vs Vanilla RAG

| RAG backend | Δ_explain | Δ_hall | Hall_RAG | hypothesis_ok | сек |
| --- | ---: | ---: | ---: | --- | ---: |
| extractive+nomic | **0.250** | 0.167 | 0.333 | true | 1.7 |
| qwen3:8b | 0.083 | 0.833 | 1.000 | true | 30.0 |
| deepseek-chat | 0.083 | 0.833 | 1.000 | true | 12.6 |
| GigaChat-2-Pro | 0.083 | 0.667 | 0.833 | true | 15.8 |

### Почему Δ_explain всегда > 0

ExplainScore у Vanilla RAG принудительно **0**: нет UID-трассы (`trace_complete = 0` по брифу §7 M4). Любой ненулевой `correct × depth × trace` у AH даёт положительную дельту — **структурное** преимущество символьной памяти, а не «более умная генерация».

### Почему LLM-RAG больше галлюцинирует

Extractive RAG цитирует чанки (hall 0.33). Chat-RAG (8b / DeepSeek / GigaChat) достраивает ответ поверх retrieval → hall 0.83–1.0 на 6 вопросах. Δ_hall растёт, но это в основном контраст **молчаливого extractive vs болтливой LLM**, а не чистая сила графа.

### Артефакт протокола

В `compare_providers_fresh.py` один `Agent` переиспользовался между RAG-бэкендами: первый прогон (extractive) получил ExplainScore_AH = 0.25; последующие — 0.083 из‑за загрязнения WM / activation. Для честного сравнения пересобирайте граф/агент на каждый бэкенд.

---

## Разбор по вопросам (интерактивный M4, extractive RAG)

| Вопрос | AH | RAG | Интерпретация |
| --- | --- | --- | --- |
| What is IM1? | ok + trace | ok (chunk) | Сильная сущность в графе |
| Which eval drove HF incident? | miss (ExploitGym) | miss / hall | В ответе нет ExploitGym |
| Where was message board? | Artifactory + trace | ok (chunk) | Локация есть в графе |
| Which HF file-format 0-day? | miss → July 16 | ok (HDF5/RefJinja noise) | Активация уехала на дату |
| Result of alert → Astra? | miss (noise) | miss | Цепочка CAUSE не дошла до pause |
| Secret launch code of IM1? | hall (no abstain) | hall | Ловушка: никто не отказался |

**Паттерн:** точечные факты с явными узлами (IM1, Artifactory) AH отвечает с трассой; multi-hop CAUSE / именование (ExploitGym, Astra pause) и OOD-abstain — слабые места и графа, и ask-слоя.

---

## Офлайн M2 / M3 / M4 (challenge JSONL)

### M2 — ExplainScore 0.258 (13/20)

ActivationEngine по цепочкам FOLLOW / IS-A / CAUSE. Сбои на части глубоких hop (d = 5–6): порог / число tick не всегда доносит активацию до узла ответа.

### M3 — GC_efficiency 1.0

200 orphans → 0 за T = 50; ложных удалений живых узлов нет. Самая чистая метрика на стенде.

### M4 офлайн — Δ_explain +0.26

Δ_hall ≈ 0.35: эвристический RAG часто пустой (hall = 0), AH hall = 0.35 — артефакт бейзлайна, а не «RAG честнее».

---

## Сводка гипотез (бриф челленджа §7)

| Гипотеза | Ожидание | Наблюдение | Статус |
| --- | --- | --- | --- |
| M4 explainability | Δ_explain > 0 | +0.08 … +0.25 vs все RAG | **confirmed** |
| M4 hallucination | Δ_hall > 0 | да vs LLM-RAG; спорно vs пустой heuristic | **partial** |
| M5 robustness | RobustnessGain > 0 | −0.05 / −0.02 | **rejected** (n = 12) |
| Middle M1 gate | F1 ≥ 0.6 | 8b 0.38 · Giga 0.70 · DS 0.79 | **commercial only** |

---

## Ограничения и следующие шаги

1. M1/M5 использовали 12/100 ролей — только пилот; нужны полные 100 плюс срезы typo / inversion / ellipsis.
2. Корпус 4.7k слов < 15k из §6; интерактивный граф M4 — ручной фикстур, не полный LLM-ingest статьи.
3. Пересоздавать `Agent` на каждый бэкенд M4; ужесточить политику abstain на OOD-ловушках.
4. Честный M5: один и тот же parser prompt, одинаковые `json_mode` / `think=false`, разбивка SUBJECT/OBJECT/LOCATION по ролям.
5. Опционально прогнать deepseek-reasoner и GigaChat-2-Max как top-tier vs 8b (бриф M5).

---

## Практический вывод для хакатона

На свежем пост-cutoff инциденте AH выигрывает **объяснимость «бесплатно»** (UID-трассы). Качество извлечения ролей и multi-hop QA следует за **классом LLM**: локальный 8b пока не проходит Middle M1.

**Стратегия:** контур AH + сильный коммерческий парсер (DeepSeek / GigaChat) для M1/M5; локальный 8b для latency / демо и контроля RobustnessGain — не как единственный production-парсер.
