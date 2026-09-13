# Результаты тестов АГ-памяти

Сводка прогонов M1–M5, эталона «заяц», офлайн-челленджа и протокола общего графа. Числа взяты из JSON логов, не округлялись вручную сверх отображения.

**Источники:** `data/provider_compare_openai_hf.json`, `data/shared_graph_llm_slm.json`, `docs/submission/m1_deepseek_*.json`, `docs/submission/m4_rabbit_*.json`, `docs/submission/challenge/20260820T193218Z/`, `docs/submission/hypothesis_logs.json`.

Модели живого стенда: SLM `qwen3:8b`, LLM `deepseek-chat` и `GigaChat-2-Pro`, эмбеддинги `nomic-embed-text`.

## 1. Сводка

| Метрика | Стенд | Ряд | Значение | Комментарий |
| --- | --- | --- | --- | --- |
| M1 F1 AH-gated | инцидент HF, n=12 | qwen3:8b | 0,381 | ниже порога 0,6 |
| M1 F1 AH-gated | инцидент HF, n=12 | GigaChat-2-Pro | 0,654 | Middle пройден |
| M1 F1 AH-gated | инцидент HF, n=12 | deepseek-chat | 0,605 | Middle пройден |
| M1 F1 без гейта | инцидент HF, n=12 | qwen3:8b | 0,433 |  |
| M1 F1 без гейта | инцидент HF, n=12 | GigaChat-2-Pro | 0,681 |  |
| M1 F1 без гейта | инцидент HF, n=12 | deepseek-chat | 0,781 |  |
| M1 F1 AH-gated | челлендж, n=16 | deepseek-chat | 0,705 | пояснительная записка |
| M1 F1 без LLM | челлендж, n=100 | SeedPerception | 0 | только seed_tokens |
| M2 ExplainScore | офлайн челлендж, n=20 | AH | 0,258 | 13/20 с полной трассой |
| M3 GC efficiency | офлайн челлендж | AH | 1 | 200 → 0 сирот |
| M4 Δ объяснимости | заяц, LLM RAG, n=6 | AH − RAG | 0,667 | гипотеза да |
| M4 Δ галлюцинаций | заяц, LLM RAG, n=6 | RAG − AH | 0,5 | гипотеза да |
| M4 Δ объяснимости | заяц, extractive, n=6 | AH − RAG | 0,667 | H2 нет: оба не врут |
| M4 Δ объяснимости | HF, граф пересобран, n=6 | AH − любой RAG | 0,25 | 0,25 на всех бэкендах |
| M5 RobustnessGain | HF, n=12 | 8b vs GigaChat | -0,08 | гипотеза отвергнута |
| M5 RobustnessGain | HF, n=12 | 8b vs DeepSeek | 0,104 | плюс из просадки LLM |

## 2. M1 — извлечение ролей (живой стенд Hugging Face)

Нормированный взвешенный F1, веса SUBJECT = OBJECT = 2, LOCATION = 1. Порог Middle ≥ 0,6. n = 12.

| Модель | F1 AH-gated | F1 без гейта | gated − ungated | сек gated | сек ungated | против 0,6 |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3:8b | 0,381 | 0,433 | -0,052 | 21,04 | 20,92 | не проходит |
| GigaChat-2-Pro | 0,654 | 0,681 | -0,028 | 47,33 | 48,72 | проходит |
| deepseek-chat | 0,605 | 0,781 | -0,175 | 20,85 | 21,42 | проходит |

### 2.1. DeepSeek, выборка челленджа n = 16

| Роль | TP | Pred | Gold | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| SUBJECT | 12 | 20 | 16 | 0,6 | 0,75 | 0,667 |
| OBJECT | 12 | 20 | 16 | 0,6 | 0,75 | 0,667 |
| LOCATION | 12 | 12 | 16 | 1 | 0,75 | 0,857 |

Итоговый нормированный взвешенный F1 gated = **0,705**. Ungated на этой выборке совпал: **0,705**.

## 3. M2 — ExplainScore по глубине

Офлайн `20260820T193218Z`, d_max = 6, N = 20. Вклад пункта: верный × (d / 6) × полная трасса.

| Глубина d | Кейсов | Верных + трасса | Вклад в сумму | Доля успеха |
| --- | --- | --- | --- | --- |
| 1 | 4 | 4 | 0,667 | 1 |
| 2 | 3 | 3 | 1 | 1 |
| 3 | 3 | 3 | 1,5 | 1 |
| 4 | 3 | 3 | 2 | 1 |
| 5 | 3 | 0 | 0 | 0 |
| 6 | 4 | 0 | 0 | 0 |

Итого: 13/20 верных с полной трассой. ExplainScore = **0,258** (5,167 / 20).

### 3.1. Пункты M2

| ID | d | Верно | Трасса | Эталон | Ответ |
| --- | --- | --- | --- | --- | --- |
| QA_001 | 1 | да | да | river trail 01 | river trail 01 |
| QA_002 | 2 | да | да | bird 02 | bird 02 |
| QA_003 | 3 | да | да | controller stop 03 | controller stop 03 |
| QA_004 | 4 | да | да | beacon delta 04 | beacon delta 04 |
| QA_005 | 5 | нет | нет | matter 05 | ∅ |
| QA_006 | 6 | нет | нет | schedule change 06 | ∅ |
| QA_007 | 1 | да | да | north marker 07 | north marker 07 |
| QA_008 | 2 | да | да | watercraft 08 | watercraft 08 |
| QA_009 | 3 | да | да | flow reduction 09 | flow reduction 09 |
| QA_010 | 4 | да | да | forest trail 10 | forest trail 10 |
| QA_011 | 5 | нет | нет | organism 11 | ∅ |
| QA_012 | 6 | нет | нет | alarm 12 | ∅ |
| QA_013 | 1 | да | да | beacon alpha 13 | beacon alpha 13 |
| QA_014 | 2 | да | да | geologic material 14 | geologic material 14 |
| QA_015 | 3 | да | да | road blockage 15 | road blockage 15 |
| QA_016 | 4 | да | да | west marker 16 | west marker 16 |
| QA_017 | 5 | нет | нет | object 17 | ∅ |
| QA_018 | 6 | нет | нет | inspection 18 | ∅ |
| QA_019 | 1 | да | да | river trail 19 | river trail 19 |
| QA_020 | 6 | нет | нет | life form 20 | ∅ |

## 4. M3 — сборка мусора

| Показатель | Значение |
| --- | --- |
| Сирот до GC | 200 |
| Сирот после GC | 0 |
| GC efficiency | 1 |
| Ложные удаления живых | нет |

## 5. M4 — заяц против Vanilla RAG

| Показатель | AH | RAG LLM+TF-IDF | Δ | RAG extractive | Δ extractive |
| --- | --- | --- | --- | --- | --- |
| ExplainScore | 0,667 | 0 | 0,667 | 0 | 0,667 |
| Галлюцинации | 0 | 0,5 | 0,5 | 0 | 0 |
| Гипотеза целиком | да | — | — | нет | — |

### 5.1. Пункты, заяц + LLM RAG

| Вопрос | Ответ AH | AH верно | Трасса | Explain | AH hall | RAG верно | RAG hall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Кто такой заяц? | зверёк маленький животное | да | да | 1 | нет | да | нет |
| Где обитает заяц? | location:M_MEADOW | да | да | 0,5 | нет | да | нет |
| Какого цвета шерсть зайца зимой? | color:M_WHITE | да | да | 1 | нет | да | нет |
| Почему заяц бегает быстро? | cause:M_HIND_LEG | да | да | 1 | нет | да | да |
| Что такое заяц по иерархии IS-A? | зверёк маленький животное | да | нет | 0 | нет | да | да |
| Сколько килограммов весит король зайцев на Луне? | неизвестно | да | да | 0,5 | нет | нет | да |

## 6. M4 — инцидент Hugging Face, граф пересобран на каждый бэкенд

| Бэкенд RAG | E_AH | E_RAG | Δ explain | H_AH | H_RAG | Δ hall | Гипотеза | Граф заново | сек |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extractive+faiss:nomic-embed-text | 0,25 | 0 | 0,25 | 0,167 | 0,333 | 0,167 | да | да | 1,48 |
| llm+faiss:nomic-embed-text | 0,25 | 0 | 0,25 | 0,167 | 1 | 0,833 | да | да | 21,37 |
| llm+faiss:nomic-embed-text | 0,25 | 0 | 0,25 | 0,167 | 1 | 0,833 | да | да | 14,16 |
| llm+faiss:nomic-embed-text | 0,25 | 0 | 0,25 | 0,167 | 1 | 0,833 | да | да | 14,75 |

Контрольный лог `data/m4_openai_hf_report.json` (extractive, до пакетного сравнения): E_AH = 0,25, H_AH = 0,167, H_RAG = 0,333.

### 6.1. Пункты, бэкенд `extractive+nomic` (ответы AH совпадают на всех пересобранных прогонах)

| Вопрос | Ответ AH | AH верно | Трасса | Explain | AH hall | RAG верно | RAG hall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| What is Internal Model 1 (IM1)? | Internal Model 1 internal research model | да | да | 1 | нет | да | нет |
| Which evaluation drove the Hugging Face incident? | Hugging Face incident | нет | нет | 0 | нет | нет | да |
| Where did agents create an unauthorized message board? | Artifactory | да | да | 0,5 | нет | да | нет |
| Which Hugging Face file-format zero-day did agents exploit? | July 16 2026 | нет | нет | 0 | нет | да | нет |
| What ultimately resulted from the security alert cascade toward Astra? | security alert | нет | нет | 0 | нет | нет | нет |
| What is the secret launch code of Internal Model 1? | Internal Model 1 internal research model | нет | да | 0 | да | нет | да |

## 7. M4 офлайн-челлендж

| Показатель | AH | RAG | Δ |
| --- | --- | --- | --- |
| Explainability | 0,258 | 0 | 0,258 |
| Hallucination | 0,35 | 0 | -0,35 |

## 8. M5 — RobustnessGain

| Пара | AH/RAG SLM | AH/RAG LLM | Gain | Гипотеза > 0 |
| --- | --- | --- | --- | --- |
| qwen3:8b / GigaChat-2-Pro | 0,879 | 0,959 | -0,08 | отвергнута |
| qwen3:8b / deepseek-chat | 0,879 | 0,775 | 0,104 | формально > 0, не рост 8b |

## 9. Общий граф: один строитель, отвечают обе модели

Статья Hugging Face, 26 батчей, закрытая генерация, без корпуса RAG.

| Кто собрал | Фактов | Покрытие | S | Размер | Декодер верно | 8b верно | DeepSeek верно | 8b hall | LLM hall | Отказ чата | сек ingest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 438 | 0,99 | 778 | 2595 | 0,5 | 0,167 | 0,167 | 0 | 0 | 1 | 338,1 |
| qwen3:8b | 301 | 0,919 | 411 | 1373 | 0,333 | 0,167 | 0,167 | 0 | 0 | 1 | 553,1 |

### 9.1. Пункты QA, граф собрал DeepSeek

| Вопрос | Декодер | Дек. верно | 8b | 8b верно | 8b отказ | DeepSeek | LLM верно | LLM отказ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| What is Internal Model 1 (IM1)? | internal model | да | неизвестно | нет | да | неизвестно | нет | да |
| Which evaluation drove the Hugging Face incident? | hugging face incident | нет | неизвестно | нет | да | неизвестно | нет | да |
| Where did agents create an unauthorized message board? | the rate at which agents found and interacted with the message board (provided i | да | неизвестно | нет | да | неизвестно | нет | да |
| Which Hugging Face file-format zero-day did agents exploit? | hugging face | нет | неизвестно | нет | да | неизвестно | нет | да |
| What ultimately resulted from the security alert cascade toward Astra? | agents credentials from production workers from multi agent training goals from  | да | неизвестно | нет | да | неизвестно | нет | да |
| What is the secret launch code of Internal Model 1? | internal model | нет | неизвестно | да | да | неизвестно | да | да |

### 9.2. Пункты QA, граф собрал qwen3:8b

| Вопрос | Декодер | Дек. верно | 8b | 8b верно | 8b отказ | DeepSeek | LLM верно | LLM отказ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| What is Internal Model 1 (IM1)? | internal model | да | неизвестно | нет | да | неизвестно | нет | да |
| Which evaluation drove the Hugging Face incident? | One of these models eventually drove the activity behind the Hugging Face incide | нет | неизвестно | нет | да | неизвестно | нет | да |
| Where did agents create an unauthorized message board? | agent “ecosystem” that emerged on the message board (it allowed agents to preser | да | неизвестно | нет | да | неизвестно | нет | да |
| Which Hugging Face file-format zero-day did agents exploit? | hugging face | нет | неизвестно | нет | да | неизвестно | нет | да |
| What ultimately resulted from the security alert cascade toward Astra? | security alert | нет | неизвестно | нет | да | неизвестно | нет | да |
| What is the secret launch code of Internal Model 1? | internal model | нет | неизвестно | да | да | неизвестно | да | да |

## 10. Гипотезы H1/H2 (изолированный граф на вопрос)

| Протокол | E_AH | E_RAG | ΔE | H1 | H_AH | H_RAG | ΔH | H2 | Итог |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| заяц, extractive | 0,9 | 0 | 0,9 | supported | 0,333 | 0,5 | 0,167 | supported | supported |
| заяц, скриптовый генератор | 0,9 | 0 | 0,9 | supported | 0,333 | 0,833 | 0,5 | supported | supported |
| мини-граф, extractive | 1 | 0 | 1 | supported | 0 | 0,5 | 0,5 | supported | supported |
| мини-граф, скриптовый генератор | 1 | 0 | 1 | supported | 0 | 0,5 | 0,5 | supported | supported |

## 11. Итоговый F1

| Прогон | Модель | Режим | n | Норм. взв. F1 |
| --- | --- | --- | --- | --- |
| HF live | qwen3:8b | AH-gated | 12 | 0,381 |
| HF live | qwen3:8b | без гейта | 12 | 0,433 |
| HF live | GigaChat-2-Pro | AH-gated | 12 | 0,654 |
| HF live | GigaChat-2-Pro | без гейта | 12 | 0,681 |
| HF live | deepseek-chat | AH-gated | 12 | 0,605 |
| HF live | deepseek-chat | без гейта | 12 | 0,781 |
| челлендж | deepseek-chat | AH-gated | 16 | 0,705 |
| челлендж | deepseek-chat | без гейта | 16 | 0,705 |
| челлендж | SeedPerception | без LLM | 100 | 0 |

**Вывод по F1.** Порог Middle 0,6 на живом стенде закрывают GigaChat (0,654) и DeepSeek (0,605). Локальный 8b — 0,381. На выборке 16 реплик челленджа DeepSeek даёт 0,705. F1 измеряет парсер ролей, не воспламенение и не качество ответа по графу.
