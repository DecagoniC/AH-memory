# Комплект сдачи

1. Пояснительная записка: [POYASNITELNAYA_ZAPISKA.md](POYASNITELNAYA_ZAPISKA.md)  
2. Корпус и дамп графа: [corpus.txt](corpus.txt), [ah_dump.json](ah_dump.json), [graph_structure.json](graph_structure.json)

| Файл | Назначение |
|------|------------|
| [POYASNITELNAYA_ZAPISKA.md](POYASNITELNAYA_ZAPISKA.md) | Архитектура, область, гиперпараметры, таблицы M1–M5, логи vs RAG |
| [corpus.txt](corpus.txt) | Текст наполнения АГ-памяти («заяц») |
| [ah_dump.json](ah_dump.json) | Сериализация AH: S, L, факторы, события |
| [graph_structure.json](graph_structure.json) | Узлы / рёбра / гиперрёбра |
| [dump_meta.json](dump_meta.json) | Сводка: 8/8 фактов, размеры секций |
| [m4_rabbit_llm.json](m4_rabbit_llm.json) | M4 AH vs DeepSeek+TF-IDF |
| [m4_rabbit_extractive.json](m4_rabbit_extractive.json) | M4 AH vs extractive TF-IDF |
| [hypothesis_logs.json](hypothesis_logs.json) | H1/H2: isolated AH vs extractive и vs генератор |
| [m1_deepseek_gated.json](m1_deepseek_gated.json) | M1, восприятие с gate |
| [m1_deepseek_ungated.json](m1_deepseek_ungated.json) | M1, ungated LLM |
| [challenge/20260820T193218Z/](challenge/20260820T193218Z/) | Offline M2, M3, M4 стенда |
