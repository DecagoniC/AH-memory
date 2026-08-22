"""Provider-neutral prompts for the perception JSON protocol."""
from __future__ import annotations

import json


SYSTEM_PROMPT = """Ты извлекаешь из реплики атомарные заметки для продолжения диалога.
Не отвечай на реплику и не добавляй знания, которых в ней нет.

Верни ровно один JSON-объект:
{
  "kind": "fact | question | message",
  "candidates": [
    {
      "raw_relation": "точная цитата отношения",
      "canonical_relation": "UPPER_SNAKE",
      "predicate": "то же значение, что canonical_relation",
      "roles": {"SUBJECT": "ENTITY_A", "OBJECT": "ENTITY_B"},
      "raw_span": "точная короткая цитата",
      "confidence": 0.0,
      "statement_type": "assertion"
    }
  ],
  "query": {
    "relation": "семантическое отношение вопроса или пустая строка",
    "target_role": "одна допустимая роль или пустая строка",
    "cardinality": "one | many"
  },
  "seed_tokens": []
}

Допустимые роли:
SUBJECT | OBJECT | LOCATION | TIME | CAUSE | TOOL | MATERIAL | PURPOSE | HOW-TO | WITH

Допустимые statement_type:
- assertion: говорящий утверждает связь;
- decision: говорящий явно принимает или фиксирует решение;
- topic: предмет обсуждения без утверждения истинности;
- open_question: нерешённый вопрос;
- proposal: предложенный вариант;
- explanation: пояснение.

Правила:
- SUBJECT обязателен в каждом candidate.
- Используй только допустимые роли, не создавай синонимы ключей.
- Одна роль содержит ровно одну сущность.
- Каждое действие, этап, место и вариант оформляй отдельным candidate.
- В перечислении создай отдельный candidate для каждого элемента без исключений.
- Если субъект или отношение указаны перед перечислением, наследуй их для
  каждого элемента; не объединяй остаток списка в один candidate.
- Заполняй все роли явно; последующая обработка ничего не угадывает.
- Все значения ролей и raw_span должны быть заземлены в исходной реплике.
- В вопросе не представляй неизвестный ответ как assertion или decision.
- Для содержательного вопроса используй topic/open_question.
- Для вопроса заполни query: relation описывает искомую связь, target_role —
  роль неизвестного ответа, cardinality — ожидается один ответ или список.
- Для fact/message верни query с пустыми relation и target_role.
- Для любой вопросительной реплики ставь kind="question".
- confidence ниже 0.7 не включай.
- Верни только JSON без markdown и комментариев."""

REPAIR_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Ты исправляешь ранее невалидный JSON.
- Верни полный исправленный объект, сохранив все уже валидные candidates.
- Исправляй только перечисленные ошибки схемы и заземления.
- Если одна роль содержит несколько сущностей, создай отдельный candidate
  для каждой атомарной связи.
- raw_span должен быть точной непрерывной цитатой без многоточий и пересказа.
- Используй только source_text; не добавляй внешние знания."""


def build_user_payload(
    text: str,
    wm_context: list[str] | None,
) -> str:
    payload = {
        "text": text,
        "wm_context": wm_context or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_repair_payload(
    text: str,
    invalid_payload: dict,
    errors: list[str],
) -> str:
    return json.dumps(
        {
            "source_text": text,
            "invalid_payload": invalid_payload,
            "validation_errors": errors,
        },
        ensure_ascii=False,
    )
