"""Entity types and name dictionaries for synthetic worlds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ENTITY_TYPES: tuple[str, ...] = (
    "Person",
    "Company",
    "Place",
    "Object",
    "Event",
    "Document",
)

PERSON_NAMES: tuple[str, ...] = (
    "Иван",
    "Пётр",
    "Анна",
    "Мария",
    "Сергей",
    "Ольга",
    "Алексей",
    "Елена",
    "Дмитрий",
    "Наталья",
    "Андрей",
    "Екатерина",
    "Михаил",
    "Татьяна",
    "Николай",
    "Ирина",
    "Владимир",
    "Светлана",
    "Павел",
    "Юлия",
)

COMPANY_NAMES: tuple[str, ...] = (
    "Яндекс",
    "Сбер",
    "Газпром",
    "Ростех",
    "Mail",
    "Тинькофф",
    "ВТБ",
    "Роснефть",
    "Касперский",
    "Ozon",
    "Avito",
    "Wildberries",
    "МТС",
    "Билайн",
    "Мегафон",
    "РЖД",
)

PLACE_NAMES: tuple[str, ...] = (
    "Москва",
    "Санкт-Петербург",
    "Казань",
    "Новосибирск",
    "Екатеринбург",
    "Владивосток",
    "Сочи",
    "Калининград",
    "Россия",
    "Германия",
    "Франция",
    "Европа",
    "Азия",
    "Берлин",
    "Париж",
    "Лондон",
)

OBJECT_NAMES: tuple[str, ...] = (
    "BMW",
    "Audi",
    "Opel",
    "Toyota",
    "Lada",
    "Ноутбук",
    "Телефон",
    "Планшет",
    "Сервер",
    "Книга",
    "Документ",
    "Мотоцикл",
    "Велосипед",
    "Часы",
)

EVENT_NAMES: tuple[str, ...] = (
    "Инцидент",
    "Сделка",
    "Переезд",
    "Встреча",
    "Авария",
    "Презентация",
    "Конференция",
    "Запуск",
)

DOCUMENT_NAMES: tuple[str, ...] = (
    "Отчёт",
    "Контракт",
    "Письмо",
    "Записка",
    "Протокол",
    "Справка",
)

TYPE_NAME_POOLS: Mapping[str, tuple[str, ...]] = {
    "Person": PERSON_NAMES,
    "Company": COMPANY_NAMES,
    "Place": PLACE_NAMES,
    "Object": OBJECT_NAMES,
    "Event": EVENT_NAMES,
    "Document": DOCUMENT_NAMES,
}


@dataclass(frozen=True)
class Entity:
    uid: str
    type: str
    name: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "type": self.type,
            "name": self.name,
            "attributes": dict(self.attributes),
        }


def make_entity_uid(entity_type: str, index: int) -> str:
    prefix = entity_type.strip().lower()
    return f"{prefix}_{index:06d}"


def pick_name(entity_type: str, index: int) -> str:
    pool = TYPE_NAME_POOLS.get(entity_type, ("Entity",))
    base = pool[index % len(pool)]
    suffix = index // len(pool)
    return base if suffix == 0 else f"{base}-{suffix + 1}"
