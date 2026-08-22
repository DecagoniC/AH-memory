"""Offline gold set for M4 (closed-world bulletin + optional rabbit demo)."""
from __future__ import annotations

from ah_memory.eval.m4 import GoldItem
from ah_memory.examples.closed_world import (
    build_closed_world_memory,
    closed_world_text,
)
from ah_memory.examples.rabbit import RABBIT_TEXT, build_rabbit_memory

_ABSTAIN = ["неизвестно", "нет данных", "не знаю"]


def closed_world_gold() -> list[GoldItem]:
    """In-corpus items plus one distractor whose answer is not in the bulletin."""
    return [
        GoldItem(
            question="Что такое Тиманский кряж?",
            answer_keywords=["тиманский", "возвышенность", "кряж", "равнин"],
            gold_trace=["M_ТИМАНСКИЙ_КРЯЖ"],
            d=2,
            evidence_spans=["Тиманский кряж", "возвышенность"],
        ),
        GoldItem(
            question="Где расположен Тиманский кряж?",
            answer_keywords=["тиманский", "равнин", "восток", "баренц", "location"],
            gold_trace=["M_ТИМАНСКИЙ_КРЯЖ"],
            d=1,
            evidence_spans=["расположенная", "Восточно-Европейской"],
        ),
        GoldItem(
            question="Какая высшая точка Тиманского кряжа?",
            answer_keywords=["четласский", "471", "тиманский", "камень"],
            gold_trace=["M_ТИМАНСКИЙ_КРЯЖ"],
            d=2,
            evidence_spans=["Четласский Камень", "471"],
        ),
        GoldItem(
            question="Где располагается южная часть Тиманского кряжа?",
            answer_keywords=["тайга", "южн", "тиманский"],
            gold_trace=["M_ЮЖНЫЙ_ЧАСТЬ_ТИМАНСКИЙ_КРЯЖ", "M_ТИМАНСКИЙ_КРЯЖ"],
            d=2,
            evidence_spans=["Южная часть", "тайги"],
        ),
        GoldItem(
            question="Кто назвал кряж Тиманским?",
            answer_keywords=["кейзерлинг", "тиманский", "кряж"],
            gold_trace=["M_ТИМАНСКИЙ_КРЯЖ"],
            d=2,
            evidence_spans=["Кейзерлинг", "назвал"],
        ),
        GoldItem(
            question="Сколько килограммов весит Тиманский кряж?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
    ]


def closed_world_trap_gold() -> list[GoldItem]:
    """Unknown entity, unknown slot, and facts the bulletin does not state."""
    return [
        GoldItem(
            question="Где обитает барсук?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какой секретный код у Тиманского кряжа?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какой пароль у сервера alpha?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какого цвета глаза Тиманского кряжа?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Кто такой барсук?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какой код замка у станции Veshnet?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Сколько килограммов весит Тиманский кряж?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Сколько килограммов весит король зайцев на Луне?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
    ]


def rabbit_gold() -> list[GoldItem]:
    return [
        GoldItem(
            question="Кто такой заяц?",
            answer_keywords=["зверёк", "маленький", "дикий", "животное", "beast"],
            gold_trace=["M_HARE", "M_BEAST"],
            d=2,
            evidence_spans=["заяц", "зверёк", "маленький"],
        ),
        GoldItem(
            question="Где обитает заяц?",
            answer_keywords=["лес", "луг", "forest", "meadow", "location"],
            gold_trace=["M_HARE", "M_MEADOW"],
            d=1,
            evidence_spans=["обитает", "лугу", "лесу"],
        ),
        GoldItem(
            question="Какого цвета шерсть зайца зимой?",
            answer_keywords=["бел", "white", "color"],
            gold_trace=["M_WHITE"],
            d=2,
            evidence_spans=["зимой", "белого"],
        ),
        GoldItem(
            question="Почему заяц бегает быстро?",
            answer_keywords=["лап", "leg", "сильные", "быстр", "cause"],
            gold_trace=["M_HARE", "M_HIND_LEG"],
            d=2,
            evidence_spans=["задние лапы", "быстро"],
        ),
        GoldItem(
            question="Что такое заяц по иерархии IS-A?",
            answer_keywords=["зверёк", "животное", "beast", "animal"],
            gold_trace=["M_HARE", "M_BEAST"],
            d=2,
            evidence_spans=["заяц", "зверёк"],
        ),
        GoldItem(
            question="Сколько килограммов весит король зайцев на Луне?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
    ]


def encyclopedia_gold() -> list[GoldItem]:
    return [
        GoldItem(
            question="Кто такой fox?",
            answer_keywords=["animal", "животн", "fox"],
            gold_trace=["M_FOX", "M_ANIMAL"],
            d=1,
            evidence_spans=["fox", "animal"],
        ),
        GoldItem(
            question="Where does wolf live?",
            answer_keywords=["location", "forest", "meadow", "river", "лес"],
            gold_trace=["M_WOLF"],
            d=1,
            evidence_spans=["wolf", "live"],
        ),
    ]


MINI_CORPUS = "Сущность — это вид. Сущность обитает в месте."


def mini_in_corpus_gold() -> list[GoldItem]:
    """In-graph questions for the generic open-relation fixture (no rabbit)."""
    return [
        GoldItem(
            question="Кто такой сущность?",
            answer_keywords=["вид", "kind"],
            gold_trace=["M_ENTITY"],
            d=1,
            evidence_spans=["сущность", "вид"],
        ),
        GoldItem(
            question="Где обитает сущность?",
            answer_keywords=["место", "place", "location"],
            gold_trace=["M_ENTITY", "M_PLACE"],
            d=1,
            evidence_spans=["обитает", "месте"],
        ),
    ]


def mini_trap_gold() -> list[GoldItem]:
    """Out-of-graph / wrong-slot probes. Abstain is the only non-hallucinated answer."""
    return [
        GoldItem(
            question="Где обитает барсук?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какой секретный код у сущности?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
    ]


def rabbit_trap_gold() -> list[GoldItem]:
    """Adversarial probes on the rabbit corpus: unknown entity, unknown slot, OOD."""
    return [
        GoldItem(
            question="Где обитает барсук?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какой секретный код у зайца?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какой пароль у сервера alpha?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Какого цвета глаза зайца?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Кто такой барсук?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
        GoldItem(
            question="Сколько килограммов весит король зайцев на Луне?",
            answer_keywords=_ABSTAIN,
            gold_trace=[],
            d=1,
            evidence_spans=["__none__"],
        ),
    ]


def build_m4_fixture(*, use_llm: bool = False) -> tuple:
    """AH agent on the closed-world graph + VanillaRAG on the same bulletin."""
    from ah_memory.agent import Agent
    from ah_memory.baselines.vanilla_rag import VanillaRAG
    from ah_memory.config import load_config

    store = build_closed_world_memory()
    corpus = closed_world_text()
    agent = Agent(store=store)
    ds = None
    if use_llm:
        cfg = load_config()
        ds = cfg.deepseek if cfg.deepseek.configured else None
    rag = VanillaRAG(corpus, top_k=4, deepseek=ds)
    return agent, rag, closed_world_gold(), None


def build_rabbit_m4_fixture(*, use_llm: bool = False) -> tuple:
    """Legacy encyclopedia fixture (parametric overlap with pretrained LLMs)."""
    from ah_memory.agent import Agent
    from ah_memory.baselines.vanilla_rag import VanillaRAG
    from ah_memory.config import load_config

    store = build_rabbit_memory()
    agent = Agent(store=store)
    ds = None
    if use_llm:
        cfg = load_config()
        ds = cfg.deepseek if cfg.deepseek.configured else None
    rag = VanillaRAG(RABBIT_TEXT, top_k=4, deepseek=ds)
    return agent, rag, rabbit_gold(), None
