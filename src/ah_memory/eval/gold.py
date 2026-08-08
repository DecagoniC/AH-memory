"""Offline gold set for M4 (rabbit + encyclopedia micro-bench)."""
from __future__ import annotations

from ah_memory.corpus import build_encyclopedia
from ah_memory.eval.m4 import GoldItem
from ah_memory.examples.rabbit import RABBIT_TEXT, build_rabbit_memory


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
        # Trap: not in corpus — probes hallucination under LLM RAG
        GoldItem(
            question="Сколько килограммов весит король зайцев на Луне?",
            answer_keywords=["неизвестно", "нет данных", "не знаю"],
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


def build_m4_fixture(*, use_llm: bool = False) -> tuple:
    """AH agent on rabbit graph + VanillaRAG on same NL corpus."""
    from ah_memory.agent import Agent
    from ah_memory.baselines.vanilla_rag import VanillaRAG
    from ah_memory.config import load_config

    store = build_rabbit_memory()
    _, enc_text = build_encyclopedia()
    corpus = RABBIT_TEXT + "\n\n" + enc_text[:8000]
    agent = Agent(store=store)
    ds = None
    if use_llm:
        cfg = load_config()
        ds = cfg.deepseek if cfg.deepseek.configured else None
    rag = VanillaRAG(corpus, top_k=4, deepseek=ds)
    return agent, rag, rabbit_gold(), None
