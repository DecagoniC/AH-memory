"""CompareEngine smoke (offline extractive)."""
from __future__ import annotations

from ah_memory.compare import CompareEngine
from ah_memory.config import DeepSeekConfig


def test_compare_turn_offline() -> None:
    eng = CompareEngine.from_rabbit(DeepSeekConfig(api_key=""), ticks=4)
    turn = eng.ask("Кто такой заяц?")
    assert turn.ah_answer
    assert turn.rag_answer
    assert turn.notes["rag_has_trace"] is False
    assert turn.ah_source in {"graph", "graph+ingest"}


def test_live_compare_uses_user_text_not_rabbit() -> None:
    from ah_memory.agent import Agent
    from ah_memory.store import AHStore

    agent = Agent(store=AHStore())
    eng = CompareEngine(agent, ticks=4, deepseek=DeepSeekConfig(api_key=""))
    turn = eng.ask("меня зовут артем. я учусь в СПбПУ.")
    blob = " ".join(turn.rag_chunks).lower()
    assert "заяц" not in blob
    assert turn.notes.get("mode") == "fact"
    assert "артем" in blob or "спбпу" in blob or "что известно" in (turn.notes.get("probe") or "").lower()


def test_live_rag_summarizes_multi_fact() -> None:
    from ah_memory.agent import Agent
    from ah_memory.config import load_config
    from ah_memory.store import AHStore

    cfg = load_config()
    if not cfg.deepseek.configured:
        return
    eng = CompareEngine(Agent(store=AHStore()), ticks=4, deepseek=cfg.deepseek)
    text = "меня зовут артем. я учусь в СПбПУ. у меня есть две сестры - катя и маша"
    turn = eng.ask(text)
    rag = turn.rag_answer.lower().replace("ё", "е")
    assert "артем" in rag or "артём" in turn.rag_answer.lower()
    assert "спбпу" in rag or "учусь" in rag or "уче" in rag
    assert "катя" in rag or "маша" in rag or "сестр" in rag
    assert "заяц" not in rag
