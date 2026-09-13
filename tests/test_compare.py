"""CompareEngine smoke (offline extractive)."""
from __future__ import annotations

from ah_memory.compare import CompareEngine
from ah_memory.config import DeepSeekConfig
from ah_memory.dialogue import AH_CONTEXT_PREAMBLE, DIALOGUE_SYSTEM, DialogueAgent


def test_compare_turn_offline() -> None:
    eng = CompareEngine.from_m4_gold(DeepSeekConfig(api_key=""), ticks=4)
    turn = eng.ask("Что такое Тиманский кряж?")
    assert turn.ah_answer
    assert turn.rag_answer
    assert turn.notes["rag_has_trace"] is False
    assert turn.ah_source in {"graph", "graph+ingest"}


def test_rag_corpus_does_not_include_graph_dump() -> None:
    from ah_memory.compare import CompareEngine, build_rag_corpus
    from ah_memory.examples.closed_world import build_closed_world_memory, closed_world_text

    source = closed_world_text()
    store = build_closed_world_memory()
    corpus = build_rag_corpus(source_docs=[source])
    assert "полезные ископаемые" in corpus.lower() or "боксит" in corpus.lower()
    assert "HAS_MINERAL" not in corpus
    eng = CompareEngine.from_m4_gold(DeepSeekConfig(api_key=""), ticks=4)
    assert "HAS_MINERAL" not in eng.rag.corpus
    assert any(
        "HAS_MINERAL" in (
            f.relation.canonical_label.upper() if f.relation else ""
        )
        for f in store.list_semantic_factors()
    )


def test_live_compare_rag_uses_user_text_not_store() -> None:
    eng = CompareEngine.from_m4_gold(DeepSeekConfig(api_key=""), ticks=4)
    before = (
        eng.agent.store.graph_size(),
        len(eng.agent.store.list_semantic_factors()),
        eng.rag.corpus.strip(),
    )

    turns = [
        eng.ask(
            "какие ресурсы есть в тиманском кряже",
            mode="raw",
        )
        for _ in range(5)
    ]

    assert {turn.notes["mode"] for turn in turns} == {"question"}
    assert len({turn.ah_answer for turn in turns}) == 1
    assert len({tuple(turn.ah_trace_uids) for turn in turns}) == 1
    assert {turn.ah_source for turn in turns} == {"graph"}
    assert (
        eng.agent.store.graph_size(),
        len(eng.agent.store.list_semantic_factors()),
        eng.rag.corpus.strip(),
    ) == before


def test_compare_modes_separate_raw_and_shared_generation() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def chat(self, messages, *, json_mode=False) -> str:
            self.calls.append(messages)
            return "сгенерированный ответ"

    eng = CompareEngine.from_m4_gold(DeepSeekConfig(api_key=""), ticks=4)
    client = RecordingClient()
    eng.rag.client = client
    eng.rebuild_rag = lambda **_: None  # type: ignore[method-assign]

    raw = eng.ask("Какие ресурсы есть на Тиманском кряже?", mode="raw")
    assert client.calls == []
    assert raw.notes["comparison_mode"] == "raw"
    assert raw.rag_source == "extractive_rag"

    generated = eng.ask(
        "Какие ресурсы есть на Тиманском кряже?",
        mode="generated",
    )
    assert len(client.calls) == 2
    assert generated.ah_answer == "сгенерированный ответ"
    assert generated.rag_answer == "сгенерированный ответ"
    assert generated.notes["comparison_mode"] == "generated"
    assert generated.ah_source.endswith("+llm")
    assert generated.rag_source == "faiss+llm"
    assert (
        client.calls[0][0]["content"]
        == client.calls[1][0]["content"]
    )


def test_live_compare_reuses_dialogue_ah_prompt_pipeline() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def chat(self, messages, *, json_mode=False) -> str:
            self.calls.append(messages)
            return "единый ответ"

    eng = CompareEngine.from_m4_gold(
        DeepSeekConfig(api_key=""),
        ticks=4,
    )
    client = RecordingClient()
    dialogue = DialogueAgent(
        eng.agent,
        chat_client=client,
        provider="recording",
    )
    eng.dialogue = dialogue
    question = "Какие ресурсы есть на Тиманском кряже?"

    chat_preview = dialogue.answer_read_only(
        question,
        ticks=4,
        generate=True,
    )
    compared = eng.ask(question, ticks=4, mode="generated")

    assert compared.notes["shared_ah_pipeline"] is True
    assert compared.ah_prompt == chat_preview.system_prompt
    assert compared.ah_prompt.startswith(DIALOGUE_SYSTEM)
    assert compared.rag_prompt.startswith(DIALOGUE_SYSTEM)
    assert AH_CONTEXT_PREAMBLE in compared.ah_prompt
    assert "единственный источник фактов" in compared.rag_prompt
    assert compared.ah_full_trace["question"] == question
    assert compared.ah_perception["kind"] == "question"
    assert compared.ah_answer == "единый ответ"
    assert compared.rag_answer == "единый ответ"


def test_live_rag_summarizes_multi_fact() -> None:
    from ah_memory.config import load_config

    cfg = load_config()
    if not cfg.deepseek.configured:
        return
    eng = CompareEngine.from_m4_gold(cfg.deepseek, ticks=4)
    turn = eng.ask(
        "какие ресурсы есть в тиманском кряже",
        mode="generated",
    )
    rag = turn.rag_answer.lower().replace("ё", "е")
    assert "боксит" in rag or "нефт" in rag or "ресурс" in rag


def test_dialogue_system_forbids_world_knowledge_when_context_present() -> None:
    text = DIALOGUE_SYSTEM.lower()
    assert "опирайся только на него" in text
    assert "не восполняй пробелы общими знаниями" in text
    assert "неизвестн" in text
    assert "не отмалчивайся" not in DIALOGUE_SYSTEM
    assert "единственный источник фактов" in AH_CONTEXT_PREAMBLE.lower()
