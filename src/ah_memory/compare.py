"""Side-by-side: АГ-память vs БЯМ+RAG на живом диалоге/графе."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

from ah_memory.agent import Agent
from ah_memory.baselines.rag_embedder import RagEmbedder, resolve_rag_embedder
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.config import DeepSeekConfig, load_config
from ah_memory.eval.gold import closed_world_gold
from ah_memory.eval.m4 import GoldItem, M4Report, evaluate_m4
from ah_memory.examples.closed_world import (
    build_closed_world_memory,
    closed_world_text,
)
from ah_memory.perception import PerceptionResult, content_entity_uids
from ah_memory.store import AHStore

if TYPE_CHECKING:
    from ah_memory.dialogue import DialogueAgent


COMPARE_GENERATION_SYSTEM = """Ты формируешь ответ только по переданному контексту.
Не используй внешние знания. Если контекст не содержит ответа, ответь ровно:
неизвестно. Перечисления передавай полностью. Отвечай по-русски кратко."""


@dataclass
class CompareTurn:
    question: str
    ah_answer: str
    rag_answer: str
    ah_trace_uids: list[str] = field(default_factory=list)
    rag_chunks: list[str] = field(default_factory=list)
    rag_scores: list[float] = field(default_factory=list)
    ah_source: str = "graph"
    rag_source: str = "vanilla_rag"
    ah_prompt: str = ""
    rag_prompt: str = ""
    ah_full_trace: dict[str, Any] = field(default_factory=dict)
    ah_perception: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def history_to_corpus(history: list[dict[str, str]]) -> list[str]:
    out: list[str] = []
    for m in history:
        c = (m.get("content") or "").strip()
        if c:
            out.append(c)
    return out


def build_rag_corpus(
    *,
    source_docs: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    extra: list[str] | None = None,
) -> str:
    """Join RAG-only documents. Never reads AHStore / graph factors."""
    parts: list[str] = []
    if source_docs:
        parts.extend(t.strip() for t in source_docs if t and t.strip())
    parts.extend(history_to_corpus(history or []))
    if extra:
        parts.extend(t.strip() for t in extra if t and t.strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return "\n\n".join(uniq) if uniq else "Корпус пуст."


class CompareEngine:
    """AH graph vs RAG documents. Two arms, no shared memory."""

    def __init__(
        self,
        agent: Agent,
        rag: VanillaRAG | None = None,
        *,
        ticks: int = 6,
        deepseek: DeepSeekConfig | None = None,
        history: list[dict[str, str]] | None = None,
        source_docs: list[str] | None = None,
        dialogue: "DialogueAgent | None" = None,
    ) -> None:
        self.agent = agent
        self.ticks = ticks
        self.deepseek = deepseek
        self.dialogue = dialogue
        self.history = history if history is not None else []
        self._extra_docs: list[str] = []
        self.source_docs: list[str] = [
            t.strip() for t in (source_docs or []) if t and t.strip()
        ]
        ds = deepseek
        if rag is not None:
            self.rag = rag
            self._embedder: RagEmbedder = getattr(rag, "embedder", None) or resolve_rag_embedder()
            if not self.source_docs and rag.corpus.strip() and rag.corpus != "Корпус пуст.":
                self.source_docs = [rag.corpus]
        else:
            self._embedder = resolve_rag_embedder()
            corpus = build_rag_corpus(
                source_docs=self.source_docs,
                history=self.history,
            )
            self.rag = VanillaRAG(
                corpus,
                top_k=4,
                deepseek=ds if ds and ds.configured else None,
                strict=True,
                embedder=self._embedder,
            )

    @classmethod
    def from_m4_gold(
        cls, deepseek: DeepSeekConfig | None = None, ticks: int = 6
    ) -> CompareEngine:
        """Офлайн M4: один исходный текст → граф AH и отдельно чанки RAG."""
        cfg = load_config()
        ds = deepseek if deepseek is not None else cfg.deepseek
        source = closed_world_text()
        store = build_closed_world_memory()
        agent = Agent(store=store)
        rag = VanillaRAG(
            source,
            top_k=4,
            deepseek=ds if ds.configured else None,
            strict=False,
        )
        engine = cls(agent, rag, ticks=ticks, deepseek=ds, source_docs=[source])
        return engine

    from_rabbit = from_m4_gold

    def bind_history(self, history: list[dict[str, str]]) -> None:
        self.history = history

    def set_source_docs(self, texts: list[str]) -> None:
        self.source_docs = [t.strip() for t in texts if t and t.strip()]
        self.rebuild_rag()

    def clear(self) -> None:
        self.history = []
        self._extra_docs = []
        self.source_docs = []
        self.rebuild_rag()

    def rebuild_rag(self, *, extra: list[str] | None = None) -> None:
        if extra:
            for e in extra:
                e = (e or "").strip()
                if e and e not in self._extra_docs:
                    self._extra_docs.append(e)
        corpus = build_rag_corpus(
            source_docs=self.source_docs,
            history=self.history,
            extra=self._extra_docs,
        )
        self.rag = VanillaRAG(
            corpus,
            top_k=6,
            deepseek=self.deepseek if self.deepseek and self.deepseek.configured else None,
            strict=True,
            embedder=self._embedder,
        )

    def ask(
        self,
        text: str,
        *,
        ticks: int | None = None,
        mode: str = "generated",
    ) -> CompareTurn:
        """Сравнить произвольный пользовательский запрос на живых данных."""
        if mode not in {"raw", "generated"}:
            raise ValueError(f"unknown comparison mode: {mode}")
        t = ticks if ticks is not None else self.ticks
        text = text.strip()
        prepared = None
        if self.dialogue is not None:
            prepared = self.dialogue.answer_read_only(
                text,
                ticks=t,
                generate=mode == "generated",
            )
            ah = prepared.ask
        else:
            query_perception = PerceptionResult(
                kind="question",
                candidates=[],
                seed_tokens=content_entity_uids(text)[:8],
                meta={"backend": "compare_query", "interaction": "query"},
            )
            ah = self.agent.ask(
                text,
                ticks=t,
                perception=query_perception,
            )
        self.rebuild_rag()
        vr = self.rag.ask(text, generate=False)
        ah_prompt = ""
        rag_prompt = ""
        if prepared is not None:
            ah_prompt = prepared.system_prompt
            if mode == "raw":
                ah_answer = ah.answer
                rag_answer = vr.answer
                ah_source = ah.source
                rag_source = "extractive_rag"
            else:
                ah_answer = prepared.reply
                rag_context = "\n\n".join(
                    f"[{index}] {chunk}"
                    for index, chunk in enumerate(vr.chunks, start=1)
                )
                rag_answer, rag_prompt = self.dialogue.generate_from_context(
                    text,
                    rag_context,
                    source="БЯМ+RAG",
                )
                generated = self.dialogue.client is not None
                ah_source = (
                    f"{ah.source}+llm" if generated else ah.source
                )
                rag_source = (
                    "faiss+llm" if generated else "extractive_rag"
                )
        else:
            ah_answer, rag_answer, ah_source, rag_source = (
                self._comparison_output(
                    text,
                    ah.answer,
                    vr.answer,
                    vr.chunks,
                    mode,
                    ah_source=ah.source,
                )
            )
        return CompareTurn(
            question=text,
            ah_answer=ah_answer,
            rag_answer=rag_answer,
            ah_trace_uids=list(ah.trace_uids),
            rag_chunks=list(vr.chunks),
            rag_scores=list(vr.scores),
            ah_source=ah_source,
            rag_source=rag_source,
            ah_prompt=ah_prompt,
            rag_prompt=rag_prompt,
            ah_full_trace=dict(ah.full_trace),
            ah_perception=dict(ah.perception),
            notes={
                "mode": "question",
                "comparison_mode": mode,
                "ah_has_trace": bool(ah.trace_uids),
                "rag_has_trace": False,
                "rag_llm": (
                    self.dialogue.client is not None
                    if self.dialogue is not None
                    else self.rag.client is not None
                ),
                "generation_provider": (
                    self.dialogue.provider
                    if self.dialogue is not None
                    else "deepseek"
                ),
                "corpus_chars": len(self.rag.corpus),
                "corpus_preview": self.rag.corpus[:280],
                "live": True,
                "shared_ah_pipeline": prepared is not None,
            },
        )

    def _comparison_output(
        self,
        question: str,
        ah_context: str,
        rag_fallback: str,
        rag_chunks: list[str],
        mode: str,
        *,
        ah_source: str,
    ) -> tuple[str, str, str, str]:
        if mode == "raw" or self.rag.client is None:
            return (
                ah_context,
                rag_fallback,
                ah_source,
                "extractive_rag",
            )
        rag_context = "\n\n".join(
            f"[{index}] {chunk}"
            for index, chunk in enumerate(rag_chunks, start=1)
        )
        return (
            self._generate_from_context(question, ah_context),
            self._generate_from_context(question, rag_context),
            f"{ah_source}+llm",
            "faiss+llm",
        )

    def _generate_from_context(self, question: str, context: str) -> str:
        if not context.strip() or context.strip() == "неизвестно":
            return "неизвестно"
        assert self.rag.client is not None
        return self.rag.client.chat(
            [
                {
                    "role": "system",
                    "content": COMPARE_GENERATION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": (
                        f"Контекст:\n{context}\n\n"
                        f"Вопрос: {question}"
                    ),
                },
            ],
            json_mode=False,
        ).strip()

    def run_m4(self, gold: list[GoldItem] | None = None) -> M4Report:
        """Для бенчмарка — закрытый корпус M4 (отдельный контур)."""
        bench = CompareEngine.from_m4_gold(self.deepseek, ticks=self.ticks)
        return evaluate_m4(
            bench.agent, bench.rag, gold or closed_world_gold(), ticks=self.ticks
        )
