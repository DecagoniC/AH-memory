"""Side-by-side: АГ-память vs БЯМ+RAG на живом диалоге/графе (не фиксированный «заяц»)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.config import DeepSeekConfig, load_config
from ah_memory.eval.gold import rabbit_gold
from ah_memory.eval.m4 import GoldItem, M4Report, evaluate_m4
from ah_memory.examples.rabbit import RABBIT_TEXT, build_rabbit_memory
from ah_memory.factor_graph import Factor
from ah_memory.perception import _is_question, _norm
from ah_memory.store import AHStore
from ah_memory.types import Section


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
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def facts_to_corpus(store: AHStore) -> list[str]:
    """Человекочитаемые факты из semantic factors / events → чанки для RAG."""
    lines: list[str] = []
    for factor in store.list_semantic_factors():
        line = _fmt_factor(store, factor)
        if line:
            lines.append(line)
    for event in store.list_events():
        span = (event.raw_span or "").strip()
        if span and span not in lines:
            lines.append(span)
    return lines


def history_to_corpus(history: list[dict[str, str]]) -> list[str]:
    out: list[str] = []
    for m in history:
        c = (m.get("content") or "").strip()
        if c:
            out.append(c)
    return out


def build_live_corpus(
    store: AHStore,
    history: list[dict[str, str]] | None = None,
    *,
    extra: list[str] | None = None,
) -> str:
    parts: list[str] = []
    parts.extend(history_to_corpus(history or []))
    parts.extend(facts_to_corpus(store))
    if extra:
        parts.extend(t.strip() for t in extra if t and t.strip())
    # dedupe keep order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return "\n\n".join(uniq) if uniq else "Корпус пуст."


def _label_of(store: AHStore, uid: str) -> str:
    bare = uid[2:] if uid.startswith("M_") else uid
    try:
        m = store.get_symbol(uid if uid.startswith("M_") else f"M_{bare}")
        for p in m.Pr:
            if p.name == "label" and p.value:
                return p.value
    except Exception:
        pass
    if bare in store.ah.S:
        forms = store.ah.S[bare].R.get("TEXT") or set()
        if forms:
            return next(iter(forms))
    return bare.replace("_", " ").lower()


def _fmt_factor(store: AHStore, factor: Factor) -> str | None:
    pred = (
        factor.relation.canonical_label.upper()
        if factor.relation is not None
        else ""
    )
    if not pred:
        return None
    roles = {role: _label_of(store, uid) for role, uid in factor.roles.items()}
    subj = roles.get("SUBJECT", "?")
    if pred in {"IS", "IS_A"} and "OBJECT" in roles:
        return f"{subj} — {roles['OBJECT']}"
    if pred in {"LIVE_IN", "LIVEIN"} and "LOCATION" in roles:
        return f"{subj} учится/обитает в {roles['LOCATION']}"
    if pred == "HAVE" and "OBJECT" in roles:
        return f"у {subj} есть {roles['OBJECT']}"
    if pred == "CREATE" and "OBJECT" in roles:
        return f"{roles.get('SUBJECT', '?')} создал(и) {roles['OBJECT']}"
    extras = ", ".join(f"{k}: {v}" for k, v in roles.items() if k != "SUBJECT")
    return f"{pred}: {subj}" + (f" — {extras}" if extras else "")


class CompareEngine:
    """Живой агент + RAG по корпусу диалога/фактов (без чужого эталона «заяц»)."""

    def __init__(
        self,
        agent: Agent,
        rag: VanillaRAG | None = None,
        *,
        ticks: int = 6,
        deepseek: DeepSeekConfig | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> None:
        self.agent = agent
        self.ticks = ticks
        self.deepseek = deepseek
        self.history = history if history is not None else []
        self._extra_docs: list[str] = []
        corpus = build_live_corpus(agent.store, self.history)
        ds = deepseek
        self.rag = rag or VanillaRAG(
            corpus,
            top_k=4,
            deepseek=ds if ds and ds.configured else None,
            strict=True,
        )

    @classmethod
    def from_rabbit(cls, deepseek: DeepSeekConfig | None = None, ticks: int = 6) -> CompareEngine:
        """Офлайн M4-бенчмарк на эталоне «заяц» (не для UI-сравнения)."""
        cfg = load_config()
        ds = deepseek if deepseek is not None else cfg.deepseek
        store = build_rabbit_memory()
        agent = Agent(store=store)
        # loose prompt допустим только в gold-бенчмарке
        rag = VanillaRAG(
            RABBIT_TEXT,
            top_k=4,
            deepseek=ds if ds.configured else None,
            strict=False,
        )
        return cls(agent, rag, ticks=ticks, deepseek=ds)

    def bind_history(self, history: list[dict[str, str]]) -> None:
        self.history = history

    def clear(self) -> None:
        self.history = []
        self._extra_docs = []
        self.rebuild_rag()

    def rebuild_rag(self, *, extra: list[str] | None = None) -> None:
        if extra:
            for e in extra:
                e = (e or "").strip()
                if e and e not in self._extra_docs:
                    self._extra_docs.append(e)
        corpus = build_live_corpus(
            self.agent.store,
            self.history,
            extra=self._extra_docs,
        )
        # Живой UI: strict — без параметрических «зайцев»
        self.rag = VanillaRAG(
            corpus,
            top_k=6,
            deepseek=self.deepseek if self.deepseek and self.deepseek.configured else None,
            strict=True,
        )

    def ask(self, text: str, *, ticks: int | None = None) -> CompareTurn:
        """Сравнить произвольный пользовательский запрос на живых данных."""
        t = ticks if ticks is not None else self.ticks
        text = text.strip()
        low = _norm(text)
        is_q = _is_question(text, low)

        if not is_q:
            # факт/сообщение: пишем в АГ и в корпус RAG, затем оба отвечают на вопрос по содержанию
            rep = self.agent.ingest(text, section=Section.H)
            self.rebuild_rag(extra=[text])
            probe = _probe_question(text)
            ah = self.agent.ask(probe, ticks=t)
            fact_bits = _summarize_ingest(self.agent, rep.created_n)
            ah_answer = fact_bits or ah.answer
            if fact_bits and ah.answer and ah.answer != "неизвестно":
                ah_answer = f"{fact_bits}\n=> {ah.answer}"
            vr = self.rag.ask(probe)
            return CompareTurn(
                question=text,
                ah_answer=ah_answer,
                rag_answer=vr.answer,
                ah_trace_uids=list(dict.fromkeys(rep.created_n + ah.trace_uids + rep.seed_uids[:8])),
                rag_chunks=list(vr.chunks),
                rag_scores=list(vr.scores),
                ah_source="graph+ingest",
                rag_source=vr.source,
                notes={
                    "mode": "fact",
                    "probe": probe,
                    "ah_has_trace": bool(ah.trace_uids or rep.created_n),
                    "rag_has_trace": False,
                    "rag_llm": self.rag.client is not None,
                    "corpus_chars": len(self.rag.corpus),
                    "corpus_preview": self.rag.corpus[:280],
                    "ingested_n": len(rep.created_n),
                    "live": True,
                },
            )

        # вопрос: только живой граф + актуальный корпус диалога
        self.rebuild_rag()
        ah = self.agent.ask(text, ticks=t)
        vr = self.rag.ask(text)
        return CompareTurn(
            question=text,
            ah_answer=ah.answer,
            rag_answer=vr.answer,
            ah_trace_uids=list(ah.trace_uids),
            rag_chunks=list(vr.chunks),
            rag_scores=list(vr.scores),
            ah_source=ah.source,
            rag_source=vr.source,
            notes={
                "mode": "question",
                "ah_has_trace": bool(ah.trace_uids),
                "rag_has_trace": False,
                "rag_llm": self.rag.client is not None,
                "corpus_chars": len(self.rag.corpus),
                "corpus_preview": self.rag.corpus[:280],
                "live": True,
            },
        )

    def run_m4(self, gold: list[GoldItem] | None = None) -> M4Report:
        """Для бенчмарка — эталон rabbit (отдельный контур)."""
        bench = CompareEngine.from_rabbit(self.deepseek, ticks=self.ticks)
        return evaluate_m4(bench.agent, bench.rag, gold or rabbit_gold(), ticks=self.ticks)


def _probe_question(fact_text: str) -> str:
    """Превращает утверждение в вопрос для честного сравнения ответов."""
    low = fact_text.lower().replace("ё", "е")
    # Несколько фактов в одной реплике → сводный вопрос (иначе RAG отвечает только на «как зовут»)
    clauses = [c.strip() for c in re.split(r"[.!?;]+", fact_text) if c.strip()]
    hits = sum(
        1
        for key in ("зовут", "имя", "учусь", "универ", "спбпу", "вуз", "сестр", "брат", "живу", "работа")
        if key in low
    )
    if len(clauses) >= 2 or hits >= 2 or len(fact_text) > 60:
        return (
            "Что известно из корпуса о собеседнике? "
            "Перечисли кратко все факты: имя, учёба/работа, родственники и прочее."
        )
    if "зовут" in low or "имя" in low:
        return "Как зовут собеседника?"
    if "учусь" in low or "универ" in low or "спбпу" in low or "вуз" in low:
        return "Где учится собеседник?"
    if "сестр" in low or "брат" in low:
        return "Кто родственники собеседника?"
    return "Что известно из только что сказанного? Перечисли все факты."


def _summarize_ingest(agent: Agent, created: list[str]) -> str:
    if not created:
        return ""
    by_uid = {factor.uid: factor for factor in agent.store.list_semantic_factors()}
    bits: list[str] = []
    for uid in created[:12]:
        factor = by_uid.get(uid)
        if factor is not None:
            line = _fmt_factor(agent.store, factor)
            if line:
                bits.append(line)
                continue
        try:
            m = agent.store.get_symbol(uid if uid.startswith("M_") else f"M_{uid}")
            for p in m.Pr:
                if p.name == "label":
                    bits.append(p.value)
        except Exception:
            continue
    return "записано: " + "; ".join(bits) if bits else f"записано узлов: {len(created)}"
