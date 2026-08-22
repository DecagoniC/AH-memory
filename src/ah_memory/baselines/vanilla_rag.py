"""Vanilla RAG: chunk corpus → FAISS dense retrieve → LLM/extractive answer.

No actant roles, no AH hypergraph, no ignition. Same corpus as AH agent.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ah_memory.baselines.rag_embedder import RagEmbedder, resolve_rag_embedder
from ah_memory.baselines.vector_store import FaissVectorStore
from ah_memory.config import DeepSeekConfig
from ah_memory.deepseek import DeepSeekClient

RagGenerator = Callable[[str, list[str]], str]

_TOKEN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", re.UNICODE)

RAG_SYSTEM = """Ты — классический RAG-ассистент поверх векторного поиска.
Используй фрагменты корпуса как основной источник. Если фрагментов недостаточно —
можно опираться на общие знания, но помечай это явно.
Отвечай по-русски кратко."""

RAG_SYSTEM_STRICT = """Ты отвечаешь ТОЛЬКО по приведённым фрагментам корпуса.
Если во фрагментах нет ответа — ответь ровно: неизвестно
Если вопрос просит перечислить факты — перечисли ВСЕ факты из фрагментов, кратко, по пунктам.
Запрещено выдумывать факты. Не используй внешние знания. Отвечай по-русски кратко."""


@dataclass
class VanillaRAGReply:
    answer: str
    chunks: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    source: str = "vanilla_rag"
    trace_uids: list[str] = field(default_factory=list)  # always empty by design


class VanillaRAG:
    def __init__(
        self,
        corpus: str,
        *,
        chunk_size: int = 280,
        chunk_overlap: int = 40,
        top_k: int = 8,
        deepseek: DeepSeekConfig | None = None,
        strict: bool = False,
        generator: RagGenerator | None = None,
        embedder: RagEmbedder | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self.corpus = corpus
        self.chunks = _chunk_text(corpus, chunk_size=chunk_size, overlap=chunk_overlap)
        self.top_k = top_k
        self.embedder = embedder or resolve_rag_embedder()
        self.store = FaissVectorStore(persist_path)
        if self.chunks:
            vectors = self.embedder.embed_many(self.chunks)
            self.store.replace(self.chunks, vectors)
        self.generator = generator
        self.client = (
            None
            if generator is not None
            else (DeepSeekClient(deepseek) if deepseek and deepseek.configured else None)
        )
        answerer = (
            "scripted" if generator is not None else ("llm" if self.client is not None else "extractive")
        )
        self.backend = f"{answerer}+faiss:{self.embedder.name}"
        self.system_prompt = RAG_SYSTEM_STRICT if strict else RAG_SYSTEM

    def ask(
        self,
        question: str,
        *,
        generate: bool = True,
    ) -> VanillaRAGReply:
        if not self.chunks:
            return VanillaRAGReply(
                answer="неизвестно",
                chunks=[],
                scores=[],
                source="empty_retrieve",
            )
        qv = self.embedder.embed_many([question])[0]
        k = self.top_k
        qlow = question.lower()
        if len(self.chunks) <= 12 or any(
            w in qlow for w in ("все факт", "перечисл", "что известно", "расскажи", "что ты знаешь")
        ):
            k = max(k, len(self.chunks))
        picked = self.store.query(qv, top_k=max(k, 8), min_score=-1.0)
        if not picked:
            return VanillaRAGReply(
                answer="неизвестно",
                chunks=[],
                scores=[],
                source="empty_retrieve",
            )
        texts = [c for c, _ in picked[: max(k, 8)]]
        scores = [s for _, s in picked[: max(k, 8)]]
        if not generate:
            answer = _extractive_answer(question, texts)
            source = "extractive_rag"
        elif self.generator is not None:
            answer = self.generator(question, texts).strip()
            source = "scripted_rag"
        elif self.client is not None:
            ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
            answer = self.client.chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Фрагменты:\n{ctx}\n\nВопрос: {question}",
                    },
                ],
                json_mode=False,
            ).strip()
            source = "llm_rag"
        else:
            answer = _extractive_answer(question, texts)
            source = "extractive_rag"
        return VanillaRAGReply(answer=answer, chunks=texts, scores=scores, source=source)


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Реплики/абзацы (\\n\\n) — атомарные чанки; длинные режем по словам."""
    text = (text or "").strip()
    if not text or text == "Корпус пуст.":
        return []
    docs = [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]
    out: list[str] = []
    for doc in docs:
        words = doc.split()
        if len(words) <= chunk_size:
            out.append(doc)
            continue
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + chunk_size]).strip()
            if piece:
                out.append(piece)
            if i + chunk_size >= len(words):
                break
    return out


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _extractive_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return "неизвестно"
    q = set(_tokenize(question))
    content_q = {t for t in q if len(t) > 3} - {
        "сколько",
        "какого",
        "какая",
        "какие",
        "почему",
        "такой",
        "такое",
        "весит",
    }
    blob = " ".join(_tokenize(" ".join(chunks)))
    if content_q and sum(1 for t in content_q if t in blob) < max(1, len(content_q) // 3):
        return "неизвестно"
    best = chunks[0]
    best_s = -1
    for ch in chunks:
        s = len(q & set(_tokenize(ch)))
        if s > best_s:
            best_s = s
            best = ch
    if best_s <= 0:
        return "неизвестно"
    parts = re.split(r"(?<=(?<![A-ZА-ЯЁ])[.!?])\s+", best)
    return parts[0] if parts else best
