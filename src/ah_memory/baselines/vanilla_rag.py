"""Vanilla vector RAG baseline (M4): chunk corpus → TF-IDF retrieve → LLM/extractive answer.

No actant roles, no AH hypergraph, no ignition. Same corpus as AH agent.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from ah_memory.config import DeepSeekConfig
from ah_memory.deepseek import DeepSeekClient

_TOKEN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", re.UNICODE)

RAG_SYSTEM = """Ты — классический RAG-ассистент поверх векторного поиска.
Используй фрагменты корпуса как основной источник. Если фрагментов недостаточно —
можно опираться на общие знания, но помечай это явно.
Отвечай по-русски кратко."""

# Живой UI-compare: только фрагменты, без «общих знаний» и без чужих тем
RAG_SYSTEM_STRICT = """Ты отвечаешь ТОЛЬКО по приведённым фрагментам корпуса.
Если во фрагментах нет ответа — ответь ровно: неизвестно
Если вопрос просит перечислить факты — перечисли ВСЕ факты из фрагментов, кратко, по пунктам.
Запрещено: выдумывать факты; упоминать зайца/животных/энциклопедию, если их нет во фрагментах.
Не используй внешние знания. Отвечай по-русски кратко."""


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
    ) -> None:
        self.corpus = corpus
        self.chunks = _chunk_text(corpus, chunk_size=chunk_size, overlap=chunk_overlap)
        self.top_k = top_k
        self._df = _doc_freq(self.chunks)
        self._n = max(1, len(self.chunks))
        self._vecs = [_tfidf(ch, self._df, self._n) for ch in self.chunks]
        self.client = DeepSeekClient(deepseek) if deepseek and deepseek.configured else None
        self.backend = "llm+tfidf" if self.client else "extractive+tfidf"
        self.system_prompt = RAG_SYSTEM_STRICT if strict else RAG_SYSTEM

    def ask(self, question: str) -> VanillaRAGReply:
        qv = _tfidf(question, self._df, self._n)
        ranked = sorted(
            ((_cosine(qv, v), i) for i, v in enumerate(self._vecs)),
            reverse=True,
        )
        # Малый корпус / сводный вопрос → отдаём все релевантные чанки
        k = self.top_k
        qlow = question.lower()
        if len(self.chunks) <= 12 or any(
            w in qlow for w in ("все факт", "перечисл", "что известно", "расскажи", "что ты знаешь")
        ):
            k = max(k, len(self.chunks))
        ranked = ranked[:k]
        picked = [(self.chunks[i], s) for s, i in ranked if s > 0]
        # Имя/сущность из вопроса — подтянуть чанки с прямым вхождением
        q_ents = [t for t in _tokenize(question) if len(t) >= 3]
        for i, ch in enumerate(self.chunks):
            ch_l = ch.lower()
            if any(e in ch_l for e in q_ents) and all(ch != p for p, _ in picked):
                picked.append((ch, 0.01))
        if not picked:
            return VanillaRAGReply(
                answer="неизвестно",
                chunks=[],
                scores=[],
                source="empty_retrieve",
            )
        # стабильный порядок: по score убыв.
        picked.sort(key=lambda x: x[1], reverse=True)
        texts = [c for c, _ in picked[: max(k, 8)]]
        scores = [s for _, s in picked[: max(k, 8)]]
        if self.client is not None:
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


def _doc_freq(chunks: list[str]) -> Counter[str]:
    df: Counter[str] = Counter()
    for ch in chunks:
        df.update(set(_tokenize(ch)))
    return df


def _tfidf(text: str, df: Counter[str], n_docs: int) -> dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    tf = Counter(toks)
    L = len(toks)
    vec: dict[str, float] = {}
    for term, c in tf.items():
        idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1.0
        vec[term] = (c / L) * idf
    return vec


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _extractive_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return "неизвестно"
    q = set(_tokenize(question))
    # abstain on out-of-corpus traps with no lexical overlap beyond stopwords
    content_q = {t for t in q if len(t) > 3} - {
        "сколько", "какого", "какая", "какие", "почему", "такой", "такое", "весит", "король"
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
    parts = re.split(r"(?<=[.!?])\s+", best)
    return parts[0] if parts else best
