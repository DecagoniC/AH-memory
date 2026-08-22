from __future__ import annotations

from ah_memory.baselines.rag_embedder import DeterministicRagEmbedder
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.baselines.vector_store import FaissVectorStore


def test_faiss_index_ranks_related_chunk_first() -> None:
    embedder = DeterministicRagEmbedder(dimensions=64)
    store = FaissVectorStore()
    chunks = [
        "Тиманский кряж — возвышенность на северо-востоке равнины.",
        "Отдельная заметка про погоду и транспортное сообщение.",
    ]
    store.replace(chunks, embedder.embed_many(chunks))
    hits = store.query(embedder.embed_many(["Где расположен Тиманский кряж?"])[0], top_k=2)
    assert hits
    assert "тиманский" in hits[0][0].lower()
    assert hits[0][1] >= hits[-1][1]


def test_faiss_store_roundtrip(tmp_path) -> None:
    embedder = DeterministicRagEmbedder(dimensions=32)
    texts = ["первый фрагмент про кряж", "второй фрагмент про реку"]
    path = tmp_path / "index.faiss"
    original = FaissVectorStore(path)
    original.replace(texts, embedder.embed_many(texts))
    loaded = FaissVectorStore()
    loaded.load(path)
    assert loaded.count() == 2
    assert loaded.texts() == texts
    hits = loaded.query(embedder.embed_many(["кряж"])[0], top_k=1)
    assert hits and "кряж" in hits[0][0]


def test_vanilla_rag_uses_faiss_backend() -> None:
    rag = VanillaRAG(
        "Тиманский кряж тянется на 900 километров.\n\nВысшая точка — Четласский Камень.",
        top_k=2,
        embedder=DeterministicRagEmbedder(),
    )
    assert rag.backend.startswith("extractive+faiss:")
    reply = rag.ask("Какая высшая точка Тиманского кряжа?")
    assert reply.trace_uids == []
    assert reply.chunks
    assert reply.scores[0] > 0
