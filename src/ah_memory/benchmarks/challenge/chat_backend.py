"""Structured chat adapter used by the graph-free RAG benchmark arm."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Sequence

from ah_memory.benchmarks.challenge.protocols import (
    BenchmarkQuery,
    SourceDocument,
    StructuredAnswer,
)


RAG_SYSTEM_PROMPT = """Answer only from the supplied source documents.
Return exactly one JSON object:
{"answer": "short answer", "support_ids": ["document UID"]}
Use only document UIDs present in the input. If the answer is unsupported, return
an empty answer and an empty support_ids list. Do not add explanations."""


class JsonChatAnswerBackend:
    def __init__(
        self,
        chat: Callable[[list[dict[str, str]]], str],
    ) -> None:
        self._chat = chat

    def answer(
        self,
        query: BenchmarkQuery,
        documents: Sequence[SourceDocument],
    ) -> StructuredAnswer:
        payload = {
            "question": query.text,
            "documents": [
                {"uid": str(document.document_id), "text": document.text}
                for document in documents
            ],
        }
        raw = self._chat(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ]
        )
        data = _json_object(raw)
        answer = str(data.get("answer") or "").strip()
        support_raw = data.get("support_ids") or []
        if not isinstance(support_raw, list):
            raise ValueError("support_ids must be a list")
        allowed = {str(document.document_id) for document in documents}
        support_ids = tuple(str(uid) for uid in support_raw)
        if any(uid not in allowed for uid in support_ids):
            raise ValueError("answer cites a document that was not retrieved")
        return StructuredAnswer(
            query_id=query.query_id,
            text=answer,
            support_ids=support_ids,
        )


def _json_object(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("chat backend must return one JSON object")
    return data
