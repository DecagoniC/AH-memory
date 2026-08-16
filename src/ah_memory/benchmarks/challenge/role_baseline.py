"""Ungated LLM role parser used as the graph-free M5 baseline."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ah_memory.morph import seeds_from_roles
from ah_memory.perception import (
    PerceptionResult,
    candidates_from_llm_json,
    llm_payload_errors,
)
from ah_memory.perception_prompt import SYSTEM_PROMPT, build_user_payload


class UngatedLLMPerception:
    """Use the same extraction request without AH grounding or graph validation."""

    def __init__(
        self,
        chat: Callable[[list[dict[str, str]]], str],
        *,
        backend: str,
    ) -> None:
        self._chat = chat
        self._backend = backend

    def parse(
        self,
        text: str,
        wm_context: list[str] | None = None,
    ) -> PerceptionResult:
        raw = self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_payload(text, wm_context),
                },
            ]
        )
        data = _json_object(raw)
        candidates = candidates_from_llm_json(data)
        kind = str(data.get("kind") or "message")
        if kind not in {"fact", "question", "message"}:
            kind = "message"
        return PerceptionResult(
            kind=kind,  # type: ignore[arg-type]
            candidates=candidates,
            seed_tokens=seeds_from_roles(candidates),
            meta={
                "backend": self._backend,
                "llm_raw": data,
                "validation_errors": llm_payload_errors(data),
                "ungated": True,
            },
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
        raise ValueError("role baseline must return one JSON object")
    return data
