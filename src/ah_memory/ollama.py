"""Provider-neutral Ollama HTTP client and perception adapter."""
from __future__ import annotations

from typing import Any

import httpx

from ah_memory.config import OllamaConfig
from ah_memory.gigachat_llm import _parse_json
from ah_memory.morph import seeds_from_roles, slug_uid
from ah_memory.perception import (
    PerceptionResult,
    _is_question,
    _norm,
    candidates_from_llm_json,
    content_entity_uids,
    gate_candidates,
    llm_payload_errors,
)
from ah_memory.perception_prompt import SYSTEM_PROMPT, build_user_payload


class OllamaClient:
    """Small client for Ollama's native chat and embedding endpoints."""

    def __init__(
        self,
        cfg: OllamaConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cfg = cfg or OllamaConfig()
        self._transport = transport

    def _http(self, *, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            timeout=self.cfg.timeout_sec if timeout is None else timeout,
            transport=self._transport,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if response.status_code < 400:
            return
        detail = (response.text or response.reason_phrase or "")[:500]
        raise httpx.HTTPStatusError(
            f"Ollama {operation} failed ({response.status_code}): {detail}",
            request=response.request,
            response=response,
        )

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        if json_mode:
            payload["format"] = "json"
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        with self._http() as client:
            response = client.post(url, json=payload)
            self._raise_for_status(response, "chat")
            data = response.json()
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama chat response has no message content") from exc

    def embeddings(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": model or self.cfg.embedding_model,
            "input": texts,
        }
        url = self.cfg.base_url.rstrip("/") + "/api/embed"
        with self._http() as client:
            response = client.post(url, json=payload)
            self._raise_for_status(response, "embeddings")
            data = response.json()
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            size = len(vectors) if isinstance(vectors, list) else 0
            raise RuntimeError(
                f"Ollama embeddings size mismatch: got {size} for {len(texts)} inputs"
            )
        try:
            return [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ollama embeddings response contains invalid vectors") from exc


def is_ollama_available(
    cfg: OllamaConfig | None = None,
    *,
    timeout_sec: float = 1.0,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Return whether the configured Ollama server responds to its version endpoint."""

    config = cfg or OllamaConfig()
    try:
        with httpx.Client(timeout=timeout_sec, transport=transport) as client:
            response = client.get(config.base_url.rstrip("/") + "/api/version")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


class OllamaPerception:
    """Perception backend using one Ollama chat request per parse."""

    def __init__(
        self,
        cfg: OllamaConfig | None = None,
        *,
        require_grounding: bool = True,
        client: OllamaClient | None = None,
    ) -> None:
        self.client = client or OllamaClient(cfg)
        self.require_grounding = require_grounding

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        user = build_user_payload(text, wm_context)
        raw = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        data = _parse_json(raw)
        candidates = candidates_from_llm_json(data)
        gated, gate_report = gate_candidates(
            text,
            candidates,
            require_grounding=self.require_grounding,
            allow_open_relations=True,
            report=True,
        )
        validation_errors = llm_payload_errors(data)
        validation_errors.extend(
            f"candidate rejected: {item.get('reason', 'validation')}"
            for item in gate_report.get("dropped", [])
        )
        kind = data.get("kind", "fact")
        if kind not in {"fact", "question", "message"}:
            kind = "fact"
        normalized = _norm(text.strip())
        if _is_question(text.strip(), normalized):
            kind = "question"
        elif gated:
            kind = "fact"
        seeds = seeds_from_roles(
            gated,
            extra=[slug_uid(str(seed)) for seed in data.get("seed_tokens", [])][:12],
        )
        if not seeds:
            seeds = content_entity_uids(text)[:8]
        if kind != "question" and not gated:
            kind = "message"
        return PerceptionResult(
            kind=kind,
            candidates=gated,
            seed_tokens=seeds,
            meta={
                "backend": "ollama",
                "llm_raw": data,
                "gate_report": gate_report,
                "validation_errors": validation_errors,
                "system_prompt": SYSTEM_PROMPT,
            },
        )
