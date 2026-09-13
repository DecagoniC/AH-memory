"""Provider-neutral Ollama HTTP client and perception adapter."""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from ah_memory.config import OllamaConfig
from ah_memory.gigachat_llm import _parse_json
from ah_memory.morph import seeds_from_roles, slug_uid
from ah_memory.perception import (
    PerceptionResult,
    SeedPerception,
    candidates_from_llm_json,
    classify_utterance,
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
        options: dict[str, Any] = {"temperature": self.cfg.temperature}
        if self.cfg.num_ctx is not None:
            options["num_ctx"] = int(self.cfg.num_ctx)
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
            # Qwen3 (and similar) otherwise spend the budget on CoT before JSON.
            payload["think"] = False
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        with self._http() as client:
            response = client.post(url, json=payload)
            self._raise_for_status(response, "chat")
            data = response.json()
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama chat response has no message content") from exc

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> Iterator[str]:
        """Yield assistant text deltas from Ollama native streaming chat."""
        options: dict[str, Any] = {"temperature": self.cfg.temperature}
        if self.cfg.num_ctx is not None:
            options["num_ctx"] = int(self.cfg.num_ctx)
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
            payload["think"] = False
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        with self._http() as client:
            with client.stream("POST", url, json=payload) as response:
                self._raise_for_status(response, "chat")
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Ollama stream returned invalid JSON: {line[:200]!r}"
                        ) from exc
                    message = data.get("message")
                    if isinstance(message, dict):
                        piece = message.get("content")
                        if piece:
                            yield str(piece)
                    if data.get("done"):
                        break

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


def list_ollama_models(
    cfg: OllamaConfig | None = None,
    *,
    timeout_sec: float = 3.0,
    transport: httpx.BaseTransport | None = None,
) -> list[dict[str, Any]]:
    """Return locally available Ollama models from GET /api/tags."""

    config = cfg or OllamaConfig()
    with httpx.Client(timeout=timeout_sec, transport=transport) as client:
        response = client.get(config.base_url.rstrip("/") + "/api/tags")
    if response.status_code >= 400:
        detail = (response.text or response.reason_phrase or "")[:500]
        raise httpx.HTTPStatusError(
            f"Ollama tags failed ({response.status_code}): {detail}",
            request=response.request,
            response=response,
        )
    data = response.json()
    models = data.get("models")
    if not isinstance(models, list):
        return []
    out: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        size = item.get("size")
        out.append(
            {
                "name": name,
                "size": int(size) if isinstance(size, (int, float)) else None,
                "digest": str(item.get("digest") or "") or None,
                "modified_at": str(item.get("modified_at") or "") or None,
            }
        )
    out.sort(key=lambda row: row["name"].lower())
    return out


def is_likely_embedding_model(name: str) -> bool:
    low = name.strip().lower()
    return "embed" in low


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
        kind = classify_utterance(
            text,
            declared_kind=str(data.get("kind") or ""),
            candidates=gated,
            query=data.get("query"),
        )
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


class OllamaHybridPerception:
    """LLM → morph gate; offline / error → SeedPerception."""

    def __init__(
        self,
        cfg: OllamaConfig | None = None,
        fallback: bool = True,
        *,
        client: OllamaClient | None = None,
    ) -> None:
        self.cfg = cfg or OllamaConfig()
        self.fallback = fallback
        self.seeds = SeedPerception()
        if client is not None:
            self.llm = OllamaPerception(self.cfg, client=client)
        elif self.cfg.configured:
            self.llm = OllamaPerception(self.cfg)
        else:
            self.llm = None

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        offline = self.seeds.parse(text, wm_context)
        if self.llm is None:
            return offline
        try:
            llm = self.llm.parse(text, wm_context)
        except Exception as exc:  # noqa: BLE001
            if not self.fallback:
                raise
            return PerceptionResult(
                kind=offline.kind,
                candidates=[],
                seed_tokens=offline.seed_tokens,
                meta={**offline.meta, "backend": "seeds", "llm_error": str(exc)},
            )

        seeds = seeds_from_roles(llm.candidates, extra=list(llm.seed_tokens)[:8])
        if not seeds:
            seeds = list(offline.seed_tokens)[:8]
        return PerceptionResult(
            kind=llm.kind,
            candidates=list(llm.candidates),
            seed_tokens=seeds,
            meta={
                "backend": "hybrid-llm" if llm.candidates else "hybrid-seeds",
                "llm_raw": llm.meta.get("llm_raw"),
                "system_prompt": llm.meta.get("system_prompt", SYSTEM_PROMPT),
                "llm_candidates": len(llm.candidates),
                "validation_errors": list(llm.meta.get("validation_errors") or []),
            },
        )
