"""DeepSeek OpenAI-compatible client + HybridPerception."""
from __future__ import annotations

from typing import Any

import httpx

from ah_memory.config import DeepSeekConfig
from ah_memory.gigachat_llm import _parse_json
from ah_memory.morph import seeds_from_roles, slug_uid
from ah_memory.perception import (
    PerceptionResult,
    SeedPerception,
    _is_question,
    _norm,
    candidates_from_llm_json,
    content_entity_uids,
    gate_candidates,
    llm_payload_errors,
)
from ah_memory.perception_prompt import SYSTEM_PROMPT, build_user_payload


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig) -> None:
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self.cfg.timeout_sec) as client:
            r = client.post(
                self.cfg.base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if r.status_code >= 400:
                detail = (r.text or r.reason_phrase or "")[:500]
                raise httpx.HTTPStatusError(
                    f"DeepSeek chat failed ({r.status_code}): {detail}",
                    request=r.request,
                    response=r,
                )
            data = r.json()
        return data["choices"][0]["message"]["content"]


class DeepSeekPerception:
    def __init__(self, cfg: DeepSeekConfig, *, require_grounding: bool = True) -> None:
        self.client = DeepSeekClient(cfg)
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
        cands = candidates_from_llm_json(data)
        gated_result = gate_candidates(
            text,
            cands,
            require_grounding=self.require_grounding,
            allow_open_relations=True,
            report=True,
        )
        gated, gate_report = gated_result
        validation_errors = llm_payload_errors(data)
        validation_errors.extend(
            f"candidate rejected: {item.get('reason', 'validation')}"
            for item in gate_report.get("dropped", [])
        )
        kind = data.get("kind", "fact")
        if kind not in {"fact", "question", "message"}:
            kind = "fact"
        low = _norm(text.strip())
        if _is_question(text.strip(), low):
            kind = "question"
        elif gated:
            kind = "fact"
        seeds = seeds_from_roles(
            gated,
            extra=[slug_uid(str(s)) for s in data.get("seed_tokens", [])][:12],
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
                "backend": "deepseek",
                "llm_raw": data,
                "gate_report": gate_report,
                "validation_errors": validation_errors,
                "system_prompt": SYSTEM_PROMPT,
            },
        )


class DeepSeekHybridPerception:
    """LLM → morph gate; offline / error → SeedPerception."""

    def __init__(self, cfg: DeepSeekConfig, fallback: bool = True) -> None:
        self.cfg = cfg
        self.fallback = fallback
        self.seeds = SeedPerception()
        self.llm = DeepSeekPerception(cfg) if cfg.configured else None

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
            },
        )
