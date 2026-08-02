"""DeepSeek OpenAI-compatible client + HybridPerception."""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ah_memory.config import DeepSeekConfig
from ah_memory.morph import filter_entity_uids, slug_uid
from ah_memory.perception import (
    FactCandidate,
    PerceptionResult,
    RulePerception,
    _finalize_candidates,
    _is_question,
    _norm,
)

SYSTEM_PROMPT = """Ты модуль восприятия АГ-памяти.
Извлеки только явные утверждения. Не добавляй сущности вне текста.
JSON:
{
  "kind": "fact" | "question" | "message",
  "candidates": [
    {
      "predicate": "IS|LIVE_IN|HAVE|RUN|BE_COLORED|CREATE|CAUSE_EVENT|USE|MOVE",
      "roles": {"SUBJECT":"UID", "OBJECT":"UID", ...},
      "raw_span": "фрагмент",
      "confidence": 0.0-1.0
    }
  ],
  "seed_tokens": ["UID", ...]
}
UID — лемма в UPPER_SNAKE (им. падеж), кириллица ок: СКУЛЬПТУРНАЯ_ЛЕПКА, МОСКВА, ДУШКИН.
SUBJECT/OBJECT/LOCATION — только сущности: существительные, имена, названия. НЕ глаголы (занимаюсь), НЕ союзы (если, когда, чтобы), НЕ вводные.
Один факт = одна микротема: заполняй ВСЕ уместные роли сразу (SUBJECT+OBJECT+LOCATION+TIME+…), не дроби на пары, если это одно утверждение.
«Я занимаюсь лепкой» → USE(SUBJECT=Я, OBJECT=СКУЛЬПТУРНАЯ_ЛЕПКА) или HAVE, не IS(ЗАНИМАЮСЬ,…).
«Работаю с Душкиным в МИФИ» → USE/LIVE_IN с SUBJECT, OBJECT/LOCATION в одном candidate.
«Если понадобится…» — не факт (candidates=[]).
Роли: SUBJECT OBJECT LOCATION TIME CAUSE TOOL MATERIAL PURPOSE HOW-TO.
Верни только JSON."""


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig) -> None:
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if json_mode else min(0.7, self.cfg.temperature + 0.4),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self.cfg.timeout_sec) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


class DeepSeekPerception:
    def __init__(self, cfg: DeepSeekConfig) -> None:
        self.client = DeepSeekClient(cfg)

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        user = json.dumps(
            {"text": text, "wm_context": wm_context or []},
            ensure_ascii=False,
        )
        raw = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        data = _parse_json(raw)
        cands = [
            FactCandidate(
                predicate=str(c.get("predicate", "")).upper(),
                roles={str(k).upper(): slug_uid(str(v)) for k, v in (c.get("roles") or {}).items()},
                raw_span=c.get("raw_span"),
                confidence=float(c.get("confidence", 0.8)),
            )
            for c in data.get("candidates", [])
            if c.get("predicate")
        ]
        cands = _finalize_candidates(cands)
        kind = data.get("kind", "fact")
        if kind not in {"fact", "question", "message"}:
            kind = "fact"
        low = _norm(text.strip())
        if _is_question(text.strip(), low):
            kind = "question"
        seeds = filter_entity_uids(
            [slug_uid(str(s)) for s in data.get("seed_tokens", [])],
            allow_pronoun=True,
        )
        if not seeds:
            seeds = RulePerception()._content_tokens(low)
        return PerceptionResult(
            kind=kind, candidates=cands, seed_tokens=seeds, meta={"backend": "deepseek"}
        )


class HybridPerception:
    """DeepSeek first; merge RulePerception if LLM missed factors."""

    def __init__(self, cfg: DeepSeekConfig, fallback: bool = True) -> None:
        self.cfg = cfg
        self.fallback = fallback
        self.rules = RulePerception()
        self.llm = DeepSeekPerception(cfg) if cfg.configured else None

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        rules = self.rules.parse(text, wm_context)
        if self.llm is None:
            return rules
        try:
            llm = self.llm.parse(text, wm_context)
        except Exception as exc:  # noqa: BLE001
            if not self.fallback:
                raise
            return PerceptionResult(
                kind=rules.kind,
                candidates=rules.candidates,
                seed_tokens=rules.seed_tokens,
                meta={**rules.meta, "backend": "rules", "llm_error": str(exc)},
            )

        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        merged: list[FactCandidate] = []
        for c in _finalize_candidates(list(llm.candidates) + list(rules.candidates)):
            key = (c.predicate, tuple(sorted(c.roles.items())))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
        seeds = filter_entity_uids(
            list(llm.seed_tokens) + list(rules.seed_tokens),
            allow_pronoun=True,
        )
        return PerceptionResult(
            kind=llm.kind if llm.kind != "message" else rules.kind,
            candidates=merged,
            seed_tokens=seeds,
            meta={"backend": "hybrid"},
        )


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))
