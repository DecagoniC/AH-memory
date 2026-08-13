"""Perception: текст → FactCandidate / seed_tokens.

Слой между NL и Transform. LLM (или offline seeds) предлагает факты;
gate_candidates отбрасывает мусор (роли не из текста, слишком широкие UID…).
Читать после types/store; дальше — transform.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ah_memory.morph import (
    STOP,
    filter_entity_uids,
    is_entity_token,
    lemma,
    sanitize_roles,
    seeds_from_roles,
    slug_uid,
    uid_too_wide,
)


# ── Выход perception ─────────────────────────────────────────────────────────
# Зачем: единый контракт для Transform. predicate — legacy-имя; raw/canonical —
# для open relations. seed_tokens — «о чём речь», даже если факта нет.


@dataclass(frozen=True)
class FactCandidate:
    predicate: str
    roles: dict[str, str]
    raw_span: str | None = None
    confidence: float = 1.0
    raw_relation: str | None = None
    canonical_relation: str | None = None


@dataclass(frozen=True)
class PerceptionResult:
    kind: Literal["fact", "question", "message"]
    candidates: list[FactCandidate]
    seed_tokens: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_graph_json(self) -> dict[str, Any]:
        """Нормализованный JSON для Transform / UI."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "candidates": [
                {
                    "predicate": c.predicate,
                    "raw_relation": c.raw_relation or c.predicate,
                    "canonical_relation": c.canonical_relation or c.predicate,
                    "roles": dict(c.roles),
                    "raw_span": c.raw_span,
                    "confidence": c.confidence,
                }
                for c in self.candidates
            ],
            "seed_tokens": list(self.seed_tokens),
            "meta": {k: v for k, v in self.meta.items() if k != "llm_raw"},
        }
        if "llm_raw" in self.meta:
            out["llm_raw"] = self.meta["llm_raw"]
        return out


class PerceptionBackend(Protocol):
    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult: ...


# ── Reserved labels (не entity UID) ──────────────────────────────────────────
# Не whitelist ingest: open relations принимает любой predicate.
# Нужны лишь чтобы seed/токен «IS»/«HAVE» не становились сущностями.

PREDICATES = {
    "CREATE",
    "IS",
    "IS_A",
    "LIVE_IN",
    "LIVES_IN",
    "LOCATED_IN",
    "BE_BORN",
    "HAVE",
    "RUN",
    "BE_COLORED",
    "CAUSE_EVENT",
    "CAUSE",
    "USE",
    "MOVE",
    "PURCHASE",
    "SELL",
    "OWNS",
    "WORKS_FOR",
    "PART_OF",
    "FOLLOW",
    "BEFORE",
    "AFTER",
    "DURING",
    "RELATED_TO",
    "ASSOC",
    "BIND",
}

# ── Текстовые хелперы / grounding ────────────────────────────────────────────
# Зачем: не пускать в граф сущности, которых нет в исходной фразе (галлюцинации LLM).

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
_SHORT_KEEP = frozenset({"я", "мы", "ты", "он", "а"})


def _norm(text: str) -> str:
    t = text.lower().replace("ё", "е")
    return re.sub(r"[—–−]", "-", t)


def _is_question(raw: str, low: str) -> bool:
    if raw.endswith("?"):
        return True
    return low.startswith(
        ("кто ", "что ", "где ", "когда ", "почему ", "как ", "зачем ",
         "what ", "when ", "where ", "why ", "how ")
    )


def _collect_text_lemmas(text: str) -> set[str]:
    low = _norm(text)
    lemmas: set[str] = set()
    for w in _TOKEN_RE.findall(low):
        if len(w) < 2 and w not in _SHORT_KEEP:
            continue
        lem = lemma(w)
        if lem:
            lemmas.add(lem.replace("ё", "е"))
    return lemmas


def _uid_grounded(uid: str, text_lemmas: set[str], raw_tokens: set[str]) -> bool:
    bare = uid[2:] if uid.startswith("M_") else uid
    if bare.isdigit():
        return bare in raw_tokens
    parts = [p.lower().replace("ё", "е") for p in bare.split("_") if p]
    if not parts:
        return False
    for p in parts:
        if p.isdigit():
            if p not in raw_tokens:
                return False
            continue
        pl = lemma(p).replace("ё", "е")
        if p not in text_lemmas and pl not in text_lemmas:
            return False
    return True


def _roles_grounded(roles: dict[str, str], text_lemmas: set[str], raw: str) -> bool:
    low = _norm(raw)
    raw_tokens = set(_TOKEN_RE.findall(low))
    for val in roles.values():
        bare = val[2:] if str(val).startswith("M_") else str(val)
        if not _uid_grounded(bare, text_lemmas, raw_tokens):
            return False
    return True


def _normalize_place_roles(pred: str, roles: dict[str, str]) -> dict[str, str]:
    """LIVE_IN / BE_BORN: place goes to LOCATION, not OBJECT."""
    if pred not in {"LIVE_IN", "BE_BORN"}:
        return roles
    if "LOCATION" not in roles and "OBJECT" in roles:
        roles = dict(roles)
        roles["LOCATION"] = roles.pop("OBJECT")
    return roles


def _finalize_candidates(cands: list[FactCandidate]) -> list[FactCandidate]:
    # Зачем: sanitize roles, place-heuristic, dedup. Canonical не схлопываем.
    out: list[FactCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for c in cands:
        raw_relation = (c.raw_relation or c.predicate).strip()
        pred = (c.canonical_relation or c.predicate).strip().upper()
        if not pred:
            continue
        roles = sanitize_roles(dict(c.roles))
        if roles is None:
            continue
        roles = _normalize_place_roles(pred, roles)
        if any(uid_too_wide(v) for v in roles.values()):
            continue
        if pred in {"LIVE_IN", "LIVES_IN", "BE_BORN", "LOCATED_IN"} and "LOCATION" not in roles:
            continue
        key = (pred, tuple(sorted(roles.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            FactCandidate(
                pred,
                roles,
                c.raw_span,
                c.confidence,
                raw_relation=raw_relation,
                canonical_relation=pred,
            )
        )
    return out


def gate_candidates(
    text: str,
    cands: list[FactCandidate],
    *,
    min_confidence: float = 0.5,
    require_grounding: bool = True,
    allow_open_relations: bool = True,
) -> list[FactCandidate]:
    """Акцептор: confidence + lemma-in-text. Open relations по умолчанию."""
    del allow_open_relations  # всегда open; флаг оставлен для совместимости вызовов
    finalized = _finalize_candidates(cands)
    text_lemmas = _collect_text_lemmas(text) if require_grounding else set()
    out: list[FactCandidate] = []
    for c in finalized:
        if c.confidence < min_confidence:
            continue
        if require_grounding and not _roles_grounded(c.roles, text_lemmas, text):
            continue
        out.append(c)
    return out


def content_entity_uids(text: str, *, limit: int = 32) -> list[str]:
    """POS-фильтр сущностей из текста → seed UID (без изобретения фактов)."""
    low = _norm(text)
    out: list[str] = []
    for w in _TOKEN_RE.findall(low):
        if len(w) < 2 and w not in _SHORT_KEEP:
            continue
        if w in STOP or not is_entity_token(w, allow_pronoun=True):
            continue
        uid = slug_uid(w)
        if uid in PREDICATES or uid in out:
            continue
        out.append(uid)
        if len(out) >= limit:
            break
    return out


# ── Бэкенды ──────────────────────────────────────────────────────────────────
# SeedPerception — без LLM (тесты / offline).
# JsonLLMPerception — любой call_fn, возвращающий JSON candidates.


class SeedPerception:
    """Offline: только seed_tokens, никогда не создаёт FactCandidate."""

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        raw = text.strip()
        low = _norm(raw)
        question = _is_question(raw, low)
        seeds = content_entity_uids(raw)[:8]
        if question:
            kind: Literal["fact", "question", "message"] = "question"
        elif seeds:
            kind = "message"
        else:
            kind = "message"
        return PerceptionResult(
            kind=kind,
            candidates=[],
            seed_tokens=seeds,
            meta={"backend": "seeds"},
        )


# Backward-compatible name (was regex RulePerception).
RulePerception = SeedPerception


class JsonLLMPerception:
    """Обёртка: call_fn(prompt) → JSON → gate_candidates → PerceptionResult."""

    def __init__(
        self,
        call_fn,
        *,
        require_grounding: bool = True,
        allow_open_relations: bool = True,
    ) -> None:
        self.call_fn = call_fn
        self.require_grounding = require_grounding
        self.allow_open_relations = allow_open_relations

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        prompt = {
            "task": "extract_facts",
            "text": text,
            "wm": wm_context or [],
            "schema": {
                "raw_relation": "str",
                "canonical_relation": "str|null",
                "predicate": "legacy canonical fallback",
                "roles": "dict",
            },
        }
        raw = self.call_fn(json.dumps(prompt, ensure_ascii=False))
        data = json.loads(raw) if isinstance(raw, str) else raw
        cands = [
            FactCandidate(
                predicate=str(
                    c.get("canonical_relation")
                    or c.get("predicate")
                    or c.get("raw_relation")
                    or c.get("relation", "")
                ).upper(),
                roles={str(k).upper(): slug_uid(str(v)) for k, v in (c.get("roles") or {}).items()},
                raw_span=c.get("raw_span"),
                confidence=float(c.get("confidence", 0.8)),
                raw_relation=str(
                    c.get("raw_relation")
                    or c.get("relation")
                    or c.get("predicate", "")
                ),
                canonical_relation=c.get("canonical_relation"),
            )
            for c in data.get("candidates", [])
            if c.get("predicate") or c.get("raw_relation") or c.get("relation")
        ]
        gated = gate_candidates(
            text,
            cands,
            require_grounding=self.require_grounding,
            allow_open_relations=self.allow_open_relations,
        )
        kind = data.get("kind", "fact")
        if kind not in {"fact", "question", "message"}:
            kind = "fact"
        low = _norm(text.strip())
        if _is_question(text.strip(), low):
            kind = "question"
            gated = []
        seeds = seeds_from_roles(
            gated,
            extra=filter_entity_uids(
                [slug_uid(str(s)) for s in data.get("seed_tokens", [])],
                allow_pronoun=True,
            )[:12],
        )
        if not seeds and kind != "question":
            seeds = content_entity_uids(text)[:8]
        if kind != "question" and not gated:
            kind = "message"
        return PerceptionResult(
            kind=kind,
            candidates=gated,
            seed_tokens=seeds,
            meta={"backend": "json_llm", "llm_raw": data},
        )
