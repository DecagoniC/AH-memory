"""Perception: текст → FactCandidate / seed_tokens.

Слой между NL и Transform. LLM (или offline seeds) предлагает факты;
gate_candidates отбрасывает мусор (роли не из текста, слишком широкие UID…).
Читать после types/store; дальше — transform.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from ah_memory.morph import (
    STOP,
    _is_acronym,
    _parse,
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


EpistemicStatus = Literal[
    "assertion",
    "decision",
    "topic",
    "open_question",
    "proposal",
    "explanation",
]


@dataclass(frozen=True)
class FactCandidate:
    predicate: str
    roles: dict[str, str]
    raw_span: str | None = None
    confidence: float = 1.0
    raw_relation: str | None = None
    canonical_relation: str | None = None
    statement_type: EpistemicStatus = "assertion"
    source: Literal["user", "assistant"] = "user"


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
                    "statement_type": c.statement_type,
                    "source": c.source,
                }
                for c in self.candidates
            ],
            "seed_tokens": list(self.seed_tokens),
            "meta": {k: v for k, v in self.meta.items() if k not in ("llm_raw", "gate_report")},
        }
        if "llm_raw" in self.meta:
            out["llm_raw"] = self.meta["llm_raw"]
        if "gate_report" in self.meta:
            out["gate_report"] = self.meta["gate_report"]
        return out


class PerceptionBackend(Protocol):
    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult: ...


# ── Текстовые хелперы / grounding ────────────────────────────────────────────
# Зачем: не пускать в граф сущности, которых нет в исходной фразе (галлюцинации LLM).

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
_SHORT_KEEP = frozenset({"я", "мы", "ты", "он", "а"})

_ALLOWED_ROLES = frozenset(
    {
        "SUBJECT",
        "OBJECT",
        "LOCATION",
        "TIME",
        "CAUSE",
        "TOOL",
        "MATERIAL",
        "PURPOSE",
        "HOW-TO",
        "WITH",
    }
)

def _canonical_role_key(role: str) -> str:
    r = str(role).upper().strip().replace("_", "-")
    if r in {"HOWTO", "HOW TO"}:
        return "HOW-TO"
    return r


def normalize_llm_roles(
    roles: dict[str, str],
) -> dict[str, str]:
    """Keep only roles from the explicit perception protocol."""
    mapped: dict[str, str] = {}
    for role, val in roles.items():
        if not val:
            continue
        canon = _canonical_role_key(role)
        if canon in _ALLOWED_ROLES and canon not in mapped:
            mapped[canon] = str(val)
    return mapped


def roles_from_llm_json(roles: dict | None) -> dict[str, str]:
    normalized = normalize_llm_roles(
        {str(k): str(v) for k, v in (roles or {}).items()}
    )
    return {role: slug_uid(val) for role, val in normalized.items()}


_ROLE_LIST_RE = re.compile(
    r"\s*(?:,|;|→|->|\b(?:и|или|and|or)\b)\s*",
    re.IGNORECASE,
)


def _llm_role_value_is_compound(value: str) -> bool:
    """True when a role contains a list/clause instead of one entity."""
    raw = str(value).strip()
    if not raw:
        return False
    if _ROLE_LIST_RE.search(raw):
        return True
    if re.search(r"(?:^|[\s_])(?:и|или|and|or)(?:[\s_]|$)", raw, re.I):
        return True

    # Models sometimes remove separators from a list of identifiers.
    tokens = raw.split()
    identifiers = [
        token
        for token in tokens
        if not re.fullmatch(r"[A-ZА-Я]\.", token)
        and (
            re.fullmatch(r"[A-ZА-Я0-9_.-]{2,}", token)
            or re.search(r"[a-zа-я][A-ZА-Я]", token)
            or re.search(r"\d", token)
        )
    ]
    return len(tokens) >= 3 and len(identifiers) >= 2


def role_variants_from_llm_json(
    roles: dict | None,
    *,
    statement_type: EpistemicStatus = "assertion",
) -> list[dict[str, str]]:
    """Validate one complete atomic role mapping without semantic guessing."""
    del statement_type  # Atomic role validation is identical for every epistemic type.
    raw_roles = {str(k): str(v) for k, v in (roles or {}).items()}
    canonical_keys = [_canonical_role_key(role) for role in raw_roles]
    if (
        not raw_roles
        or any(role not in _ALLOWED_ROLES for role in canonical_keys)
        or len(set(canonical_keys)) != len(canonical_keys)
    ):
        return []
    normalized = normalize_llm_roles(raw_roles)
    if "SUBJECT" not in normalized:
        return []
    for role, value in list(normalized.items()):
        parenthetical = re.fullmatch(r"\s*(.+?)\s*\(([^()]+)\)\s*", value)
        if parenthetical and role not in {"SUBJECT", "TIME"}:
            if "WITH" in normalized:
                return []
            normalized[role] = parenthetical.group(1).strip()
            normalized["WITH"] = parenthetical.group(2).strip()
    if len(normalized) < 2:
        return []
    for role, value in normalized.items():
        if role != "TIME" and _llm_role_value_is_compound(value):
            return []
    return [{role: slug_uid(value) for role, value in normalized.items()}]


def llm_payload_errors(data: dict[str, Any]) -> list[str]:
    """Return schema/atomicity errors suitable for a domain-neutral repair prompt."""
    errors: list[str] = []
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return ["candidates must be a list"]
    allowed_statuses = {
        "assertion",
        "decision",
        "topic",
        "open_question",
        "proposal",
        "explanation",
    }
    for index, candidate in enumerate(candidates):
        prefix = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        predicate = (
            candidate.get("canonical_relation")
            or candidate.get("predicate")
            or candidate.get("raw_relation")
        )
        if not isinstance(predicate, str) or not predicate.strip():
            errors.append(f"{prefix} requires a relation label")
        status = str(candidate.get("statement_type", "")).lower()
        if status not in allowed_statuses:
            errors.append(f"{prefix}.statement_type is invalid")
        roles = candidate.get("roles")
        if not isinstance(roles, dict):
            errors.append(f"{prefix}.roles must be an object")
            continue
        canonical_roles = [_canonical_role_key(str(role)) for role in roles]
        unknown = sorted(
            role for role in canonical_roles if role not in _ALLOWED_ROLES
        )
        if unknown:
            errors.append(f"{prefix}.roles contains unsupported roles: {unknown}")
        if "SUBJECT" not in canonical_roles:
            errors.append(f"{prefix}.roles requires SUBJECT")
        if len(canonical_roles) < 2:
            errors.append(f"{prefix}.roles requires at least two roles")
        if len(set(canonical_roles)) != len(canonical_roles):
            errors.append(f"{prefix}.roles contains duplicate canonical roles")
        for role, value in roles.items():
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.roles.{role} must be a non-empty string")
            elif (
                _canonical_role_key(str(role)) != "TIME"
                and _llm_role_value_is_compound(value)
            ):
                errors.append(f"{prefix}.roles.{role} must contain one entity")
    return errors


def candidates_from_llm_json(data: dict[str, Any]) -> list[FactCandidate]:
    """Build typed, atomic notes; epistemic status prevents topics becoming facts."""
    out: list[FactCandidate] = []
    utterance_kind = str(data.get("kind", "fact")).lower()
    for candidate in data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        raw_statement_type = candidate.get("statement_type")
        if raw_statement_type is None and utterance_kind == "question":
            # In a question, an untyped candidate is usually the question itself.
            continue
        statement_type_raw = str(raw_statement_type or "assertion").lower()
        allowed_statuses = {
            "assertion",
            "decision",
            "topic",
            "open_question",
            "proposal",
            "explanation",
        }
        if statement_type_raw not in allowed_statuses:
            continue
        statement_type = cast(EpistemicStatus, statement_type_raw)
        predicate = str(
            candidate.get("canonical_relation")
            or candidate.get("predicate")
            or candidate.get("raw_relation")
            or candidate.get("relation", "")
        ).upper()
        if not predicate:
            continue
        predicate_parts = [part for part in re.split(r"[_\s-]+", predicate) if part]
        if len(predicate_parts) > 5:
            predicate = {
                "topic": "DISCUSSES",
                "open_question": "ASKS_ABOUT",
                "proposal": "PROPOSES",
                "explanation": "EXPLAINS",
                "decision": "DECIDES",
                "assertion": "RELATED_TO",
            }[statement_type]
        source: Literal["user", "assistant"] = (
            "assistant"
            if str(candidate.get("source", "user")).lower() == "assistant"
            else "user"
        )
        raw_relation = str(
            candidate.get("raw_relation")
            or candidate.get("relation")
            or candidate.get("predicate", "")
        )
        raw_span = candidate.get("raw_span")
        for roles in role_variants_from_llm_json(
            candidate.get("roles"),
            statement_type=statement_type,
        ):
            out.append(
                FactCandidate(
                    predicate=predicate,
                    roles=roles,
                    raw_span=raw_span,
                    confidence=float(candidate.get("confidence", 0.8)),
                    raw_relation=raw_relation,
                    canonical_relation=predicate,
                    statement_type=statement_type,
                    source=source,
                )
            )
    return out


def _norm(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[—–−]", "-", t)
    t = t.replace("&", "")
    return t


def _text_tokens(raw: str) -> set[str]:
    """Токены текста для grounding (d&d → dd, днд остаётся)."""
    low = _norm(raw)
    return set(_TOKEN_RE.findall(low))


def classify_utterance(
    text: str,
    *,
    declared_kind: str | None = None,
    candidates: list[FactCandidate] | None = None,
    query: Any = None,
    interaction: Literal["auto", "query"] = "auto",
) -> Literal["fact", "question", "message"]:
    """Combine protocol evidence without language-specific phrase lists."""
    if interaction == "query":
        return "question"
    if text.rstrip().endswith("?"):
        return "question"
    if declared_kind == "question":
        return "question"
    if isinstance(query, dict) and any(
        str(query.get(field) or "").strip()
        for field in ("relation", "target_role")
    ):
        return "question"
    candidates_now = candidates or []
    if any(
        candidate.statement_type in {"topic", "open_question"}
        for candidate in candidates_now
    ):
        return "question"
    if any(
        candidate.statement_type in {"assertion", "decision"}
        for candidate in candidates_now
    ):
        return "fact"
    return "message"


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


def _uid_grounded(
    uid: str,
    text_lemmas: set[str],
    raw_tokens: set[str],
    raw: str,
) -> bool:
    bare = uid[2:] if uid.startswith("M_") else uid
    if bare.isdigit():
        return bare in raw_tokens
    parts = [p.lower().replace("ё", "е") for p in bare.split("_") if p]
    if not parts:
        return False
    grounded = 0
    for p in parts:
        if p.isdigit():
            if p not in raw_tokens:
                return False
            grounded += 1
            continue
        pl = lemma(p).replace("ё", "е")
        if p in raw_tokens or p in text_lemmas or pl in text_lemmas:
            grounded += 1
            continue
        return False
    compact = bare.replace("_", "").lower()
    if _is_acronym(compact):
        if compact in raw_tokens:
            return True
        flat = re.sub(r"[^a-zа-я0-9]", "", _norm(raw))
        if compact in flat:
            return True
    return grounded == len(parts)


def _roles_grounded(
    roles: dict[str, str],
    text_lemmas: set[str],
    raw: str,
    statement_type: EpistemicStatus = "assertion",
) -> bool:
    raw_tokens = _text_tokens(raw)
    for role, val in roles.items():
        bare = val[2:] if str(val).startswith("M_") else str(val)
        if (
            role == "SUBJECT"
            and bare in {"ДИАЛОГ", "АССИСТЕНТ"}
            and statement_type in {"topic", "open_question", "proposal", "explanation"}
        ):
            continue
        if (
            role == "SUBJECT"
            and bare in {"Я", "МЫ"}
            and (
                _subject_is_explicit_or_implied(bare, raw_tokens)
                or (
                    statement_type == "decision"
                    and bare == "МЫ"
                    and raw_tokens.intersection(
                        {"ок", "ладно", "хорошо", "тогда", "выбираем", "решили"}
                    )
                )
            )
        ):
            continue
        if not _uid_grounded(bare, text_lemmas, raw_tokens, raw):
            return False
    return True


def _subject_is_explicit_or_implied(subject: str, raw_tokens: set[str]) -> bool:
    expected_number = "sing" if subject == "Я" else "plur"
    explicit = "я" if subject == "Я" else "мы"
    if explicit in raw_tokens:
        return True
    for token in raw_tokens:
        tag = str(_parse(token).tag)
        if "VERB" in tag and "1per" in tag and expected_number in tag:
            return True
    return False


def _finalize_candidates(cands: list[FactCandidate]) -> list[FactCandidate]:
    # Зачем: sanitize roles and deduplicate. Canonical labels remain open.
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
        if any(uid_too_wide(v) for v in roles.values()):
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
                statement_type=c.statement_type,
                source=c.source,
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
    report: bool = False,
) -> list[FactCandidate] | tuple[list[FactCandidate], dict[str, Any]]:
    """Акцептор: confidence + lemma-in-text. Open relations по умолчанию."""
    del allow_open_relations  # всегда open; флаг оставлен для совместимости вызовов
    finalized = _finalize_candidates(cands)
    text_lemmas = _collect_text_lemmas(text) if require_grounding else set()
    out: list[FactCandidate] = []
    dropped: list[dict[str, Any]] = []
    for c in finalized:
        if c.confidence < min_confidence:
            if report:
                dropped.append(
                    {
                        "reason": "low_confidence",
                        "confidence": c.confidence,
                        "predicate": c.canonical_relation or c.predicate,
                        "roles": dict(c.roles),
                        "raw_span": c.raw_span,
                    }
                )
            continue
        if require_grounding and not _roles_grounded(
            c.roles, text_lemmas, text, c.statement_type
        ):
            if report:
                dropped.append(
                    {
                        "reason": "grounding",
                        "predicate": c.canonical_relation or c.predicate,
                        "roles": dict(c.roles),
                        "raw_span": c.raw_span,
                    }
                )
            continue
        out.append(c)
    if report:
        return out, {
            "llm_candidate_count": len(cands),
            "finalized_count": len(finalized),
            "gated_count": len(out),
            "dropped": dropped,
        }
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
        if uid in out:
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
        seeds = content_entity_uids(raw)[:8]
        kind = classify_utterance(raw)
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
                "roles": sorted(_ALLOWED_ROLES),
                "statement_type": [
                    "assertion",
                    "decision",
                    "topic",
                    "open_question",
                    "proposal",
                    "explanation",
                ],
            },
        }
        raw = self.call_fn(json.dumps(prompt, ensure_ascii=False))
        data = json.loads(raw) if isinstance(raw, str) else raw
        cands = candidates_from_llm_json(data)
        gated_result = gate_candidates(
            text,
            cands,
            require_grounding=self.require_grounding,
            allow_open_relations=self.allow_open_relations,
            report=True,
        )
        gated, gate_report = gated_result
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
            meta={
                "backend": "json_llm",
                "llm_raw": data,
                "gate_report": gate_report,
                "validation_errors": validation_errors,
            },
        )
