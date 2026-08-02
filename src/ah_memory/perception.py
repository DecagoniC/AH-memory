"""Perception: NL → FactCandidate. Lemmas via pymorphy3; roles = entities only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ah_memory.morph import (
    STOP,
    filter_entity_uids,
    head_entity,
    is_entity_token,
    lemma,
    sanitize_roles,
    slug_uid,
)


@dataclass(frozen=True)
class FactCandidate:
    predicate: str
    roles: dict[str, str]
    raw_span: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class PerceptionResult:
    kind: Literal["fact", "question", "message"]
    candidates: list[FactCandidate]
    seed_tokens: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)


class PerceptionBackend(Protocol):
    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult: ...


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


def _finalize_candidates(cands: list[FactCandidate]) -> list[FactCandidate]:
    out: list[FactCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for c in cands:
        roles = sanitize_roles(dict(c.roles))
        if roles is None:
            continue
        key = (c.predicate.upper(), tuple(sorted(roles.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            FactCandidate(c.predicate.upper(), roles, c.raw_span, c.confidence)
        )
    return out


class RulePerception:
    """Pattern parser; entity UIDs = lemmas, no verb/conj subjects."""

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        raw = text.strip()
        low = _norm(raw)
        question = _is_question(raw, low)

        cands: list[FactCandidate] = []
        for sent in re.split(r"[.!;?\n]\s*", raw):
            s = sent.strip()
            if not s:
                continue
            cands.extend(self._parse_sentence(s))
        cands = _finalize_candidates(cands)

        seeds = self._content_tokens(low)
        for c in cands:
            seeds.extend(c.roles.values())
        seed_tokens = filter_entity_uids(seeds, allow_pronoun=True)

        kind: Literal["fact", "question", "message"] = "question" if question else "fact"
        if not question and not cands:
            kind = "message"
        return PerceptionResult(kind=kind, candidates=cands, seed_tokens=seed_tokens, meta={})

    def _content_tokens(self, low: str) -> list[str]:
        out: list[str] = []
        for w in re.findall(r"[a-zа-я0-9]{2,}", low):
            if w in STOP or not is_entity_token(w, allow_pronoun=True):
                continue
            uid = slug_uid(w)
            if uid not in out and uid not in PREDICATES:
                out.append(uid)
        return out[:32]

    def _parse_sentence(self, sent: str) -> list[FactCandidate]:
        low = _norm(sent)
        out: list[FactCandidate] = []

        if _is_question(sent, low) and not re.search(
            r"\b(?:является|это|обитает|живет|имеет|появил)\b", low
        ):
            return out

        # «меня зовут X» / «зовут X»
        m = re.search(r"(?:меня\s+)?зовут\s+([a-zа-яё\-]+)", low)
        if m:
            name = slug_uid(m.group(1))
            out.append(FactCandidate("IS", {"SUBJECT": "Я", "OBJECT": name}, sent, 0.9))

        # «мой брат — X» / «брата зовут X»
        m = re.search(r"(?:моего\s+)?брат[ауи]?\s+зовут\s+([a-zа-яё\-]+)", low)
        if m:
            out.append(
                FactCandidate("IS", {"SUBJECT": "БРАТ", "OBJECT": slug_uid(m.group(1))}, sent, 0.9)
            )

        # «занимаюсь X» → USE(Я, lemma(X))
        m = re.search(r"занима(?:юсь|ется|емся|ются)\s+(.+)", low)
        if m:
            words = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            obj = head_entity(words, allow_pronoun=False)
            if obj:
                out.append(
                    FactCandidate("USE", {"SUBJECT": "Я", "OBJECT": obj}, sent, 0.8)
                )

        m = re.search(r"^(.+?)\s*-\s*это\s+(.+)$", low)
        if not m:
            m = re.search(r"^(.+?)\s+это\s+(.+)$", low)
        if not m:
            m = re.search(r"^(.+?)\s*-\s*(.+)$", low)
        if not m:
            m = re.search(r"^(.+?)\s+является\s+(.+)$", low)
        if not m:
            m = re.search(r"^(.+?)\s+is\s+(?:an?\s+)?(.+)$", low)
        if m:
            left = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            right = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(2))]
            # skip conditional / discourse left sides
            if left and left[0] in STOP:
                left = left[1:]
            subj = head_entity(left, allow_pronoun=True)
            objs = filter_entity_uids([slug_uid(x) for x in right], allow_pronoun=False)[:4]
            if subj and objs:
                for obj in objs:
                    if obj != subj:
                        out.append(
                            FactCandidate("IS", {"SUBJECT": subj, "OBJECT": obj}, sent, 0.85)
                        )

        m = re.search(
            r"(.+?)\s+(?:обитает|живет|живёт|lives|live[sd]?|работа[ею]т)\s+(?:в|на|in|at)\s+(.+)",
            low,
        )
        if m:
            left = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            subj = head_entity(left, allow_pronoun=True)
            locs = filter_entity_uids(self._content_tokens(m.group(2)), allow_pronoun=False)[:2]
            if subj:
                for loc in locs:
                    out.append(
                        FactCandidate("LIVE_IN", {"SUBJECT": subj, "LOCATION": loc}, sent, 0.85)
                    )

        m = re.search(r"(?:у)\s+(.+?)\s+(?:есть|имеется)\s+(.+)", low)
        if not m:
            m = re.search(r"(.+?)\s+(?:имеет|have|has)\s+(.+)", low)
        if m:
            left = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            subj = head_entity(left, allow_pronoun=True)
            if subj:
                for obj in filter_entity_uids(self._content_tokens(m.group(2)), allow_pronoun=False)[:3]:
                    out.append(
                        FactCandidate("HAVE", {"SUBJECT": subj, "OBJECT": obj}, sent, 0.8)
                    )

        m = re.search(r"(.+?)\s+(?:бегает|бежит|runs?|running)\s*(.*)", low)
        if m:
            left = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            subj = head_entity(left, allow_pronoun=True)
            if subj:
                roles: dict[str, str] = {"SUBJECT": subj}
                rest = filter_entity_uids(self._content_tokens(m.group(2)), allow_pronoun=False)
                if rest:
                    roles["HOW-TO"] = rest[0]
                out.append(FactCandidate("RUN", roles, sent, 0.8))

        m = re.search(
            r"(.+?)\s+появил(?:ся|ась|ось|ись)\s+(?:в\s+)?(\d{4})",
            low,
        )
        if m:
            left = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            subj = head_entity(left, allow_pronoun=False)
            year = m.group(2)
            if subj:
                out.append(FactCandidate("IS", {"SUBJECT": subj, "OBJECT": year}, sent, 0.7))
                out.append(
                    FactCandidate(
                        "CREATE",
                        {"SUBJECT": year, "OBJECT": subj, "TIME": year},
                        sent,
                        0.7,
                    )
                )

        m = re.search(r"(.+?)\s+(\w+)\s+цвет", low)
        if m:
            left = [lemma(w) for w in re.findall(r"[a-zа-я0-9]{2,}", m.group(1))]
            subj = head_entity(left, allow_pronoun=False)
            color = slug_uid(m.group(2))
            if subj and is_entity_token(m.group(2)):
                out.append(
                    FactCandidate(
                        "BE_COLORED",
                        {"SUBJECT": subj, "OBJECT": color},
                        sent,
                        0.7,
                    )
                )

        return out


PREDICATES = {
    "CREATE",
    "IS",
    "LIVE_IN",
    "HAVE",
    "RUN",
    "BE_COLORED",
    "CAUSE_EVENT",
    "USE",
    "MOVE",
}


class JsonLLMPerception:
    def __init__(self, call_fn) -> None:
        self.call_fn = call_fn

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        prompt = {
            "task": "extract_facts",
            "text": text,
            "wm": wm_context or [],
            "schema": {"predicate": "str", "roles": "dict"},
        }
        raw = self.call_fn(json.dumps(prompt, ensure_ascii=False))
        data = json.loads(raw)
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
        seeds = filter_entity_uids(
            [slug_uid(str(s)) for s in data.get("seed_tokens", [])],
            allow_pronoun=True,
        )
        return PerceptionResult(kind=kind, candidates=cands, seed_tokens=seeds)
