"""Strict schemas for the offline challenge benchmark corpora."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROLE_VARIANTS = frozenset({"clean", "typo", "inversion", "ellipsis"})
MANDATORY_ROLES = frozenset({"SUBJECT", "OBJECT", "LOCATION"})
QA_RELATIONS = frozenset({"FOLLOW", "IS-A", "CAUSE"})


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], kind: str) -> None:
    if set(data) != expected:
        raise ValueError(
            f"{kind} fields must be exactly {sorted(expected)}; got {sorted(data)}"
        )


@dataclass(frozen=True)
class RoleCorpusItem:
    item_id: str
    text: str
    variant: str
    domain: str
    expected_roles: Mapping[str, str]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoleCorpusItem":
        _exact_keys(
            raw,
            {"id", "text", "variant", "domain", "expected_roles"},
            "role item",
        )
        roles = _mapping(raw["expected_roles"], "expected_roles")
        if set(roles) != MANDATORY_ROLES:
            raise ValueError(
                "expected_roles must contain exactly SUBJECT, OBJECT, and LOCATION"
            )
        atomic_roles = {
            role: _text(value, f"expected_roles.{role}")
            for role, value in roles.items()
        }
        variant = _text(raw["variant"], "variant")
        if variant not in ROLE_VARIANTS:
            raise ValueError(f"unsupported role variant: {variant}")
        return cls(
            item_id=_text(raw["id"], "id"),
            text=_text(raw["text"], "text"),
            variant=variant,
            domain=_text(raw["domain"], "domain"),
            expected_roles=atomic_roles,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "text": self.text,
            "variant": self.variant,
            "domain": self.domain,
            "expected_roles": dict(self.expected_roles),
        }


@dataclass(frozen=True)
class SourceFact:
    uid: str
    subject: str
    relation: str
    object: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceFact":
        _exact_keys(raw, {"uid", "subject", "relation", "object"}, "source fact")
        relation = _text(raw["relation"], "source fact relation")
        if relation not in QA_RELATIONS:
            raise ValueError(f"unsupported source fact relation: {relation}")
        return cls(
            uid=_text(raw["uid"], "source fact uid"),
            subject=_text(raw["subject"], "source fact subject"),
            relation=relation,
            object=_text(raw["object"], "source fact object"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "uid": self.uid,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
        }


@dataclass(frozen=True)
class SourceDocument:
    uid: str
    text: str
    fact_uids: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceDocument":
        _exact_keys(raw, {"uid", "text", "fact_uids"}, "source document")
        fact_uids = raw["fact_uids"]
        if not isinstance(fact_uids, list) or not fact_uids:
            raise ValueError("source document fact_uids must be a non-empty list")
        return cls(
            uid=_text(raw["uid"], "source document uid"),
            text=_text(raw["text"], "source document text"),
            fact_uids=tuple(
                _text(uid, "source document fact uid") for uid in fact_uids
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "text": self.text,
            "fact_uids": list(self.fact_uids),
        }


@dataclass(frozen=True)
class QAItem:
    item_id: str
    question: str
    answer: str
    depth: int
    relation_type: str
    proof_path: tuple[str, ...]
    source_facts: tuple[SourceFact, ...]
    source_documents: tuple[SourceDocument, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QAItem":
        _exact_keys(
            raw,
            {
                "id",
                "question",
                "answer",
                "depth",
                "relation_type",
                "proof_path",
                "source_facts",
                "source_documents",
            },
            "QA item",
        )
        depth = raw["depth"]
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 6:
            raise ValueError("depth must be an integer from 1 through 6")
        relation = _text(raw["relation_type"], "relation_type")
        if relation not in QA_RELATIONS:
            raise ValueError(f"unsupported QA relation: {relation}")
        proof_path_raw = raw["proof_path"]
        if not isinstance(proof_path_raw, list):
            raise ValueError("proof_path must be a list")
        proof_path = tuple(_text(uid, "proof UID") for uid in proof_path_raw)
        if len(proof_path) != depth or len(set(proof_path)) != depth:
            raise ValueError("proof_path must contain exactly depth unique UIDs")
        facts_raw = raw["source_facts"]
        documents_raw = raw["source_documents"]
        if not isinstance(facts_raw, list) or not isinstance(documents_raw, list):
            raise ValueError("source_facts and source_documents must be lists")
        facts = tuple(SourceFact.from_dict(_mapping(item, "source fact")) for item in facts_raw)
        documents = tuple(
            SourceDocument.from_dict(_mapping(item, "source document"))
            for item in documents_raw
        )
        fact_by_uid = {fact.uid: fact for fact in facts}
        if len(fact_by_uid) != len(facts):
            raise ValueError("source fact UIDs must be unique within an item")
        if set(proof_path) - set(fact_by_uid):
            raise ValueError("every proof UID must identify a source fact")
        if any(fact_by_uid[uid].relation != relation for uid in proof_path):
            raise ValueError("all proof facts must use relation_type")
        for left_uid, right_uid in zip(proof_path, proof_path[1:]):
            if fact_by_uid[left_uid].object != fact_by_uid[right_uid].subject:
                raise ValueError("proof facts must form an ordered chain")
        answer = _text(raw["answer"], "answer")
        if fact_by_uid[proof_path[-1]].object != answer:
            raise ValueError("answer must equal the final proof fact object")
        documented = {
            fact_uid for document in documents for fact_uid in document.fact_uids
        }
        if set(proof_path) - documented:
            raise ValueError("source documents must cover the complete proof path")
        if documented - set(fact_by_uid):
            raise ValueError("source documents reference unknown fact UIDs")
        return cls(
            item_id=_text(raw["id"], "id"),
            question=_text(raw["question"], "question"),
            answer=answer,
            depth=depth,
            relation_type=relation,
            proof_path=proof_path,
            source_facts=facts,
            source_documents=documents,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "question": self.question,
            "answer": self.answer,
            "depth": self.depth,
            "relation_type": self.relation_type,
            "proof_path": list(self.proof_path),
            "source_facts": [fact.to_dict() for fact in self.source_facts],
            "source_documents": [
                document.to_dict() for document in self.source_documents
            ],
        }


CHALLENGE_DATA_DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "challenge"
