"""Case types and dataclasses for entity resolution benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CaseType(str, Enum):
    MORPHOLOGY = "MORPHOLOGY"
    SYNONYM = "SYNONYM"
    SEMANTIC_NEAR = "SEMANTIC_NEAR"
    NEGATIVE = "NEGATIVE"
    CONTEXTUAL = "CONTEXTUAL"
    AMBIGUOUS = "AMBIGUOUS"
    SURFACE = "SURFACE"
    HOLD_OUT = "HOLD_OUT"


class ExpectedRelation(str, Enum):
    SAME_ENTITY = "SAME_ENTITY"
    SAME_CONCEPT = "SAME_CONCEPT"
    DIFFERENT_ENTITY = "DIFFERENT_ENTITY"
    DIFFERENT_CONCEPT = "DIFFERENT_CONCEPT"


@dataclass(frozen=True)
class SymbolSpec:
    uid: str
    name: str
    aliases: tuple[str, ...] = ()
    kind: str = "entity"  # entity | person | place | brand | concept
    context_cues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityResolutionCase:
    case_id: str
    mention: str
    target_uid: str | None
    candidate_uids: tuple[str, ...]
    case_type: CaseType
    expected_relation: ExpectedRelation
    context: str | None = None
    difficulty: str = "easy"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mention": self.mention,
            "target_uid": self.target_uid,
            "candidate_uids": list(self.candidate_uids),
            "case_type": self.case_type.value,
            "expected_relation": self.expected_relation.value,
            "context": self.context,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }


@dataclass
class CandidateScore:
    uid: str
    similarity: float
    method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolutionResult:
    selected_uid: str | None
    confidence: float
    method: str
    candidates: list[CandidateScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_uid": self.selected_uid,
            "confidence": self.confidence,
            "method": self.method,
            "candidates": [c.to_dict() for c in self.candidates],
        }
