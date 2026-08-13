"""Модель данных AH (монография §3 + open-semantics).

Актуальный путь:
  S (слова) → M (сущности) → Event/Factor (open relations) → L (BIND/ASSOC/…) → AH
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


class Section(str, Enum):
    """Секции гиперслоя: C — общее, P — личное, H — история."""

    C = "C"
    P = "P"
    H = "H"


class RefKind(str, Enum):
    S = "S"
    M = "M"


class LinkId(str, Enum):
    IS_A = "IS-A"
    FOLLOW = "FOLLOW"
    ASSOC = "ASSOC"
    BIND = "BIND"
    CAUSE = "CAUSE"


class Role(str, Enum):
    """Слоты Event/Factor — кто/где/чем."""

    SUBJECT = "SUBJECT"
    OBJECT = "OBJECT"
    AUXILIARY = "AUXILIARY"
    RECIPIENT = "RECIPIENT"
    SOURCE = "SOURCE"
    ABSENTEE = "ABSENTEE"
    LOCATION = "LOCATION"
    STATE = "STATE"
    TIME = "TIME"
    DURATION = "DURATION"
    CAUSE = "CAUSE"
    PURPOSE = "PURPOSE"
    TOOL = "TOOL"
    MATERIAL = "MATERIAL"
    AMOUNT = "AMOUNT"
    HOW_TO = "HOW-TO"


@dataclass
class Property:
    name: str
    value: str
    type: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.value:
            raise ValueError("Property.name and Property.value must be non-empty")


@dataclass(frozen=True)
class SRef:
    """Ссылка на AbstractSymbol в S (лексика)."""

    target_uid: str
    kind: RefKind = RefKind.S

    def __post_init__(self) -> None:
        if self.kind is not RefKind.S:
            raise ValueError("SRef.kind must be 'S'")


@dataclass(frozen=True)
class MRef:
    """Ссылка на сущность в C∪P∪H (обычно SecondOrderSymbol)."""

    target_uid: str
    kind: RefKind = RefKind.M

    def __post_init__(self) -> None:
        if self.kind is not RefKind.M:
            raise ValueError("MRef.kind must be 'M'")


ElementRef = Union[SRef, MRef]


@dataclass
class AbstractSymbol:
    """s_i = ⟨UID, R⟩ — лексика 1-го порядка."""

    uid: str
    R: dict[str, set[str]] = field(default_factory=dict)
    x: float = 0.0
    created_tau: int = 0
    added_at: str = ""

    def modality_partition_ok(self) -> bool:
        seen: set[str] = set()
        for symbols in self.R.values():
            if seen & symbols:
                return False
            seen |= symbols
        return True


@dataclass
class SecondOrderSymbol:
    """m = ⟨UID, Pr, Mt⟩ — сущность / смысл 2-го порядка."""

    uid: str
    Pr: list[Property] = field(default_factory=list)
    Mt: list[Property] = field(default_factory=list)
    x: float = 0.0
    created_tau: int = 0
    added_at: str = ""


@dataclass
class AssocLink:
    """l = ⟨UID, ID, w, (e1*, e2*)⟩ — бинарная связь (IS-A, BIND, ASSOC…)."""

    uid: str
    id: str
    w: float
    e1: ElementRef
    e2: ElementRef
    created_tau: int = 0
    added_at: str = ""


HyperElement = Union[SecondOrderSymbol, SRef, MRef]


@dataclass
class AH:
    """AH = ⟨S, C, P, H, L⟩. Event/Factor живут в AHStore, не в AH."""

    S: dict[str, AbstractSymbol] = field(default_factory=dict)
    C: dict[str, HyperElement] = field(default_factory=dict)
    P: dict[str, HyperElement] = field(default_factory=dict)
    H: dict[str, HyperElement] = field(default_factory=dict)
    L: dict[str, AssocLink] = field(default_factory=dict)
    tau: int = 0
    revision: int = 0

    def section(self, name: Section) -> dict[str, HyperElement]:
        return {Section.C: self.C, Section.P: self.P, Section.H: self.H}[name]

    def all_hyper(self) -> dict[str, HyperElement]:
        out: dict[str, HyperElement] = {}
        out.update(self.C)
        out.update(self.P)
        out.update(self.H)
        return out

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
