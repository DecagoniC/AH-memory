"""AH memory core types (monograph §3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


class Section(str, Enum):
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
    """s* = ⟨'S, UID, UID*⟩"""

    target_uid: str
    ref_uid: str
    kind: RefKind = RefKind.S

    def __post_init__(self) -> None:
        if self.kind is not RefKind.S:
            raise ValueError("SRef.kind must be 'S'")


@dataclass(frozen=True)
class MRef:
    """m* = ⟨'M, UID, UID*⟩"""

    target_uid: str
    ref_uid: str
    kind: RefKind = RefKind.M

    def __post_init__(self) -> None:
        if self.kind is not RefKind.M:
            raise ValueError("MRef.kind must be 'M'")


ElementRef = Union[SRef, MRef]


@dataclass
class AbstractSymbol:
    """s_i = ⟨UID, R⟩"""

    uid: str
    R: dict[str, set[str]] = field(default_factory=dict)
    x: float = 0.0
    created_tau: int = 0

    def modality_partition_ok(self) -> bool:
        seen: set[str] = set()
        for symbols in self.R.values():
            if seen & symbols:
                return False
            seen |= symbols
        return True


@dataclass
class SecondOrderSymbol:
    """m = ⟨UID, Pr, Mt⟩"""

    uid: str
    Pr: list[Property] = field(default_factory=list)
    Mt: list[Property] = field(default_factory=list)
    x: float = 0.0
    created_tau: int = 0


@dataclass
class FunctionalSymbol:
    """g = ⟨UID, ID, {e_i*}⟩"""

    uid: str
    id: str
    operands: list[ElementRef] = field(default_factory=list)
    x: float = 0.0
    created_tau: int = 0


@dataclass
class ElementList:
    """k = ⟨UID, {e_i*}, Pr, Mt⟩"""

    uid: str
    items: list[ElementRef] = field(default_factory=list)
    Pr: list[Property] = field(default_factory=list)
    Mt: list[Property] = field(default_factory=list)
    x: float = 0.0
    created_tau: int = 0


@dataclass
class ActantSlot:
    role: Role
    filler: ElementRef | None = None


@dataclass
class Template:
    """T = ⟨UID, s*, A⟩"""

    uid: str
    predicate: SRef
    actants: list[ActantSlot] = field(default_factory=list)
    x: float = 0.0
    created_tau: int = 0


@dataclass
class Hyperlink:
    """N = ⟨UID, w, t*, {fillers}, Pr, Mt⟩ — fact = filled control model."""

    uid: str
    w: float
    template: ElementRef
    fillers: dict[Role, ElementRef] = field(default_factory=dict)
    Pr: list[Property] = field(default_factory=list)
    Mt: list[Property] = field(default_factory=list)
    x: float = 0.0
    created_tau: int = 0


@dataclass
class AssocLink:
    """l = ⟨UID, ID, w, (e1*, e2*)⟩"""

    uid: str
    id: str
    w: float
    e1: ElementRef
    e2: ElementRef


HyperElement = Union[
    SecondOrderSymbol,
    FunctionalSymbol,
    ElementList,
    Template,
    Hyperlink,
    SRef,
    MRef,
]


@dataclass
class AH:
    """AH = ⟨S, C, P, H, L⟩"""

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
