"""Query-conditioned ranking for compact graph context."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Iterable, Mapping, TYPE_CHECKING

from ah_memory.morph import STOP, humanize_symbol_text, lemma
from ah_memory.store import AHStore

if TYPE_CHECKING:
    from ah_memory.factor_graph import Factor
    from ah_memory.ignition import WorkingMemoryEntry


_TOKEN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def semantic_terms(text: str) -> set[str]:
    """Return generic normalized content terms without domain vocabulary."""
    terms: set[str] = set()
    for token in _TOKEN.findall(text.lower()):
        normalized = lemma(token)
        if normalized and normalized not in STOP and len(normalized) > 1:
            terms.add(normalized)
    return terms


def lexical_relevance(query: str, document: str) -> float:
    query_terms = semantic_terms(query)
    document_terms = semantic_terms(document)
    if not query_terms or not document_terms:
        return 0.0
    overlap = len(query_terms.intersection(document_terms))
    if not overlap:
        return 0.0
    return min(
        1.0,
        overlap / math.sqrt(len(query_terms) * len(document_terms)),
    )


def _slug_like(text: str) -> bool:
    value = str(text).strip()
    return bool(
        not value
        or value.startswith("M_")
        or "_" in value
        or (value.isupper() and " " not in value)
    )


def symbol_label(store: AHStore, uid: str, *, compact: bool = False) -> str:
    """Resolve a stable human-readable label for a graph variable."""
    raw = str(uid or "").strip()
    if not raw:
        return ""
    m_uid = raw if raw.startswith("M_") else f"M_{raw}"
    bare = m_uid[2:]
    try:
        symbol = store.get_symbol(m_uid)
    except Exception:
        symbol = None
    forms: list[str] = []
    if symbol is not None:
        forms.extend(
            str(prop.value).strip()
            for prop in symbol.Pr
            if prop.name == "label" and str(prop.value).strip()
        )
    abstract = store.ah.S.get(bare)
    if abstract is not None:
        forms.extend(
            str(value).strip()
            for value in abstract.R.get("TEXT", set())
            if str(value).strip()
        )
    ranked = sorted(
        dict.fromkeys(forms),
        key=lambda item: (_slug_like(item), -len(item), item.lower()),
    )
    chosen = ranked[0] if ranked else bare
    return humanize_symbol_text(chosen, uid=m_uid, compact=compact) or chosen


@dataclass(frozen=True)
class RankedSymbol:
    uid: str
    label: str
    score: float
    activation: float
    lexical: float
    gate: float
    support: int

    def prompt_card(self) -> str:
        """Provider-neutral compact card; omit internal UIDs from the prompt."""
        return json.dumps(
            {
                "entity": self.label,
                "relevance": round(self.score, 4),
                "activation": round(self.activation, 4),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class GraphContextRanker:
    """Rank symbols and factors using only generic structural signals."""

    def __init__(self, store: AHStore) -> None:
        self.store = store

    def rank_symbols(
        self,
        query: str,
        entries: Iterable["WorkingMemoryEntry"],
        *,
        factor_gates: Mapping[str, float] | None = None,
        limit: int = 12,
    ) -> list[RankedSymbol]:
        if limit <= 0:
            return []
        symbol_gates = self._symbol_gates(factor_gates or {})
        ranked: list[RankedSymbol] = []
        for entry in entries:
            label = symbol_label(self.store, entry.uid)
            lexical = lexical_relevance(query, label)
            activation = min(1.0, max(0.0, float(entry.activation)))
            gate = symbol_gates.get(entry.uid, 0.0)
            support = len(entry.support)
            support_score = min(1.0, support / 4.0)
            score = (
                0.55 * lexical
                + 0.30 * activation
                + 0.10 * gate
                + 0.05 * support_score
            )
            ranked.append(
                RankedSymbol(
                    uid=entry.uid,
                    label=label,
                    score=score,
                    activation=activation,
                    lexical=lexical,
                    gate=gate,
                    support=support,
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.lexical,
                -item.activation,
                item.label,
                item.uid,
            )
        )
        return ranked[:limit]

    def factor_score(
        self,
        factor: "Factor",
        query: str,
        *,
        activation: Mapping[str, float] | None = None,
        factor_gates: Mapping[str, float] | None = None,
    ) -> float:
        relation = (
            factor.relation.canonical_label
            if factor.relation is not None
            else ""
        )
        meta = factor.metadata or {}
        span = str(meta.get("raw_span") or "").strip()
        if not span:
            event = self.store.events.get(str(meta.get("event_uid") or ""))
            if event is not None:
                span = str(event.raw_span or "").strip()
        document = " ".join(
            [
                relation.replace("_", " "),
                span,
                *(
                    symbol_label(self.store, uid)
                    for uid in factor.roles.values()
                ),
            ]
        )
        lexical = lexical_relevance(query, document)
        active = max(
            (
                float((activation or {}).get(uid, 0.0))
                for uid in factor.variables
            ),
            default=0.0,
        )
        gate = float((factor_gates or {}).get(factor.uid, 0.0))
        authoritative = (
            1.0
            if (factor.metadata or {}).get("statement_type")
            in {"assertion", "decision"}
            else 0.0
        )
        return (
            0.50 * lexical
            + 0.30 * min(1.0, max(0.0, gate))
            + 0.15 * min(1.0, max(0.0, active))
            + 0.05 * authoritative
        )

    def _symbol_gates(
        self,
        factor_gates: Mapping[str, float],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for factor_uid, raw_score in factor_gates.items():
            factor = self.store.semantic_factors.get(factor_uid)
            if factor is None:
                continue
            score = min(1.0, max(0.0, float(raw_score)))
            for uid in factor.variables:
                scores[uid] = max(scores.get(uid, 0.0), score)
        return scores
