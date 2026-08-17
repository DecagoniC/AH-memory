"""Minimal DSL interpreter over store operations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ah_memory.store import AHError, AHStore


@dataclass
class DSLResult:
    op: str
    value: Any


class DSLInterpreter:
    """
    Expressions:
      findRoles(ROLE, VALUE)     — semantic factors с данной ролью
      findSymbolsByKind(kind=Episode)
      getSymbol(UID)
      findLinks(UID)
      findSymbols(query)
      intersect(expr, expr)
      answer_who(UID)
    """

    def __init__(self, store: AHStore) -> None:
        self.store = store

    def execute(self, source: str) -> DSLResult:
        src = source.strip()
        if src.startswith("intersect("):
            left, right = self._split_args(src[len("intersect(") : -1])
            a = self.execute(left).value
            b = self.execute(right).value
            return DSLResult("intersect", sorted(set(a) & set(b)))

        m = re.match(r"(\w+)\((.*)\)$", src, re.DOTALL)
        if not m:
            raise AHError(f"bad DSL: {source}")
        op, args_raw = m.group(1), m.group(2).strip()
        args = [a.strip().strip("'\"") for a in self._split_top(args_raw)] if args_raw else []

        if op == "findRoles":
            role, value = args[0], args[1]
            role_u = role.upper()
            uids = [
                factor.uid
                for factor in self.store.list_semantic_factors()
                if factor.roles.get(role_u) == value
            ]
            return DSLResult(op, list(dict.fromkeys(uids)))
        if op in {"findLists", "findSymbolsByKind"}:
            kind = None
            if args:
                kind = args[0].split("=", 1)[-1]
            return DSLResult(op, [x.uid for x in self.store.find_symbols_by_kind(kind)])
        if op == "getSymbol":
            return DSLResult(op, self.store.get_symbol(args[0]).uid)
        if op == "findLinks":
            return DSLResult(op, [l.uid for l in self.store.find_links(args[0])])
        if op == "findSymbols":
            q = args[0] if args else ""
            return DSLResult(op, [s.uid for s in self.store.find_symbols(q)])
        if op == "answer_who":
            return DSLResult(op, self._answer_who(args[0]))
        raise AHError(f"unknown DSL op: {op}")

    def _answer_who(self, subject_m: str) -> str:
        labels: list[str] = []
        for factor in self.store.list_semantic_factors():
            if factor.relation is None:
                continue
            pred = factor.relation.canonical_label.upper()
            if pred not in {"IS", "IS_A"}:
                continue
            if factor.roles.get("SUBJECT") != subject_m:
                continue
            obj = factor.roles.get("OBJECT")
            if not obj:
                continue
            try:
                m = self.store.get_symbol(obj)
            except AHError:
                continue
            for p in m.Pr:
                if p.name == "label":
                    labels.append(p.value.lower())
        for link in self.store.find_links(subject_m):
            if link.id != "IS-A":
                continue
            if link.e1.target_uid != subject_m:
                continue
            parent = link.e2.target_uid
            child_ans = self._answer_who(parent)
            labels.extend(child_ans.split() if child_ans != "неизвестно" else [])
            try:
                m = self.store.get_symbol(parent)
                for p in m.Pr:
                    if p.name == "label":
                        labels.append(p.value.lower())
            except AHError:
                pass
        seen: set[str] = set()
        ordered = []
        for x in labels:
            if x and x not in seen and x != "неизвестно":
                seen.add(x)
                ordered.append(x)
        return " ".join(ordered) if ordered else "неизвестно"

    @staticmethod
    def _split_top(args_raw: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in args_raw:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf))
        return parts

    def _split_args(self, inner: str) -> tuple[str, str]:
        parts = self._split_top(inner)
        if len(parts) != 2:
            raise AHError("intersect expects 2 args")
        return parts[0], parts[1]
