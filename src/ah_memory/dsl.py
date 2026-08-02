"""Minimal DSL interpreter over table-3 operations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ah_memory.store import AHError, AHStore
from ah_memory.types import Role, Section


@dataclass
class DSLResult:
    op: str
    value: Any


class DSLInterpreter:
    """
    Expressions:
      findRoles(ROLE, VALUE)
      findLists(kind=Episode)
      getSymbol(UID)
      getHypernode(UID)
      findLinks(UID)
      findSymbols(query)
      intersect(expr, expr)
      answer_who(UID)   # syntactic helper
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
            nodes = self.store.find_roles(role, value)
            return DSLResult(op, [n.uid for n in nodes])
        if op == "findLists":
            kind = None
            if args:
                kind = args[0].split("=", 1)[-1]
            return DSLResult(op, [x.uid for x in self.store.find_lists(kind)])
        if op == "getSymbol":
            return DSLResult(op, self.store.get_symbol(args[0]).uid)
        if op == "getHypernode":
            return DSLResult(op, self.store.get_hypernode(args[0]).uid)
        if op == "findLinks":
            return DSLResult(op, [l.uid for l in self.store.find_links(args[0])])
        if op == "findSymbols":
            q = args[0] if args else ""
            return DSLResult(op, [s.uid for s in self.store.find_symbols(q)])
        if op == "findHypernodes":
            return DSLResult(op, [n.uid for n in self.store.find_hypernodes()])
        if op == "answer_who":
            return DSLResult(op, self._answer_who(args[0]))
        raise AHError(f"unknown DSL op: {op}")

    def _answer_who(self, subject_m: str) -> str:
        labels: list[str] = []
        for n in self.store.find_roles(Role.SUBJECT, subject_m):
            try:
                tpl = self.store.get_template(n.template.target_uid)
            except AHError:
                continue
            if tpl.predicate.target_uid != "IS":
                continue
            obj = n.fillers.get(Role.OBJECT)
            if not obj:
                continue
            try:
                m = self.store.get_symbol(obj.target_uid)
            except AHError:
                continue
            for p in m.Pr:
                if p.name == "label":
                    labels.append(p.value.lower())
        # IS-A parents (oriented e1 → e2)
        for link in self.store.find_links(subject_m):
            if link.id != "IS-A":
                continue
            if link.e1.target_uid != subject_m:
                continue
            parent = link.e2.target_uid
            child_ans = self._answer_who(parent)
            labels.extend(child_ans.split() if child_ans != "неизвестно" else [])
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
