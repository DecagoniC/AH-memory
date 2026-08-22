"""Agent: perceive → transform → ignite → (опц.) ответить из графа.

Оркестратор без LLM-диалога (диалог — dialogue.DialogueAgent поверх Agent).
Читать после transform/identity; ignition — чёрный ящик «посеяли UID → тики».
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Callable, Sequence

from ah_memory.dsl import DSLInterpreter
from ah_memory.gc import collect
from ah_memory.graph_query import GraphQueryPlanner, QueryPlan
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine, TickTrace
from ah_memory.perception import PerceptionBackend, PerceptionResult, SeedPerception, FactCandidate
from ah_memory.store import AHStore
from ah_memory.transform import IngestReport, Transform
from ah_memory.types import Section


def _filter_assistant_perception(perc: PerceptionResult) -> PerceptionResult:
    """Из ответа ассистента — только уверенные proposals/explanations."""
    from ah_memory.morph import is_nounish, sanitize_roles, seeds_from_roles

    kept: list[FactCandidate] = []
    for c in perc.candidates:
        if c.confidence < 0.75 or c.statement_type not in {"proposal", "explanation"}:
            continue
        roles = sanitize_roles(c.roles)
        if roles is None:
            continue
        # OBJECT/LOCATION must stay nounish (evaluative adjectives already rejected in sanitize)
        for role in ("OBJECT", "LOCATION"):
            val = roles.get(role)
            if val and not any(is_nounish(p, allow_pronoun=False) for p in val.lower().split("_")):
                roles = None
                break
        if roles is None:
            continue
        kept.append(
            FactCandidate(
                c.predicate,
                roles,
                c.raw_span,
                c.confidence,
                raw_relation=c.raw_relation,
                canonical_relation=c.canonical_relation,
                statement_type=c.statement_type,
                source="assistant",
            )
        )
    seeds = seeds_from_roles(kept)
    kind = perc.kind
    if not kept:
        kind = "message"
    return PerceptionResult(kind=kind, candidates=kept, seed_tokens=seeds, meta=dict(perc.meta))


@dataclass
class AgentReply:
    """Ответ ask/step + трасса активации для UI."""

    answer: str
    trace_uids: list[str]
    traces: list[TickTrace] = field(default_factory=list)
    source: str = "graph"
    seed_uids: list[str] = field(default_factory=list)
    perception: dict = field(default_factory=dict)
    full_trace: dict = field(default_factory=dict)


class Agent:
    # ── Сборка пайплайна ─────────────────────────────────────────────────────
    # perception → Transform(+identity) → IgnitionEngine; dsl — эвристики ответа.

    def __init__(
        self,
        store: AHStore | None = None,
        perception: PerceptionBackend | None = None,
        hp: HyperParams | None = None,
        *,
        relation_normalizer=None,
        parameter_generator=None,
        state_engine=None,
        identity=None,
        query_embed: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self.store = store or AHStore()
        self.hp = hp or HyperParams()
        self.perception = perception or SeedPerception()
        self.transform = Transform(
            self.store,
            self.hp,
            relation_normalizer=relation_normalizer,
            parameter_generator=parameter_generator,
            state_engine=state_engine,
            identity=identity,
        )
        self.ignition = IgnitionEngine(self.store, self.hp)
        self.dsl = DSLInterpreter(self.store)
        self.query_planner = GraphQueryPlanner(
            self.store,
            embed=query_embed,
        )

    def adopt_store(self, store: AHStore) -> None:
        """Swap graph contents in place and rebuild ignition over the new structure."""
        if store is not self.store:
            self.store.replace_from(store)
        self.transform.store = self.store
        self.transform.relation_normalizer.registry = self.store.relations
        if self.transform.identity is not None:
            self.transform.identity.store = self.store
        self.ignition = IgnitionEngine(self.store, self.hp)
        self.dsl = DSLInterpreter(self.store)
        self.query_planner.bind_store(self.store)

    def ingest(
        self,
        text: str,
        section: Section = Section.C,
        *,
        source: str = "user",
        perception: PerceptionResult | None = None,
    ) -> IngestReport:
        """Текст → граф (+ короткий прогон активации по новым seeds)."""
        if perception is None:
            perc = self.perception.parse(text, list(self.ignition.wm.contents()))
        else:
            perc = perception
        if source == "assistant":
            perc = _filter_assistant_perception(perc)
        elif perc.kind == "question":
            # Keep explicit factual clauses in mixed utterances, never the query itself.
            from ah_memory.morph import seeds_from_roles

            factual = [
                candidate
                for candidate in perc.candidates
                if candidate.statement_type not in {"topic", "open_question"}
            ]
            perc = PerceptionResult(
                kind="message",
                candidates=factual,
                seed_tokens=seeds_from_roles(factual),
                meta={**perc.meta, "not_persisted": "question"},
            )
        report = self.transform.apply(perc, section=section)
        # Зачем: новые символы сразу «подогреть», чтобы следующий ask их видел в WM.
        seeds = [
            ActivationSeed(uid=u, delta_x=self.hp.seed_delta) for u in report.seed_uids
        ]
        if seeds:
            self.ignition.seed(seeds)
            self.ignition.run(2)
        collect(self.store, self.hp)
        return report

    def ask(
        self,
        question: str,
        ticks: int = 6,
        *,
        perception: PerceptionResult | None = None,
    ) -> AgentReply:
        # Зачем: не писать вопрос как факт, а найти UID по seeds → ignition → compose.
        if perception is None:
            perc = self.perception.parse(question, list(self.ignition.wm.contents()))
        else:
            perc = perception
        entry_uids = self._resolve_entry_uids(
            perc.seed_tokens,
            text=question,
        )
        seeds = [
            ActivationSeed(uid=uid, delta_x=self.hp.seed_delta)
            for uid in entry_uids
        ]
        plan = self.query_planner.plan(question, perc, entry_uids)
        if plan.factor_scores:
            self.ignition.set_factor_gates(
                plan.factor_scores,
                reset_state=True,
            )
        else:
            self.ignition.set_factor_gates(None)
        self.ignition.seed(seeds)
        traces = self.ignition.run(ticks)
        trace_uids: list[str] = []
        for t in traces:
            for uid in t.activated:
                if uid not in trace_uids:
                    trace_uids.append(uid)

        answer, support = self._answer_from_plan(plan)
        if answer == "неизвестно":
            answer, support = self._compose_answer(
                question,
                perc.seed_tokens,
                trace_uids,
            )
        self.ignition.set_factor_gates(None)
        for uid in support:
            if uid not in trace_uids:
                trace_uids.append(uid)
        collect(self.store, self.hp)
        full_trace = self._full_trace(traces, trace_uids, answer=answer)
        full_trace["query_plan"] = {
            "anchors": list(plan.anchors),
            "relation_text": plan.relation_text,
            "target_role": plan.target_role,
            "cardinality": plan.cardinality,
            "relation_scores": dict(plan.relation_scores),
            "factor_gates": dict(plan.factor_scores),
        }
        return AgentReply(
            answer=answer,
            trace_uids=trace_uids,
            traces=traces,
            source="graph",
            seed_uids=[s.uid for s in seeds],
            perception=perc.to_graph_json(),
            full_trace=full_trace,
        )

    def _answer_from_plan(self, plan: QueryPlan) -> tuple[str, list[str]]:
        if not plan.factor_scores:
            return "неизвестно", []
        return self.query_planner.decode(plan)

    def _resolve_entry_uids(
        self,
        tokens: list[str],
        *,
        text: str = "",
    ) -> list[str]:
        mentions = self._resolve_text_mentions(text)
        if mentions:
            return mentions
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            bare = token[2:] if token.startswith("M_") else token
            if bare.count("_") > 3:
                continue
            query = bare.lower().replace("_", " ")
            candidates: list[str] = []
            if bare in self.store.ah.S:
                candidates.append(bare)
            m_uid = f"M_{bare}"
            if m_uid in self.store.ah.all_hyper():
                candidates.append(m_uid)
            candidates.extend(symbol.uid for symbol in self.store.find_symbols(query))
            for form_uid, abstract in self.store.ah.S.items():
                forms = " ".join(abstract.R.get("TEXT", set())).lower()
                if query and (query in forms or query in form_uid.lower()):
                    candidates.extend([form_uid, f"M_{form_uid}"])
            for uid in candidates:
                if uid in seen:
                    continue
                if uid not in self.store.ah.S and uid not in self.store.ah.all_hyper():
                    continue
                seen.add(uid)
                out.append(uid)
            if len(out) >= 16:
                break
        return out[:16]

    def _resolve_text_mentions(self, text: str) -> list[str]:
        """Resolve longest entity phrases before broad token lookup."""
        from ah_memory.morph import lemma

        text_lemmas = [
            normalized
            for token in re.findall(r"[a-zа-яё0-9]+", text.lower())
            if (normalized := lemma(token))
        ]
        if not text_lemmas:
            return []
        matches: list[tuple[int, int, str]] = []
        for bare, abstract in self.store.ah.S.items():
            surfaces = set(abstract.R.get("TEXT") or set())
            m_uid = f"M_{bare}"
            if m_uid in self.store.ah.all_hyper():
                try:
                    symbol = self.store.get_symbol(m_uid)
                    surfaces.update(
                        prop.value
                        for prop in symbol.Pr
                        if prop.name == "label" and prop.value
                    )
                except Exception:
                    pass
            for surface in surfaces:
                phrase = [
                    normalized
                    for token in re.findall(
                        r"[a-zа-яё0-9]+",
                        surface.lower(),
                    )
                    if (normalized := lemma(token))
                ]
                if len(phrase) < 2 or len(phrase) > len(text_lemmas):
                    continue
                width = len(phrase)
                for start in range(len(text_lemmas) - width + 1):
                    if text_lemmas[start : start + width] == phrase:
                        matches.append((start, start + width, bare))
        matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
        occupied: set[int] = set()
        selected: list[str] = []
        for start, end, bare in matches:
            span = set(range(start, end))
            if occupied.intersection(span):
                continue
            occupied.update(span)
            m_uid = f"M_{bare}"
            if m_uid in self.store.ah.all_hyper():
                selected.append(m_uid)
            if bare in self.store.ah.S:
                selected.append(bare)
            if len(selected) >= 8:
                break
        return list(dict.fromkeys(selected))

    def step_message(self, text: str, ticks: int = 3) -> AgentReply:
        """Один шаг цикла: вопрос → ask, иначе → ingest."""
        perc = self.perception.parse(text, list(self.ignition.wm.contents()))
        if perc.kind == "question":
            return self.ask(text, ticks=ticks, perception=perc)
        report = self.ingest(text)
        wm = list(self.ignition.wm.contents())
        return AgentReply(
            answer=f"ingested:{len(report.created_n)} wm:{len(wm)}",
            trace_uids=wm,
            traces=self.ignition.traces[-ticks:],
            source="graph",
            full_trace=self._full_trace(
                self.ignition.traces[-ticks:],
                wm,
                answer=f"ingested:{len(report.created_n)} wm:{len(wm)}",
            ),
        )

    def _full_trace(
        self,
        traces: list[TickTrace],
        activated_nodes: list[str],
        *,
        answer: str = "",
    ) -> dict:
        # Зачем: единый JSON для web/dialogue — узлы, факторы, events, state.
        requested = set(activated_nodes)
        semantic_factor_ids = set(self.store.semantic_factors)
        actual_nodes = [
            uid for uid in activated_nodes if uid not in semantic_factor_ids
        ]
        activated_set = set(actual_nodes)
        factor_uids = list(
            dict.fromkeys(
                factor_uid
                for trace in traces
                for factor_uid in trace.trace_factors
            )
        )
        semantic_factors = [
            factor
            for factor in self.store.list_semantic_factors()
            if factor.uid in requested
            or activated_set.intersection(factor.variables)
        ]
        relation_data = []
        seen_relations: set[str] = set()
        for factor in semantic_factors:
            if factor.relation is None:
                continue
            canonical = factor.relation.canonical_label
            if canonical in seen_relations:
                continue
            seen_relations.add(canonical)
            relation_data.append(factor.relation.to_dict())
        relevant_events = [
            event.to_dict()
            for event in self.store.list_events()
            if activated_set.intersection(
                reference.uid for reference in event.arguments.values()
            )
        ]
        return {
            "answer": answer,
            "activated_nodes": actual_nodes,
            "activated_factors": list(
                dict.fromkeys(
                    factor_uids
                    + [factor.uid for factor in semantic_factors]
                )
            ),
            "timesteps": [
                {
                    "tau": trace.tau,
                    "activation": dict(trace.activation_top),
                    "messages": [
                        asdict(event) for event in trace.events
                    ],
                }
                for trace in traces
            ],
            "relations": relation_data,
            "events": relevant_events,
            "state": self.store.state.to_dict(),
            "state_transitions": list(self.store.state_transitions),
            "final_evidence": actual_nodes,
        }

    def _compose_answer(
        self, question: str, seeds: list[str], trace: list[str]
    ) -> tuple[str, list[str]]:
        # Эвристический ответ без LLM (кто/где/роли факторов + labels активированных M).
        q = question.lower()
        subjects = self._resolve_subjects(seeds + trace)
        support: list[str] = []
        if "кто" in q or "что" in q:
            for subj in subjects:
                ans = self.dsl.execute(f"answer_who({subj})").value
                if ans != "неизвестно":
                    support.extend([subj, *trace[:4]])
                    return str(ans), support
        if any(w in q for w in ("сколько", "когда родился", "на луне")):
            return "неизвестно", []
        if "где" in q or "обита" in q:
            for subj in subjects:
                for factor in self.store.list_semantic_factors():
                    if factor.roles.get("SUBJECT") != subj:
                        continue
                    loc = factor.roles.get("LOCATION")
                    if loc:
                        support.extend([subj, factor.uid, loc])
                        return f"location:{loc}", support
        if "цвет" in q or "шерст" in q:
            for factor in self.store.list_semantic_factors():
                pred = (
                    factor.relation.canonical_label.upper()
                    if factor.relation is not None
                    else ""
                )
                if pred not in {"BE_COLORED", "COLORED", "HAS_COLOR"}:
                    continue
                obj = factor.roles.get("OBJECT")
                time_r = factor.roles.get("TIME")
                if not obj:
                    continue
                if "зим" in q and time_r and "WINTER" not in time_r.upper():
                    continue
                if "лет" in q and time_r and "SUMMER" not in time_r.upper():
                    continue
                support.extend([factor.uid, obj] + ([time_r] if time_r else []))
                return f"color:{obj}", support
        if "почему" in q or "быстр" in q:
            for subj in subjects:
                for factor in self.store.list_semantic_factors():
                    if factor.roles.get("SUBJECT") != subj:
                        continue
                    obj = factor.roles.get("OBJECT")
                    cause = factor.roles.get("CAUSE")
                    how = factor.roles.get("HOW-TO")
                    pred = (
                        factor.relation.canonical_label.upper()
                        if factor.relation is not None
                        else ""
                    )
                    if cause:
                        support.extend([subj, factor.uid, cause])
                        return f"cause:{cause}", support
                    if how:
                        support.extend([subj, factor.uid, how])
                        return f"cause:{how}", support
                    if pred == "HAVE" and obj:
                        support.extend([subj, factor.uid, obj])
                        return f"cause:{obj}", support
        labels: list[str] = []
        for uid in trace:
            try:
                m = self.store.get_symbol(uid)
            except Exception:
                continue
            for p in m.Pr:
                if p.name == "label":
                    labels.append(p.value)
                    support.append(uid)
        if labels:
            return " ".join(labels[:8]), support
        if trace:
            return "activated:" + ",".join(trace[:12]), list(trace[:12])
        return "неизвестно", []

    def _resolve_subjects(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        for t in tokens:
            bare = t[2:] if t.startswith("M_") else t
            m_uid = f"M_{bare}"
            if m_uid not in out and (
                m_uid in self.store.ah.C
                or m_uid in self.store.ah.P
                or m_uid in self.store.ah.H
                or bare in self.store.ah.S
            ):
                out.append(m_uid)
            for s in self.store.find_symbols(bare.lower().replace("_", " ")):
                mu = f"M_{s.uid}"
                if mu not in out:
                    out.append(mu)
        return out[:8]
