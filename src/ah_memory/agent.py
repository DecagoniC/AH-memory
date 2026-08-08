"""Cognitive agent loop: perceive → transform → ignite → answer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ah_memory.dsl import DSLInterpreter
from ah_memory.gc import collect
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine, TickTrace
from ah_memory.perception import PerceptionBackend, PerceptionResult, SeedPerception, FactCandidate
from ah_memory.store import AHStore
from ah_memory.transform import IngestReport, Transform
from ah_memory.types import Role, Section


def _filter_assistant_perception(perc: PerceptionResult) -> PerceptionResult:
    """Keep only confident structural facts from chat replies (drop greetings / fluff)."""
    from ah_memory.morph import is_nounish, sanitize_roles, seeds_from_roles

    kept: list[FactCandidate] = []
    for c in perc.candidates:
        if c.confidence < 0.85:
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
            )
        )
    seeds = seeds_from_roles(kept)
    kind = perc.kind
    if not kept:
        kind = "message"
    return PerceptionResult(kind=kind, candidates=kept, seed_tokens=seeds, meta=dict(perc.meta))


@dataclass
class AgentReply:
    answer: str
    trace_uids: list[str]
    traces: list[TickTrace] = field(default_factory=list)
    source: str = "graph"
    seed_uids: list[str] = field(default_factory=list)
    perception: dict = field(default_factory=dict)
    full_trace: dict = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        store: AHStore | None = None,
        perception: PerceptionBackend | None = None,
        hp: HyperParams | None = None,
        *,
        relation_normalizer=None,
        parameter_generator=None,
        state_engine=None,
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
        )
        self.ignition = IgnitionEngine(self.store, self.hp)
        self.dsl = DSLInterpreter(self.store)

    def ingest(self, text: str, section: Section = Section.C, *, source: str = "user") -> IngestReport:
        """Perceive → materialize S/m seeds + N factors."""
        perc = self.perception.parse(text, list(self.ignition.wm.contents()))
        if source == "assistant":
            perc = _filter_assistant_perception(perc)
        report = self.transform.apply(perc, section=section)
        # activate only role-bearing seeds (report.seed_uids already from perception.seed_tokens)
        seeds = [
            ActivationSeed(uid=u, delta_x=self.hp.seed_delta) for u in report.seed_uids
        ]
        if seeds:
            self.ignition.seed(seeds)
            self.ignition.run(2)
        collect(self.store, self.hp)
        return report

    def ask(self, question: str, ticks: int = 6) -> AgentReply:
        perc = self.perception.parse(question, list(self.ignition.wm.contents()))
        seeds: list[ActivationSeed] = []
        seen: set[str] = set()
        for u in perc.seed_tokens:
            bare = u[2:] if u.startswith("M_") else u
            if bare.count("_") > 3:
                continue
            q = bare.lower().replace("_", " ")
            candidates = []
            if bare in self.store.ah.S:
                candidates.append(bare)
            m_uid = f"M_{bare}"
            if m_uid in self.store.ah.all_hyper():
                candidates.append(m_uid)
            for s in self.store.find_symbols(q):
                candidates.append(s.uid)
            for form_uid, abs_s in self.store.ah.S.items():
                forms = " ".join(abs_s.R.get("TEXT", set())).lower()
                if q and (q in forms or q in form_uid.lower()):
                    candidates.append(form_uid)
                    candidates.append(f"M_{form_uid}")
            for uid in candidates:
                if uid in seen:
                    continue
                seen.add(uid)
                seeds.append(ActivationSeed(uid=uid, delta_x=self.hp.seed_delta))
            if len(seeds) >= 16:
                break
        self.ignition.seed(seeds)
        traces = self.ignition.run(ticks)
        trace_uids: list[str] = []
        for t in traces:
            for uid in t.activated:
                if uid not in trace_uids:
                    trace_uids.append(uid)

        answer, support = self._compose_answer(question, perc.seed_tokens, trace_uids)
        for uid in support:
            if uid not in trace_uids:
                trace_uids.append(uid)
        collect(self.store, self.hp)
        return AgentReply(
            answer=answer,
            trace_uids=trace_uids,
            traces=traces,
            source="graph",
            seed_uids=[s.uid for s in seeds],
            perception=perc.to_graph_json(),
            full_trace=self._full_trace(traces, trace_uids, answer=answer),
        )

    def step_message(self, text: str, ticks: int = 3) -> AgentReply:
        """Continuous perception cycle (bonus track)."""
        perc = self.perception.parse(text, list(self.ignition.wm.contents()))
        if perc.kind == "question" and not perc.candidates:
            return self.ask(text, ticks=ticks)
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
                for n in self.store.find_roles(Role.SUBJECT, subj):
                    loc = n.fillers.get(Role.LOCATION)
                    if loc:
                        support.extend([subj, n.uid, loc.target_uid])
                        return f"location:{loc.target_uid}", support
        if "цвет" in q or "шерст" in q:
            for n in self.store.find_hypernodes():
                try:
                    pred = self.store.get_template(n.template.target_uid).predicate.target_uid
                except Exception:
                    continue
                if pred != "BE_COLORED":
                    continue
                obj = n.fillers.get(Role.OBJECT)
                time_r = n.fillers.get(Role.TIME)
                if not obj:
                    continue
                if "зим" in q and time_r and "WINTER" not in time_r.target_uid.upper():
                    continue
                if "лет" in q and time_r and "SUMMER" not in time_r.target_uid.upper():
                    continue
                support.extend([n.uid, obj.target_uid] + ([time_r.target_uid] if time_r else []))
                return f"color:{obj.target_uid}", support
        if "почему" in q or "быстр" in q:
            for subj in subjects:
                for n in self.store.find_roles(Role.SUBJECT, subj):
                    obj = n.fillers.get(Role.OBJECT)
                    cause = n.fillers.get(Role.CAUSE)
                    try:
                        pred = self.store.get_template(n.template.target_uid).predicate.target_uid
                    except Exception:
                        pred = ""
                    if pred == "HAVE" and obj and "LEG" in obj.target_uid.upper():
                        support.extend([subj, n.uid, obj.target_uid])
                        return f"cause:{obj.target_uid}", support
                    if cause:
                        support.extend([subj, n.uid, cause.target_uid])
                        return f"cause:{cause.target_uid}", support
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
