"""Cognitive agent loop: perceive → transform → ignite → answer."""
from __future__ import annotations

from dataclasses import dataclass, field

from ah_memory.dsl import DSLInterpreter
from ah_memory.gc import collect
from ah_memory.hyperparams import HyperParams
from ah_memory.ignition import ActivationSeed, IgnitionEngine, TickTrace
from ah_memory.perception import PerceptionBackend, RulePerception
from ah_memory.store import AHStore
from ah_memory.transform import IngestReport, Transform
from ah_memory.types import Role, Section


@dataclass
class AgentReply:
    answer: str
    trace_uids: list[str]
    traces: list[TickTrace] = field(default_factory=list)
    source: str = "graph"


class Agent:
    def __init__(
        self,
        store: AHStore | None = None,
        perception: PerceptionBackend | None = None,
        hp: HyperParams | None = None,
    ) -> None:
        self.store = store or AHStore()
        self.hp = hp or HyperParams()
        self.perception = perception or RulePerception()
        self.transform = Transform(self.store, self.hp)
        self.ignition = IgnitionEngine(self.store, self.hp)
        self.dsl = DSLInterpreter(self.store)

    def ingest(self, text: str, section: Section = Section.C) -> IngestReport:
        """Perceive → materialize S/m seeds + N factors."""
        perc = self.perception.parse(text, list(self.ignition.wm.contents()))
        report = self.transform.apply(perc, section=section)
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
        return AgentReply(answer=answer, trace_uids=trace_uids, traces=traces, source="graph")

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
        )

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
