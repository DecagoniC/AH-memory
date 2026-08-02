"""LLM dialogue: both user and assistant turns are ingested into AH."""
from __future__ import annotations

from dataclasses import dataclass, field

from ah_memory.config import DeepSeekConfig
from ah_memory.deepseek import DeepSeekClient
from ah_memory.ignition import TickTrace
from ah_memory.store import AHStore
from ah_memory.transform import IngestReport
from ah_memory.types import AssocLink, ElementList, Hyperlink, LinkId, Property, Section


DIALOGUE_SYSTEM = """Ты обычный полезный собеседник. Отвечай по-русски ясно и по делу.

У тебя есть фоновый контекст «АГ-память» — извлечённые ранее факты из диалога.
Используй их молча, если они помогают ответу. Если контекст пуст или нерелевантен —
просто отвечай из своих знаний, без оговорок про «память», «базу», UID и «не хватает данных».
Не спрашивай пользователя «записать ли факт в память».
Не используй JSON. Не перечисляй внутренние идентификаторы."""


def _tick_dict(t: TickTrace) -> dict:
    return {
        "tau": t.tau,
        "seeds": t.seeds_applied,
        "evidence": t.evidence,
        "beliefs_top": t.beliefs_top,
        "activated": t.activated,
        "wm": t.wm,
        "trace_factors": t.trace_factors,
        "weight_updates": t.weight_updates,
        "stats": t.z_stats,
    }


@dataclass
class TurnResult:
    reply: str
    user_facts: list[str] = field(default_factory=list)
    assistant_facts: list[str] = field(default_factory=list)
    trace_uids: list[str] = field(default_factory=list)
    wm: list[str] = field(default_factory=list)
    backend: str = "deepseek"
    history_len: int = 0
    system_prompt: str = ""
    activation: dict = field(default_factory=dict)


class DialogueAgent:
    """Wraps Agent: chat with LLM + ingest both sides into AH."""

    def __init__(self, agent, deepseek: DeepSeekConfig | None = None) -> None:
        self.agent = agent
        self.cfg = deepseek
        self.client = DeepSeekClient(deepseek) if deepseek and deepseek.configured else None
        self.history: list[dict[str, str]] = []
        self._ep_prev: str | None = None
        self._turn = 0
        self.last_activation: dict = {}

    @property
    def store(self) -> AHStore:
        return self.agent.store

    def reset_history(self) -> None:
        self.history.clear()
        self._ep_prev = None
        self._turn = 0
        self.last_activation = {}

    def talk(self, user_text: str, ticks: int = 6) -> TurnResult:
        user_text = user_text.strip()
        self._turn += 1
        ign = self.agent.ignition

        i0 = len(ign.traces)
        user_rep = self.agent.ingest(user_text, section=Section.H)
        user_ticks = [_tick_dict(t) for t in ign.traces[i0:]]
        self._record_episode("USER", user_text, user_rep)

        mem = self._memory_context(user_text)
        ask = self.agent.ask(user_text, ticks=ticks)
        graph_hint = ask.answer if ask.answer and ask.answer != "неизвестно" else ""

        if self.client is not None:
            reply, system_prompt = self._llm_reply(user_text, mem, graph_hint)
            backend = "deepseek+ah"
        else:
            reply, system_prompt = self._fallback_reply(user_text, mem, graph_hint)
            backend = "rules+ah"

        i1 = len(ign.traces)
        asst_rep = self.agent.ingest(reply, section=Section.H)
        asst_ticks = [_tick_dict(t) for t in ign.traces[i1:]]
        self._record_episode("ASSISTANT", reply, asst_rep)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > 24:
            self.history = self.history[-24:]

        wm = sorted(ign.wm.contents())
        trace = list(
            dict.fromkeys(
                ask.trace_uids + wm + user_rep.created_n + asst_rep.created_n + user_rep.seed_uids[:8]
            )
        )
        activation = {
            "turn": self._turn,
            "threshold_t": self.agent.hp.threshold_t,
            "user_ingest": {
                "created_n": user_rep.created_n,
                "seeds": user_rep.seed_uids[:24],
                "skipped": user_rep.skipped[:12],
                "ticks": user_ticks,
            },
            "ask": {
                "graph_hint": graph_hint,
                "seed_uids": ask.seed_uids[:24],
                "ticks": [_tick_dict(t) for t in ask.traces],
            },
            "assistant_ingest": {
                "created_n": asst_rep.created_n,
                "seeds": asst_rep.seed_uids[:24],
                "ticks": asst_ticks,
            },
            "final_wm": wm,
            "memory_brief": mem,
        }
        self.last_activation = activation
        return TurnResult(
            reply=reply,
            user_facts=user_rep.created_n,
            assistant_facts=asst_rep.created_n,
            trace_uids=trace,
            wm=wm,
            backend=backend,
            history_len=len(self.history),
            system_prompt=system_prompt,
            activation=activation,
        )

    def _compose_system_blocks(self, mem: str, graph_hint: str) -> list[str]:
        blocks = [DIALOGUE_SYSTEM]
        if mem or graph_hint:
            ctx_parts = ["[Контекст из АГ-памяти — учитывай по возможности, не цитируй как отчёт]"]
            if mem:
                ctx_parts.append(mem)
            if graph_hint:
                ctx_parts.append(f"Активировано: {graph_hint}")
            blocks.append("\n".join(ctx_parts))
        return blocks

    def _llm_reply(self, user_text: str, mem: str, graph_hint: str) -> tuple[str, str]:
        sys_blocks = self._compose_system_blocks(mem, graph_hint)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": b} for b in sys_blocks
        ]
        messages.extend(self.history[-12:])
        messages.append({"role": "user", "content": user_text})
        reply = self.client.chat(messages, json_mode=False).strip()
        prompt_view = "\n\n──── system ────\n\n".join(sys_blocks)
        return reply, prompt_view

    def _fallback_reply(self, user_text: str, mem: str, graph_hint: str) -> tuple[str, str]:
        sys_blocks = self._compose_system_blocks(mem, graph_hint)
        prompt_view = "\n\n──── system ────\n\n".join(sys_blocks)
        prompt_view += "\n\n──── mode ────\n\nrules fallback (LLM недоступен)"
        if graph_hint and graph_hint != "неизвестно":
            return graph_hint, prompt_view
        if mem:
            return (mem.split("\n", 1)[-1] if "\n" in mem else mem), prompt_view
        return "Хорошо. Можешь уточнить вопрос или рассказать больше — отвечу.", prompt_view

    def _label(self, uid: str) -> str:
        bare = uid[2:] if uid.startswith("M_") else uid
        try:
            m = self.store.get_symbol(uid if uid.startswith("M_") else f"M_{bare}")
            for p in m.Pr:
                if p.name == "label" and p.value:
                    return p.value
        except Exception:
            pass
        if bare in self.store.ah.S:
            forms = self.store.ah.S[bare].R.get("TEXT") or set()
            if forms:
                return next(iter(forms))
        return bare.replace("_", " ").lower()

    def _fmt_fact(self, n: Hyperlink) -> str | None:
        try:
            tpl = self.store.get_template(n.template.target_uid)
            pred = tpl.predicate.target_uid
        except Exception:
            return None
        roles = {r.value: self._label(f.target_uid) for r, f in n.fillers.items()}
        subj = roles.get("SUBJECT", "?")
        if pred == "IS" and "OBJECT" in roles:
            return f"{subj} — {roles['OBJECT']}"
        if pred == "LIVE_IN" and "LOCATION" in roles:
            return f"{subj} обитает в {roles['LOCATION']}"
        if pred == "HAVE" and "OBJECT" in roles:
            return f"у {subj} есть {roles['OBJECT']}"
        if pred == "CREATE" and "OBJECT" in roles:
            t = roles.get("TIME")
            base = f"{roles.get('SUBJECT', '?')} создал(и) {roles['OBJECT']}"
            return f"{base} ({t})" if t else base
        if pred == "RUN":
            how = roles.get("HOW-TO")
            return f"{subj} бегает" + (f" ({how})" if how else "")
        if pred == "BE_COLORED" and "OBJECT" in roles:
            return f"{subj} цвета {roles['OBJECT']}"
        extras = ", ".join(f"{k}: {v}" for k, v in roles.items() if k != "SUBJECT")
        return f"{pred}: {subj}" + (f" — {extras}" if extras else "")

    def _memory_context(self, user_text: str, max_facts: int = 16) -> str:
        """Human-readable facts for soft RAG; prefer nodes touching query seeds."""
        store = self.store
        nodes = store.find_hypernodes()
        if not nodes:
            return ""

        q = user_text.lower()
        scored: list[tuple[int, Hyperlink]] = []
        for n in nodes:
            score = 0
            try:
                tpl = store.get_template(n.template.target_uid)
                blob = tpl.predicate.target_uid + " "
            except Exception:
                blob = ""
            for f in n.fillers.values():
                lab = self._label(f.target_uid)
                blob += lab + " " + f.target_uid + " "
                if lab.lower() in q or f.target_uid.lower().replace("m_", "") in q:
                    score += 3
            if score == 0:
                score = 1  # keep recent tail
            scored.append((score, n))
        scored.sort(key=lambda x: (x[0], x[1].uid), reverse=True)

        lines: list[str] = []
        seen: set[str] = set()
        for score, n in scored:
            if score < 1:
                continue
            line = self._fmt_fact(n)
            if not line or line in seen:
                continue
            low = line.lower()
            if low.startswith(("чтобы —", "точная —", "этом —", "нет —")):
                continue
            if any(low.endswith(f" — {w}") for w in ("all", "you", "need", "self", "attention")):
                continue
            seen.add(line)
            lines.append(f"• {line}")
            if len(lines) >= max_facts:
                break
        return "\n".join(lines)

    def _record_episode(self, speaker: str, text: str, report: IngestReport) -> None:
        """Episode links to extracted N + m seeds (not the raw utterance blob)."""
        store = self.store
        ep_uid = store.new_uid(f"EP_{speaker}")
        item_uids: list[str] = []
        for u in report.created_n:
            if u not in item_uids:
                item_uids.append(u)
        for u in report.seed_uids:
            if u.startswith("M_") and u not in item_uids:
                item_uids.append(u)
        items = [store.m_ref(u) for u in item_uids[:12]]
        short = text.strip().replace("\n", " ")
        if len(short) > 48:
            short = short[:45] + "…"
        store.add_element(
            Section.H,
            ElementList(
                uid=ep_uid,
                items=items,
                Pr=[
                    Property(name="label", value=f"{speaker}:{short}"),
                    Property(name="speaker", value=speaker),
                ],
                Mt=[Property(name="kind", value="Episode")],
            ),
        )
        if self._ep_prev is not None:
            store.add_link(
                AssocLink(
                    uid=store.new_uid("L_FOLLOW"),
                    id=LinkId.FOLLOW.value,
                    w=0.8,
                    e1=store.m_ref(self._ep_prev),
                    e2=store.m_ref(ep_uid),
                )
            )
        self._ep_prev = ep_uid
