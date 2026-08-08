"""LLM dialogue: answer from existing AH graph, then ingest both turns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ah_memory.config import GigaChatConfig
from ah_memory.gigachat_llm import GigaChatClient
from ah_memory.ignition import TickTrace
from ah_memory.store import AHStore
from ah_memory.types import Hyperlink, Section


DIALOGUE_SYSTEM = """Ты обычный полезный собеседник. Отвечай по-русски ясно и по делу.

Контекст «АГ-память» / «Активировано» (если есть ниже) — приоритетный источник для личных фактов о пользователе
и связей из этой сессии (имя, родные, учёба, места, задачи). Если там есть релевантное — используй.

На обычные вопросы (общие знания, пояснения, small talk, бытовые советы) отвечай как обычный ассистент:
не отмалчивайся и не требуй наличия АГ-памяти.

Не выдумывай личные факты о пользователе, которых нет в контексте АГ-памяти.
Не упоминай внутреннюю память, UID, JSON и не спрашивай, записать ли что-то в память."""


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
        "chains": t.chains,
    }


def _collect_chains(*tick_groups: list[dict], limit: int = 24) -> list[str]:
    """Dedup human chains across ticks (prefer longer / later)."""
    seen: set[str] = set()
    out: list[str] = []
    for ticks in tick_groups:
        for t in ticks or []:
            for line in t.get("chains") or []:
                # normalize by node UIDs roughly: keep first occurrence of same end target
                key = line
                if key in seen:
                    continue
                seen.add(key)
                out.append(line)
                if len(out) >= limit:
                    return out
    return out


@dataclass
class TurnResult:
    reply: str
    user_facts: list[str] = field(default_factory=list)
    assistant_facts: list[str] = field(default_factory=list)
    trace_uids: list[str] = field(default_factory=list)
    wm: list[str] = field(default_factory=list)
    backend: str = "gigachat"
    history_len: int = 0
    system_prompt: str = ""
    activation: dict = field(default_factory=dict)
    graph_build_json: dict = field(default_factory=dict)


class DialogueAgent:
    """Wraps Agent: answer from AH, then ingest both turns."""

    def __init__(
        self,
        agent,
        *,
        chat_client: Any = None,
        provider: str = "rules",
        gigachat: GigaChatConfig | None = None,
    ) -> None:
        self.agent = agent
        # backward compat: gigachat=...
        if chat_client is not None:
            self.client = chat_client
            self.provider = provider
        elif gigachat is not None and gigachat.configured:
            self.client = GigaChatClient(gigachat)
            self.provider = "gigachat"
        else:
            self.client = None
            self.provider = "rules"
        self.cfg = gigachat
        self.history: list[dict[str, str]] = []
        self._turn = 0
        self.last_activation: dict = {}
        self.last_graph_build_json: dict = {}

    @property
    def store(self) -> AHStore:
        return self.agent.store

    def reset_history(self) -> None:
        self.history.clear()
        self._turn = 0
        self.last_activation = {}
        self.last_graph_build_json = {}

    def talk(self, user_text: str, ticks: int = 6) -> TurnResult:
        user_text = user_text.strip()
        self._turn += 1
        ign = self.agent.ignition

        # 1) Ответ LLM по текущему графу (ещё без фактов из этой реплики)
        mem = self._memory_context(user_text)
        i_ask = len(ign.traces)
        ask = self.agent.ask(user_text, ticks=ticks)
        ask_ticks = [_tick_dict(t) for t in ign.traces[i_ask:]]
        graph_hint = ask.answer if ask.answer and ask.answer != "неизвестно" else ""

        if self.client is not None:
            reply, system_prompt = self._llm_reply(user_text, mem, graph_hint)
            backend = f"{self.provider}+ah"
        else:
            reply, system_prompt = self._fallback_reply(user_text, mem, graph_hint)
            backend = "rules+ah"

        # 2) После ответа — запись реплик в граф
        i0 = len(ign.traces)
        user_rep = self.agent.ingest(user_text, section=Section.H)
        user_ticks = [_tick_dict(t) for t in ign.traces[i0:]]

        i1 = len(ign.traces)
        asst_rep = self.agent.ingest(reply, section=Section.H, source="assistant")
        asst_ticks = [_tick_dict(t) for t in ign.traces[i1:]]

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
        graph_build_json = {
            "turn": self._turn,
            "user_text": user_text,
            "user_perception": user_rep.perception,
            "ask_perception": ask.perception,
            "assistant_text": reply,
            "assistant_perception": asst_rep.perception,
            "created": {
                "user_n": user_rep.created_n,
                "assistant_n": asst_rep.created_n,
                "user_skipped": user_rep.skipped,
                "assistant_skipped": asst_rep.skipped,
            },
        }
        activation = {
            "turn": self._turn,
            "threshold_t": self.agent.hp.threshold_t,
            "chains": _collect_chains(ask_ticks, user_ticks, asst_ticks),
            "ask": {
                "graph_hint": graph_hint,
                "seed_uids": ask.seed_uids[:24],
                "ticks": ask_ticks,
            },
            "user_ingest": {
                "created_n": user_rep.created_n,
                "seeds": user_rep.seed_uids[:24],
                "skipped": user_rep.skipped[:12],
                "ticks": user_ticks,
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
        self.last_graph_build_json = graph_build_json
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
            graph_build_json=graph_build_json,
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
        system_content = "\n\n".join(sys_blocks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
        ]
        reply = self.client.chat(messages, json_mode=False).strip()
        return reply, system_content

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
        if pred == "BE_BORN" and "LOCATION" in roles:
            return f"{subj} родился(ась) в {roles['LOCATION']}"
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
            seen.add(line)
            lines.append(f"• {line}")
            if len(lines) >= max_facts:
                break
        return "\n".join(lines)
