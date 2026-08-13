"""DialogueAgent: ответ LLM с контекстом AH, затем ingest обеих реплик.

Поверх Agent: talk() = ask(+LLM) → ingest(user) → ingest(assistant).
Читать после agent.py; детали GigaChat — gigachat_llm.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ah_memory.config import GigaChatConfig
from ah_memory.gigachat_llm import GigaChatClient
from ah_memory.ignition import TickTrace
from ah_memory.store import AHStore
from ah_memory.types import Section


def _format_added_at(iso: str, tau: int | None = None) -> str:
    """Человекочитаемое время для промпта LLM."""
    pretty = iso
    if iso:
        try:
            dt = datetime.fromisoformat(iso)
            pretty = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            pretty = iso
    if pretty and tau is not None:
        return f"{pretty}, τ={tau}"
    if pretty:
        return pretty
    if tau is not None:
        return f"τ={tau}"
    return ""


# ── Системный промпт диалога ─────────────────────────────────────────────────
# Зачем: LLM отвечает как чат; AH — приоритет только для личных фактов сессии.

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
        "activation_top": t.activation_top,
        "events": [asdict(event) for event in t.events],
        "convergence": t.convergence,
        "timings_ms": t.timings_ms,
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
    full_trace: dict = field(default_factory=dict)


class DialogueAgent:
    """Обёртка Agent: ответ по AH (+LLM), затем запись user/assistant в граф."""

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
        # Порядок важен: сначала ответить по СТАРОМУ графу, потом ingest реплик
        # (иначе вопрос пользователя уже «загрязняет» контекст ответа).
        user_text = user_text.strip()
        self._turn += 1
        ign = self.agent.ignition

        # 1) Активация + LLM/fallback по текущей памяти
        mem = self._memory_context(user_text)
        i_ask = len(ign.traces)
        ask = self.agent.ask(user_text, ticks=ticks)
        ask_ticks = [_tick_dict(t) for t in ign.traces[i_ask:]]
        graph_hint = ask.answer if ask.answer and ask.answer != "неизвестно" else ""

        if self.client is not None:
            reply, system_prompt = self._llm_reply(
                user_text,
                mem,
                graph_hint,
                ask.full_trace,
            )
            backend = f"{self.provider}+ah"
        else:
            reply, system_prompt = self._fallback_reply(
                user_text,
                mem,
                graph_hint,
                ask.full_trace,
            )
            backend = "rules+ah"

        # 2) Запись user и assistant в Section.H
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
        full_trace = self.agent._full_trace(
            ign.traces[i_ask:],
            trace,
            answer=reply,
        )
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
            "full_trace": full_trace,
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
            full_trace=full_trace,
        )

    def _compose_system_blocks(
        self,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> list[str]:
        blocks = [DIALOGUE_SYSTEM]
        compact = self._compact_memory_for_llm(prepared_context)
        if mem or graph_hint or compact:
            ctx_parts = [
                "[Контекст из АГ-памяти — учитывай по возможности, не цитируй как отчёт]"
            ]
            if mem:
                ctx_parts.append(mem)
            if compact:
                ctx_parts.append(compact)
            elif graph_hint:
                # Fallback only when compact WM is empty.
                ctx_parts.append(f"Активировано: {graph_hint}")
            blocks.append("\n".join(ctx_parts))
        return blocks

    def _compact_memory_for_llm(
        self,
        prepared_context: dict | None,
        *,
        max_facts: int = 12,
        max_state: int = 8,
        max_nodes: int = 10,
    ) -> str:
        """Human-readable WM for the chat model — no raw UIDs/JSON dumps."""
        if not prepared_context:
            return ""
        lines: list[str] = []
        seen: set[str] = set()

        # Prefer semantic factors touching activated nodes.
        activated = [
            uid
            for uid in prepared_context.get("activated_nodes") or []
            if not str(uid).startswith(("PRIOR::", "SF::"))
        ]
        activated_set = set(activated)
        fact_lines: list[str] = []
        for factor in self.store.list_semantic_factors():
            if activated_set and not activated_set.intersection(factor.variables):
                continue
            if factor.relation is None:
                continue
            roles = {
                role: self._label(uid) for role, uid in factor.roles.items()
            }
            pred = factor.relation.canonical_label
            meta = factor.metadata or {}
            when = _format_added_at(
                str(meta.get("added_at") or ""),
                int(meta["created_tau"])
                if meta.get("created_tau") is not None
                else None,
            )
            if not when and factor.variables:
                when = self._format_uid_added_at(factor.variables[0])
            line = self._fmt_semantic_fact(pred, roles, when=when)
            if not line or line in seen:
                continue
            seen.add(line)
            fact_lines.append(f"• {line}")
            if len(fact_lines) >= max_facts:
                break

        # Fallback: compact event summaries (no nested relation objects).
        if not fact_lines:
            for event in (prepared_context.get("events") or [])[:max_facts]:
                pred = (
                    (event.get("predicate") or {}).get("canonical_label")
                    or event.get("predicate")
                    or "?"
                )
                if isinstance(pred, dict):
                    pred = pred.get("canonical_label", "?")
                args = event.get("arguments") or {}
                roles = {
                    role: self._label(
                        (ref.get("uid") if isinstance(ref, dict) else str(ref))
                    )
                    for role, ref in args.items()
                }
                meta = event.get("metadata") or {}
                when = _format_added_at(
                    str(meta.get("added_at") or ""),
                    int(meta["created_tau"])
                    if meta.get("created_tau") is not None
                    else None,
                )
                line = self._fmt_semantic_fact(str(pred), roles, when=when)
                if line and line not in seen:
                    seen.add(line)
                    fact_lines.append(f"• {line}")

        if fact_lines:
            lines.append("Факты (с временем добавления):")
            lines.extend(fact_lines)

        state = prepared_context.get("state") or {}
        values = state.get("values") or {}
        state_lines: list[str] = []
        for key, value in list(values.items())[: max_state * 2]:
            pretty = self._fmt_state_item(str(key), value)
            if pretty and pretty not in seen:
                seen.add(pretty)
                state_lines.append(f"• {pretty}")
            if len(state_lines) >= max_state:
                break
        if state_lines:
            lines.append("Состояние:")
            lines.extend(state_lines)

        # Short focus list — labels + added_at, capped.
        focus = []
        seen_focus: set[str] = set()
        for uid in activated[:max_nodes]:
            lab = self._label(uid)
            if not lab or lab in seen_focus:
                continue
            seen_focus.add(lab)
            when = self._format_uid_added_at(uid)
            focus.append(f"{lab} [{when}]" if when else lab)
        if focus:
            lines.append("Фокус: " + "; ".join(focus))

        return "\n".join(lines)

    def _format_uid_added_at(self, uid: str) -> str:
        added, tau = self.store.get_added_at(uid)
        return _format_added_at(added, tau)

    def _fmt_semantic_fact(
        self,
        pred: str,
        roles: dict[str, str],
        *,
        when: str = "",
    ) -> str | None:
        subj = roles.get("SUBJECT") or roles.get("BUYER") or next(
            iter(roles.values()), "?"
        )
        obj = roles.get("OBJECT")
        loc = roles.get("LOCATION")
        pred_u = pred.upper()
        if pred_u in {"PURCHASE", "BUY"} and obj:
            base = f"{subj} купил(а) {obj}"
        elif pred_u == "SELL" and obj:
            base = f"{subj} продал(а) {obj}"
        elif pred_u in {"WORKS_FOR", "WORK"} and obj:
            base = f"{subj} работает в {obj}"
        elif pred_u in {"LIVES_IN", "LIVE_IN"} and (obj or loc):
            base = f"{subj} живёт в {obj or loc}"
        elif pred_u == "LOCATED_IN" and (obj or loc):
            base = f"{subj} находится в {obj or loc}"
        elif pred_u == "OWNS" and obj:
            base = f"{subj} владеет {obj}"
        elif pred_u in {"IS", "IS_A"} and obj:
            base = f"{subj} — {obj}"
        elif pred_u == "HAVE" and obj:
            base = f"у {subj} есть {obj}"
        elif pred_u == "BE_COLORED" and obj:
            base = f"{subj} цвета {obj}"
        elif pred_u == "RUN":
            how = roles.get("HOW-TO")
            base = f"{subj} бегает" + (f" ({how})" if how else "")
        elif pred_u == "BE_BORN" and (obj or loc):
            base = f"{subj} родился(ась) в {obj or loc}"
        elif pred_u == "PART_OF" and obj:
            base = f"{subj} входит в {obj}"
        elif pred_u == "CAUSE" and obj:
            base = f"{subj} вызывает {obj}"
        elif pred_u == "FOLLOW" and obj:
            base = f"{subj} → {obj}"
        elif pred_u == "KNOW" and obj:
            base = f"{subj} знает {obj}"
        elif pred_u == "MOVE" and obj:
            base = f"{subj} переехал(а) в {obj}"
        else:
            extras = ", ".join(
                f"{k}: {v}" for k, v in roles.items() if k not in {"SUBJECT"} and v
            )
            if not extras and subj == "?":
                return None
            base = f"{pred_u}: {subj}" + (f" — {extras}" if extras else "")
        if when:
            return f"{base} [добавлено: {when}]"
        return base

    def _fmt_state_item(self, key: str, value: Any) -> str | None:
        parts = key.split(":")
        if parts[0] == "OWNS" and len(parts) >= 3:
            owner = self._label(parts[1])
            obj = self._label(parts[2])
            owned = bool(value)
            return f"{owner} {'владеет' if owned else 'не владеет'} {obj}"
        if parts[0] == "LAST_PURCHASE" and len(parts) >= 2:
            return f"последняя покупка {self._label(parts[1])}: {self._label(str(value))}"
        if isinstance(value, (list, tuple)):
            return None
        if value is None or value == "":
            return None
        # Skip opaque synthetic keys.
        if key.startswith(("PURCHASE_HISTORY", "SF::", "PRIOR::")):
            return None
        return f"{key} = {value}"

    def _llm_reply(
        self,
        user_text: str,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> tuple[str, str]:
        sys_blocks = self._compose_system_blocks(
            mem,
            graph_hint,
            prepared_context,
        )
        system_content = "\n\n".join(sys_blocks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
        ]
        reply = self.client.chat(messages, json_mode=False).strip()
        return reply, system_content

    def _fallback_reply(
        self,
        user_text: str,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> tuple[str, str]:
        sys_blocks = self._compose_system_blocks(
            mem,
            graph_hint,
            prepared_context,
        )
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

    def _memory_context(self, user_text: str, max_facts: int = 16) -> str:
        """Soft RAG из open semantic factors."""
        store = self.store
        factors = list(store.list_semantic_factors())
        if not factors:
            return ""

        q = user_text.lower()
        scored: list[tuple[int, Any]] = []
        for factor in factors:
            if factor.relation is None:
                continue
            score = 0
            for uid in factor.variables:
                lab = self._label(uid)
                if lab.lower() in q or uid.lower().replace("m_", "") in q:
                    score += 3
            scored.append((score, factor))
        scored.sort(key=lambda x: (x[0], x[1].uid), reverse=True)

        lines: list[str] = []
        seen: set[str] = set()
        for score, factor in scored:
            if score < 3:
                continue
            roles = {role: self._label(uid) for role, uid in factor.roles.items()}
            meta = factor.metadata or {}
            when = _format_added_at(
                str(meta.get("added_at") or ""),
                int(meta["created_tau"])
                if meta.get("created_tau") is not None
                else None,
            )
            line = self._fmt_semantic_fact(
                factor.relation.canonical_label, roles, when=when
            )
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(f"• {line}")
            if len(lines) >= max_facts:
                break
        if not lines:
            return ""
        return "Факты (с временем добавления):\n" + "\n".join(lines)
