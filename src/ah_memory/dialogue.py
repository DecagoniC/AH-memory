"""DialogueAgent: ответ LLM с контекстом AH, затем ingest обеих реплик.

Поверх Agent: talk() = ask(+LLM) → ingest(user) → ingest(assistant).
Читать после agent.py; детали GigaChat — gigachat_llm.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Iterator

from ah_memory.config import GigaChatConfig
from ah_memory.context_ranker import GraphContextRanker, lexical_relevance, symbol_label
from ah_memory.gigachat_llm import GigaChatClient, _parse_json
from ah_memory.ignition import TickTrace
from ah_memory.perception import (
    PerceptionResult,
    SeedPerception,
    candidates_from_llm_json,
    content_entity_uids,
    gate_candidates,
    llm_payload_errors,
)
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


_EPISTEMIC_PREFIX = {
    "decision": "РЕШЕНИЕ",
    "topic": "ТЕМА",
    "open_question": "ОТКРЫТЫЙ ВОПРОС",
    "proposal": "ПРЕДЛОЖЕНИЕ АССИСТЕНТА",
    "explanation": "ПОЯСНЕНИЕ АССИСТЕНТА",
}


def _mark_epistemic(line: str | None, metadata: dict) -> str | None:
    if not line:
        return line
    prefix = _EPISTEMIC_PREFIX.get(str(metadata.get("statement_type") or ""))
    return f"{prefix}: {line}" if prefix else line


def _is_recap_request(text: str) -> bool:
    low = text.lower()
    return any(
        cue in low
        for cue in (
            "напомни",
            "подведи итог",
            "подведем итог",
            "подведём итог",
            "итог",
            "что решили",
            "резюме",
            "recap",
            "summary",
        )
    )


# ── Системный промпт диалога ─────────────────────────────────────────────────
# Зачем: LLM пересказывает только выданный контекст AH; без контекста — обычный чат.

DIALOGUE_SYSTEM = """Ты отвечаешь по-русски ясно, кратко и по делу.

Если ниже есть блок контекста (АГ-память, «Активировано» или другой помеченный источник),
это закрытый набор фактов:
- опирайся только на него;
- не добавляй сущности, места, даты, числа, причины, свойства и связи, которых там нет;
- не восполняй пробелы общими знаниями, даже если они кажутся очевидными;
- если нужного факта нет в контексте — ответь ровно: неизвестно;
- можно связно перефразировать факты из контекста, не превращая ответ в дамп списка.

Если блока контекста нет, отвечай как обычный собеседник.
Не выдумывай факты о пользователе, которых нет в репликах.
Не упоминай внутреннюю память, UID, JSON и не спрашивай, записать ли что-то в память."""

AH_CONTEXT_PREAMBLE = (
    "[Контекст из АГ-памяти — единственный источник фактов. "
    "Не добавляй сущности, места, даты, числа, причины и связи, которых нет в списке. "
    "Если ответа нет — неизвестно]"
)


ASSISTANT_MEMORY_SYSTEM = """Ты редактор блокнотика диалога.
Из ответа ассистента извлеки 1–5 атомарных заметок, полезных в следующих репликах.
Верни только JSON: {"kind":"message","candidates":[...],"seed_tokens":[...]}.

Каждый candidate содержит raw_relation, canonical_relation, predicate, roles, raw_span,
confidence и statement_type. statement_type только:
- "explanation" — определение, причинная или процедурная связь, изложенная ассистентом;
- "proposal" — рекомендация, вариант или предлагаемое действие.

Роли только SUBJECT | OBJECT | LOCATION | TIME | CAUSE | TOOL | MATERIAL | PURPOSE | HOW-TO | WITH.
Одна роль — одна сущность; SUBJECT обязателен. Для общей рекомендации допустим
SUBJECT:"АССИСТЕНТ". Разбивай списки и этапы на отдельные candidates.
Если предложено несколько вариантов, верни отдельный proposal для каждого:
не OBJECT:"X, Y", а два candidates с OBJECT:"X" и OBJECT:"Y".
Не используй assertion/decision: ответ ассистента не является подтверждённым фактом пользователя.
Не извлекай риторические фразы, оговорки и повторы. Не добавляй знания вне ответа."""


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


@dataclass
class ReadOnlyTurn:
    """Prepared AH answer before any user/assistant graph ingestion."""

    reply: str
    system_prompt: str
    backend: str
    user_perception: PerceptionResult
    ask: Any
    ask_ticks: list[dict] = field(default_factory=list)
    ask_traces: list[Any] = field(default_factory=list)
    memory_context: str = ""
    graph_hint: str = ""


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
        final: TurnResult | None = None
        for event in self.talk_stream(user_text, ticks=ticks):
            if event.get("event") == "done":
                final = event["turn"]
        if final is None:
            raise RuntimeError("dialogue stream ended without a done event")
        return final

    def talk_stream(self, user_text: str, ticks: int = 6) -> Iterator[dict[str, Any]]:
        """Yield status/token/done events; final ``done.turn`` is a TurnResult."""
        # Порядок важен: сначала ответить по СТАРОМУ графу, потом ingest реплик
        # (иначе вопрос пользователя уже «загрязняет» контекст ответа).
        user_text = user_text.strip()
        self._turn += 1
        ign = self.agent.ignition

        # Shared read-only path (also used by UI comparison), then stream reply.
        yield {"event": "status", "phase": "perception"}
        prepared = self.answer_read_only(user_text, ticks=ticks, generate=False)
        user_perc = prepared.user_perception
        ask = prepared.ask
        ask_ticks = prepared.ask_ticks
        graph_hint = prepared.graph_hint
        mem = prepared.memory_context

        yield {"event": "status", "phase": "reply"}
        if self.client is not None:
            messages, system_prompt = self._reply_messages(
                user_text,
                mem,
                graph_hint,
                ask.full_trace,
            )
            stream_fn = getattr(self.client, "chat_stream", None)
            if callable(stream_fn):
                chunks: list[str] = []
                for piece in stream_fn(messages, json_mode=False):
                    if not piece:
                        continue
                    chunks.append(piece)
                    yield {"event": "token", "text": piece}
                reply = "".join(chunks).strip()
            else:
                reply = self.client.chat(messages, json_mode=False).strip()
                if reply:
                    yield {"event": "token", "text": reply}
            backend = f"{self.provider}+ah"
        else:
            reply, system_prompt = self._fallback_reply(
                user_text,
                mem,
                graph_hint,
                ask.full_trace,
            )
            if reply:
                yield {"event": "token", "text": reply}
            backend = "rules+ah"

        yield {"event": "status", "phase": "ingest"}
        # 2) Запись user и assistant в Section.H
        i0 = len(ign.traces)
        user_rep = self.agent.ingest(
            user_text, section=Section.H, perception=user_perc
        )
        user_ticks = [_tick_dict(t) for t in ign.traces[i0:]]

        i1 = len(ign.traces)
        # Ответ ассистента хранится как proposals/explanations, не как пользовательская истина.
        asst_perc = (
            SeedPerception().parse(
                reply,
                self.agent.perception_context(reply),
            )
            if _is_recap_request(user_text)
            else self._parse_assistant_memory(
                reply,
                self.agent.perception_context(reply),
            )
        )
        asst_rep = self.agent.ingest(
            reply,
            section=Section.H,
            source="assistant",
            perception=asst_perc,
        )
        asst_ticks = [_tick_dict(t) for t in ign.traces[i1:]]

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > 24:
            self.history = self.history[-24:]

        wm = ign.wm.ranked_uids()
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
            "parser": {
                "llm_raw": user_perc.meta.get("llm_raw"),
                "gate_report": user_perc.meta.get("gate_report"),
                "gated_candidates": user_perc.to_graph_json().get("candidates", []),
                "ingest_skipped": list(user_rep.skipped),
                "ingest_created": list(user_rep.created_n),
            },
            "created": {
                "user_n": user_rep.created_n,
                "assistant_n": asst_rep.created_n,
                "user_skipped": user_rep.skipped,
                "assistant_skipped": asst_rep.skipped,
            },
        }
        full_trace = self.agent._full_trace(
            prepared.ask_traces,
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
        turn = TurnResult(
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
        yield {"event": "done", "turn": turn}

    def answer_read_only(
        self,
        user_text: str,
        *,
        ticks: int = 6,
        generate: bool = True,
    ) -> ReadOnlyTurn:
        """Run the same perception → AH → prompt path without ingesting the turn."""
        user_text = user_text.strip()
        # Perception and prompt preparation must not depend on activation left by
        # whichever UI mode happened to run immediately before this query.
        self.agent.ignition.set_factor_gates(None, reset_state=True)
        wm_ctx = self.agent.perception_context(user_text)
        user_perc = self.agent.perception.parse(user_text, wm_ctx)
        ask_ticks_count = (
            ticks if user_perc.kind == "question" else min(ticks, 2)
        )
        memory_context = self._memory_context(user_text)
        ignition = self.agent.ignition
        trace_start = len(ignition.traces)
        ask = self.agent.ask(
            user_text,
            ticks=ask_ticks_count,
            perception=user_perc,
        )
        ask_traces = list(ignition.traces[trace_start:])
        ask_ticks = [_tick_dict(item) for item in ask_traces]
        graph_hint = (
            ask.answer
            if ask.answer and ask.answer != "неизвестно"
            else ""
        )
        if generate and self.client is not None:
            reply, system_prompt = self._llm_reply(
                user_text,
                memory_context,
                graph_hint,
                ask.full_trace,
            )
            backend = f"{self.provider}+ah"
        elif generate:
            reply, system_prompt = self._fallback_reply(
                user_text,
                memory_context,
                graph_hint,
                ask.full_trace,
            )
            backend = "rules+ah"
        else:
            system_prompt = self._prompt_view(
                memory_context,
                graph_hint,
                ask.full_trace,
                mode="generation disabled",
            )
            reply = ask.answer
            backend = "graph"
        return ReadOnlyTurn(
            reply=reply,
            system_prompt=system_prompt,
            backend=backend,
            user_perception=user_perc,
            ask=ask,
            ask_ticks=ask_ticks,
            ask_traces=ask_traces,
            memory_context=memory_context,
            graph_hint=graph_hint,
        )

    def _parse_assistant_memory(
        self,
        text: str,
        wm_context: list[str],
    ) -> PerceptionResult:
        fallback = SeedPerception().parse(text, wm_context)
        if self.client is None:
            return fallback
        try:
            payload: dict[str, Any] = {"assistant_text": text}
            raw = self.client.chat(
                [
                    {"role": "system", "content": ASSISTANT_MEMORY_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                json_mode=True,
            )
            data = _parse_json(raw)
            parsed = candidates_from_llm_json(data)
            invalid_statuses = [
                candidate.statement_type
                for candidate in parsed
                if candidate.statement_type not in {"proposal", "explanation"}
            ]
            typed = [
                replace(candidate, source="assistant")
                for candidate in parsed
                if candidate.statement_type in {"proposal", "explanation"}
            ]
            gated_result = gate_candidates(text, typed, report=True)
            gated, gate_report = gated_result
            validation_errors = llm_payload_errors(data)
            if invalid_statuses:
                validation_errors.append(
                    "assistant candidates require proposal or explanation"
                )
            validation_errors.extend(
                f"candidate rejected: {item.get('reason', 'validation')}"
                for item in gate_report.get("dropped", [])
            )
            seeds = list(
                dict.fromkeys(
                    [
                        uid
                        for candidate in gated
                        for uid in candidate.roles.values()
                    ]
                    + content_entity_uids(text)[:8]
                )
            )[:16]
            return PerceptionResult(
                kind="message",
                candidates=list(gated),
                seed_tokens=seeds,
                meta={
                    "backend": f"{self.provider}_assistant_memory",
                    "llm_raw": data,
                    "gate_report": gate_report,
                    "validation_errors": validation_errors,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return PerceptionResult(
                kind=fallback.kind,
                candidates=[],
                seed_tokens=fallback.seed_tokens,
                meta={
                    **fallback.meta,
                    "backend": "assistant_seeds",
                    "llm_error": str(exc),
                },
            )

    def _compose_system_blocks(
        self,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> list[str]:
        blocks = [DIALOGUE_SYSTEM]
        compact = self._compact_memory_for_llm(prepared_context)
        # Compact WM is the query-ranked fact list. `mem` is the older
        # pre-ask lexical RAG over the same factors — never both.
        context = compact or mem
        if context or graph_hint:
            ctx_parts = [AH_CONTEXT_PREAMBLE]
            if context:
                ctx_parts.append(context)
            elif graph_hint:
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

        # Rank facts by this query's lexical intent, factor gates and activation.
        # Global durable facts are only a small fallback when no relevant item exists.
        activated = [
            uid
            for uid in prepared_context.get("activated_nodes") or []
            if not str(uid).startswith(("PRIOR::", "SF::"))
        ]
        fact_lines: list[str] = []
        all_factors = [
            factor
            for factor in self.store.list_semantic_factors()
            if factor.relation is not None
        ]
        def created_tau(factor) -> int:
            return int((factor.metadata or {}).get("created_tau", -1))

        query = str(prepared_context.get("question") or "")
        if query:
            query_plan = prepared_context.get("query_plan") or {}
            factor_gates = query_plan.get("factor_gates") or {}
            timesteps = prepared_context.get("timesteps") or []
            activation = (
                dict(timesteps[-1].get("activation") or {})
                if timesteps
                else {uid: 1.0 for uid in activated}
            )
            ranker = GraphContextRanker(self.store)
            scored_factors = [
                (
                    ranker.factor_score(
                        factor,
                        query,
                        activation=activation,
                        factor_gates=factor_gates,
                    ),
                    factor,
                )
                for factor in all_factors
            ]
            scored_factors.sort(
                key=lambda item: (
                    -item[0],
                    -created_tau(item[1]),
                    item[1].uid,
                )
            )
            ordered_factors = [
                factor.uid
                for score, factor in scored_factors
                if score > 0.05
            ]
        else:
            # Compatibility for callers that provide only an activation trace.
            # Shared context UIDs are excluded from semantic relevance.
            relevant = [
                factor
                for factor in all_factors
                if set(activated).intersection(
                    {
                        uid
                        for uid in factor.variables
                        if uid
                        != (factor.metadata or {}).get("context_uid")
                    }
                )
            ]
            decisions = sorted(
                [
                    factor
                    for factor in all_factors
                    if (factor.metadata or {}).get("statement_type")
                    == "decision"
                ],
                key=lambda item: (-created_tau(item), item.uid),
            )
            recent_other = sorted(
                [
                    factor
                    for factor in all_factors
                    if factor not in relevant and factor not in decisions
                ],
                key=lambda item: (-created_tau(item), item.uid),
            )[:4]
            ordered_factors = list(
                dict.fromkeys(
                    factor.uid
                    for factor in [
                        *relevant,
                        *decisions,
                        *recent_other,
                    ]
                )
            )
        if not ordered_factors:
            ordered_factors = [
                factor.uid
                for factor in sorted(
                    [
                        item
                        for item in all_factors
                        if (item.metadata or {}).get("statement_type")
                        in {"assertion", "decision"}
                    ],
                    key=lambda item: (-created_tau(item), item.uid),
                )[:4]
            ]
        factors_by_uid = {factor.uid: factor for factor in all_factors}
        for factor_uid in ordered_factors:
            factor = factors_by_uid[factor_uid]
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
            line = self._fmt_semantic_fact(
                pred,
                roles,
                when=when,
                span=self._factor_raw_span(factor),
            )
            line = _mark_epistemic(line, meta)
            key = (self._factor_raw_span(factor) or line).casefold()
            if not line or key in seen:
                continue
            seen.add(key)
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
                line = self._fmt_semantic_fact(
                    str(pred),
                    roles,
                    when=when,
                    span=str(event.get("raw_span") or "").strip(),
                )
                line = _mark_epistemic(line, meta)
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

    def _factor_raw_span(self, factor) -> str:
        meta = factor.metadata or {}
        span = str(meta.get("raw_span") or "").strip()
        if span:
            return span
        event = self.store.events.get(str(meta.get("event_uid") or ""))
        if event is not None:
            return str(event.raw_span or "").strip()
        return ""

    def _fmt_semantic_fact(
        self,
        pred: str,
        roles: dict[str, str],
        *,
        when: str = "",
        span: str = "",
    ) -> str | None:
        subj = roles.get("SUBJECT")
        pred_u = pred.upper()
        role_parts = [
            f"{role}: {value}"
            for role, value in roles.items()
            if role != "SUBJECT" and value
        ]
        if not subj and not role_parts and not span:
            return None
        if span:
            structured = pred_u
            if subj:
                structured += f": {subj}"
            if role_parts:
                structured += " — " + ", ".join(role_parts)
            base = f"«{span}» ({structured})" if (subj or role_parts) else f"«{span}»"
        else:
            if not subj and not role_parts:
                return None
            base = pred_u
            if subj:
                base += f": {subj}"
            if role_parts:
                base += " — " + ", ".join(role_parts)
        if when:
            return f"{base} [добавлено: {when}]"
        return base

    def _fmt_state_item(self, key: str, value: Any) -> str | None:
        if isinstance(value, (list, tuple)):
            return None
        if value is None or value == "":
            return None
        readable_key = ":".join(self._label(part) for part in key.split(":"))
        readable_value = self._label(str(value)) if isinstance(value, str) else value
        return f"{readable_key} = {readable_value}"

    def _reply_messages(
        self,
        user_text: str,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> tuple[list[dict[str, str]], str]:
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
        return messages, system_content

    def _llm_reply(
        self,
        user_text: str,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> tuple[str, str]:
        messages, system_content = self._reply_messages(
            user_text,
            mem,
            graph_hint,
            prepared_context,
        )
        reply = self.client.chat(messages, json_mode=False).strip()
        return reply, system_content

    def generate_from_context(
        self,
        user_text: str,
        context: str,
        *,
        source: str,
    ) -> tuple[str, str]:
        """Generate another comparison arm with the same dialogue policy."""
        blocks = [DIALOGUE_SYSTEM]
        if context.strip():
            blocks.append(
                f"[Контекст {source} — единственный источник фактов. "
                "Не добавляй ничего сверх этого текста. Если ответа нет — неизвестно]\n"
                f"{context.strip()}"
            )
        system_content = "\n\n".join(blocks)
        if self.client is None:
            return (
                context.strip() or "неизвестно",
                system_content
                + "\n\n──── mode ────\n\nrules fallback (LLM недоступен)",
            )
        reply = self.client.chat(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_text},
            ],
            json_mode=False,
        ).strip()
        return reply, system_content

    def _prompt_view(
        self,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None,
        *,
        mode: str,
    ) -> str:
        blocks = self._compose_system_blocks(
            mem,
            graph_hint,
            prepared_context,
        )
        return (
            "\n\n──── system ────\n\n".join(blocks)
            + f"\n\n──── mode ────\n\n{mode}"
        )

    def _fallback_reply(
        self,
        user_text: str,
        mem: str,
        graph_hint: str,
        prepared_context: dict | None = None,
    ) -> tuple[str, str]:
        prompt_view = self._prompt_view(
            mem,
            graph_hint,
            prepared_context,
            mode="rules fallback (LLM недоступен)",
        )
        if graph_hint and graph_hint != "неизвестно":
            return graph_hint, prompt_view
        if mem:
            return (mem.split("\n", 1)[-1] if "\n" in mem else mem), prompt_view
        return "Хорошо. Можешь уточнить вопрос или рассказать больше — отвечу.", prompt_view

    def _label(self, uid: str) -> str:
        return symbol_label(self.store, uid, compact=True)

    def _memory_context(self, user_text: str, max_facts: int = 16) -> str:
        """Soft RAG из open semantic factors."""
        store = self.store
        factors = list(store.list_semantic_factors())
        if not factors:
            return ""

        recall_requested = _is_recap_request(user_text)
        activation = {
            entry.uid: entry.activation
            for entry in self.agent.ignition.wm.entries()
        }
        scored: list[tuple[float, Any]] = []
        for factor in factors:
            if factor.relation is None:
                continue
            meta = factor.metadata or {}
            statement_type = meta.get("statement_type")
            document = " ".join(
                [
                    factor.relation.canonical_label.replace("_", " "),
                    *(self._label(uid) for uid in factor.roles.values()),
                ]
            )
            lexical = lexical_relevance(user_text, document)
            if not recall_requested and lexical <= 0.0:
                continue
            score = lexical
            if recall_requested and statement_type == "decision":
                score += 10.0
            elif recall_requested and statement_type == "assertion":
                score += 2.0
            context_uid = meta.get("context_uid")
            for uid in factor.variables:
                if context_uid and uid == context_uid:
                    continue
                score += 0.15 * float(activation.get(uid, 0.0))
            scored.append((score, factor))
        scored.sort(
            key=lambda x: (
                x[0],
                int((x[1].metadata or {}).get("created_tau", -1)),
            ),
            reverse=True,
        )

        lines: list[str] = []
        seen: set[str] = set()
        for score, factor in scored:
            if score <= 0.05:
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
                factor.relation.canonical_label,
                roles,
                when=when,
                span=self._factor_raw_span(factor),
            )
            line = _mark_epistemic(line, meta)
            key = (self._factor_raw_span(factor) or line).casefold()
            if not line or key in seen:
                continue
            seen.add(key)
            lines.append(f"• {line}")
            if len(lines) >= max_facts:
                break
        if not lines:
            return ""
        return "Факты (с временем добавления):\n" + "\n".join(lines)
