"""FastAPI web UI: chat + graph dump + live AH vs RAG compare."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ah_memory.agent import Agent
from ah_memory.baselines.vanilla_rag import VanillaRAG
from ah_memory.compare import CompareEngine
from ah_memory.config import LlmProvider, load_config
from ah_memory.corpus import build_encyclopedia
from ah_memory.deepseek import DeepSeekClient, DeepSeekHybridPerception
from ah_memory.dialogue import DialogueAgent
from ah_memory.examples.rabbit import build_rabbit_memory
from ah_memory.gigachat_llm import GigaChatClient, HybridPerception
from ah_memory.graph_export import dump_ah_json, dump_graph
from ah_memory.perception import SeedPerception
from ah_memory.store import AHStore

STATIC = Path(__file__).parent / "static"
cfg = load_config()
app = FastAPI(title="AH Memory", version="0.4.2")

# runtime provider (UI can switch without restart)
llm_provider: LlmProvider = cfg.agent.llm_provider


def _provider_ready(provider: LlmProvider) -> bool:
    if not cfg.agent.use_llm:
        return False
    if provider == "deepseek":
        return cfg.deepseek.configured
    return cfg.gigachat.configured


def _make_perception(provider: LlmProvider | None = None) -> Any:
    p = provider or llm_provider
    if not cfg.agent.use_llm:
        return SeedPerception()
    if p == "deepseek" and cfg.deepseek.configured:
        return DeepSeekHybridPerception(cfg.deepseek, fallback=cfg.agent.fallback_rules)
    if p == "gigachat" and cfg.gigachat.configured:
        return HybridPerception(cfg.gigachat, fallback=cfg.agent.fallback_rules)
    return SeedPerception()


def _make_chat_client(provider: LlmProvider | None = None) -> tuple[Any | None, str]:
    p = provider or llm_provider
    if not cfg.agent.use_llm:
        return None, "rules"
    if p == "deepseek" and cfg.deepseek.configured:
        return DeepSeekClient(cfg.deepseek), "deepseek"
    if p == "gigachat" and cfg.gigachat.configured:
        return GigaChatClient(cfg.gigachat), "gigachat"
    return None, "rules"


def _store_for_preload(mode: str) -> AHStore:
    mode = (mode or "empty").strip().lower()
    if mode == "encyclopedia":
        store, _ = build_encyclopedia()
        return store
    if mode == "rabbit":
        return build_rabbit_memory()
    store = AHStore()
    store.clear()
    return store


def _build_core() -> Agent:
    mode = cfg.agent.preload if cfg.agent.preload in {"empty", "rabbit", "encyclopedia"} else "empty"
    return Agent(store=_store_for_preload(mode), perception=_make_perception())


def _build_dialogue(core: Agent) -> DialogueAgent:
    client, name = _make_chat_client()
    return DialogueAgent(core, chat_client=client, provider=name)


def _apply_llm_provider(provider: LlmProvider) -> None:
    """Swap perception + dialogue client; keep current AH store/graph."""
    global agent, dialogue, llm_provider
    llm_provider = provider
    agent.perception = _make_perception(provider)
    hist = list(dialogue.history)
    turn = dialogue._turn
    last_act = dialogue.last_activation
    last_gb = dialogue.last_graph_build_json
    client, name = _make_chat_client(provider)
    dialogue = DialogueAgent(agent, chat_client=client, provider=name)
    dialogue.history = hist
    dialogue._turn = turn
    dialogue.last_activation = last_act
    dialogue.last_graph_build_json = last_gb


def _build_compare(core: Agent) -> CompareEngine:
    """Сравнение на том же живом агенте, что и чат (корпус = диалог + факты)."""
    ds = cfg.deepseek if (cfg.agent.use_llm and cfg.deepseek.configured) else None
    return CompareEngine(core, ticks=cfg.agent.ticks, deepseek=ds)


agent = _build_core()
dialogue = _build_dialogue(agent)
comparer = _build_compare(agent)


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    ticks: int | None = None


class CompareIn(BaseModel):
    question: str = Field(min_length=1)
    ticks: int | None = None


class ChatOut(BaseModel):
    reply: str
    kind: str
    trace_uids: list[str]
    wm: list[str]
    backend: str
    stats: dict[str, Any]
    user_facts: list[str] = []
    assistant_facts: list[str] = []
    system_prompt: str = ""
    activation: dict[str, Any] = Field(default_factory=dict)
    graph_build_json: dict[str, Any] = Field(default_factory=dict)


class ProviderIn(BaseModel):
    provider: Literal["gigachat", "deepseek"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    model = None
    ready = _provider_ready(llm_provider)
    if llm_provider == "deepseek" and cfg.deepseek.configured:
        model = cfg.deepseek.model
    elif llm_provider == "gigachat" and cfg.gigachat.configured:
        model = cfg.gigachat.model
    return {
        "ok": True,
        "version": "0.4.3-merged",
        "llm_provider": llm_provider,
        "llm_configured": ready,
        "use_llm": cfg.agent.use_llm,
        "model": model if ready else None,
        "providers": {
            "gigachat": {
                "configured": cfg.gigachat.configured,
                "model": cfg.gigachat.model if cfg.gigachat.configured else None,
            },
            "deepseek": {
                "configured": cfg.deepseek.configured,
                "model": cfg.deepseek.model if cfg.deepseek.configured else None,
            },
        },
        "preload": cfg.agent.preload,
        "rag_backend": comparer.rag.backend,
        "corpus_chars": len(comparer.rag.corpus),
        "tau": agent.store.ah.tau,
        "history": len(dialogue.history),
        "graph_size": len(agent.store.ah.S) + len(agent.store.ah.L) + len(agent.store.find_hypernodes()),
        "|S|": len(agent.store.ah.S),
    }


@app.post("/api/llm-provider")
def set_llm_provider(body: ProviderIn) -> dict[str, Any]:
    if not _provider_ready(body.provider):
        raise HTTPException(
            400,
            f"provider '{body.provider}' not configured (check API key / credentials)",
        )
    _apply_llm_provider(body.provider)
    return health()


@app.get("/api/graph")
def graph(limit: int | None = 400, mode: str = "hyper") -> dict[str, Any]:
    if mode not in {"hyper", "all"}:
        mode = "hyper"
    return dump_graph(agent.store, limit_nodes=limit, mode=mode)


@app.get("/api/dump")
def full_dump() -> dict[str, Any]:
    return dump_ah_json(agent.store)


@app.get("/api/trace")
def last_trace() -> dict[str, Any]:
    """Last turn activation trace (evidence / BP ticks / WM)."""
    act = getattr(dialogue, "last_activation", None) or {}
    build = getattr(dialogue, "last_graph_build_json", None) or {}
    return {
        "ok": True,
        "tau": agent.store.ah.tau,
        "wm": sorted(agent.ignition.wm.contents()),
        "activation": act,
        "graph_build_json": build,
        "recent_ticks": [
            {
                "tau": t.tau,
                "seeds": t.seeds_applied,
                "evidence": t.evidence,
                "beliefs_top": t.beliefs_top,
                "wm": t.wm,
                "activated": t.activated,
                "trace_factors": t.trace_factors,
                "weight_updates": t.weight_updates,
                "stats": t.z_stats,
                "chains": t.chains,
            }
            for t in agent.ignition.traces[-12:]
        ],
    }


@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    ticks = body.ticks or cfg.agent.ticks
    try:
        turn = dialogue.talk(text, ticks=ticks)
        # держим RAG-корпус в синхроне с диалогом
        comparer.bind_history(dialogue.history)
        comparer.rebuild_rag()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"dialogue failed: {exc}") from exc

    return ChatOut(
        reply=turn.reply,
        kind="dialogue",
        trace_uids=turn.trace_uids,
        wm=turn.wm,
        backend=turn.backend,
        stats=dump_graph(agent.store, limit_nodes=1)["stats"],
        user_facts=turn.user_facts,
        assistant_facts=turn.assistant_facts,
        system_prompt=turn.system_prompt,
        activation=turn.activation,
        graph_build_json=turn.graph_build_json,
    )


@app.post("/api/compare")
def compare(body: CompareIn) -> dict[str, Any]:
    """Произвольный запрос пользователя: живая АГ vs RAG по тому же диалогу/графу."""
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    ticks = body.ticks or cfg.agent.ticks
    try:
        comparer.bind_history(dialogue.history)
        turn = comparer.ask(q, ticks=ticks)
        # факты, записанные compare, уже в agent; обновим history-корпус без дублирования chat
        comparer.rebuild_rag()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"compare failed: {exc}") from exc
    return turn.as_dict()


@app.post("/api/compare/m4")
def compare_m4() -> dict[str, Any]:
    """Эталонный gold M4 (заяц) — отдельный бенчмарк, не UI-диалог."""
    try:
        report = comparer.run_m4()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"m4 failed: {exc}") from exc
    return {
        "summary": report.as_dict(),
        "rag_backend": "llm+tfidf (rabbit gold)",
        "note": "M4 — контрольный корпус «заяц». Кнопка Сравнить использует ваш живой диалог.",
        "items": [
            {
                "q": i.question,
                "ah": i.ah_answer,
                "rag": i.rag_answer,
                "ah_correct": i.ah_correct,
                "rag_correct": i.rag_correct,
                "ah_trace_complete": i.ah_trace_complete,
                "ah_hall": i.ah_hallucinated,
                "rag_hall": i.rag_hallucinated,
                "ah_explain": round(i.ah_explain, 4),
                "trace": i.ah_trace[:16],
            }
            for i in report.items
        ],
    }


@app.post("/api/reset")
def reset(preload: str = "empty") -> dict[str, Any]:
    """Пересборка агента; по умолчанию empty — чистый граф под ваши запросы."""
    global agent, dialogue, comparer, cfg, llm_provider
    cfg = load_config()
    llm_provider = cfg.agent.llm_provider
    mode = (preload or "empty").strip().lower()
    if mode not in {"empty", "rabbit", "encyclopedia"}:
        mode = "empty"

    agent = Agent(store=_store_for_preload(mode), perception=_make_perception())
    dialogue = _build_dialogue(agent)
    dialogue.reset_history()
    comparer = _build_compare(agent)
    comparer.clear()

    stats = dump_graph(agent.store)["stats"]
    return {
        "ok": True,
        "preload": mode,
        "version": "0.4.3-merged",
        "llm_provider": llm_provider,
        "rag_backend": comparer.rag.backend,
        "corpus_chars": len(comparer.rag.corpus),
        "stats": stats,
    }


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _pids_listening_on(port: int) -> list[int]:
    """PIDs with TCP LISTEN on port (Windows netstat / Unix lsof|ss)."""
    import subprocess
    import sys

    pids: list[int] = []
    if sys.platform == "win32":
        r = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            check=False,
        )
        needle = f":{port}"
        for line in r.stdout.splitlines():
            if "LISTENING" not in line.upper() or needle not in line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            local = parts[1] if parts[0].upper().startswith("TCP") else parts[0]
            if not local.endswith(needle):
                continue
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid > 0 and pid not in pids:
                pids.append(pid)
        return pids

    for cmd in (
        ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
        ["fuser", f"{port}/tcp"],
    ):
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0 or not r.stdout.strip():
            continue
        for tok in r.stdout.replace(":", " ").split():
            try:
                pid = int(tok)
            except ValueError:
                continue
            if pid > 0 and pid not in pids:
                pids.append(pid)
        if pids:
            break
    return pids


def _free_port(port: int) -> None:
    import os
    import signal
    import sys
    import time

    me = os.getpid()
    for pid in _pids_listening_on(port):
        if pid == me:
            continue
        print(f"[ah-web] port {port} busy -> stop pid {pid}")
        try:
            if sys.platform == "win32":
                import subprocess

                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F", "/T"],
                    capture_output=True,
                    check=False,
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            print(f"[ah-web] could not stop {pid}: {exc}")
    for _ in range(20):
        left = [p for p in _pids_listening_on(port) if p != me]
        if not left:
            break
        time.sleep(0.15)


def main() -> None:
    import uvicorn

    c = load_config()
    _free_port(c.web.port)
    print(f"[ah-web] http://{c.web.host}:{c.web.port}")
    uvicorn.run("web.app:app", host=c.web.host, port=c.web.port, reload=False)


if __name__ == "__main__":
    main()
