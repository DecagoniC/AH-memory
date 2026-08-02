"""FastAPI web UI: chat + graph dump."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ah_memory.agent import Agent
from ah_memory.config import load_config
from ah_memory.corpus import build_encyclopedia
from ah_memory.deepseek import HybridPerception
from ah_memory.dialogue import DialogueAgent
from ah_memory.examples.rabbit import build_rabbit_memory
from ah_memory.graph_export import dump_ah_json, dump_graph
from ah_memory.perception import RulePerception
from ah_memory.store import AHStore

STATIC = Path(__file__).parent / "static"
cfg = load_config()
app = FastAPI(title="AH Memory", version="0.3.0")


def _make_perception() -> Any:
    if cfg.agent.use_llm and cfg.deepseek.configured:
        return HybridPerception(cfg.deepseek, fallback=cfg.agent.fallback_rules)
    return RulePerception()


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
    # never silently fall through to rabbit
    mode = cfg.agent.preload if cfg.agent.preload in {"empty", "rabbit", "encyclopedia"} else "empty"
    return Agent(store=_store_for_preload(mode), perception=_make_perception())


def _build_dialogue(core: Agent) -> DialogueAgent:
    ds = cfg.deepseek if (cfg.agent.use_llm and cfg.deepseek.configured) else None
    return DialogueAgent(core, deepseek=ds)


agent = _build_core()
dialogue = _build_dialogue(agent)


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "0.3.1-no-bootstrap",
        "llm_configured": cfg.deepseek.configured,
        "use_llm": cfg.agent.use_llm,
        "model": cfg.deepseek.model if cfg.deepseek.configured else None,
        "preload": cfg.agent.preload,
        "tau": agent.store.ah.tau,
        "history": len(dialogue.history),
        "graph_size": len(agent.store.ah.S) + len(agent.store.ah.L) + len(agent.store.find_hypernodes()),
        "|S|": len(agent.store.ah.S),
    }


@app.get("/api/graph")
def graph(limit: int | None = 400, mode: str = "hyper") -> dict[str, Any]:
    if mode not in {"hyper", "all"}:
        mode = "hyper"
    return dump_graph(agent.store, limit_nodes=limit, mode=mode)


@app.get("/api/dump")
def full_dump() -> dict[str, Any]:
    return dump_ah_json(agent.store)


@app.post("/api/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    ticks = body.ticks or cfg.agent.ticks
    try:
        turn = dialogue.talk(text, ticks=ticks)
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
    )


@app.post("/api/reset")
def reset(preload: str = "empty") -> dict[str, Any]:
    """Always rebuild agent; default preload=empty wipes the graph."""
    global agent, dialogue, cfg
    cfg = load_config()
    mode = (preload or "empty").strip().lower()
    if mode not in {"empty", "rabbit", "encyclopedia"}:
        mode = "empty"

    agent = Agent(store=_store_for_preload(mode), perception=_make_perception())
    dialogue = DialogueAgent(
        agent,
        deepseek=cfg.deepseek if (cfg.agent.use_llm and cfg.deepseek.configured) else None,
    )
    dialogue.reset_history()
    stats = dump_graph(agent.store)["stats"]
    return {"ok": True, "preload": mode, "version": "0.3.1-no-bootstrap", "stats": stats}


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
            # match ...:8000 only as local port end
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
    """Stop foreign listeners on port so restart does not hit WinError 10048."""
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
    # brief wait for TIME_WAIT / handle release
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
