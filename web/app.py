"""FastAPI web UI: chat + graph dump."""
from __future__ import annotations

import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ah_memory.agent import Agent
from ah_memory.config import LlmProvider, load_config
from ah_memory.deepseek import DeepSeekClient, DeepSeekHybridPerception
from ah_memory.dialogue import DialogueAgent
from ah_memory.factor_parameters import (
    EmbeddingParameterGenerator,
    FixedParameterGenerator,
    RuleBasedParameterGenerator,
)
from ah_memory.gigachat_llm import GigaChatClient, HybridPerception
from ah_memory.graph_export import dump_ah_json, dump_graph
from ah_memory.perception import SeedPerception
from ah_memory.relation_normalizer import (
    EmbeddingNormalizer,
    ExactNormalizer,
    RelationNormalizer,
)
from ah_memory.semantic_activation import (
    DecayActivation,
    LinearActivation,
    SaturatingReLUActivation,
    SigmoidActivation,
)
from ah_memory.store import AHStore
from ah_memory.synthetic import (
    SyntheticGraphConfig,
    SyntheticGraphGenerator,
    export_zip,
    get_preset,
    ingest_world,
    proof_view,
    run_benchmark,
)
from ah_memory.synthetic.config import DEFAULT_RELATION_TYPES, merge_config
from ah_memory.synthetic.ingest import IngestResult
from ah_memory.synthetic.ground_truth import SyntheticWorld

STATIC = Path(__file__).parent / "static"
cfg = load_config()
app = FastAPI(title="AH Memory", version="0.4.0")

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


def _empty_store() -> AHStore:
    store = AHStore()
    store.clear()
    return store


def _build_identity(store: AHStore):
    from ah_memory.identity import build_identity_service

    embed = None
    use_emb = bool(cfg.identity.use_embeddings)
    if use_emb:
        try:
            from ah_memory.benchmarks.entity_resolution.resolvers import make_embed_fn

            _name, embed, _dim = make_embed_fn(
                cfg.embedding.model, dimensions=cfg.embedding.dimensions
            )
        except Exception:
            use_emb = False
            embed = None
    return build_identity_service(
        store,
        enabled=cfg.identity.enabled,
        use_embeddings=use_emb,
        embed=embed,
        safety_threshold=cfg.identity.safety_threshold,
        margin=cfg.identity.margin,
    )


def _build_core(preload: str | None = None) -> Agent:
    del preload  # preload-режимы (rabbit/encyclopedia) сняты
    store = _empty_store()
    semantics_mode = cfg.open_semantics.normalization_mode.lower()
    if semantics_mode == "learned":
        strategies = (
            ExactNormalizer(),
            EmbeddingNormalizer(
                similarity_threshold=cfg.open_semantics.embedding_similarity_threshold
            ),
        )
        parameter_generator = EmbeddingParameterGenerator(
            seed=cfg.open_semantics.parameter_seed
        )
    elif semantics_mode == "fixed":
        strategies = (ExactNormalizer(),)
        parameter_generator = FixedParameterGenerator()
    else:
        strategies = (ExactNormalizer(),)
        parameter_generator = RuleBasedParameterGenerator()
    return Agent(
        store=store,
        perception=_make_perception(),
        relation_normalizer=RelationNormalizer(store.relations, strategies),
        parameter_generator=parameter_generator,
        identity=_build_identity(store),
    )


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


agent = _build_core()
dialogue = _build_dialogue(agent)

_synthetic_world: SyntheticWorld | None = None
_synthetic_ingest: IngestResult | None = None
_synthetic_report: dict[str, Any] | None = None
_synthetic_zip: Path | None = None
_ACTIVATION_FUNCS = {
    "linear": LinearActivation,
    "sigmoid": SigmoidActivation,
    "relu": SaturatingReLUActivation,
    "decay": DecayActivation,
}


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
    activation: dict[str, Any] = Field(default_factory=dict)
    graph_build_json: dict[str, Any] = Field(default_factory=dict)
    full_trace: dict[str, Any] = Field(default_factory=dict)


class ProviderIn(BaseModel):
    provider: Literal["gigachat", "deepseek"]


class SyntheticGenerateIn(BaseModel):
    preset: str = "small"
    num_entities: int | None = None
    num_events: int | None = None
    num_factors: int | None = None
    max_hop_depth: int | None = None
    distractor_ratio: float | None = None
    num_queries: int | None = None
    random_seed: int = 42
    relation_types: list[str] | None = None


class SyntheticBenchmarkIn(BaseModel):
    limit: int | None = None
    activation: Literal["linear", "sigmoid", "relu", "decay"] = "linear"
    timesteps: int = 4
    threshold: float = 0.05


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


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
        "version": "0.4.0-open-semantics",
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
        "open_semantics_mode": cfg.open_semantics.normalization_mode,
        "tau": agent.store.ah.tau,
        "history": len(dialogue.history),
        "graph_size": len(agent.store.ah.S) + len(agent.store.ah.L) + len(agent.store.semantic_factors),
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
    return dump_graph(
        agent.store,
        limit_nodes=limit,
        mode=mode,
        activation=agent.ignition.state.activation,
    )


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
        "full_trace": act.get("full_trace", {}),
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
                "activation_top": t.activation_top,
                "events": [asdict(event) for event in t.events],
                "convergence": t.convergence,
                "timings_ms": t.timings_ms,
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
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"dialogue failed: {exc}") from exc

    return ChatOut(
        reply=turn.reply,
        kind="dialogue",
        trace_uids=turn.trace_uids,
        wm=turn.wm,
        backend=turn.backend,
        stats=dump_graph(
            agent.store,
            limit_nodes=1,
            activation=agent.ignition.state.activation,
        )["stats"],
        user_facts=turn.user_facts,
        assistant_facts=turn.assistant_facts,
        system_prompt=turn.system_prompt,
        activation=turn.activation,
        graph_build_json=turn.graph_build_json,
        full_trace=turn.full_trace,
    )


@app.post("/api/reset")
def reset(preload: str = "empty") -> dict[str, Any]:
    """Always rebuild agent; default preload=empty wipes the graph."""
    global agent, dialogue, cfg, llm_provider
    global _synthetic_world, _synthetic_ingest, _synthetic_report, _synthetic_zip
    cfg = load_config()
    llm_provider = cfg.agent.llm_provider
    agent = _build_core()
    dialogue = _build_dialogue(agent)
    dialogue.reset_history()
    _synthetic_world = None
    _synthetic_ingest = None
    _synthetic_report = None
    _synthetic_zip = None
    stats = dump_graph(
        agent.store,
        activation=agent.ignition.state.activation,
    )["stats"]
    return {
        "ok": True,
        "preload": preload,
        "version": "0.4.0-open-semantics",
        "llm_provider": llm_provider,
        "stats": stats,
    }


@app.post("/api/synthetic/generate")
def synthetic_generate(body: SyntheticGenerateIn) -> dict[str, Any]:
    global agent, dialogue
    global _synthetic_world, _synthetic_ingest, _synthetic_report, _synthetic_zip
    started = time.perf_counter()
    try:
        base = get_preset(body.preset)
    except KeyError:
        base = SyntheticGraphConfig(preset="custom", random_seed=body.random_seed)
    overrides: dict[str, Any] = {
        "random_seed": body.random_seed,
        "preset": body.preset,
    }
    for key in (
        "num_entities",
        "num_events",
        "num_factors",
        "max_hop_depth",
        "distractor_ratio",
        "num_queries",
    ):
        value = getattr(body, key)
        if value is not None:
            overrides[key] = value
    if body.relation_types:
        overrides["relation_types"] = tuple(body.relation_types)
    else:
        overrides["relation_types"] = DEFAULT_RELATION_TYPES
    config = merge_config(base, overrides)
    world = SyntheticGraphGenerator(config).generate()
    agent = _build_core("empty")
    dialogue = _build_dialogue(agent)
    dialogue.reset_history()
    ingest = ingest_world(world, agent.store)
    zip_path = Path(tempfile.gettempdir()) / f"ah_synthetic_{config.random_seed}.zip"
    export_zip(world, zip_path)
    _synthetic_world = world
    _synthetic_ingest = ingest
    _synthetic_report = None
    _synthetic_zip = zip_path
    summary = world.to_summary()
    summary["generation_time_sec"] = round(
        world.generation_time_sec + (time.perf_counter() - started - world.generation_time_sec),
        4,
    )
    summary["ingest"] = ingest.to_dict()["stats"]
    summary["graph_stats"] = dump_graph(
        agent.store,
        activation=agent.ignition.state.activation,
    )["stats"]
    return {"ok": True, **summary}


@app.get("/api/synthetic/status")
def synthetic_status() -> dict[str, Any]:
    if _synthetic_world is None:
        return {"ok": True, "ready": False}
    return {
        "ok": True,
        "ready": True,
        **_synthetic_world.to_summary(),
        "benchmark": (
            {
                "aggregate": _synthetic_report.get("aggregate"),
                "activation_name": _synthetic_report.get("activation_name"),
            }
            if _synthetic_report
            else None
        ),
    }


@app.post("/api/synthetic/benchmark")
def synthetic_benchmark(body: SyntheticBenchmarkIn) -> dict[str, Any]:
    global _synthetic_report
    if _synthetic_world is None or _synthetic_ingest is None:
        raise HTTPException(400, "synthetic graph is not generated")
    fn_cls = _ACTIVATION_FUNCS.get(body.activation, LinearActivation)
    report = run_benchmark(
        agent.store,
        _synthetic_world,
        _synthetic_ingest,
        limit=body.limit,
        activation_function=fn_cls(),
        timesteps=body.timesteps,
        threshold=body.threshold,
        activation_name=body.activation,
    )
    _synthetic_report = report.to_dict()
    return {"ok": True, **_synthetic_report}


@app.get("/api/synthetic/queries")
def synthetic_queries() -> dict[str, Any]:
    if _synthetic_world is None:
        raise HTTPException(400, "synthetic graph is not generated")
    results_by_id = {}
    if _synthetic_report:
        results_by_id = {
            item["query_id"]: item for item in _synthetic_report.get("results") or []
        }
    items = []
    for query in _synthetic_world.queries:
        row = query.to_dict()
        if query.query_id in results_by_id:
            row["result"] = results_by_id[query.query_id]
        items.append(row)
    return {"ok": True, "queries": items}


@app.get("/api/synthetic/query/{query_id}")
def synthetic_query_detail(query_id: str) -> dict[str, Any]:
    if _synthetic_world is None:
        raise HTTPException(400, "synthetic graph is not generated")
    detail = None
    if _synthetic_report:
        detail = next(
            (
                item
                for item in _synthetic_report.get("results") or []
                if item.get("query_id") == query_id
            ),
            None,
        )
    try:
        view = proof_view(
            _synthetic_world,
            query_id,
            ingest=_synthetic_ingest,
        )
    except KeyError as exc:
        raise HTTPException(404, f"unknown query: {query_id}") from exc
    return {"ok": True, "proof": view, "result": detail}


@app.get("/api/synthetic/proof/{query_id}")
def synthetic_proof(query_id: str) -> dict[str, Any]:
    if _synthetic_world is None:
        raise HTTPException(400, "synthetic graph is not generated")
    try:
        view = proof_view(
            _synthetic_world,
            query_id,
            ingest=_synthetic_ingest,
        )
    except KeyError as exc:
        raise HTTPException(404, f"unknown query: {query_id}") from exc
    highlight = [
        node["ah_uid"] for node in view["nodes"] if node.get("ah_uid")
    ]
    for step in view["steps"]:
        if step.get("ah_factor_uid"):
            highlight.append(step["ah_factor_uid"])
    return {"ok": True, "proof": view, "highlight": highlight}


@app.get("/api/synthetic/download")
def synthetic_download() -> FileResponse:
    if _synthetic_zip is None or not _synthetic_zip.exists():
        raise HTTPException(400, "synthetic dataset is not available")
    return FileResponse(
        path=str(_synthetic_zip),
        filename=_synthetic_zip.name,
        media_type="application/zip",
    )


_er_report: dict[str, Any] | None = None


class EntityResolutionIn(BaseModel):
    embedding_model: str | None = None
    dimensions: int | None = None
    resolver: Literal["exact", "morphology", "embedding", "hybrid"] | None = None
    run_activation: bool = True


@app.post("/api/entity-resolution/benchmark")
def entity_resolution_benchmark(body: EntityResolutionIn) -> dict[str, Any]:
    """Run EntityResolutionBenchmark (separate from synthetic aggregation)."""
    global _er_report
    from ah_memory.benchmarks.entity_resolution.resolvers import (
        default_resolvers,
        make_embed_fn,
    )
    from ah_memory.benchmarks.entity_resolution.runner import (
        run_entity_resolution_benchmark,
    )

    model = body.embedding_model or cfg.embedding.model
    dims = body.dimensions or cfg.embedding.dimensions
    name, embed_fn, dims = make_embed_fn(model, dimensions=dims)
    resolvers = default_resolvers(embed_fn, dimensions=dims)
    if body.resolver:
        resolvers = {body.resolver: resolvers[body.resolver]}
    report = run_entity_resolution_benchmark(
        output_dir="results/entity_resolution",
        embedding_name=name,
        embed_fn=embed_fn,
        dimensions=dims,
        run_activation=body.run_activation,
        resolvers=resolvers,
    )
    _er_report = report
    return {"ok": True, **report}


@app.get("/api/entity-resolution/status")
def entity_resolution_status() -> dict[str, Any]:
    if _er_report is None:
        return {"ok": False, "ready": False}
    return {
        "ok": True,
        "ready": True,
        "dataset": _er_report.get("dataset"),
        "cases": _er_report.get("cases"),
        "embedding": _er_report.get("embedding"),
        "resolvers": _er_report.get("resolvers"),
        "threshold_sweep": _er_report.get("threshold_sweep"),
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
