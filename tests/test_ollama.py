from __future__ import annotations

import json

import httpx

from ah_memory.config import OllamaConfig, _norm_provider, load_config
from ah_memory.ollama import (
    OllamaClient,
    OllamaHybridPerception,
    OllamaPerception,
    is_ollama_available,
)
from ah_memory.perception_prompt import SYSTEM_PROMPT, build_user_payload


def test_client_sends_native_chat_json_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"kind":"message"}'}},
        )

    cfg = OllamaConfig(model="chat-model", temperature=0.25)
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))
    messages = [{"role": "user", "content": "input"}]

    assert client.chat(messages) == '{"kind":"message"}'
    assert len(requests) == 1
    assert requests[0].url == "http://127.0.0.1:11434/api/chat"
    assert json.loads(requests[0].content) == {
        "model": "chat-model",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.25, "num_ctx": 2048},
        "format": "json",
        "think": False,
    }


def test_client_chat_stream_yields_deltas() -> None:
    requests: list[httpx.Request] = []
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Hel"}}),
        json.dumps({"message": {"role": "assistant", "content": "lo"}}),
        json.dumps(
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode("utf-8"))

    cfg = OllamaConfig(model="chat-model", temperature=0.1, num_ctx=1024)
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))
    messages = [{"role": "user", "content": "hi"}]

    assert list(client.chat_stream(messages, json_mode=False)) == ["Hel", "lo"]
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {
        "model": "chat-model",
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 1024},
    }


def test_client_sends_one_batched_embedding_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"embeddings": [[1, 2.5], [3.25, 4]]})

    cfg = OllamaConfig(embedding_model="embedding-model")
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))

    assert client.embeddings(["first", "second"]) == [[1.0, 2.5], [3.25, 4.0]]
    assert len(requests) == 1
    assert requests[0].url == "http://127.0.0.1:11434/api/embed"
    assert json.loads(requests[0].content) == {
        "model": "embedding-model",
        "input": ["first", "second"],
    }


def test_perception_uses_shared_prompt_and_exactly_one_request() -> None:
    requests: list[httpx.Request] = []
    response_payload = {
        "kind": "fact",
        "candidates": [
            {
                "raw_relation": "links",
                "canonical_relation": "LINKS",
                "predicate": "LINKS",
                "roles": {"SUBJECT": "Entity A", "OBJECT": "Entity B"},
                "raw_span": "Entity A links Entity B",
                "confidence": 0.9,
                "statement_type": "assertion",
            }
        ],
        "seed_tokens": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(response_payload)}},
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    backend = OllamaPerception(require_grounding=False, client=client)
    result = backend.parse("Entity A links Entity B", ["context"])

    assert len(requests) == 1
    sent = json.loads(requests[0].content)
    assert sent["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_payload("Entity A links Entity B", ["context"]),
        },
    ]
    assert result.kind == "fact"
    assert len(result.candidates) == 1
    assert result.meta["backend"] == "ollama"


def test_availability_helper_handles_success_and_connection_failure() -> None:
    available = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"version": "test"})
    )

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    assert is_ollama_available(transport=available)
    assert not is_ollama_available(transport=httpx.MockTransport(fail))


def test_config_loads_ollama_yaml_and_environment_overrides(
    tmp_path, monkeypatch
) -> None:
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_CHAT_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_TIMEOUT_SEC",
        "OLLAMA_TEMPERATURE",
        "OLLAMA_NUM_CTX",
        "LLM_PROVIDER",
    ):
        monkeypatch.setenv(name, "")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ollama:
  base_url: http://localhost:22000
  model: yaml-chat
  embedding_model: yaml-embed
  timeout_sec: 12
  temperature: 0.2
agent:
  llm_provider: deepseek
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "env-chat")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SEC", "3.5")

    cfg = load_config(config_path)

    assert cfg.ollama == OllamaConfig(
        base_url="http://localhost:22000",
        model="env-chat",
        embedding_model="yaml-embed",
        timeout_sec=3.5,
        temperature=0.2,
        num_ctx=2048,
    )
    assert cfg.agent.llm_provider == "deepseek"


def test_norm_provider_accepts_ollama_aliases() -> None:
    assert _norm_provider("ollama") == "ollama"
    assert _norm_provider("LOCAL") == "ollama"
    assert _norm_provider("deepseek") == "deepseek"
    assert _norm_provider("gigachat") == "gigachat"


def test_config_loads_ollama_as_llm_provider(tmp_path, monkeypatch) -> None:
    for name in (
        "LLM_PROVIDER",
        "OLLAMA_CHAT_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT_SEC",
        "OLLAMA_NUM_CTX",
    ):
        monkeypatch.setenv(name, "")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm_provider: ollama
ollama:
  model: yaml-local
  num_ctx: 2048
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.agent.llm_provider == "ollama"
    assert cfg.ollama.model == "yaml-local"
    assert cfg.ollama.num_ctx == 2048
    assert cfg.ollama.timeout_sec == 300


def test_client_omits_num_ctx_when_disabled() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "{}"}},
        )

    cfg = OllamaConfig(model="chat-model", num_ctx=None)
    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))
    client.chat([{"role": "user", "content": "x"}])
    sent = json.loads(requests[0].content)
    assert sent["options"] == {"temperature": 0.1}


def test_hybrid_perception_falls_back_when_server_offline() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = OllamaClient(transport=httpx.MockTransport(fail))
    backend = OllamaHybridPerception(fallback=True, client=client)
    result = backend.parse("Entity A links Entity B")
    assert result.meta["backend"] == "seeds"
    assert "llm_error" in result.meta
    assert result.candidates == []


def test_hybrid_perception_uses_llm_when_chat_succeeds() -> None:
    response_payload = {
        "kind": "fact",
        "candidates": [
            {
                "raw_relation": "links",
                "canonical_relation": "LINKS",
                "predicate": "LINKS",
                "roles": {"SUBJECT": "Entity A", "OBJECT": "Entity B"},
                "raw_span": "Entity A links Entity B",
                "confidence": 0.9,
                "statement_type": "assertion",
            }
        ],
        "seed_tokens": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(response_payload)}},
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    backend = OllamaHybridPerception(
        fallback=True,
        client=client,
    )
    result = backend.parse("Entity A links Entity B")
    assert result.kind == "fact"
    assert len(result.candidates) == 1
    assert result.meta["backend"] == "hybrid-llm"


def test_list_ollama_models_parses_tags_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/tags")
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:8b", "size": 5200000000},
                    {"name": "nomic-embed-text:latest", "size": 274000000},
                    {"name": "qwen3:30b-a3b", "size": 18000000000},
                ]
            },
        )

    from ah_memory.ollama import is_likely_embedding_model, list_ollama_models

    models = list_ollama_models(transport=httpx.MockTransport(handler))
    names = [m["name"] for m in models]
    assert names == ["nomic-embed-text:latest", "qwen3:30b-a3b", "qwen3:8b"]
    assert is_likely_embedding_model("nomic-embed-text:latest")
    assert not is_likely_embedding_model("qwen3:8b")


def test_make_embed_fn_reports_offline_ollama(monkeypatch) -> None:
    from ah_memory.benchmarks.entity_resolution.resolvers import make_embed_fn

    monkeypatch.setattr(
        "ah_memory.ollama.is_ollama_available",
        lambda *args, **kwargs: False,
    )
    try:
        make_embed_fn("nomic-embed-text")
    except RuntimeError as exc:
        assert "Ollama is not reachable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when Ollama is offline")
