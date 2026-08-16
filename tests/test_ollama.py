from __future__ import annotations

import json

import httpx

from ah_memory.config import OllamaConfig, load_config
from ah_memory.ollama import OllamaClient, OllamaPerception, is_ollama_available
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
        "options": {"temperature": 0.25},
        "format": "json",
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
    ):
        monkeypatch.delenv(name, raising=False)
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
    )
    assert cfg.agent.llm_provider == "deepseek"
