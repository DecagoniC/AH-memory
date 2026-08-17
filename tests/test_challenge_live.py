from __future__ import annotations

import pytest

from ah_memory.benchmarks.challenge.loader import load_role_corpus
from ah_memory.benchmarks.challenge_evaluation import RoleBenchmarkItem, run_m1_benchmark
from ah_memory.config import load_config


pytestmark = pytest.mark.integration


def test_live_gigachat_parses_one_role_item() -> None:
    from ah_memory.gigachat_llm import GigaChatPerception

    cfg = load_config()
    if not cfg.gigachat.configured:
        pytest.skip("GigaChat is not configured")
    item = load_role_corpus()[0]
    report = run_m1_benchmark(
        GigaChatPerception(cfg.gigachat),
        [
            RoleBenchmarkItem(
                item.item_id,
                item.text,
                (dict(item.expected_roles),),
                item.variant,
            )
        ],
        model=cfg.gigachat.model,
    )
    assert report.items[0].error == ""
    assert report.items[0].raw_output is not None


def test_live_ollama_parses_one_role_item() -> None:
    from ah_memory.ollama import OllamaPerception, is_ollama_available

    cfg = load_config()
    if not is_ollama_available(cfg.ollama):
        pytest.skip("Ollama is not available")
    item = load_role_corpus()[0]
    report = run_m1_benchmark(
        OllamaPerception(cfg.ollama),
        [
            RoleBenchmarkItem(
                item.item_id,
                item.text,
                (dict(item.expected_roles),),
                item.variant,
            )
        ],
        model=cfg.ollama.model,
    )
    assert report.items[0].error == ""
    assert report.items[0].raw_output is not None
