from __future__ import annotations

from pathlib import Path


CORE_MODULES = (
    "perception.py",
    "perception_prompt.py",
    "gigachat_llm.py",
    "deepseek.py",
    "dialogue.py",
    "agent.py",
    "transform.py",
    "relation_registry.py",
    "relation_normalizer.py",
    "state_engine.py",
)

FORBIDDEN_DOMAIN_TOKENS = (
    "OUTBOX",
    "POSTGRESQL",
    "KAFKA",
    "STORE_EVENT",
    "ARCHIVE_TO",
    "EVENT_SOURCING",
    "PURCHASE",
    "LAST_PURCHASE",
    "PURCHASE_HISTORY",
    "LIVE_IN",
    "DESIGNING",
)


def test_core_modules_contain_no_fixture_vocabulary() -> None:
    source_root = Path(__file__).parents[1] / "src" / "ah_memory"
    violations: list[str] = []
    for module in CORE_MODULES:
        text = (source_root / module).read_text(encoding="utf-8").upper()
        for token in FORBIDDEN_DOMAIN_TOKENS:
            if token in text:
                violations.append(f"{module}: {token}")
    assert violations == []
