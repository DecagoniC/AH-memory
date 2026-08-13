"""Entity resolution benchmark — separate from synthetic aggregation."""
from __future__ import annotations

from ah_memory.benchmarks.entity_resolution.cases import (
    CaseType,
    EntityResolutionCase,
    ExpectedRelation,
    ResolutionResult,
)

__all__ = [
    "CaseType",
    "EntityResolutionCase",
    "ExpectedRelation",
    "ResolutionResult",
    "EntityResolutionBenchmark",
    "run_entity_resolution_benchmark",
]


def __getattr__(name: str):
    if name in {"EntityResolutionBenchmark", "run_entity_resolution_benchmark"}:
        from ah_memory.benchmarks.entity_resolution import runner

        return getattr(runner, name)
    raise AttributeError(name)
