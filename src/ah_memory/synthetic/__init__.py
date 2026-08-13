"""Synthetic world generator for AH-memory activation research."""
from __future__ import annotations

from ah_memory.synthetic.benchmark import (
    BenchmarkReport,
    QueryEvalResult,
    proof_view,
    run_benchmark,
)
from ah_memory.synthetic.config import SyntheticGraphConfig
from ah_memory.synthetic.graph_generator import SyntheticGraphGenerator
from ah_memory.synthetic.ground_truth import (
    SyntheticDocument,
    SyntheticQuery,
    SyntheticWorld,
)
from ah_memory.synthetic.ingest import IngestResult, ingest_world
from ah_memory.synthetic.presets import PRESETS, get_preset
from ah_memory.synthetic.serializer import export_dataset, export_zip, world_to_json

__all__ = [
    "BenchmarkReport",
    "IngestResult",
    "PRESETS",
    "QueryEvalResult",
    "SyntheticDocument",
    "SyntheticGraphConfig",
    "SyntheticGraphGenerator",
    "SyntheticQuery",
    "SyntheticWorld",
    "export_dataset",
    "export_zip",
    "get_preset",
    "ingest_world",
    "proof_view",
    "run_benchmark",
    "world_to_json",
]
