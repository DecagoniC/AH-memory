"""Named size presets for synthetic worlds."""
from __future__ import annotations

from ah_memory.synthetic.config import SyntheticGraphConfig


PRESETS: dict[str, SyntheticGraphConfig] = {
    "tiny": SyntheticGraphConfig(
        num_entities=24,
        num_factors=48,
        num_events=24,
        num_queries=16,
        max_hop_depth=2,
        distractor_ratio=0.2,
        random_seed=7,
        preset="tiny",
    ),
    "small": SyntheticGraphConfig(
        num_entities=100,
        num_factors=500,
        num_events=200,
        num_queries=50,
        max_hop_depth=3,
        distractor_ratio=0.3,
        random_seed=42,
        preset="small",
    ),
    "medium": SyntheticGraphConfig(
        num_entities=1000,
        num_factors=5000,
        num_events=2000,
        num_queries=500,
        max_hop_depth=5,
        distractor_ratio=0.3,
        random_seed=42,
        preset="medium",
    ),
    "large": SyntheticGraphConfig(
        num_entities=10000,
        num_factors=50000,
        num_events=20000,
        num_queries=5000,
        max_hop_depth=8,
        distractor_ratio=0.3,
        random_seed=42,
        preset="large",
    ),
    "stress": SyntheticGraphConfig(
        num_entities=25000,
        num_factors=100000,
        num_events=40000,
        num_queries=8000,
        max_hop_depth=8,
        distractor_ratio=0.35,
        random_seed=42,
        preset="stress",
    ),
}


def get_preset(name: str) -> SyntheticGraphConfig:
    key = (name or "small").strip().lower()
    if key not in PRESETS:
        raise KeyError(f"unknown preset: {name}")
    return PRESETS[key]
