from __future__ import annotations

from dataclasses import dataclass

from .protocol import DATASET_SEED


@dataclass(frozen=True)
class LoraControlConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    fixed_updates: int = 32
    effective_batch_size: int = 32
    seed: int = DATASET_SEED
