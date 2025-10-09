"""Random seed management utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeedConfig:
    seed: int
    deterministic: bool = True


def set_seed(config: SeedConfig) -> None:
    """Set seeds for random, numpy, and torch modules."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["SeedConfig", "set_seed"]
