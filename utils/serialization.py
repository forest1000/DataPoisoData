"""Model serialization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch


def save_state_dict(state_dict: Dict[str, Any], path: Path) -> None:
    """Save a model state_dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_state_dict(path: Path) -> Dict[str, Any]:
    """Load a model state_dict from disk."""
    if not path.exists():
        raise FileNotFoundError(f"State dict file not found: {path}")
    return torch.load(path, map_location="cpu")


__all__ = ["save_state_dict", "load_state_dict"]
