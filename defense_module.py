"""Defense mechanisms against malicious client updates."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from utils.logging import get_logger


class Defense(ABC):
    """Abstract base class for server-side defenses."""

    @abstractmethod
    def filter_updates(self, client_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a filtered list of client updates."""


class CosineSimilarityDefense(Defense):
    """Filter updates based on cosine similarity to the mean update."""

    def __init__(self, threshold: float | None = None, topk: int | None = None) -> None:
        if threshold is None and topk is None:
            raise ValueError("Either threshold or topk must be provided")
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [-1, 1]")
        if topk is not None and topk <= 0:
            raise ValueError("topk must be positive")
        self.threshold = threshold
        self.topk = topk
        self.logger = get_logger()

    def filter_updates(self, client_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not client_updates:
            return []
        vectors = [self._flatten(update["state_dict"]) for update in client_updates]
        stacked = torch.stack(vectors)
        mean_vector = stacked.mean(dim=0)
        similarities = F.cosine_similarity(stacked, mean_vector.unsqueeze(0))

        if self.threshold is not None:
            mask = similarities >= self.threshold
        else:
            topk = min(self.topk or len(similarities), len(similarities))
            _, indices = torch.topk(similarities, k=topk)
            mask = torch.zeros_like(similarities, dtype=torch.bool)
            mask[indices] = True

        filtered = [update for update, keep in zip(client_updates, mask) if bool(keep)]
        if not filtered:
            self.logger.warning(
                "All client updates filtered out; falling back to using all updates."
            )
            return client_updates
        return filtered

    def _flatten(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        tensors = []
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"State dict entry {key} is not a tensor")
            tensors.append(value.detach().flatten())
        return torch.cat(tensors)


__all__ = ["Defense", "CosineSimilarityDefense"]
