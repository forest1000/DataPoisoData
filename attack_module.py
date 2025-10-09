"""Attack simulations for federated learning."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Set, List

import numpy as np
import torch
from torch.utils.data import Dataset


class Attack(ABC):
    """Base class for adversarial attacks on client datasets."""

    @abstractmethod
    def apply_to_client_dataset(self, client_id: int, dataset: Any) -> Any:
        """Return a possibly modified dataset for the given client."""


class LabelFlipDataset(Dataset[Any]):
    """Dataset wrapper that flips labels for selected indices."""

    def __init__(
        self,
        base_dataset: Dataset[Any],
        poisoned_indices: Set[int],
        mapping: Dict[int, int],
    ) -> None:
        self.base_dataset = base_dataset
        self.poisoned_indices = poisoned_indices
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self.base_dataset[index]
        if not isinstance(sample, Iterable) or isinstance(sample, torch.Tensor):
            raise TypeError("Dataset samples must be tuple-like (data, label)")
        data, *rest = sample
        if not rest:
            raise ValueError("Dataset sample must contain a label component")
        label = rest[0]
        label_int = int(label)
        if index in self.poisoned_indices:
            label_int = self.mapping.get(label_int, label_int)
        if isinstance(label, torch.Tensor):
            label = torch.as_tensor(label_int, dtype=label.dtype)
        else:
            label = type(label)(label_int)
        if len(rest) == 1:
            return data, label
        return (data, label, *rest[1:])


class LabelFlipAttack(Attack):
    """Label flipping attack targeting a subset of clients."""

    def __init__(
        self,
        target_clients: Set[int],
        poison_ratio: float,
        mapping: Dict[int, int] | None = None,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= poison_ratio <= 1.0:
            raise ValueError("poison_ratio must be within [0, 1]")
        self.target_clients = target_clients
        self.poison_ratio = poison_ratio
        self.mapping = mapping
        self.rng = np.random.default_rng(seed)

    def apply_to_client_dataset(self, client_id: int, dataset: Any) -> Any:
        if client_id not in self.target_clients or self.poison_ratio == 0.0:
            return dataset
        if not isinstance(dataset, Dataset):
            raise TypeError("LabelFlipAttack requires torch.utils.data.Dataset instances")
        mapping = self.mapping or self._default_mapping(dataset)
        indices = self._select_poisoned_indices(len(dataset))
        return LabelFlipDataset(dataset, indices, mapping)

    def _default_mapping(self, dataset: Dataset[Any]) -> Dict[int, int]:
        labels = self._collect_labels(dataset)
        unique = sorted(set(labels))
        if len(unique) < 2:
            raise ValueError("LabelFlipAttack requires at least two unique labels")
        mapping: Dict[int, int] = {}
        for label in unique:
            idx = unique.index(label)
            mapping[label] = unique[(idx + 1) % len(unique)]
        return mapping

    def _collect_labels(self, dataset: Dataset[Any]) -> List[int]:
        labels = []
        for _, label, *rest in dataset:  # type: ignore[misc]
            labels.append(int(label))
        return labels

    def _select_poisoned_indices(self, dataset_size: int) -> Set[int]:
        poison_count = math.floor(dataset_size * self.poison_ratio)
        if poison_count == 0:
            return set()
        indices = self.rng.choice(dataset_size, size=poison_count, replace=False)
        return set(int(i) for i in indices)


__all__ = ["Attack", "LabelFlipAttack"]
