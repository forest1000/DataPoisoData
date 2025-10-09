from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch.utils.data import TensorDataset

from attack_module import LabelFlipAttack


def test_label_flip_attack_changes_expected_ratio() -> None:
    features = torch.zeros((10, 1))
    labels = torch.tensor([0] * 5 + [1] * 5)
    dataset = TensorDataset(features, labels)

    attack = LabelFlipAttack(target_clients={0}, poison_ratio=0.4, mapping={0: 1, 1: 0}, seed=1)
    poisoned = attack.apply_to_client_dataset(0, dataset)

    original_labels = [int(label) for _, label in dataset]
    poisoned_labels = [int(label) for _, label in poisoned]
    flipped = sum(o != p for o, p in zip(original_labels, poisoned_labels))
    assert flipped == 4


def test_label_flip_attack_non_target_client() -> None:
    features = torch.zeros((5, 1))
    labels = torch.tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(features, labels)

    attack = LabelFlipAttack(target_clients={1}, poison_ratio=1.0, mapping={0: 1, 1: 0}, seed=2)
    unchanged = attack.apply_to_client_dataset(0, dataset)

    original_labels = [int(label) for _, label in dataset]
    new_labels = [int(label) for _, label in unchanged]
    assert original_labels == new_labels
