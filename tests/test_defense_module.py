from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from defense_module import CosineSimilarityDefense


def test_cosine_similarity_defense_filters_outliers() -> None:
    base = {"w": torch.zeros(4), "b": torch.zeros(1)}
    update_good_1 = {"state_dict": {k: v.clone() for k, v in base.items()}, "num_samples": 10}
    update_good_2 = {"state_dict": {k: v.clone() + 0.01 for k, v in base.items()}, "num_samples": 10}
    update_bad = {
        "state_dict": {k: torch.randn_like(v) * 10 for k, v in base.items()},
        "num_samples": 10,
    }

    defense = CosineSimilarityDefense(topk=2)
    filtered = defense.filter_updates([update_good_1, update_good_2, update_bad])
    assert len(filtered) == 2
    assert update_bad not in filtered
