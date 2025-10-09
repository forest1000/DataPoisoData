from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate_global_model, summarize_experiment


def _build_single_client_dataloaders() -> dict[int, dict[str, DataLoader]]:
    features = torch.tensor([[0.0], [1.0], [0.0], [1.0]]).unsqueeze(-1).unsqueeze(-1)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return {0: {"val": loader, "test": loader, "train": loader}}


def test_evaluate_global_model_returns_expected_metrics() -> None:
    dataloaders = _build_single_client_dataloaders()
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 2))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model[1].bias.copy_(torch.tensor([0.5, -0.5]))

    metrics = evaluate_global_model(model, dataloaders, split="val")
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_summarize_experiment_writes_artifacts(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "round": [0, 1, 0, 1],
            "attack": [False, False, True, True],
            "defense": [False, False, True, True],
            "split": ["val", "val", "val", "val"],
            "accuracy": [0.8, 0.85, 0.5, 0.6],
            "precision": [0.8, 0.85, 0.5, 0.6],
            "recall": [0.8, 0.85, 0.5, 0.6],
            "f1": [0.8, 0.85, 0.5, 0.6],
        }
    )
    summarize_experiment(data, str(tmp_path))
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "curves.png").exists()
