"""Evaluation and visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def evaluate_global_model(
    model: Any,
    dataloaders: Dict[int, Dict[str, DataLoader[Any]]],
    split: str,
    average: str = "macro",
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Evaluate the global model on the specified split across clients."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for client_loaders in dataloaders.values():
            loader = client_loaders.get(split)
            if loader is None:
                continue
            for batch in loader:
                inputs, targets = batch[:2]
                inputs = inputs.to(device)
                logits = model(inputs)
                predictions = torch.argmax(logits, dim=1).cpu()
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(predictions.tolist())

    if not y_true:
        raise ValueError(f"No samples available for split '{split}'")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def summarize_experiment(metrics_over_rounds: pd.DataFrame, out_dir: str) -> None:
    """Persist metrics to CSV and plot performance curves."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    metrics_over_rounds.to_csv(csv_path, index=False)

    if metrics_over_rounds.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = metrics_over_rounds.groupby(["attack", "defense", "split"])
    for (attack, defense, split), group in grouped:
        ax.plot(group["round"], group["f1"], label=f"{split}|attack={attack}|defense={defense}")

    ax.set_xlabel("Round")
    ax.set_ylabel("F1 score")
    ax.set_title("Federated Learning Performance")
    ax.legend(loc="best")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "curves.png")
    plt.close(fig)


__all__ = ["evaluate_global_model", "summarize_experiment"]
