"""Federated learning strategies and client/server implementations."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


class Client(ABC):
    """Abstract client participating in federated learning."""

    @abstractmethod
    def local_train(self, global_state_dict: Dict[str, Any], epochs: int) -> Dict[str, Any]:
        """Train locally and return update payload."""


class Server(ABC):
    """Abstract server coordinating clients."""

    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client updates into a new global model state."""


class TorchClient(Client):
    """PyTorch client for local training."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader[Any],
        criterion: nn.Module,
        optimizer_builder: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        device: Optional[torch.device] = None,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer_builder = optimizer_builder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def local_train(self, global_state_dict: Dict[str, Any], epochs: int) -> Dict[str, Any]:
        self.model.load_state_dict(global_state_dict)
        self.model.train()
        optimizer = self.optimizer_builder(self.model.parameters())

        num_samples = 0
        total_batches = 0
        for _ in range(epochs):
            for batch in self.train_loader:
                inputs, targets = batch[:2]
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if total_batches == 0:  # 最初のエポックのみカウント
                    num_samples += inputs.size(0)
                total_batches += 1

        state_dict = copy.deepcopy(self.model.state_dict())
        return {"state_dict": state_dict, "num_samples": num_samples}

class FedAvgServer(Server):
    """Server implementing the FedAvg aggregation."""

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not client_updates:
            raise ValueError("client_updates must not be empty")
        reference = client_updates[0]["state_dict"]
        aggregated = {key: torch.zeros_like(value) for key, value in reference.items()}
        total_weight = 0.0

        for update in client_updates:
            state_dict = update["state_dict"]
            num_samples = float(update.get("num_samples", 1))
            total_weight += num_samples
            self._validate_state_dict(reference, state_dict)
            for key, value in state_dict.items():
                aggregated[key] += value * num_samples

        if total_weight == 0:
            raise ValueError("Total weight of updates is zero")
        for key in aggregated:
            aggregated[key] /= total_weight
        return aggregated

    def _validate_state_dict(
        self, reference: Dict[str, Any], candidate: Dict[str, Any]
    ) -> None:
        if reference.keys() != candidate.keys():
            raise ValueError("State dict keys mismatch during aggregation")
        for key in reference:
            ref_tensor = reference[key]
            cand_tensor = candidate[key]
            if ref_tensor.shape != cand_tensor.shape:
                raise ValueError(f"Tensor shape mismatch for key '{key}'")


__all__ = ["Client", "Server", "TorchClient", "FedAvgServer"]
