from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from data_loader import register_dataset


class DummyFederatedDataset:
    def __init__(self, num_clients: int = 2) -> None:
        self.num_clients = num_clients
        self._data = {}
        for client_id in range(num_clients):
            torch.manual_seed(client_id)
            for split, length in {"train": 12, "val": 6, "test": 6}.items():
                features = torch.randn(length, 1, 8, 8)
                labels = torch.randint(0, 2, (length,))
                self._data[(split, client_id)] = TensorDataset(features, labels)

    def _loader(self, split: str, center: int, batch_size: int, num_workers: int) -> DataLoader:
        dataset = self._data[(split, center)]
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

    def train_dataloader(self, center: int, batch_size: int, num_workers: int) -> DataLoader:
        return self._loader("train", center, batch_size, num_workers)

    def val_dataloader(self, center: int, batch_size: int, num_workers: int) -> DataLoader:
        return self._loader("val", center, batch_size, num_workers)

    def test_dataloader(self, center: int, batch_size: int, num_workers: int) -> DataLoader:
        return self._loader("test", center, batch_size, num_workers)


register_dataset("dummy", lambda: DummyFederatedDataset())
