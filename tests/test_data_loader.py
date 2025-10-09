from __future__ import annotations

import pytest

from data_loader import build_client_dataloaders


def test_build_client_dataloaders_dummy_dataset() -> None:
    dataloaders = build_client_dataloaders(
        dataset_name="dummy",
        num_clients=2,
        batch_size=4,
        num_workers=0,
        resize=None,
        normalize=None,
        augment=False,
        use_weighted_sampler=False,
    )
    assert len(dataloaders) == 2
    for client_id, splits in dataloaders.items():
        assert set(splits.keys()) == {"train", "val", "test"}
        for name, loader in splits.items():
            assert loader.batch_size == 4
            assert len(loader.dataset) > 0, f"{name} dataset for client {client_id} should not be empty"


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        build_client_dataloaders(
            dataset_name="unknown",
            num_clients=1,
            batch_size=2,
            num_workers=0,
        )
