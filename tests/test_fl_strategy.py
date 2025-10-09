from __future__ import annotations

import pytest

pytest.importorskip("torch")
from torch import nn

from data_loader import build_client_dataloaders
from fl_strategy import FedAvgServer, TorchClient
from main import build_model, infer_data_properties, make_optimizer_builder


def test_fedavg_round_progression() -> None:
    dataloaders = build_client_dataloaders(
        dataset_name="dummy",
        num_clients=2,
        batch_size=4,
        num_workers=0,
    )
    input_shape, num_classes = infer_data_properties(dataloaders)
    global_model = build_model(input_shape, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_builder = make_optimizer_builder("sgd", 0.01)

    clients = []
    for client_id, loaders in dataloaders.items():
        model = build_model(input_shape, num_classes)
        clients.append(
            TorchClient(
                client_id=client_id,
                model=model,
                train_loader=loaders["train"],
                criterion=criterion,
                optimizer_builder=optimizer_builder,
            )
        )

    server = FedAvgServer()
    global_state = global_model.state_dict()
    updates = [client.local_train(global_state, epochs=1) for client in clients]
    aggregated = server.aggregate(updates)

    assert set(aggregated.keys()) == set(global_state.keys())
    for key, tensor in aggregated.items():
        assert tensor.shape == global_state[key].shape
