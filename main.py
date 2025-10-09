"""Main entry point for federated learning experiments."""
from __future__ import annotations

import argparse
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from attack_module import Attack, LabelFlipAttack
from data_loader import build_client_dataloaders, clone_dataloader
from defense_module import CosineSimilarityDefense, Defense
from evaluation import evaluate_global_model, summarize_experiment
from fl_strategy import FedAvgServer, TorchClient
from utils.logging import setup_logging
from utils.registry import Registry
from utils.seed import SeedConfig, set_seed
from utils.serialization import save_state_dict

STRATEGY_REGISTRY: Registry[type[object]] = Registry("strategy")
ATTACK_REGISTRY: Registry[type[Attack]] = Registry("attack")
DEFENSE_REGISTRY: Registry[type[Defense]] = Registry("defense")

STRATEGY_REGISTRY.register("fedavg", FedAvgServer)
ATTACK_REGISTRY.register("label_flip", LabelFlipAttack)
DEFENSE_REGISTRY.register("cosine", CosineSimilarityDefense)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a mapping")
    return config


def build_model(input_shape: torch.Size, num_classes: int) -> nn.Module:
    input_dim = int(torch.tensor(list(input_shape)).prod().item())
    if input_dim == 0:
        raise ValueError("Input dimension is zero")
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def make_optimizer_builder(name: str, lr: float) -> Any:
    name = name.lower()

    def builder(params: Any) -> torch.optim.Optimizer:
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr)
        raise ValueError(f"Unsupported optimizer '{name}'")

    return builder


def instantiate_attack(config: Dict[str, Any]) -> Attack | None:
    name = (config.get("name") or "").strip().lower()
    if not name:
        return None
    attack_cls = ATTACK_REGISTRY.get(name)
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return attack_cls(**kwargs)


def instantiate_defense(config: Dict[str, Any]) -> Defense | None:
    name = (config.get("name") or "").strip().lower()
    if not name:
        return None
    defense_cls = DEFENSE_REGISTRY.get(name)
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return defense_cls(**kwargs)


def select_clients(num_clients: int, fraction: float) -> List[int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("client_fraction must be within (0, 1]")
    count = max(1, math.ceil(num_clients * fraction))
    client_ids = list(range(num_clients))
    random.shuffle(client_ids)
    return sorted(client_ids[:count])


def apply_attack_to_dataloaders(attack: Attack, dataloaders: Dict[int, Dict[str, DataLoader[Any]]]) -> None:
    for client_id, splits in dataloaders.items():
        train_loader = splits.get("train")
        if train_loader is None:
            continue
        poisoned_dataset = attack.apply_to_client_dataset(client_id, train_loader.dataset)
        if poisoned_dataset is train_loader.dataset:
            continue
        sampler = train_loader.sampler if isinstance(train_loader.sampler, WeightedRandomSampler) else None
        shuffle = sampler is None
        splits["train"] = clone_dataloader(
            train_loader,
            dataset=poisoned_dataset,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            sampler=sampler,
            shuffle=shuffle,
        )


def infer_data_properties(dataloaders: Dict[int, Dict[str, DataLoader[Any]]]) -> tuple[torch.Size, int]:
    for loaders in dataloaders.values():
        train_loader = loaders.get("train")
        if train_loader is None:
            continue
        batch = next(iter(train_loader))
        inputs, targets = batch[:2]
        input_shape = inputs.shape[1:]
        max_label = int(torch.max(targets).item())
        num_classes = max_label + 1
        return input_shape, num_classes
    raise ValueError("Unable to infer data properties from dataloaders")


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated learning runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    experiment_config = config.get("experiment", {})
    output_root = Path(experiment_config.get("output_dir", "experiments/runs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_config = config.get("logging", {})
    logger = setup_logging(output_dir / "train.log", logging_config.get("level", "INFO"))

    seed_value = experiment_config.get("seed", 42)
    set_seed(SeedConfig(seed=seed_value))

    data_config = config.get("data", {})
    dataloaders = build_client_dataloaders(**data_config)

    attack_config = config.get("attack", {})
    attack = instantiate_attack(attack_config)
    attack_enabled = attack is not None
    if attack:
        apply_attack_to_dataloaders(attack, dataloaders)

    input_shape, num_classes = infer_data_properties(dataloaders)
    global_model = build_model(input_shape, num_classes)
    criterion = nn.CrossEntropyLoss()

    fl_config = config.get("fl", {})
    rounds = int(fl_config.get("rounds", 1))
    local_epochs = int(fl_config.get("local_epochs", 1))
    lr = float(fl_config.get("lr", 0.001))
    optimizer_name = fl_config.get("optimizer", "sgd")
    client_fraction = float(fl_config.get("client_fraction", 1.0))

    optimizer_builder = make_optimizer_builder(optimizer_name, lr)

    clients: Dict[int, TorchClient] = {}
    for client_id, loaders in dataloaders.items():
        model = build_model(input_shape, num_classes)
        clients[client_id] = TorchClient(
            client_id=client_id,
            model=model,
            train_loader=loaders["train"],
            criterion=criterion,
            optimizer_builder=optimizer_builder,
        )

    defense_config = config.get("defense", {})
    defense = instantiate_defense(defense_config)
    defense_enabled = defense is not None

    server_cls = STRATEGY_REGISTRY.get(fl_config.get("strategy", "fedavg"))
    server = server_cls()
    global_state = global_model.state_dict()

    eval_config = config.get("eval", {})
    eval_splits = eval_config.get("splits", ["val"])
    average = eval_config.get("average", "macro")

    records: List[Dict[str, Any]] = []

    for round_idx in range(rounds):
        logger.info("Starting round %s", round_idx)
        selected_clients = select_clients(len(clients), client_fraction)
        updates = []
        for client_id in selected_clients:
            update = clients[client_id].local_train(global_state, local_epochs)
            updates.append(update)
        if defense:
            updates = defense.filter_updates(updates)
        global_state = server.aggregate(updates)
        global_model.load_state_dict(global_state)

        for split in eval_splits:
            metrics = evaluate_global_model(
                global_model,
                dataloaders,
                split=split,
                average=average,
            )
            record = {
                "round": round_idx,
                "attack": attack_enabled,
                "defense": defense_enabled,
                "split": split,
                **metrics,
            }
            records.append(record)
            logger.info(
                "Round %s | Split %s | Acc %.4f | F1 %.4f",
                round_idx,
                split,
                metrics["accuracy"],
                metrics["f1"],
            )

    metrics_df = pd.DataFrame(records)
    summarize_experiment(metrics_df, str(output_dir))

    final_state_path = output_dir / "global_final.pt"
    save_state_dict(global_model.state_dict(), final_state_path)
    shutil.copy(config_path, output_dir / "config.used.yaml")


if __name__ == "__main__":
    main()
