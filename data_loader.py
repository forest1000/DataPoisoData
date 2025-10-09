"""Data loader utilities integrating Flamby datasets.

This module wires the project to the official `FLamby`_ repository by
resolving dataset builders directly from ``flamby.datasets`` packages.

.. _FLamby: https://github.com/owkin/FLamby
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.registry import Registry

Transforms = Callable[[Tensor], Tensor]


DATASET_REGISTRY: Registry[Callable[[], Any]] = Registry("dataset")

# Maps canonical dataset names to module paths inside the FLamby repository.
# The modules are imported lazily so that environments without FLamby installed
# still load this file, while `_load_flamby_dataset` will raise a descriptive
# error that references the official repository URL.
FLAMBY_MODULE_HINTS: Dict[str, Tuple[str, ...]] = {
    "fed-ixi": (
        "flamby.datasets.fed_ixi.fed_ixi",
        "flamby.datasets.fed_ixi",
    ),
    "fed_ixi": (
        "flamby.datasets.fed_ixi.fed_ixi",
        "flamby.datasets.fed_ixi",
    ),
    "fed-heart-disease": (
        "flamby.datasets.fed_heart_disease.fed_heart_disease",
        "flamby.datasets.fed_heart_disease",
    ),
    "fed_heart_disease": (
        "flamby.datasets.fed_heart_disease.fed_heart_disease",
        "flamby.datasets.fed_heart_disease",
    ),
    "fed-isic2019": (
        "flamby.datasets.fed_isic2019.fed_isic2019",
        "flamby.datasets.fed_isic2019",
    ),
    "fed_isic2019": (
        "flamby.datasets.fed_isic2019.fed_isic2019",
        "flamby.datasets.fed_isic2019",
    ),
}


@dataclass
class TransformConfig:
    resize: Optional[Tuple[int, int]] = None
    normalize: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    augment: bool = False


class TransformDataset(Dataset[Any]):
    """Wrap a dataset to apply a transform to the input tensor."""

    def __init__(self, base_dataset: Dataset[Any], transform: Optional[Transforms]) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        data, *rest = self.base_dataset[index]
        if self.transform is not None:
            data = self.transform(_to_tensor(data))
        return (data, *rest)


def register_dataset(name: str, builder: Callable[[], Any]) -> None:
    DATASET_REGISTRY.register(name, builder)


def build_client_dataloaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    num_workers: int = 4,
    resize: Optional[Tuple[int, int]] = None,
    normalize: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
    augment: bool = False,
    use_weighted_sampler: bool = False,
) -> Dict[int, Dict[str, DataLoader[Any]]]:
    """Return client dataloaders for train/val/test splits."""

    dataset_builder = _resolve_dataset_builder(dataset_name)
    dataset_obj = dataset_builder()
    _validate_num_clients(dataset_obj, num_clients, dataset_name)
    transform_config = TransformConfig(resize=resize, normalize=normalize, augment=augment)

    client_loaders: Dict[int, Dict[str, DataLoader[Any]]] = {}
    for client_id in range(num_clients):
        splits: Dict[str, DataLoader[Any]] = {}
        for split in ("train", "val", "test"):
            loader = _build_split_loader(
                dataset_obj,
                split=split,
                client_id=client_id,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            transform = _build_transform(transform_config, split)
            dataset = TransformDataset(loader.dataset, transform) if transform else loader.dataset
            sampler = None
            shuffle = split == "train"
            if use_weighted_sampler and split == "train":
                sampler = _build_weighted_sampler(dataset)
                shuffle = False
            splits[split] = clone_dataloader(
                loader,
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
                shuffle=shuffle,
            )
        client_loaders[client_id] = splits
    return client_loaders


def clone_dataloader(
    original: DataLoader[Any],
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler],
    shuffle: bool,
) -> DataLoader[Any]:
    """Clone a DataLoader with a new dataset and sampling strategy."""

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": getattr(original, "pin_memory", False),
        "drop_last": original.drop_last,
        "collate_fn": original.collate_fn,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(**kwargs)


def _resolve_dataset_builder(dataset_name: str) -> Callable[[], Any]:
    normalized = dataset_name.strip().lower()
    try:
        return DATASET_REGISTRY.get(normalized)
    except KeyError:
        pass
    return lambda: _load_flamby_dataset(dataset_name)


def _load_flamby_dataset(dataset_name: str) -> Any:
    normalized = dataset_name.strip().lower()
    module_candidates = _flamby_module_candidates(normalized)
    last_error: Optional[Exception] = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        dataset_cls = _find_federated_dataset_class(module)
        if dataset_cls is not None:
            return dataset_cls()
    hint = (
        "Ensure FLamby is installed from https://github.com/owkin/FLamby and "
        "that the dataset name matches an available benchmark."
    )
    if last_error is not None:
        raise ValueError(
            f"Unable to import FLamby dataset '{dataset_name}'. {hint}"
        ) from last_error
    raise ValueError(f"No federated dataset class found for '{dataset_name}'. {hint}")


def _flamby_module_candidates(dataset_name: str) -> Iterable[str]:
    if dataset_name in FLAMBY_MODULE_HINTS:
        return FLAMBY_MODULE_HINTS[dataset_name]
    slug = dataset_name.replace("-", "_").replace(" ", "_")
    return (
        f"flamby.datasets.{slug}.{slug}",
        f"flamby.datasets.{slug}",
    )


def _find_federated_dataset_class(module: Any) -> Optional[type]:
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if obj.__name__.lower().startswith("fed"):
            return obj
    return None


def _validate_num_clients(dataset_obj: Any, num_clients: int, dataset_name: str) -> None:
    possible_attrs = [
        "num_clients",
        "n_clients",
        "n_centers",
        "nb_clients",
        "number_of_clients",
    ]
    available = None
    for attr in possible_attrs:
        if hasattr(dataset_obj, attr):
            available = getattr(dataset_obj, attr)
            break
    if available is not None and int(available) != num_clients:
        raise ValueError(
            f"Requested num_clients={num_clients} but dataset '{dataset_name}' has {available} clients"
        )


def _build_split_loader(
    dataset_obj: Any,
    split: str,
    client_id: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader[Any]:
    method_name = f"{split}_dataloader"
    if not hasattr(dataset_obj, method_name):
        raise ValueError(f"Dataset does not provide split '{split}'")
    method = getattr(dataset_obj, method_name)
    loader = method(center=client_id, batch_size=batch_size, num_workers=num_workers)
    if not isinstance(loader, DataLoader):
        raise TypeError(f"Expected DataLoader for split '{split}' but received {type(loader)}")
    return loader


def _build_transform(config: TransformConfig, split: str) -> Optional[Transforms]:
    if config.resize is None and config.normalize is None and (not config.augment or split != "train"):
        return None

    def transform(tensor: Tensor) -> Tensor:
        out = tensor
        if config.resize is not None:
            out = _resize_tensor(out, config.resize)
        if config.normalize is not None:
            mean, std = config.normalize
            mean_tensor = torch.tensor(mean, dtype=out.dtype, device=out.device).view(-1, 1, 1)
            std_tensor = torch.tensor(std, dtype=out.dtype, device=out.device).view(-1, 1, 1)
            out = (out - mean_tensor) / std_tensor
        if config.augment and split == "train":
            if torch.rand(1).item() > 0.5:
                out = torch.flip(out, dims=[-1])
        return out

    return transform


def _resize_tensor(tensor: Tensor, size: Tuple[int, int]) -> Tensor:
    if tensor.dim() < 2:
        raise ValueError("Cannot resize tensor without spatial dimensions")
    tensor = tensor.unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _build_weighted_sampler(dataset: Dataset[Any]) -> WeightedRandomSampler:
    labels = []
    for _, label, *rest in dataset:  # type: ignore[misc]
        labels.append(int(label))
    if not labels:
        raise ValueError("Cannot build weighted sampler without labels")
    class_sample_count = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_sample_count.float()
    sample_weights = weights[torch.tensor(labels)]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def _to_tensor(data: Any) -> Tensor:
    if isinstance(data, Tensor):
        return data
    return torch.as_tensor(data)


__all__ = ["build_client_dataloaders", "register_dataset", "TransformDataset", "clone_dataloader"]
