"""Data loader utilities integrating Flamby datasets.

This module provides individual adapter functions for each Flamby dataset,
respecting the original API and data splits provided by each dataset.

.. _FLamby: https://github.com/owkin/FLamby
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.logging import get_logger
from utils.registry import Registry

Transforms = Callable[[Tensor], Tensor]

logger = get_logger()

DATASET_REGISTRY: Registry[Callable[[], Any]] = Registry("dataset")


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


def _to_tensor(data: Any) -> Tensor:
    """Convert data to torch.Tensor."""
    if isinstance(data, Tensor):
        return data
    return torch.as_tensor(data)


# ============================================================================
# FedIXI Dataset Adapter
# ============================================================================

class FedIXIAdapter:
    """Adapter for FedIXI and FedIXITiny datasets.
    
    API reference:
        from flamby.datasets.fed_ixi import FedIXITiny
        dataset = FedIXITiny(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    Note: No explicit validation split, using test as validation in this adapter
    """

    def __init__(self, use_tiny: bool = True) -> None:
        self.use_tiny = use_tiny
        self.num_clients = 3  # FedIXI has 3 centers
        
        try:
            if use_tiny:
                from flamby.datasets.fed_ixi import FedIXITiny
                self.dataset_cls = FedIXITiny
            else:
                from flamby.datasets.fed_ixi import FedIXI
                self.dataset_cls = FedIXI
        except ImportError as e:
            raise ImportError(
                "Failed to import FedIXI. Install with: pip install flamby[fed-ixi]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        # FedIXI doesn't have explicit validation split, using test as validation
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# FedHeartDisease Dataset Adapter
# ============================================================================

class FedHeartDiseaseAdapter:
    """Adapter for FedHeartDisease dataset.
    
    API reference:
        from flamby.datasets.fed_heart_disease import FedHeartDisease
        dataset = FedHeartDisease(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    - 4 centers (hospitals)
    """

    def __init__(self) -> None:
        self.num_clients = 4  # FedHeartDisease has 4 centers
        
        try:
            from flamby.datasets.fed_heart_disease import FedHeartDisease
            self.dataset_cls = FedHeartDisease
        except ImportError as e:
            raise ImportError(
                "Failed to import FedHeartDisease. "
                "Install with: pip install flamby[fed-heart-disease]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        # Using test split as validation
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# FedISIC2019 Dataset Adapter
# ============================================================================

class FedISIC2019Adapter:
    """Adapter for FedISIC2019 dataset.
    
    API reference:
        from flamby.datasets.fed_isic2019 import FedIsic2019
        dataset = FedIsic2019(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    - 6 centers (different institutions)
    """

    def __init__(self) -> None:
        self.num_clients = 6  # FedISIC2019 has 6 centers
        
        try:
            from flamby.datasets.fed_isic2019 import FedIsic2019
            self.dataset_cls = FedIsic2019
        except ImportError as e:
            raise ImportError(
                "Failed to import FedISIC2019. "
                "Install with: pip install flamby[fed-isic2019]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        # Using test split as validation
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# FedCamelyon16 Dataset Adapter
# ============================================================================

class FedCamelyon16Adapter:
    """Adapter for FedCamelyon16 dataset.
    
    API reference:
        from flamby.datasets.fed_camelyon16 import FedCamelyon16
        dataset = FedCamelyon16(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    - 2 centers
    """

    def __init__(self) -> None:
        self.num_clients = 2  # FedCamelyon16 has 2 centers
        
        try:
            from flamby.datasets.fed_camelyon16 import FedCamelyon16
            self.dataset_cls = FedCamelyon16
        except ImportError as e:
            raise ImportError(
                "Failed to import FedCamelyon16. "
                "Install with: pip install flamby[fed-camelyon16]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# FedKiTS19 Dataset Adapter
# ============================================================================

class FedKiTS19Adapter:
    """Adapter for FedKiTS19 dataset.
    
    API reference:
        from flamby.datasets.fed_kits19 import FedKits19
        dataset = FedKits19(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    - 6 centers
    """

    def __init__(self) -> None:
        self.num_clients = 6  # FedKiTS19 has 6 centers
        
        try:
            from flamby.datasets.fed_kits19 import FedKits19
            self.dataset_cls = FedKits19
        except ImportError as e:
            raise ImportError(
                "Failed to import FedKiTS19. "
                "Install with: pip install flamby[fed-kits19]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# FedLiTS Dataset Adapter
# ============================================================================

class FedLiTSAdapter:
    """Adapter for FedLiTS dataset.
    
    API reference:
        from flamby.datasets.fed_lidc_idri import FedLidc
        dataset = FedLidc(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    - 4 centers
    """

    def __init__(self) -> None:
        self.num_clients = 4  # FedLiTS has 4 centers
        
        try:
            from flamby.datasets.fed_lidc_idri import FedLidc
            self.dataset_cls = FedLidc
        except ImportError as e:
            raise ImportError(
                "Failed to import FedLiTS. "
                "Install with: pip install flamby[fed-lidc-idri]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# FedTCGA-BRCA Dataset Adapter
# ============================================================================

class FedTCGABRCAAdapter:
    """Adapter for FedTCGA-BRCA dataset.
    
    API reference:
        from flamby.datasets.fed_tcga_brca import FedTcgaBrca
        dataset = FedTcgaBrca(center=0, train=True, pooled=False)
    
    The dataset provides:
    - train=True: training data
    - train=False: test data
    - 6 centers
    """

    def __init__(self) -> None:
        self.num_clients = 6  # FedTCGA-BRCA has 6 centers
        
        try:
            from flamby.datasets.fed_tcga_brca import FedTcgaBrca
            self.dataset_cls = FedTcgaBrca
        except ImportError as e:
            raise ImportError(
                "Failed to import FedTCGA-BRCA. "
                "Install with: pip install flamby[fed-tcga-brca]"
            ) from e

    def train_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=True, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(
        self, center: int, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        dataset = self.dataset_cls(center=center, train=False, pooled=False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )


# ============================================================================
# Registry and Main API
# ============================================================================

def register_dataset(name: str, builder: Callable[[], Any]) -> None:
    """Register a custom dataset builder."""
    DATASET_REGISTRY.register(name, builder)


# Register all Flamby datasets
DATASET_REGISTRY.register("fed-ixi", lambda: FedIXIAdapter(use_tiny=True))
DATASET_REGISTRY.register("fed_ixi", lambda: FedIXIAdapter(use_tiny=True))
DATASET_REGISTRY.register("fed-ixi-tiny", lambda: FedIXIAdapter(use_tiny=True))
DATASET_REGISTRY.register("fed-ixi-full", lambda: FedIXIAdapter(use_tiny=False))

DATASET_REGISTRY.register("fed-heart-disease", lambda: FedHeartDiseaseAdapter())
DATASET_REGISTRY.register("fed_heart_disease", lambda: FedHeartDiseaseAdapter())

DATASET_REGISTRY.register("fed-isic2019", lambda: FedISIC2019Adapter())
DATASET_REGISTRY.register("fed_isic2019", lambda: FedISIC2019Adapter())

DATASET_REGISTRY.register("fed-camelyon16", lambda: FedCamelyon16Adapter())
DATASET_REGISTRY.register("fed_camelyon16", lambda: FedCamelyon16Adapter())

DATASET_REGISTRY.register("fed-kits19", lambda: FedKiTS19Adapter())
DATASET_REGISTRY.register("fed_kits19", lambda: FedKiTS19Adapter())

DATASET_REGISTRY.register("fed-lits", lambda: FedLiTSAdapter())
DATASET_REGISTRY.register("fed_lits", lambda: FedLiTSAdapter())
DATASET_REGISTRY.register("fed-lidc-idri", lambda: FedLiTSAdapter())

DATASET_REGISTRY.register("fed-tcga-brca", lambda: FedTCGABRCAAdapter())
DATASET_REGISTRY.register("fed_tcga_brca", lambda: FedTCGABRCAAdapter())


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
    """Return client dataloaders for train/val/test splits.
    
    Args:
        dataset_name: Name of the Flamby dataset (e.g., 'fed-ixi', 'fed-heart-disease')
        num_clients: Number of clients/centers to use
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        resize: Optional (height, width) to resize images
        normalize: Optional (mean, std) tuples for normalization
        augment: Whether to apply data augmentation (train only)
        use_weighted_sampler: Whether to use weighted sampling for class balance
    
    Returns:
        Dictionary mapping client_id to dict of split_name to DataLoader
        
    Example:
        >>> loaders = build_client_dataloaders('fed-ixi', num_clients=3, batch_size=16)
        >>> train_loader = loaders[0]['train']
        >>> for X, y in train_loader:
        ...     # Training code
    """
    # Resolve dataset adapter
    normalized_name = dataset_name.strip().lower()
    try:
        dataset_adapter = DATASET_REGISTRY.get(normalized_name)
    except KeyError as e:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        ) from e

    # Validate num_clients
    if hasattr(dataset_adapter, 'num_clients'):
        expected_clients = dataset_adapter.num_clients
        if num_clients != expected_clients:
            logger.warning(
                f"Dataset '{dataset_name}' has {expected_clients} centers, "
                f"but num_clients={num_clients} was requested. "
                f"Using first {num_clients} centers."
            )

    # Build transform config
    transform_config = TransformConfig(
        resize=resize,
        normalize=normalize,
        augment=augment
    )

    # Build dataloaders for each client
    client_loaders: Dict[int, Dict[str, DataLoader[Any]]] = {}
    
    for client_id in range(num_clients):
        splits: Dict[str, DataLoader[Any]] = {}
        
        for split in ("train", "val", "test"):
            # Get base dataloader from adapter
            method_name = f"{split}_dataloader"
            if not hasattr(dataset_adapter, method_name):
                raise ValueError(
                    f"Dataset adapter for '{dataset_name}' does not provide '{split}' split"
                )
            
            method = getattr(dataset_adapter, method_name)
            loader = method(center=client_id, batch_size=batch_size, num_workers=num_workers)
            
            # Apply transforms
            transform = _build_transform(transform_config, split)
            if transform:
                dataset = TransformDataset(loader.dataset, transform)
            else:
                dataset = loader.dataset
            
            # Apply weighted sampler if requested
            sampler = None
            shuffle = split == "train"
            if use_weighted_sampler and split == "train":
                try:
                    sampler = _build_weighted_sampler(dataset)
                    shuffle = False
                except Exception as e:
                    logger.warning(
                        f"Failed to build weighted sampler for client {client_id}: {e}. "
                        f"Using default sampling."
                    )
            
            # Create final dataloader
            splits[split] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=loader.pin_memory,
                drop_last=loader.drop_last,
                collate_fn=loader.collate_fn,
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


def _build_transform(config: TransformConfig, split: str) -> Optional[Transforms]:
    """Build a transform function based on configuration."""
    if (config.resize is None and 
        config.normalize is None and 
        (not config.augment or split != "train")):
        return None

    def transform(tensor: Tensor) -> Tensor:
        out = tensor
        
        # Resize
        if config.resize is not None:
            out = _resize_tensor(out, config.resize)
        
        # Normalize
        if config.normalize is not None:
            mean, std = config.normalize
            mean_tensor = torch.tensor(mean, dtype=out.dtype, device=out.device).view(-1, 1, 1)
            std_tensor = torch.tensor(std, dtype=out.dtype, device=out.device).view(-1, 1, 1)
            out = (out - mean_tensor) / std_tensor
        
        # Augmentation (only for training)
        if config.augment and split == "train":
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                out = torch.flip(out, dims=[-1])
        
        return out

    return transform


def _resize_tensor(tensor: Tensor, size: Tuple[int, int]) -> Tensor:
    """Resize a tensor to the specified size."""
    if tensor.dim() < 2:
        raise ValueError(
            f"Cannot resize tensor with {tensor.dim()} dimensions (need at least 2)"
        )
    
    # Add batch dimension if needed
    needs_unsqueeze = tensor.dim() == 3
    if needs_unsqueeze:
        tensor = tensor.unsqueeze(0)
    
    resized = torch.nn.functional.interpolate(
        tensor,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    
    # Remove batch dimension if we added it
    if needs_unsqueeze:
        resized = resized.squeeze(0)
    
    return resized


def _build_weighted_sampler(dataset: Dataset[Any]) -> WeightedRandomSampler:
    """Build a weighted sampler for class-balanced sampling."""
    labels = []
    
    for item in dataset:
        # Extract label from dataset item
        if isinstance(item, (tuple, list)):
            # Typically (data, label) or (data, label, ...)
            if len(item) >= 2:
                label = item[1]
            else:
                raise ValueError("Dataset item must contain at least (data, label)")
        else:
            raise TypeError(f"Unexpected dataset item type: {type(item)}")
        
        labels.append(int(label))
    
    if not labels:
        raise ValueError("Cannot build weighted sampler: no labels found in dataset")
    
    # Compute class weights
    class_sample_count = torch.bincount(torch.tensor(labels))
    if (class_sample_count == 0).any():
        raise ValueError("Cannot build weighted sampler: some classes have zero samples")
    
    weights = 1.0 / class_sample_count.float()
    sample_weights = weights[torch.tensor(labels)]
    
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=True
    )


__all__ = [
    "build_client_dataloaders",
    "register_dataset",
    "clone_dataloader",
    "TransformDataset",
    # Adapter classes
    "FedIXIAdapter",
    "FedHeartDiseaseAdapter",
    "FedISIC2019Adapter",
    "FedCamelyon16Adapter",
    "FedKiTS19Adapter",
    "FedLiTSAdapter",
    "FedTCGABRCAAdapter",
]