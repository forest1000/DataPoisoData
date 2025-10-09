# Federated Medical Imaging Baseline with Flamby

This repository provides a modular Python baseline for evaluating poisoning
attacks and server-side defenses in cross-silo federated learning (FL) for
medical imaging. The framework is designed around [Flamby](https://github.com/owkin/FLamby)
benchmarks and enables rapid experimentation with interchangeable data
pipelines, FL strategies, attacks, defenses, and evaluation routines.

## Features

- **Dataset abstraction** – Build client-specific train/val/test loaders from
  Flamby datasets with configurable preprocessing, augmentation, and optional
  class-balanced sampling.
- **Extensible FL stack** – Modular client/server separation with a FedAvg
  implementation ready for extension to algorithms such as FedProx or
  SCAFFOLD.
- **Attack simulation** – Label flip poisoning with configurable target
  clients, ratios, and label mappings, plus a registry for custom attacks.
- **Defense mechanisms** – Cosine-similarity outlier filtering with graceful
  fallback and hooks for Krum, Trimmed Mean, or custom defenses.
- **Evaluation & visualization** – Macro metrics across clients, automatic
  CSV/PNG artifact export, and experiment logging in timestamped directories.

## Repository structure

```
project_root/
  main.py
  config.yaml
  data_loader.py
  fl_strategy.py
  attack_module.py
  defense_module.py
  evaluation.py
  utils/
    registry.py
    seed.py
    logging.py
    serialization.py
  experiments/
    runs/<timestamp>/
  tests/
    ...
  README.md
```

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision scikit-learn numpy pandas matplotlib pytest mypy black isort monai
cd ..
git clone https://github.com/owkin/FLamby.git
cd FLamby
pip install -e .
```

> **Note:** The provided tests rely on lightweight dummy datasets and do not
> require downloading full Flamby corpora. When running full experiments, make
> sure the relevant Flamby datasets are installed and accessible.

## Configuration

Experiments are configured with YAML files. A minimal example is provided in
[`config.yaml`](config.yaml):

```yaml
data:
  dataset_name: "fed-ixi"
  num_clients: 5
  batch_size: 16
fl:
  strategy: "fedavg"
  rounds: 10
attack:
  name: "label_flip"
  target_clients: [1, 3]
  poison_ratio: 0.3
defense:
  name: "cosine"
  threshold: 0.2
```

### Key sections

- `experiment`: Output directory and random seed control.
- `data`: Dataset name, client count, dataloader configuration, and optional
  preprocessing controls (`resize`, `normalize`, `augment`,
  `use_weighted_sampler`).
- `fl`: Strategy key, number of rounds, local epochs, learning rate,
  optimizer, and fraction of clients participating per round.
- `attack`: Attack registry key plus attack-specific parameters. Set `name` to
  an empty string to disable attacks.
- `defense`: Defense registry key and parameters. Set `name` to an empty string
  to disable defenses.
- `eval`: Evaluation splits and averaging mode for precision/recall/F1.
- `logging`: Logging level and output controls.

## Running experiments

```bash
python main.py --config config.yaml
```

Artifacts are written to `experiments/runs/<timestamp>/` and include:

- `config.used.yaml`: Exact configuration used for the run.
- `train.log`: Round-wise training and evaluation logs.
- `metrics.csv`: Consolidated metrics per round/split/attack/defense setting.
- `curves.png`: F1-score trajectories per split and regime.
- `global_final.pt`: Serialized global model weights.

## Extending the framework

### Registering new datasets

The default [`data_loader.py`](data_loader.py) module resolves dataset builders
directly from the official [FLamby repository](https://github.com/owkin/FLamby).
Canonical names such as `"fed-ixi"`, `"fed-heart-disease"`, and
`"fed-isic2019"` map to the corresponding `flamby.datasets` packages, so using
any of these values in your configuration is sufficient to instantiate the
benchmark-provided federated datasets.

To add additional datasets or custom wrappers, register a builder function:

```python
from data_loader import register_dataset

register_dataset("my-medical-dataset", lambda: MyFederatedDataset())
```

Custom builders must return an object exposing `train_dataloader`,
`val_dataloader`, and `test_dataloader` methods that accept
`center`, `batch_size`, and `num_workers` arguments and return
`torch.utils.data.DataLoader` instances.

### Adding attacks or defenses

Each module exposes an abstract base class and a registry:

```python
from attack_module import Attack
from utils.registry import Registry

class MyAttack(Attack):
    ...

ATTACK_REGISTRY.register("my_attack", MyAttack)
```

Follow the same pattern for new server-side defenses or FL strategies. All
components are designed with pure input/output APIs and type hints to ease
static analysis (`mypy --strict`) and formatting (`black`, `isort`).

## Testing

Run the unit and smoke tests with:

```bash
pytest -q
```

The test suite covers dataloader assembly, attack behavior, defense filtering,
strategy aggregation, and evaluation metrics.

## Reproducibility and ethics

- Seeds for Python, NumPy, and PyTorch are controlled via `experiment.seed` and
  deterministic execution is requested from PyTorch when available.
- Do not log or export protected health information (PHI). Logs contain only
  aggregate metrics and file paths.
- This code is **for research use only** and is not validated for clinical
  deployment. Interpret results with care and consider downstream ethical
  implications when applying poisoning and defense algorithms to sensitive
  datasets.

## Troubleshooting

- Ensure the requested Flamby dataset is installed. Invalid dataset keys raise
  informative errors with available options from the registry.
- If all client updates are filtered by a defense, the system logs a warning
  and falls back to aggregating the unfiltered updates to prevent deadlock.

## License

This project is provided as-is for research purposes. Refer to the original
Flamby and PyTorch licenses for upstream components.
