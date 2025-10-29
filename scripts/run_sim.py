"""Run an asynchronous federated learning simulation.

This script reads a YAML configuration file describing the experiment,
builds the dataset and clients, instantiates the chosen aggregation
strategy, and executes a simple asynchronous simulation loop.  Evaluation
metrics are logged at regular intervals.

Note: This implementation provides a simplified asynchronous simulation
that samples staleness values from a distribution and applies them in
the aggregation step.  It does not model true wall‑clock delay or
priority queues.  Nevertheless, it serves as a baseline for exploring
different strategies and hyperparameters.
"""

from __future__ import annotations

import argparse
import yaml
import random
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from fed.partitioning import dirichlet_partition
from fed.delays import sample_client_delays, compute_staleness
from fed.metrics import fairness_gini, time_to_target
from fed.recorder import Recorder
from fed.strategies.fedasync import FedAsync
from fed.strategies.fedbuff import FedBuff
from fed.strategies.dynstale import DynStale
from models import squeezenet_lit, resnet_lit
from fed.clients.lightning_client import LightningClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an asynchronous federated learning simulation.")
    parser.add_argument("--cfg", type=str, default="configs/experiment.yaml", help="Path to the experiment YAML file.")
    return parser.parse_args()

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataset(dataset_name: str, data_cfg: dict, train: bool = True) -> datasets.VisionDataset:
    # Determine transforms
    if dataset_name == "cifar10":
        normalize = transforms.Normalize(mean=data_cfg["cifar10"]["normalize"]["mean"], std=data_cfg["cifar10"]["normalize"]["std"])
        if train and data_cfg["cifar10"]["augment"]:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        return datasets.CIFAR10(root=data_cfg["cifar10"]["root"], train=train, download=True, transform=transform)
    elif dataset_name == "fashion_mnist":
        normalize = transforms.Normalize(mean=data_cfg["fashion_mnist"]["normalize"]["mean"], std=data_cfg["fashion_mnist"]["normalize"]["std"])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return datasets.FashionMNIST(root=data_cfg["fashion_mnist"]["root"], train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def create_model(model_cfg: dict, dataset_name: str):
    name = model_cfg.get("name")
    num_classes = model_cfg.get("num_classes", 10)
    lr = model_cfg.get("lr", 0.001)
    if name == "squeezenet":
        return squeezenet_lit.LitSqueezeNet(num_classes=num_classes, lr=lr)
    elif name == "resnet":
        depth = model_cfg.get("depth", 18)
        return resnet_lit.LitResNet(num_classes=num_classes, lr=lr, depth=depth)
    else:
        raise ValueError(f"Unsupported model name: {name}")

def create_strategy(strategy_name: str, strategy_cfg: dict):
    if strategy_name == "fedasync":
        eta = float(strategy_cfg.get("eta", 1.0))
        # Build staleness function from string expression if provided
        fn_expr = strategy_cfg.get("staleness_fn", None)
        if fn_expr is None:
            staleness_fn = None  # will default inside FedAsync
        else:
            # Create a lambda s: <expr> with limited globals
            # s is an integer; allow numpy functions
            def _build_fn(expr: str):
                return lambda s: eval(expr, {"s": s, "np": np})
            staleness_fn = _build_fn(fn_expr)
        return FedAsync(eta=eta, staleness_fn=staleness_fn)
    elif strategy_name == "fedbuff":
        buffer_size = int(strategy_cfg.get("buffer_size", 16))
        timeout_s = strategy_cfg.get("timeout_s", None)
        timeout_s = float(timeout_s) if timeout_s is not None else None
        return FedBuff(buffer_size=buffer_size, timeout_s=timeout_s)
    elif strategy_name == "dynstale":
        init_lambda = float(strategy_cfg.get("init_lambda", 0.1))
        beta = float(strategy_cfg.get("beta", 1.0))
        ema = float(strategy_cfg.get("ema", 0.9))
        adapt_quantile = float(strategy_cfg.get("adapt_quantile", 0.75))
        drop_stale_after = strategy_cfg.get("drop_stale_after", None)
        drop_stale_after = int(drop_stale_after) if drop_stale_after is not None else None
        return DynStale(init_lambda=init_lambda, beta=beta, ema=ema, adapt_quantile=adapt_quantile, drop_stale_after=drop_stale_after)
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    # Set seed
    seed = int(cfg.get("seed", 0))
    set_random_seed(seed)
    # Load data config
    data_cfg_path = os.path.join(os.path.dirname(args.cfg), "data.yaml")
    with open(data_cfg_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    # Determine dataset
    dataset_name = cfg.get("dataset", "cifar10")
    # Load full training and test datasets
    full_train_dataset = get_dataset(dataset_name, data_cfg, train=True)
    val_dataset = get_dataset(dataset_name, data_cfg, train=False)
    labels = np.array(full_train_dataset.targets)
    # Partition data among clients
    n_clients = int(cfg.get("n_clients", 10))
    alpha = float(cfg.get("alpha", 1.0))
    client_indices = dirichlet_partition(labels, n_clients=n_clients, alpha=alpha, seed=seed)
    # Create dataloaders for each client
    batch_size = 32  # fixed batch size; can be extended in config
    clients: list[LightningClient] = []
    # Determine model settings
    model_cfg = {
        "name": cfg.get("model", {}).get("name", cfg.get("model", "squeezenet")) if isinstance(cfg.get("model"), dict) else cfg.get("model", "squeezenet"),
        "num_classes": 10,
        "lr": cfg.get("model", {}).get("lr", cfg.get("model", {}).get("lr", 0.001)),
        "depth": cfg.get("model", {}).get("depth", 18),
    }
    # Instantiate a model to obtain initial parameters for the server
    global_model = create_model(model_cfg, dataset_name)
    server_params: list[np.ndarray] = [p.detach().cpu().numpy() for p in global_model.parameters()]
    # Create dataloaders per client
    for cid in range(n_clients):
        # Subset of indices for this client
        idxs = client_indices[cid]
        subset = Subset(full_train_dataset, idxs)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        # We use the same validation set for all clients
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        client_model = create_model(model_cfg, dataset_name)
        client = LightningClient(model=client_model, train_dataloader=train_loader, val_dataloader=val_loader, epochs=int(cfg.get("model", {}).get("epochs", 1)))
        clients.append(client)
    # Create strategy
    strategy_name = cfg.get("strategy", "fedasync")
    strategy_cfg_path = os.path.join(os.path.dirname(args.cfg), f"{strategy_name}.yaml")
    if os.path.exists(strategy_cfg_path):
        with open(strategy_cfg_path, "r") as f:
            strategy_cfg = yaml.safe_load(f)
    else:
        strategy_cfg = {}
    strategy = create_strategy(strategy_name, strategy_cfg)
    # Simulation parameters
    rounds = int(cfg.get("rounds", 100))
    sample_fraction = float(cfg.get("sample_fraction", 0.1))
    num_sampled = max(1, int(sample_fraction * n_clients))
    # Delay parameters
    delay_cfg = cfg.get("delay", {})
    delay_dist = delay_cfg.get("dist", "lognormal")
    delay_params = tuple(delay_cfg.get("params", [0.0, 0.75]))
    # Evaluation settings
    eval_cfg = cfg.get("eval", {})
    eval_every = int(eval_cfg.get("every_events", 10))
    target_acc = float(eval_cfg.get("target_acc", 0.9))
    # Recorder
    recorder = Recorder()
    # Precompute delays per client
    delays = sample_client_delays(n_clients, dist=delay_dist, params=delay_params, seed=seed)
    # Maintain last update time per client for staleness computation
    last_update_event = [0 for _ in range(n_clients)]
    # Event loop
    global_history: list[tuple[int, float]] = []
    for event in range(1, rounds + 1):
        # Sample clients
        sampled_clients = np.random.choice(n_clients, size=num_sampled, replace=False)
        for cid in sampled_clients:
            client = clients[cid]
            # Simulate client training: get updated parameters from client
            client_params, num_examples, _ = client.fit(server_params, {})
            # Compute staleness: difference between current event and last update event
            staleness = event - last_update_event[cid]
            last_update_event[cid] = event
            # Aggregate
            if isinstance(strategy, FedAsync):
                server_params = strategy.aggregate(server_params, client_params, staleness)
            elif isinstance(strategy, DynStale):
                server_params = strategy.aggregate(server_params, client_params, staleness)
            elif isinstance(strategy, FedBuff):
                # For FedBuff we treat current_time as event index
                new_params, aggregated = strategy.on_update(server_params, client_params, current_time=float(event))
                if aggregated and new_params is not None:
                    server_params = new_params
            else:
                raise RuntimeError(f"Unknown strategy type: {type(strategy)}")
        # Evaluate periodically
        if event % eval_every == 0 or event == rounds:
            # Evaluate global model on validation set
            # Load server params into a temporary model
            eval_model = create_model(model_cfg, dataset_name)
            # Set weights
            param_iter = iter(server_params)
            for p in eval_model.parameters():
                p.data = torch.from_numpy(next(param_iter)).to(p.device)
            # Evaluate
            device = "cuda" if torch.cuda.is_available() else "cpu"
            eval_model.to(device)
            eval_model.eval()
            correct = 0
            total = 0
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = eval_model(x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            val_acc = correct / total
            global_history.append((event, val_acc))
            # Log metrics
            recorder.log(event=event, val_acc=val_acc)
            print(f"[Event {event}] Validation accuracy: {val_acc:.4f}")
            # Early stopping
            if val_acc >= target_acc:
                print(f"Target accuracy {target_acc} reached at event {event}.")
                break
    # Save recorder results
    df = recorder.to_df()
    results_path = os.path.join(os.path.dirname(args.cfg), "results.csv")
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()