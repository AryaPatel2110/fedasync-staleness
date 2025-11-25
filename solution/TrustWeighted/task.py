"""TrustWeighted/task.py

Model definition and data loading utilities for Flower + PyTorch Lightning + flwr-datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class LitAutoEncoder(pl.LightningModule):
    """Simple autoencoder for MNIST-like images."""

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, 1, 28, 28)
        return x_hat

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# Dataset wrapper to go from HuggingFace Datasets → PyTorch Dataset
# ---------------------------------------------------------------------------


class HFImageDataset(Dataset):
    """Wrap a HuggingFace image dataset with keys 'image' and 'label'."""

    def __init__(self, hf_dataset, transform=None) -> None:
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[int(idx)]
        img = item["image"]  # PIL image or array
        label = int(item["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    test_num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 42


DATA_CFG = DataConfig()


def _build_transform() -> transforms.Compose:
    """Transforms for MNIST-like grayscale images."""
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),  # [0,1]
        ]
    )


def load_data(partition_id: int, num_partitions: int):
    """Load federated train/val/test data for a given client partition.

    Parameters
    ----------
    partition_id : int
        The client id (0 .. num_partitions-1).
    num_partitions : int
        Total number of client partitions.

    Returns
    -------
    trainloader : DataLoader
        Dataloader for the local train split.
    valloader : DataLoader
        Dataloader for the local validation split.
    testloader : DataLoader
        Dataloader for the centralized test split.
    """
    # Federated MNIST dataset
    fds = FederatedDataset(
        dataset="mnist",
        partitioners={"train": IidPartitioner(num_partitions=num_partitions)},
    )

    # Local partition of the training data for this client
    partition = fds.load_partition(partition_id, split="train")

    # Split local partition into train/val (DatasetDict with "train" and "test")
    partition_train_valid = partition.train_test_split(
        test_size=DATA_CFG.val_fraction, seed=DATA_CFG.seed
    )

    # Centralized test split, one HF Dataset (columns: "image", "label")
    full_test = fds.load_split("test")

    transform = _build_transform()

    train_ds = HFImageDataset(partition_train_valid["train"], transform=transform)
    val_ds = HFImageDataset(partition_train_valid["test"], transform=transform)
    test_ds = HFImageDataset(full_test, transform=transform)

    trainloader = DataLoader(
        train_ds,
        batch_size=DATA_CFG.batch_size,
        shuffle=True,
        num_workers=DATA_CFG.num_workers,
        persistent_workers=True,
    )
    valloader = DataLoader(
        val_ds,
        batch_size=DATA_CFG.batch_size,
        shuffle=False,
        num_workers=DATA_CFG.num_workers,
        persistent_workers=True,
    )
    testloader = DataLoader(
        test_ds,
        batch_size=DATA_CFG.batch_size,
        shuffle=False,
        num_workers=DATA_CFG.test_num_workers,
        persistent_workers=True,
    )

    return trainloader, valloader, testloader


# ---------------------------------------------------------------------------
# Convenience factory for model (if client/server want a simple hook)
# ---------------------------------------------------------------------------


def get_model(lr: float = 1e-3) -> LitAutoEncoder:
    """Factory function to create a new model instance."""
    return LitAutoEncoder(lr=lr)
