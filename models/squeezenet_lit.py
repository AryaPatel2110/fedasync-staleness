"""PyTorch Lightning module wrapping a SqueezeNet model for CIFAR‑10."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LitSqueezeNet(pl.LightningModule):
    """LightningModule encapsulating SqueezeNet training and evaluation."""
    def __init__(self, num_classes: int = 10, lr: float = 1e-3) -> None:
        super().__init__()
        # Save hyperparameters to self.hparams for logging/serialization
        self.save_hyperparameters()
        # Load SqueezeNet 1.1 architecture from torchvision
        self.net = torch.hub.load("pytorch/vision", "squeezenet1_1", pretrained=False)
        # Replace the final classifier convolution to match the number of classes
        self.net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        # Initialize weights
        nn.init.normal_(self.net.classifier[1].weight, 0, 0.01)
        nn.init.constant_(self.net.classifier[1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", acc, prog_bar=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)