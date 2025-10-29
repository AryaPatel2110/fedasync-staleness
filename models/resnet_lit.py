"""PyTorch Lightning module wrapping a small ResNet for Fashion‑MNIST or CIFAR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl

class LitResNet(pl.LightningModule):
    """LightningModule encapsulating a ResNet model with configurable depth."""
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, depth: int = 18) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Select the appropriate ResNet architecture
        if depth == 18:
            backbone = models.resnet18(pretrained=False)
        elif depth == 34:
            backbone = models.resnet34(pretrained=False)
        else:
            # Default to resnet18 if unsupported depth
            backbone = models.resnet18(pretrained=False)
        # Modify the first conv layer for single‑channel input if needed (e.g., Fashion‑MNIST)
        # The original ResNet expects 3‑channel input; we adapt for grayscale by repeating channel
        if backbone.conv1.in_channels == 3:
            # keep conv1 as is for RGB datasets (CIFAR‑10)
            pass
        # Replace the final fully connected layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If the input has a single channel, repeat to make 3 channels
        if x.size(1) == 1 and self.net.conv1.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
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