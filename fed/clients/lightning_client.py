"""Flower client wrapper for PyTorch Lightning models."""

from __future__ import annotations

import numpy as np
import torch
import flwr as fl  # type: ignore
import pytorch_lightning as pl

class LightningClient(fl.client.NumPyClient):
    """A Flower NumPyClient that trains a PyTorch Lightning model.

    This client wraps a LightningModule and a LightningDataModule (or dataloaders)
    to participate in federated learning using Flower.  Parameters are passed
    between the Flower server and the Lightning model as NumPy arrays.
    """
    def __init__(
        self,
        model: pl.LightningModule,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 1,
        device: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Save hyperparameters for logging
        self._logger: pl.loggers.TensorBoardLogger | None = None
        if log_dir is not None:
            self._logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name="client")

    # Flower API methods
    def get_parameters(self, config: dict[str, str] | None = None) -> list[np.ndarray]:  # type: ignore[override]
        """Return current model parameters as a list of NumPy arrays."""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Update local model parameters from a list of NumPy arrays."""
        params_iter = iter(parameters)
        for param in self.model.parameters():
            p = next(params_iter)
            param.data = torch.from_numpy(p).to(param.data.device)

    def fit(self, parameters: list[np.ndarray], config: dict[str, str] | None = None) -> tuple[list[np.ndarray], int, dict[str, float]]:  # type: ignore[override]
        """Perform local training and return updated parameters.

        Returns a tuple of (updated_params, num_examples, metrics).
        """
        # Set the model parameters
        self.set_parameters(parameters)
        # Create Trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            logger=self._logger,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        # Train the model
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
        # Return updated parameters and number of samples used for training
        num_examples = len(self.train_dataloader.dataset)
        return self.get_parameters({}), num_examples, {}

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, str] | None = None) -> tuple[float, int, dict[str, float]]:  # type: ignore[override]
        """Evaluate the model on the validation set.

        Returns a tuple of (loss, num_examples, metrics).  The metrics dict
        includes the accuracy under the key 'val_acc'.
        """
        # Set the model parameters
        self.set_parameters(parameters)
        # Use a Trainer to evaluate
        trainer = pl.Trainer(
            accelerator="auto",
            logger=False,
            enable_checkpointing=False,
        )
        metrics = trainer.validate(self.model, dataloaders=self.val_dataloader, verbose=False)[0]
        # 'val_loss' and 'val_acc' should be returned; fallback to 0 if missing
        loss = float(metrics.get("val_loss", 0.0))
        acc = float(metrics.get("val_acc", 0.0))
        num_examples = len(self.val_dataloader.dataset)
        return loss, num_examples, {"val_acc": acc}