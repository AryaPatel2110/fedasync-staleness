"""TrustWeighted: A Flower / PyTorch Lightning app (ServerApp)."""

from __future__ import annotations
from pathlib import Path
import csv
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from TrustWeighted.task import _RAW_CFG, LitCifar  
from TrustWeighted.strategy import AsyncTrustFedAvg

# Create ServerApp instance (this is what pyproject.toml points to)
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config from pyproject.toml
    # These keys should match [tool.flwr.app.config]
    num_rounds: int = int(context.run_config["num-server-rounds"])
    max_epochs: int = int(context.run_config["max-epochs"])

    # Initialize global model and wrap into ArrayRecord
    model = LitCifar()
    arrays = ArrayRecord(model.state_dict())

    # Initialize trust-weighted strategy (inherits FedAvg.start)
    # You can tune fraction_train as you like
    strategy = AsyncTrustFedAvg(
        fraction_train=0.5,
        fraction_evaluate=0.5,  # keep evaluation behaviour like FedAvg
    )

    # Training configuration passed to clients
    train_cfg = {
        "max-epochs": max_epochs,
        "min_delay": _RAW_CFG.get("async", {}).get("min_delay", 0.0),
        "max_delay": _RAW_CFG.get("async", {}).get("max_delay", 0.0),
    }
    # Start training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=train_cfg,
    )

        # ------------------------------------------------------------------
    # Save per-round training metrics to CSV (TrustWeighted.csv)
    # ------------------------------------------------------------------
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "TrustWeighted.csv"

    # Decide which fields to write (must match keys in train_history)
    fieldnames = ["round", "train_loss","train_acc", "total_examples", "buffer_size", "num_accepted", "avg_staleness"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in strategy.train_history:
            # Make sure only known fields are written
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Saved per-round TrustWeighted metrics to {csv_path}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    final_state = result.arrays.to_torch_state_dict()
    torch.save(final_state, "final_model.pt")