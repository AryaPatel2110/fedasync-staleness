"""Parameter sweep driver for federated learning experiments.

This script automates running multiple simulations over a grid of hyperparameters
by generating temporary configuration files and invoking the main run_sim
script for each combination.  Use this to explore the effect of different
Dirichlet alpha values, delay distributions, strategies, and models.

Usage example:

    python scripts/sweep.py --cfg configs/experiment.yaml \
        --alphas 0.1 1.0 10.0 \
        --strategies fedasync fedbuff dynstale
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep of federated simulations.")
    parser.add_argument("--cfg", type=str, default="configs/experiment.yaml", help="Base experiment config path.")
    parser.add_argument("--alphas", type=float, nargs="*", default=[1.0], help="Dirichlet alpha values to sweep.")
    parser.add_argument("--strategies", type=str, nargs="*", default=["fedasync"], help="Aggregation strategies to test.")
    parser.add_argument("--models", type=str, nargs="*", default=["squeezenet"], help="Model architectures to test.")
    return parser.parse_args()

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_simulation(cfg_path: str) -> None:
    subprocess.run(["python", "scripts/run_sim.py", "--cfg", cfg_path], check=True)

def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.cfg)
    base_dir = os.path.dirname(args.cfg)
    # Generate combinations
    combos = list(itertools.product(args.alphas, args.strategies, args.models))
    for alpha, strategy, model in combos:
        # Create a temporary configuration by copying base config and overriding fields
        cfg = base_cfg.copy()
        cfg["alpha"] = float(alpha)
        cfg["strategy"] = strategy
        if isinstance(cfg.get("model"), dict):
            cfg["model"]["name"] = model
        else:
            cfg["model"] = {"name": model}
        # Write temporary config file
        cfg_name = f"tmp_alpha{alpha}_strategy{strategy}_model{model}.yaml"
        cfg_path = os.path.join(base_dir, cfg_name)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"Running simulation for alpha={alpha}, strategy={strategy}, model={model}")
        run_simulation(cfg_path)
        # Optionally remove the temporary config file
        os.remove(cfg_path)

if __name__ == "__main__":
    main()