# FedAsync Staleness Simulation

This repository contains a baseline implementation for exploring the impact of client staleness in asynchronous federated learning.

The goal of this project is to provide a modular, extensible framework for comparing different server–side aggregation strategies on a variety of datasets and models.  We use the Flower federated learning framework to manage client/server communication and PyTorch Lightning to encapsulate the neural network training logic.

## Structure

The project is organized into distinct top‑level directories:

- **configs/** – YAML configuration files controlling experiments, models, and strategies.
- **data/** – Placeholder for any downloaded or pre‑processed datasets (this folder is empty by default).  Datasets are loaded via torchvision within the code.
- **fed/** – Core federated learning utilities including data partitioning, delay sampling, metric computations, and custom server strategies.
- **fed/clients/** – Definitions of client classes that wrap PyTorch Lightning models into Flower clients.
- **fed/strategies/** – Implementations of FedAsync, FedBuff, and the custom dynamic staleness strategy.
- **models/** – LightningModule definitions for the models used in experiments (e.g., SqueezeNet for CIFAR‑10 and a small ResNet for Fashion‑MNIST).
- **scripts/** – Entry points for running experiments and parameter sweeps.
- **utils/** – Miscellaneous helpers for seeding, configuration parsing, and parameter serialization.

See `configs/experiment.yaml` for a template experiment configuration.  Modify the YAML files or pass parameters on the command line to explore different settings such as the Dirichlet α parameter, delay distributions, and strategy hyperparameters.

## Running an experiment

The main script `scripts/run_sim.py` orchestrates a single asynchronous simulation.  It reads an experiment configuration file, partitions data among clients using a Dirichlet distribution, instantiates the chosen server strategy, and runs an event‑driven simulation loop.  Evaluation metrics (accuracy, fairness, convergence speed) are logged at configurable intervals.

To run a simulation with default settings:

```bash
python scripts/run_sim.py --cfg configs/experiment.yaml
```

## Extending the framework

The code base is designed to be modular.  New strategies can be added in `fed/strategies/`, new models in `models/`, and new data partitions or delay distributions in `fed/partitioning.py` and `fed/delays.py`.  Experiment parameters are controlled via YAML files in the `configs/` directory.

Please see the source code and inline comments for more details.