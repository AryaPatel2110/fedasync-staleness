---
tags: [quickstart, vision, fds]
dataset: [MNIST]
framework: [lightning]
---

```shell
solution
├── TrustWeighted
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

# Create a Python virtual environment

```bash
python -m venv .venv
```

# Activate the environment

**Windows**
```bash
.venv\Scripts\activate.bat
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

# Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `TrustWeighted` package.

```bash
pip install -e .
```

## Run the Example

The _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 max-epochs=2"
```