# 🧠 Federated Asynchronous Learning (FedAsync, FedBuff & TrustWeight)

This repository implements three federated asynchronous learning frameworks:
- **FedAsync**: Asynchronous Federated Learning with staleness-aware aggregation
- **FedBuff**: Buffered Asynchronous Federated Learning with batch aggregation
- **TrustWeight**: Trust-Weighted Asynchronous Federated Learning with quality-aware aggregation

All frameworks simulate heterogeneous client behavior and perform asynchronous updates to a central server.

---

## 📦 Project Structure

```
fedasync-staleness/
│
├── FedAsync/              # FedAsync implementation
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   └── config.yaml
│
├── FedBuff/               # FedBuff implementation
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   └── config.yml
│
├── TrustWeight/           # TrustWeight implementation
│   ├── client.py
│   ├── server.py
│   ├── strategy.py
│   ├── run.py
│   └── config.yaml
│
├── utils/                 # Shared utilities
│   ├── helper.py
│   ├── model.py          # Model architectures (ResNet-18)
│   └── partitioning.py   # Data partitioning (Dirichlet)
│
├── experiments/          # Experimental work and history
│   ├── baseline/         # Baseline training (SqueezeNet → ResNet-18 evolution)
│   ├── notebooks/        # Jupyter notebooks for Colab/local execution
│   ├── analysis/         # Analysis scripts and comparison reports
│   ├── archive/          # Historical development files
│   └── outside/          # Additional experiment results (Google Colab)
│
├── Analysis/              # Analysis scripts (matching main branch)
│
├── logs/                  # Experiment results and outputs
│   ├── avinash/          # Main experiment runs (timestamped)
│   └── TrustWeight/      # TrustWeight-specific experiments
│
├── results/               # Final model weights and outputs
├── checkpoints/           # Intermediate model checkpoints
├── data/                  # Dataset storage (CIFAR-10)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

#### FedAsync
```bash
python -m FedAsync.run
```

#### FedBuff
```bash
python -m FedBuff.run
```

#### TrustWeight
```bash
python -m TrustWeight.run
```

---

## 📊 Project Evolution

### Phase 1: SqueezeNet Baseline
- Initial experiments with **SqueezeNet** architecture
- Baseline centralized training to establish benchmarks
- See `experiments/baseline/` for details

### Phase 2: ResNet-18 Upgrade
- Architecture upgraded to **ResNet-18** (adapted for CIFAR-10)
- Improved data augmentation pipeline
- Enhanced hyperparameter tuning

### Phase 3: Federated Learning
- Implemented FedAsync, FedBuff, and TrustWeight
- Comprehensive experiments across multiple configurations
- Results documented in `logs/` and `experiments/`

---

## 📈 Experiment Results

### Best Results (Avinash Branch)
- **FedAsync**: 28.54% (α=1000.0, 0% stragglers, 300 rounds)
- **FedBuff**: 68.31% ⭐ (α=100.0, 0% stragglers, 100 rounds)
- **TrustWeight**: 36.94% (α=1000.0, 0% stragglers, 504 rounds)

### Main Branch Comparison
- **FedAsync**: 75.32% (main branch)
- **FedBuff**: 62.24% (main branch)
- **TrustWeight**: 72.32% (main branch)

For detailed comparisons, see `experiments/analysis/MAIN_VS_AVINASH_RESULTS_COMPARISON.md`

---

## 📁 Directory Details

### Core Implementation
- **FedAsync/**, **FedBuff/**, **TrustWeight/**: Core method implementations
- **utils/**: Shared utilities (models, data partitioning, helpers)

### Experiments
- **experiments/baseline/**: Baseline training experiments
- **experiments/notebooks/**: Complete notebooks for Google Colab
- **experiments/analysis/**: Analysis scripts and comparison reports
- **experiments/archive/**: Historical development files
- **experiments/outside/**: Additional experiment results (138 CSV files)

### Results
- **logs/**: All experiment results and outputs
- **results/**: Final model weights
- **checkpoints/**: Intermediate checkpoints

---

## 🧪 Key Features

- **Asynchronous aggregation**: Clients update server immediately after local training
- **Client heterogeneity simulation**: Random per-client delays to mimic real-world latency
- **Staleness-aware aggregation**: Weight updates based on staleness
- **Quality-aware aggregation** (TrustWeight): Weight updates based on loss improvement and update quality
- **Config-driven**: All behavior customizable via YAML config files
- **Comprehensive logging**: Global and client-level logs in CSV format

---

## 📚 Documentation

- **Main README**: This file
- **experiments/README.md**: Experimental work overview
- **experiments/baseline/README.md**: Baseline experiments
- **experiments/notebooks/README.md**: Notebook usage
- **experiments/analysis/README.md**: Analysis tools
- **logs/README.md**: Experiment results structure

---

## ✅ Example Workflow

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run FedBuff
python -m FedBuff.run

# Check results
ls logs/avinash/run_*/
cat logs/avinash/run_*/FedBuff.csv
```

---

## 🔬 Experimental Work

This repository preserves the complete experimental history:
- **SqueezeNet → ResNet-18 evolution**: See `experiments/baseline/`
- **216+ experiment runs**: See `logs/` and `experiments/outside/`
- **Analysis and comparisons**: See `experiments/analysis/`
- **Notebooks for reproducibility**: See `experiments/notebooks/`

All experimental work is organized in `experiments/` to maintain a clean core implementation while preserving research history.
