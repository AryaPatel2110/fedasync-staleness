# Sanjana Branch Changes Summary

This document tracks all changes unique to the `sanjana` branch to ensure they are preserved during merge with main.

## Date: December 10, 2025

## New Directories and Files

### 1. Scripts Directory (`scripts/`)
All hyperparameter tuning scripts moved here:
- `scripts/hyperparameter_tuning.py` - Original SqueezeNet tuning
- `scripts/squeezenet_hyperparameter_tuning.py` - Comprehensive SqueezeNet tuning
- `scripts/squeezenet_quick_tuning.py` - Quick SqueezeNet tuning
- `scripts/resnet18_hyperparameter_tuning.py` - Full ResNet-18 grid search
- `scripts/resnet18_quick_tuning.py` - Quick ResNet-18 tuning
- `scripts/README.md` - Documentation

### 2. Experiment Results (`experiments/`)
- `experiments/squeezenet_fedbuff_results/` - 9 historical SqueezeNet FedBuff runs
- `experiments/README.md` - Documentation

### 3. Hyperparameter Tuning Results (`hyperparameter_tuning_results/`)
- SqueezeNet results: `quick_results_*.csv`, `results_*.csv`, `quick_summary_*.json`
- ResNet-18 results: `resnet18/quick_results_*.csv`, `resnet18/results_*.csv`
- `hyperparameter_tuning_results/README.md` - Documentation

### 4. Logs Archive (`logs/archive/`)
- `fedasync_test.log`
- `fedasync_test_fixed.log`
- `fedbuff_test.log`
- `fedbuff_test_fixed.log`
- `tuning_output.log`

## Modified Core Files

### FedAsync
- `FedAsync/client.py` - Updated to use ResNet-18, cleaned up comments
- `FedAsync/server.py` - Fixed CSV overwriting, improved logging
- `FedAsync/run.py` - Updated to use ResNet-18, cleaned up

### FedBuff
- `FedBuff/client.py` - Updated to use ResNet-18, cleaned up comments
- `FedBuff/server.py` - Fixed CSV overwriting, fixed stagnation bug, improved logging
- `FedBuff/run.py` - Updated to use ResNet-18, cleaned up

### Utils
- `utils/model.py` - Contains `build_resnet18` (SqueezeNet removed to match main)
- `utils/partitioning.py` - Fixed leftover sample distribution bug, cleaned up

## Key Features

1. **ResNet-18 Architecture**: All implementations use ResNet-18 (migrated from SqueezeNet)
2. **Fixed CSV Logging**: CSV files are now created fresh on each run (no overwriting)
3. **Fixed FedBuff Stagnation**: Added max_rounds check in buffer flush
4. **Organized Structure**: All scripts, logs, and results properly organized
5. **Documentation**: README files added for all new directories

## Merge Strategy

When merging with main:
1. **Preserve all new directories**: `scripts/`, `experiments/`, `hyperparameter_tuning_results/`, `logs/archive/`
2. **Preserve all documentation**: All README.md files
3. **Preserve experiment results**: All CSV and JSON files in results directories
4. **Review core file changes**: Check if main branch has conflicting changes to FedAsync/FedBuff/utils
5. **Keep ResNet-18 architecture**: Ensure main branch changes don't revert to SqueezeNet

## Files to Protect During Merge

```
scripts/
experiments/
hyperparameter_tuning_results/
logs/archive/
SANJANA_BRANCH_CHANGES.md
```

