"""Configuration loading utilities."""

from __future__ import annotations

import yaml
from pathlib import Path

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_dicts(a: dict, b: dict) -> dict:
    """Recursively merge two dictionaries, with values from b overriding a."""
    result = dict(a)
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result