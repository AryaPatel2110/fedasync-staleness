"""Model parameter serialization helpers."""

from __future__ import annotations

import numpy as np
import torch

def to_numpy_list(parameters: torch.nn.Parameter | list[torch.Tensor]) -> list[np.ndarray]:
    """Convert a list of PyTorch tensors to a list of NumPy arrays."""
    if isinstance(parameters, torch.nn.Parameter):
        return [parameters.detach().cpu().numpy()]
    return [p.detach().cpu().numpy() for p in parameters]

def to_torch_params(numpy_params: list[np.ndarray], model: torch.nn.Module) -> None:
    """Load NumPy parameter arrays into a PyTorch model in place."""
    for param, np_param in zip(model.parameters(), numpy_params):
        param.data = torch.from_numpy(np_param).to(param.data.device)