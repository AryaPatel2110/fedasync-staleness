"""Utility functions for server aggregation strategies."""

from __future__ import annotations

import numpy as np

def numpy_parameters(parameters: list[np.ndarray]) -> list[np.ndarray]:
    """Convert a list of parameters to numpy arrays without copying.

    Flower represents model weights as lists of numpy arrays.  This helper
    returns the same list, but ensures that each element is a numpy array.
    """
    return [np.array(p, copy=False) for p in parameters]

def weighted_average(server_params: list[np.ndarray], client_params: list[np.ndarray], weight: float, server_lr: float = 1.0) -> list[np.ndarray]:
    """Compute a weighted average between server and client parameters.

    The update rule is

        new_param = server_param + server_lr * weight * (client_param - server_param)

    Args:
        server_params: Current server parameters.
        client_params: Parameters received from a client.
        weight: Scalar weight applied to the difference (e.g., staleness decay).
        server_lr: Learning rate applied at the server when updating.

    Returns:
        A list of numpy arrays representing the new server parameters.
    """
    s = numpy_parameters(server_params)
    c = numpy_parameters(client_params)
    return [s_i + server_lr * weight * (c_i - s_i) for s_i, c_i in zip(s, c)]