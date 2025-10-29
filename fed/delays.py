"""Delay sampling for asynchronous federated learning.

In asynchronous simulations we assign each client a random computation delay
and transmission delay.  This module includes functions to sample delays
from various distributions and to compute a simple notion of staleness.
"""

from __future__ import annotations

import numpy as np

def sample_client_delays(n_clients: int, dist: str = "lognormal", params: tuple[float, float] | None = None, seed: int | None = None) -> np.ndarray:
    """Sample positive delay values for a set of clients.

    Args:
        n_clients: Number of clients for which to sample delays.
        dist: Name of the distribution to sample from.  Supported values: 'lognormal', 'pareto'.
        params: Parameters specific to the chosen distribution.  For 'lognormal',
            this is (mean, sigma) of the underlying normal distribution.  For 'pareto',
            this is (shape, scale).
        seed: Random seed for reproducibility.

    Returns:
        A numpy array of shape (n_clients,) containing sampled delay values.
    """
    rng = np.random.default_rng(seed)
    if dist == "lognormal":
        mu, sigma = params if params is not None else (0.0, 0.75)
        return rng.lognormal(mean=mu, sigma=sigma, size=n_clients)
    elif dist == "pareto":
        shape, scale = params if params is not None else (2.0, 1.0)
        # Pareto distribution with given shape and scale
        return scale * (1 + rng.pareto(shape, size=n_clients))
    else:
        raise ValueError(f"Unsupported delay distribution: {dist}")

def compute_staleness(current_time: float, update_time: float) -> int:
    """Compute integer staleness based on current and update times.

    Staleness is defined as the difference between the current server clock and the
    time at which a client update was generated.  Both inputs can be floats
    representing event counters or wall‑clock seconds; the result is an integer
    obtained by rounding down.

    Args:
        current_time: Current global time or event index.
        update_time: Time or event index when the update was generated.

    Returns:
        Non‑negative integer staleness.
    """
    return max(0, int(current_time) - int(update_time))