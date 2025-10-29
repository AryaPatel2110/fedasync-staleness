"""FedAsync aggregation strategy.

This module defines a lightweight aggregator implementing the FedAsync
algorithm for asynchronous federated learning.  The strategy applies
exponential decay to updates based on staleness and uses a server‑side
learning rate to control the update magnitude.

Note: This class does not directly subclass Flower's built‑in Strategy
classes.  Instead, it exposes a simple `aggregate` method used in the
simulation loop implemented in `scripts/run_sim.py`.  Adaptation to
Flower's async API would require additional integration work.
"""

from __future__ import annotations

from typing import Callable, Sequence
import numpy as np

from .utils import weighted_average

def default_staleness_fn(s: int) -> float:
    """Default staleness decay: 1/(1 + s)."""
    return 1.0 / (1.0 + float(s))

class FedAsync:
    """Asynchronous aggregation with fixed staleness decay.

    Parameters
    ----------
    eta : float, optional
        Server learning rate applied to updates.  Equivalent to the
        coefficient η in FedAsync.
    staleness_fn : Callable[[int], float], optional
        Function mapping staleness (non‑negative integer) to a scalar weight.
        By default uses 1/(1 + s).
    """
    def __init__(self, eta: float = 1.0, staleness_fn: Callable[[int], float] | None = None) -> None:
        self.eta = eta
        self.staleness_fn = staleness_fn or default_staleness_fn

    def aggregate(self, server_params: Sequence[np.ndarray], client_params: Sequence[np.ndarray], staleness: int) -> list[np.ndarray]:
        """Aggregate a single client update into the server parameters.

        Args:
            server_params: Current server parameters (list of numpy arrays).
            client_params: Client's updated parameters.
            staleness: Integer staleness of the client update (0 means fresh).

        Returns:
            New server parameters after applying the weighted update.
        """
        w = float(self.staleness_fn(int(staleness)))
        return weighted_average(server_params, client_params, w, server_lr=self.eta)