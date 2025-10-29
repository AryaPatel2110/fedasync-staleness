"""FedBuff aggregation strategy.

FedBuff (buffered asynchronous aggregation) accumulates a batch of client
updates and applies them at once.  It can also trigger aggregation
based on a timeout if too much wall‑clock time has passed since the last
aggregation.  This implementation uses uniform weighting of buffered
updates when aggregating.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from time import monotonic

from .utils import numpy_parameters

class FedBuff:
    """Buffered aggregation strategy for asynchronous federated learning.

    Parameters
    ----------
    buffer_size : int
        Number of updates to accumulate before performing an aggregation.
    timeout_s : float
        Maximum wall‑clock time (in seconds) to wait before forcing an aggregation,
        regardless of how many updates have arrived.  If None, only the buffer
        size triggers aggregation.
    """
    def __init__(self, buffer_size: int = 16, timeout_s: float | None = None) -> None:
        self.buffer_size = buffer_size
        self.timeout_s = timeout_s
        self._buffer: list[Sequence[np.ndarray]] = []
        self._last_agg_time: float = monotonic()

    def on_update(self, server_params: Sequence[np.ndarray], client_params: Sequence[np.ndarray], current_time: float) -> tuple[list[np.ndarray] | None, bool]:
        """Handle a new client update.

        Args:
            server_params: Current server parameters (list of numpy arrays).
            client_params: Client's updated parameters to buffer.
            current_time: Current wall‑clock time (seconds since epoch or simulation start).

        Returns:
            A tuple of (new_server_params, aggregated), where new_server_params is
            either the updated server parameters if an aggregation was performed or
            None if not.  The boolean ``aggregated`` indicates whether an
            aggregation occurred.
        """
        # Add the update to the buffer
        self._buffer.append(client_params)
        should_aggregate = False
        # Check buffer size trigger
        if len(self._buffer) >= self.buffer_size:
            should_aggregate = True
        # Check timeout trigger if configured
        if self.timeout_s is not None and (current_time - self._last_agg_time) >= self.timeout_s:
            should_aggregate = True
        if should_aggregate:
            new_params = self._aggregate_buffer(server_params)
            self._buffer.clear()
            self._last_agg_time = current_time
            return new_params, True
        return None, False

    def _aggregate_buffer(self, server_params: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Aggregate the buffered updates with uniform weighting.

        This implementation takes a simple arithmetic mean of the buffered
        client parameters.  It does not perform any staleness weighting.

        Args:
            server_params: Current server parameters.

        Returns:
            New server parameters after incorporating buffered updates.
        """
        if not self._buffer:
            # No updates to aggregate
            return list(numpy_parameters(server_params))
        # Start from current server params as base
        agg = [np.array(p, copy=True) for p in server_params]
        n_updates = len(self._buffer)
        # Compute mean client parameters
        # Sum all client parameter arrays element‑wise
        for client_params in self._buffer:
            for idx, c in enumerate(client_params):
                agg[idx] += np.array(c, copy=False)
        # Compute average (server + sum) / (1 + n_updates)
        # Equivalent to (server + sum(client_params)) / (n_updates + 1)
        denom = float(n_updates + 1)
        for idx in range(len(agg)):
            agg[idx] = agg[idx] / denom
        return agg