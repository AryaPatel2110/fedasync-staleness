"""Dynamic staleness control aggregation strategy.

This strategy adapts the exponential staleness decay parameter (lambda)
based on the distribution of observed staleness values and optionally
incorporates a gradient novelty term.  The aim is to reduce the impact
of heavily stale updates while still allowing fresh updates to
contribute substantially.
"""

from __future__ import annotations

from typing import Sequence, Deque
from collections import deque
import numpy as np

from .utils import weighted_average

class DynStale:
    """Adaptive staleness weighting strategy.

    Parameters
    ----------
    init_lambda : float, optional
        Initial decay rate.  Higher values lead to stronger decay.
    beta : float, optional
        Exponent applied to the gradient similarity term.  If 0, the
        similarity term is ignored.
    ema : float, optional
        Exponential moving average factor for the reference gradient and
        staleness quantile.
    adapt_quantile : float, optional
        Quantile of staleness distribution used to adapt lambda (e.g., 0.75).
    drop_stale_after : int | None, optional
        If set, updates with staleness strictly greater than this value
        contribute zero weight (i.e., are effectively dropped).
    """
    def __init__(
        self,
        init_lambda: float = 0.1,
        beta: float = 1.0,
        ema: float = 0.9,
        adapt_quantile: float = 0.75,
        drop_stale_after: int | None = None,
    ) -> None:
        self.lambda_t = init_lambda
        self.beta = beta
        self.ema = ema
        self.adapt_quantile = adapt_quantile
        self.drop_stale_after = drop_stale_after
        # maintain history of staleness values to compute quantile
        self._staleness_history: Deque[int] = deque(maxlen=100)
        # maintain running mean of reference gradient (flattened)
        self._grad_ref: np.ndarray | None = None

    def _update_lambda(self) -> None:
        """Update the decay rate lambda based on staleness history."""
        if not self._staleness_history:
            return
        # Compute current quantile of staleness values
        q = float(np.quantile(list(self._staleness_history), self.adapt_quantile))
        # Avoid zero to prevent division by zero
        q = max(q, 1e-8)
        # Set lambda such that exp(-lambda * q) ≈ 0.5 (half weight at the quantile)
        self.lambda_t = np.log(2.0) / q

    def _compute_gradient_similarity(self, grad: np.ndarray) -> float:
        """Compute cosine similarity between current gradient and reference gradient."""
        if self._grad_ref is None:
            # initialize reference gradient
            self._grad_ref = grad.copy()
            return 1.0
        # cosine similarity = (g dot g_ref) / (||g||*||g_ref||)
        dot_prod = float(np.dot(grad, self._grad_ref))
        norm_g = float(np.linalg.norm(grad))
        norm_ref = float(np.linalg.norm(self._grad_ref))
        if norm_g == 0.0 or norm_ref == 0.0:
            cos_sim = 0.0
        else:
            cos_sim = dot_prod / (norm_g * norm_ref)
        # Update reference gradient using exponential moving average
        self._grad_ref = self.ema * self._grad_ref + (1.0 - self.ema) * grad
        # Clip to [0, 1]
        return max(0.0, min(1.0, cos_sim))

    def aggregate(
        self, server_params: Sequence[np.ndarray], client_params: Sequence[np.ndarray], staleness: int
    ) -> list[np.ndarray]:
        """Aggregate a client update into server parameters using adaptive weighting.

        Args:
            server_params: Current server parameters.
            client_params: Client parameters.
            staleness: Integer staleness of the client update.

        Returns:
            Updated server parameters.
        """
        s = int(staleness)
        # Optionally drop heavily stale updates
        if self.drop_stale_after is not None and s > self.drop_stale_after:
            return list(p.copy() for p in server_params)
        # Compute gradient (flattened) for similarity term
        grad_vecs = [np.array(cp - sp, copy=False).ravel() for sp, cp in zip(server_params, client_params)]
        grad = np.concatenate(grad_vecs)
        # Update reference gradient and compute similarity
        sim = self._compute_gradient_similarity(grad) if self.beta > 0 else 1.0
        # Update staleness history and lambda
        self._staleness_history.append(s)
        self._update_lambda()
        # Compute weight: exponential staleness decay times similarity^beta
        weight = float(np.exp(-self.lambda_t * s)) * (sim ** self.beta)
        # Apply weighted average to server parameters
        return weighted_average(server_params, client_params, weight, server_lr=1.0)