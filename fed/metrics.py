"""Metric computations for federated learning experiments.

This module provides helper functions to measure various aspects of a federated
learning experiment, including fairness of per‑client accuracies and the
event/time to reach a target accuracy.
"""

from __future__ import annotations

import numpy as np

def fairness_gini(values: list[float] | np.ndarray) -> float:
    """Compute the Gini coefficient of a list of values.

    The Gini coefficient measures inequality among values.  A value of 0
    indicates perfect equality, whereas a value close to 1 indicates extreme
    inequality.  When used on per‑client accuracies, a higher Gini implies
    larger disparities among clients.

    Args:
        values: Iterable of non‑negative numbers.

    Returns:
        Gini coefficient in [0, 1].
    """
    x = np.array(values, dtype=float)
    if np.all(x == 0):
        return 0.0
    # Sort values
    x_sorted = np.sort(x)
    n = len(x_sorted)
    # Compute the Lorenz curve area
    cumulative = np.cumsum(x_sorted)
    lorenz = cumulative / cumulative[-1]
    # The area under the Lorenz curve times 2 minus 1 gives the Gini coefficient
    gini = 1.0 - 2.0 * np.trapz(lorenz, dx=1.0 / n)
    return float(gini)

def time_to_target(history: list[tuple[float, float]], target_acc: float) -> tuple[float | None, int | None]:
    """Determine the first time and index at which accuracy reaches a target.

    Args:
        history: Sequence of (time_or_event, accuracy) tuples, sorted by the first element.
        target_acc: Accuracy threshold to search for.

    Returns:
        A tuple of (time_or_event, index) corresponding to the first entry where
        accuracy >= target_acc.  If the target is never reached, returns (None, None).
    """
    for idx, (t, acc) in enumerate(history):
        if acc >= target_acc:
            return t, idx
    return None, None