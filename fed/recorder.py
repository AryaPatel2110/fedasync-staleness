"""Event recorder for asynchronous federated simulations.

This helper class collects per‑event or per‑evaluation metrics during
simulations.  It stores the data internally and can export it to
a pandas DataFrame when needed.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

class Recorder:
    """Record arbitrary events and metrics during simulation.

    The recorder accumulates a list of dictionaries.  Each call to
    :meth:`log` appends a new entry.  At the end of a simulation you can
    convert the records to a pandas DataFrame via :meth:`to_df`.
    """
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def log(self, **metrics: Any) -> None:
        """Append a set of metrics to the internal log.

        Args:
            **metrics: Arbitrary keyword arguments representing metric names and values.
        """
        self._events.append(dict(metrics))

    def to_df(self) -> pd.DataFrame:
        """Convert the recorded events into a DataFrame.

        Returns:
            A pandas DataFrame with one row per event and columns corresponding
            to the metric names.
        """
        return pd.DataFrame(self._events)