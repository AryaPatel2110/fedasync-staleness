"""Data partitioning utilities for federated learning experiments.

This module provides functions to split dataset indices among a number
of clients.  Non‑IID behavior is controlled via a Dirichlet distribution
over class proportions.  A lower α leads to more skewed partitions,
while a higher α approaches a uniform split across clients.
"""

from __future__ import annotations

import numpy as np

def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int | None = None) -> list[np.ndarray]:
    """Partition dataset indices among clients using a Dirichlet distribution.

    Given an array of labels, this function creates a list of index arrays,
    one for each client.  It first groups indices by class, samples a
    Dirichlet distribution for each class to determine how many examples
    go to each client, and then assigns those indices accordingly.

    Args:
        labels: 1‑D numpy array of class labels for the dataset.
        n_clients: Number of clients among which to split the data.
        alpha: Concentration parameter for the Dirichlet distribution.  Smaller values
            yield more heterogeneity; values approaching infinity produce uniform splits.
        seed: Random seed for reproducibility.

    Returns:
        A list of numpy arrays, where each array contains the indices assigned to one client.
    """
    rng = np.random.default_rng(seed)
    # Determine how many classes exist (assumes labels are 0-indexed integers)
    n_classes = int(labels.max()) + 1
    # Collect indices for each class
    class_indices: list[np.ndarray] = [np.where(labels == c)[0] for c in range(n_classes)]
    # Sample Dirichlet distribution per class; shape (n_classes, n_clients)
    proportions = rng.dirichlet([alpha] * n_clients, size=n_classes)
    # Initialize list for client indices
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for c, idxs in enumerate(class_indices):
        # Shuffle class indices to randomize selection order
        rng.shuffle(idxs)
        # Compute split boundaries based on proportions
        # Convert proportions into counts per client for this class
        counts = (proportions[c] * len(idxs)).astype(int)
        # Adjust counts to ensure the total matches len(idxs)
        # Increase counts one by one until the sum matches
        remainder = len(idxs) - counts.sum()
        for i in range(remainder):
            counts[i % n_clients] += 1
        # Now split the shuffled indices according to counts
        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                client_indices[client_id].extend(idxs[start:start + count].tolist())
            start += count
    # Convert each list into a sorted numpy array
    return [np.sort(np.array(id_list, dtype=int)) for id_list in client_indices]