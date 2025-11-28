# Orchestrator: partitions data, starts server, runs async clients
import logging
import random
import threading
from typing import List

import numpy as np
import torch
from torchvision import datasets, transforms

from .config import load_config
from .client import AsyncClient
from .server import AsyncServer


def _set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_train_dataset(data_dir: str) -> datasets.CIFAR10:
    """Create the CIFAR-10 training dataset with standard transforms."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return ds


def _partition_dirichlet(
    num_examples: int,
    num_clients: int,
    alpha: float,
) -> List[List[int]]:
    """Non-iid Dirichlet partitioning over example indices.

    Returns:
        A list of `num_clients` index lists; each list is the data indices for that client.
    """
    indices = np.arange(num_examples)
    props = np.random.dirichlet(alpha=[alpha] * num_clients)
    props = props / props.sum()
    cuts = (np.cumsum(props) * num_examples).astype(int)
    splits = np.split(indices, cuts[:-1])
    return [split.tolist() for split in splits]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # --------------------- load config & seed ---------------------
    cfg = load_config()
    _set_seed(cfg.seed)

    # --------------------- dataset & partition --------------------
    train_ds = _build_train_dataset(cfg.data.data_dir)
    num_examples = len(train_ds)

    partitions = _partition_dirichlet(
        num_examples=num_examples,
        num_clients=cfg.clients.total,
        alpha=cfg.partition_alpha,
    )

    # --------------------- create server -------------------------
    # AsyncServer encapsulates the trust-weighted async aggregation logic + logging.
    server = AsyncServer(cfg=cfg)

    # --------------------- create clients ------------------------
    clients: List[AsyncClient] = []
    for cid in range(cfg.clients.total):
        indices = partitions[cid] if cid < len(partitions) else []
        client = AsyncClient(
            cid=cid,
            indices=indices,
            cfg=cfg,
        )
        clients.append(client)

    # ------------------- concurrency control ---------------------
    # At most `cfg.clients.concurrent` clients train at the same time.
    sem = threading.Semaphore(cfg.clients.concurrent)

    def client_loop(cl: AsyncClient) -> None:
        """Loop for a single client: fetch global, train, send update, repeat."""
        while not server.should_stop():
            with sem:
                cont = cl.run_once(server)
            if not cont or server.should_stop():
                break

    # --------------------- start client threads ------------------
    threads: List[threading.Thread] = []
    for cl in clients:
        t = threading.Thread(target=client_loop, args=(cl,), daemon=False)
        t.start()
        threads.append(t)

    # --------------------- wait for completion -------------------
    # Server decides when to stop (based on target accuracy / max rounds).
    server.wait()
    for t in threads:
        t.join(timeout=1.0)


if __name__ == "__main__":
    main()
