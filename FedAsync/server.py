# Async FedAsync server with periodic evaluation/logging and accuracy-based stopping
import csv
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.model import state_to_list, list_to_state
from utils.helper import get_device


def _testloader(root: str, batch_size: int = 256):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / max(1, total), correct / max(1, total)


class AsyncFedServer:
    """FedAsync: alpha(staleness)=c/(staleness+1). Periodic eval/log. Stop on target accuracy."""
    def __init__(
        self,
        global_model: torch.nn.Module,
        total_train_samples: int,
        c: float = 0.5,
        mixing_alpha: float = 0.5,
        use_sample_weighing: bool = True,
        target_accuracy: float = 0.70,
        max_rounds: Optional[int] = None,
        eval_interval_s: int = 15,
        data_dir: str = "./data",
        checkpoints_dir: str = "./checkpoints",
        logs_dir: str = "./logs",
        global_log_csv: Optional[str] = None,
        client_participation_csv: Optional[str] = None,
        final_model_path: Optional[str] = None,
        resume: bool = True,
        device: Optional[torch.device] = None,
        clip_norm: Optional[float] = None,
        tau_max: Optional[int] = None,
    ):
        self.model = global_model
        self.template = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.device = device or get_device()
        self.model.to(self.device)

        self.total_train_samples = int(total_train_samples)
        self.c = float(c)
        self.mixing_alpha = float(mixing_alpha)
        self.use_sample_weighing = bool(use_sample_weighing)
        self.clip_norm = float(clip_norm) if clip_norm is not None else None
        self.tau_max = int(tau_max) if tau_max is not None else None

        self.eval_interval_s = int(eval_interval_s)
        self.target_accuracy = float(target_accuracy)
        self.max_rounds = int(max_rounds) if max_rounds is not None else None

        # I/O
        self.data_dir = data_dir
        self.ckpt_dir = Path(checkpoints_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(logs_dir); self.log_dir.mkdir(parents=True, exist_ok=True)

        # Paths supplied by config (with defaults)
        self.csv_path = Path(global_log_csv) if global_log_csv else (self.log_dir / "FedAsync.csv")
        self.participation_csv = Path(client_participation_csv) if client_participation_csv else (self.log_dir / "FedAsyncClientParticipation.csv")
        self.final_model_path = Path(final_model_path) if final_model_path else Path("./results/FedAsyncModel.pt")
        self.final_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Init CSV headers if files don't exist
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "time_sec", "round", "test_acc", "updates_per_sec",
                    "tau_bin_0", "tau_bin_1", "tau_bin_2", "tau_bin_3", "tau_bin_4", "tau_bin_5",
                    "tau_bin_6_10", "tau_bin_11_20", "tau_bin_21p",
                    "align_mean", "fairness_gini", "method", "alpha", "K", "timeout", "m", "seed", "strag_frac"
                ])

        if not self.participation_csv.exists():
            self.participation_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.participation_csv.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "client_id", "local_train_loss", "local_train_acc",
                    "local_test_loss", "local_test_acc", "total_agg"
                ])

        self._lock = threading.Lock()
        self._stop = False
        self.t_round = 0                     # increments on every merge
        self._log_count = 0                  # increments on each periodic CSV write
        self.testloader = _testloader(self.data_dir)
        self._train_loss_acc_accum: List[Tuple[float, float, int]] = []  # (loss, acc, n) since last eval
        self._start_ts = time.time()
        self._staleness_history: List[int] = []
        self._client_weights: Dict[int, float] = {}

        if resume:
            self._maybe_resume()

    # ---------- persistence ----------
    def _ckpt_file(self) -> Path:
        return self.ckpt_dir / "server_last.ckpt"

    def _maybe_resume(self) -> None:
        ck = self._ckpt_file()
        if ck.exists():
            blob = torch.load(ck, map_location="cpu")
            state = list_to_state(self.template, blob["global_params"])
            self.model.load_state_dict(state, strict=True)
            self.t_round = int(blob["t_round"])
            print(f"[resume] Loaded server checkpoint at total_agg={self.t_round}")

    def _save_ckpt(self) -> None:
        sd = state_to_list(self.model.state_dict())
        torch.save({"t_round": self.t_round, "global_params": sd}, self._ckpt_file())

    def _save_final_model(self) -> None:
        torch.save(self.model.state_dict(), self.final_model_path)

    # ---------- client/server API ----------
    def get_global(self):
        with self._lock:
            return state_to_list(self.model.state_dict()), self.t_round

    def submit_update(
        self,
        client_id: int,
        base_version: int,
        new_params: List[torch.Tensor],
        num_samples: int,
        train_time_s: float,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
    ) -> None:
        with self._lock:
            if self._stop:
                return
            if self.max_rounds is not None and self.t_round >= self.max_rounds:
                self._stop = True
                return

            # Log client participation before increment
            with self.participation_csv.open("a", newline="") as f:
                csv.writer(f).writerow([
                    client_id, f"{train_loss:.6f}", f"{train_acc:.6f}",
                    f"{test_loss:.6f}", f"{test_acc:.6f}", self.t_round
                ])

            # FedAsync merge
            staleness = max(0, self.t_round - base_version)
            if self.tau_max is not None and staleness > self.tau_max:
                return
            
            self._staleness_history.append(staleness)
            
            alpha = self.c / float(staleness + 1)
            sw = float(num_samples) / float(self.total_train_samples) if self.use_sample_weighing else 1.0
            eff = alpha * self.mixing_alpha * sw
            
            self._client_weights[client_id] = self._client_weights.get(client_id, 0.0) + eff

            g = state_to_list(self.model.state_dict())
            
            if self.clip_norm is not None:
                g_tensors = [gi.to(new_params[0].device).type_as(new_params[0]) for gi in g]
                deltas = [ci.to(g_tensors[i].device).type_as(g_tensors[i]) - g_tensors[i] for i, ci in enumerate(new_params)]
                delta_norms_sq = [torch.norm(d.view(-1)).item() ** 2 for d in deltas]
                total_norm = math.sqrt(sum(delta_norms_sq))
                if total_norm > self.clip_norm:
                    clip_coef = self.clip_norm / (total_norm + 1e-8)
                    new_params = [g_tensors[i] + deltas[i] * clip_coef for i in range(len(deltas))]
            
            merged = [(1.0 - eff) * gi + eff * ci for gi, ci in zip(g, new_params)]
            new_state = list_to_state(self.template, merged)
            self.model.load_state_dict(new_state, strict=True)

            # accumulate metrics since last eval tick
            self._train_loss_acc_accum.append((float(train_loss), float(train_acc), int(num_samples)))

            self.t_round += 1
            self._save_ckpt()

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def mark_stop(self) -> None:
        with self._lock:
            self._stop = True
            # store final model when marking stop
            self._save_final_model()
            print(f"[LOG] saved final model -> {self.final_model_path}")

    # ---------- evaluation / logging ----------
    def _compute_avg_train(self) -> Tuple[float, float]:
        if not self._train_loss_acc_accum:
            return 0.0, 0.0
        loss_sum, acc_sum, n_sum = 0.0, 0.0, 0
        for l, a, n in self._train_loss_acc_accum:
            loss_sum += l * n
            acc_sum += a * n
            n_sum += n
        return loss_sum / max(1, n_sum), acc_sum / max(1, n_sum)

    def _compute_tau_bins(self) -> List[int]:
        bins = [0] * 9
        for tau in self._staleness_history:
            if tau == 0:
                bins[0] += 1
            elif tau == 1:
                bins[1] += 1
            elif tau == 2:
                bins[2] += 1
            elif tau == 3:
                bins[3] += 1
            elif tau == 4:
                bins[4] += 1
            elif tau == 5:
                bins[5] += 1
            elif 6 <= tau <= 10:
                bins[6] += 1
            elif 11 <= tau <= 20:
                bins[7] += 1
            else:
                bins[8] += 1
        return bins

    def _compute_fairness_gini(self) -> float:
        if not self._client_weights:
            return 0.0
        weights = sorted(self._client_weights.values())
        n = len(weights)
        if n == 0 or sum(weights) == 0:
            return 0.0
        cumsum = 0.0
        gini = 0.0
        for i, w in enumerate(weights):
            cumsum += w
            gini += (i + 1) * w
        gini = (2.0 * gini) / (n * sum(weights)) - (n + 1) / n
        return max(0.0, min(1.0, gini))

    def _periodic_eval_and_log(self, method: str = "FedAsync", alpha: float = 0.5, K: int = 8, timeout: float = 0.5, m: int = 20, seed: int = 42, strag_frac: float = 0.0):
        test_loss, test_acc = _evaluate(self.model, self.testloader, self.device)
        avg_train_loss, avg_train_acc = self._compute_avg_train()
        now = time.time() - self._start_ts
        updates_per_sec = self.t_round / now if now > 0 else 0.0
        
        tau_bins = self._compute_tau_bins()
        fairness_gini = self._compute_fairness_gini()
        
        self._train_loss_acc_accum.clear()
        self._staleness_history.clear()

        # CSV and one-line console entry
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                f"{now:.3f}", self.t_round, f"{test_acc:.6f}", f"{updates_per_sec:.6f}",
                tau_bins[0], tau_bins[1], tau_bins[2], tau_bins[3], tau_bins[4], tau_bins[5],
                tau_bins[6], tau_bins[7], tau_bins[8],
                "NaN", f"{fairness_gini:.6f}", method, f"{alpha:.6f}", K, f"{timeout:.6f}", m, seed, f"{strag_frac:.6f}"
            ])
        print(f"[LOG] total_agg={self.t_round} "
              f"avg_train_loss={avg_train_loss:.4f} avg_train_acc={avg_train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} time={now:.1f}s")

        # extra checkpoint every 100 CSV logs
        self._log_count += 1
        if self._log_count % 100 == 0:
            path = self.ckpt_dir / f"global_log{self._log_count}_t{self.t_round}.pt"
            torch.save(self.model.state_dict(), path)

        if test_acc >= self.target_accuracy:
            self._stop = True

    def start_eval_timer(self):
        params = getattr(self, "_method_params", {
            "method": "FedAsync", "alpha": 0.5, "K": 8, "timeout": 15.0, "m": 20, "seed": 42
        })
        def _loop():
            next_ts = time.time() + self.eval_interval_s
            while True:
                now = time.time()
                sleep_for = max(0.0, next_ts - now)
                time.sleep(sleep_for)
                with self._lock:
                    if self._stop:
                        break
                    self._periodic_eval_and_log(**params)
                next_ts += self.eval_interval_s
        threading.Thread(target=_loop, daemon=True).start()

    def wait(self):
        try:
            while not self.should_stop():
                time.sleep(0.2)
        finally:
            self.mark_stop()
