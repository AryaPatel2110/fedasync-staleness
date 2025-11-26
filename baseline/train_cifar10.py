import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import torch
except Exception as e:
    sys.stderr.write('ERROR: PyTorch not installed or incompatible with this interpreter.\n')
    sys.stderr.write(f'{e.__class__.__name__}: {e}\n')
    sys.stderr.write('Hint: activate .venv_311 and pip install torch torchvision numpy pyyaml.\n')
    sys.exit(1)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helper import get_device, set_seed

print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} mps={getattr(torch.backends.mps,'is_available',lambda:False)()} cuda={torch.cuda.is_available()}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=val_test_transform)

    indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_size = int(0.8 * len(train_full))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    class SubsetDataset:
        def __init__(self, dataset, indices, transform):
            self.dataset, self.indices, self.transform = dataset, indices, transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            img, target = self.dataset[self.indices[idx]]
            return (self.transform(img), target) if self.transform else (img, target)

    train_dataset = SubsetDataset(train_full, train_indices, train_transform)
    val_dataset = SubsetDataset(train_full, val_indices, val_test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    logs_base = Path("logs") / "avinash" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_baseline"
    logs_base.mkdir(parents=True, exist_ok=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    csv_header = "time_sec,epoch,train_loss,val_acc,test_acc,lr,wd,batch,seed"
    with open(logs_base / "COMMIT.txt", "w") as f:
        f.write(f"{commit_hash},{csv_header}\n")
    csv_path = logs_base / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_header.split(","))

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            train_count += x.size(0)
        train_loss = train_loss_sum / train_count

        def eval_acc(loader):
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            return correct / total

        model.eval()
        val_acc, test_acc = eval_acc(val_loader), eval_acc(test_loader)
        elapsed = time.time() - start_time

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{elapsed:.3f}", epoch, f"{train_loss:.6f}", f"{val_acc:.6f}", f"{test_acc:.6f}", f"{args.lr:.6f}", f"{args.wd:.6f}", args.batch, args.seed])
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.6f} val_acc={val_acc:.6f} test_acc={test_acc:.6f}", flush=True)

if __name__ == "__main__":
    main()
