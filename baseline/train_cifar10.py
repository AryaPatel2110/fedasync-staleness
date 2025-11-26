import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import SqueezeNet1_1_Weights

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helper import get_device, set_seed


def build_squeezenet_cifar10():
    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
    return model


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

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_full = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=val_test_transform)

    indices = list(range(len(train_full)))
    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_full), generator=g).tolist()
    train_size = int(0.8 * len(train_full))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    class SubsetDataset:
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, target = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, target

    train_dataset = SubsetDataset(train_full, train_indices, train_transform)
    val_dataset = SubsetDataset(train_full, val_indices, val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_squeezenet_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_base = Path("logs") / "avinash" / f"run_{run_timestamp}_baseline"
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
        train_loss_sum = 0.0
        train_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            train_count += x.size(0)

        train_loss = train_loss_sum / train_count

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += y.size(0)
        test_acc = test_correct / test_total

        elapsed = time.time() - start_time

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                f"{elapsed:.3f}", epoch, f"{train_loss:.6f}", f"{val_acc:.6f}",
                f"{test_acc:.6f}", f"{args.lr:.6f}", f"{args.wd:.6f}", args.batch, args.seed
            ])

        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()

