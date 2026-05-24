import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import model_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/pretrain")
    parser.add_argument("--arch", default="se_resnet18")
    parser.add_argument("--result", default="results/pretrain")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize(92),
        transforms.RandomCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = ImageFolder(os.path.join(args.data_root, "train"), transform=train_tf)
    val_set = ImageFolder(os.path.join(args.data_root, "val"), transform=val_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = model_dict[args.arch](num_classes=200).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_path = result_dir / "model_best.pth"
    last_path = result_dir / "model_last.pth"
    log_path = result_dir / "train_log.jsonl"

    print(f"Device: {device},  Train samples: {len(train_set)},  Val samples: {len(val_set)}")

    with log_path.open("w") as f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                loss = F.cross_entropy(model(images), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    correct += (model(images).argmax(dim=1) == labels).sum().item()
                    total += labels.numel()
            acc = correct / total * 100.0

            if acc > best_acc:
                best_acc = acc
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_acc": best_acc}, best_path)

            row = {"epoch": epoch, "loss": round(epoch_loss / len(train_loader), 4),
                   "val_acc": round(acc, 2), "best_acc": round(best_acc, 2),
                   "lr": round(scheduler.get_last_lr()[0], 6)}
            print(json.dumps(row))
            f.write(json.dumps(row) + "\n")

    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs, "best_acc": best_acc}, last_path)
    print(f"\nDone. Best val acc: {best_acc:.2f}%, saved to {best_path}")


if __name__ == "__main__":
    main()
