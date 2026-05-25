import argparse
import json
import math
import os
import time
from itertools import cycle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import model_dict


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a compliant task student from teacher soft labels."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--arch", default="se_resnet18", choices=sorted(model_dict))
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--teacher-score", required=True,
                        help="Path to teacher soft label CSV")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="train: distill on train (k-fold teacher labels); "
                             "val: distill on val (teacher labels)")
    parser.add_argument("--result", default="results/distill")
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--query-batch-size", default=256, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--soft-weight", default=1.0, type=float)
    parser.add_argument("--hard-weight", default=1.0, type=float)
    parser.add_argument("--pseudo-hard-weight", default=0.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--query-augment", default="weak", choices=["weak", "none"])
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def train_transform():
    return transforms.Compose([
        transforms.Resize(92),
        transforms.RandomCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SoftScoreDataset(Dataset):
    """Dataset that returns (image, hard_label, teacher_soft_label)."""
    def __init__(self, root, transform, score_file):
        self.dataset = ImageFolder(root, transform=transform)
        scores = pd.read_csv(score_file, index_col=0).astype("float32")
        if len(scores) != len(self.dataset):
            raise ValueError(
                f"score rows ({len(scores)}) != dataset rows ({len(self.dataset)})"
            )
        self.scores = torch.from_numpy(scores.to_numpy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, self.scores[index]


def load_student(args, device):
    model = model_dict[args.arch](num_classes=args.num_classes)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(json.dumps({
            "loaded_pretrained": args.pretrained,
            "compatible_keys": len(compatible),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }, sort_keys=True))
    return model.to(device)


def soft_ce(logits, targets, temperature):
    log_probs = F.log_softmax(logits / temperature, dim=1)
    return -(targets * log_probs).sum(dim=1).mean() * (temperature ** 2)


def evaluate_soft(model, loader, device):
    """Evaluate on SoftScoreDataset (has teacher labels)."""
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probs = F.softmax(model(images), dim=1)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return correct / total * 100.0 if total else None


def evaluate_hard(model, loader, device):
    """Evaluate on standard ImageFolder (no teacher labels)."""
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probs = F.softmax(model(images), dim=1)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return correct / total * 100.0 if total else None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    start = time.time()

    result_dir = Path(args.result) / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)

    model = load_student(args, device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Support: always from train set with hard labels
    support = ImageFolder(os.path.join(args.data, "train"), transform=train_transform())
    support_loader = DataLoader(
        support, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.cuda,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # Query: from the chosen split with teacher soft labels
    query_transform = train_transform() if args.query_augment == "weak" else eval_transform()
    query_train = SoftScoreDataset(
        os.path.join(args.data, args.split), query_transform, args.teacher_score,
    )
    query_loader = DataLoader(
        query_train, batch_size=args.query_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.cuda,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # Eval: always on val set (never used for training, only for model selection)
    val_eval_dir = os.path.join(args.data, "val")
    if os.path.isdir(val_eval_dir):
        use_soft_eval = (args.split == "val")
        if use_soft_eval:
            query_eval = SoftScoreDataset(
                val_eval_dir, eval_transform(), args.teacher_score,
            )
            query_eval_loader = DataLoader(
                query_eval, batch_size=args.query_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=args.cuda,
                persistent_workers=args.workers > 0,
                prefetch_factor=2 if args.workers > 0 else None,
            )
        else:
            query_eval = ImageFolder(val_eval_dir, transform=eval_transform())
            query_eval_loader = DataLoader(
                query_eval, batch_size=args.query_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=args.cuda,
                persistent_workers=args.workers > 0,
                prefetch_factor=2 if args.workers > 0 else None,
            )
    else:
        query_eval_loader = None
        use_soft_eval = False

    steps_per_epoch = max(len(support_loader), len(query_loader))
    best_acc = -math.inf
    best_path = result_dir / "model_best.pth"
    last_path = result_dir / "model_last.pth"
    log_path = result_dir / "train_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            hard_iter = cycle(support_loader)
            soft_iter = cycle(query_loader)
            losses, hard_losses, soft_losses, pseudo_losses = [], [], [], []

            for _ in range(steps_per_epoch):
                hard_images, hard_labels = next(hard_iter)
                soft_images, _, soft_targets = next(soft_iter)
                hard_images = hard_images.to(device, non_blocking=True)
                hard_labels = hard_labels.to(device, non_blocking=True)
                soft_images = soft_images.to(device, non_blocking=True)
                soft_targets = soft_targets.to(device, non_blocking=True)

                hard_logits = model(hard_images)
                soft_logits = model(soft_images)
                hard_loss = F.cross_entropy(hard_logits, hard_labels)
                distill_loss = soft_ce(soft_logits, soft_targets, args.temperature)
                pseudo_loss = F.cross_entropy(soft_logits, soft_targets.argmax(dim=1))

                loss = (args.hard_weight * hard_loss
                        + args.soft_weight * distill_loss
                        + args.pseudo_hard_weight * pseudo_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                hard_losses.append(hard_loss.item())
                soft_losses.append(distill_loss.item())
                pseudo_losses.append(pseudo_loss.item())

            scheduler.step()

            # Evaluate on val set for model selection
            acc = None
            if query_eval_loader is not None:
                if use_soft_eval:
                    acc = evaluate_soft(model, query_eval_loader, device)
                else:
                    acc = evaluate_hard(model, query_eval_loader, device)
                if acc > best_acc:
                    best_acc = acc
                    torch.save({
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc,
                        "args": vars(args),
                    }, best_path)

            row = {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "hard_loss": sum(hard_losses) / len(hard_losses),
                "soft_loss": sum(soft_losses) / len(soft_losses),
                "pseudo_loss": sum(pseudo_losses) / len(pseudo_losses),
                "lr": scheduler.get_last_lr()[0],
                "val_accuracy": acc,
                "best_accuracy": best_acc if best_acc > -math.inf else None,
                "elapsed_sec": round(time.time() - start, 3),
            }
            print(json.dumps(row, sort_keys=True))
            log_f.write(json.dumps(row, sort_keys=True) + "\n")
            log_f.flush()

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": args.epochs,
        "best_acc": best_acc if best_acc > -math.inf else None,
        "args": vars(args),
    }, last_path)

    print(json.dumps({"best_model": str(best_path), "best_accuracy": best_acc},
                     sort_keys=True))


if __name__ == "__main__":
    main()
