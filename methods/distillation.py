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
from utils.test_imagefolder import TestImageFolder


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
    parser.add_argument("--teacher-score", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--result", default="results/distill")
    parser.add_argument("--score-result", default="Score_student")
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
    parser.add_argument("--cuda", default=torch.cuda.is_available(), type=bool)
    return parser.parse_args()


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize(92),
            transforms.RandomCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SoftScoreDataset(Dataset):
    def __init__(self, root, split, transform, score_file):
        if split == "test":
            self.dataset = TestImageFolder(root, transform=transform)
        else:
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
        print(
            json.dumps(
                {
                    "loaded_pretrained": args.pretrained,
                    "compatible_keys": len(compatible),
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                },
                sort_keys=True,
            )
        )
    return model.to(device)


def soft_ce(logits, targets, temperature):
    log_probs = F.log_softmax(logits / temperature, dim=1)
    return -(targets * log_probs).sum(dim=1).mean() * (temperature**2)


def evaluate(model, loader, device, write_score_path=None):
    model.eval()
    total = 0
    correct = 0
    probs_out = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probs = F.softmax(model(images), dim=1)
            probs_out.append(probs.cpu())
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    acc = correct / total * 100.0 if total else None
    if write_score_path is not None:
        probs = torch.cat(probs_out, dim=0)
        index = [f"{write_score_path.stem.replace('_score', '')}_{i:05d}" for i in range(probs.size(0))]
        columns = [f"score_{i}" for i in range(probs.size(1))]
        pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(write_score_path)
    return acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    start = time.time()

    result_dir = Path(args.result) / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir = Path(args.score_result)
    score_dir.mkdir(parents=True, exist_ok=True)

    model = load_student(args, device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    support = ImageFolder(os.path.join(args.data, "train"), transform=train_transform())
    query_transform = train_transform() if args.query_augment == "weak" else eval_transform()
    query_train = SoftScoreDataset(
        os.path.join(args.data, args.split),
        args.split,
        query_transform,
        args.teacher_score,
    )
    query_eval = SoftScoreDataset(
        os.path.join(args.data, args.split),
        args.split,
        eval_transform(),
        args.teacher_score,
    )
    support_loader = DataLoader(
        support,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.cuda,
        drop_last=True,
    )
    query_loader = DataLoader(
        query_train,
        batch_size=args.query_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.cuda,
        drop_last=True,
    )
    query_eval_loader = DataLoader(
        query_eval,
        batch_size=args.query_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.cuda,
    )

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
            losses = []
            hard_losses = []
            soft_losses = []
            pseudo_losses = []
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
                loss = (
                    args.hard_weight * hard_loss
                    + args.soft_weight * distill_loss
                    + args.pseudo_hard_weight * pseudo_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                hard_losses.append(hard_loss.item())
                soft_losses.append(distill_loss.item())
                pseudo_losses.append(pseudo_loss.item())

            scheduler.step()
            acc = None
            if args.split == "val":
                acc = evaluate(model, query_eval_loader, device)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "epoch": epoch,
                            "best_acc": best_acc,
                            "args": vars(args),
                        },
                        best_path,
                    )
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

    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": args.epochs,
            "best_acc": best_acc if best_acc > -math.inf else None,
            "args": vars(args),
        },
        last_path,
    )

    if args.split == "test":
        score_path = score_dir / f"{args.dataset}_score.csv"
        evaluate(model, query_eval_loader, device, write_score_path=score_path)
        print(json.dumps({"score_file": str(score_path)}, sort_keys=True))
    else:
        print(json.dumps({"best_model": str(best_path), "best_accuracy": best_acc}, sort_keys=True))


if __name__ == "__main__":
    main()
