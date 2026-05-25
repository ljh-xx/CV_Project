"""
K-fold teacher soft labels on the downstream TRAIN set only.
Splits each class's 16 images into K folds. For each fold:
  - Support: (K-1) folds → build prototypes
  - Query: held-out fold → compute soft labels via cosine similarity
This avoids self-match and keeps distillation purely on train data.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.data import create_transform, resolve_model_data_config


TASKS = {"Task1": 102, "Task2": 100, "Task3": 101, "Task4": 37}


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-fold teacher soft labels on train set (no val/test used)."
    )
    parser.add_argument("--teacher", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--task", default="all")
    parser.add_argument("--result", default="./Score_teacher_train_kfold")
    parser.add_argument("--k", default=4, type=int, help="Number of folds (default 4)")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--score-scale", default=30.0, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataset, batch_size, workers, pin_memory, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=pin_memory)
    feats, labels, paths = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", next(iter(out.values())))
        if out.ndim == 3:
            out = out[:, 0]
        elif out.ndim > 2:
            out = out.flatten(2).mean(dim=-1)
        feats.append(out.cpu())
        labels.append(targets.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def kfold_soft_labels(features, labels, num_classes, k, score_scale, seed):
    """Generate soft labels for all images via K-fold prototype classification."""
    rng = np.random.RandomState(seed)
    dim = features.size(1)
    all_probs = torch.zeros(features.size(0), num_classes)

    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        n = len(idx)
        if n < k:
            raise ValueError(f"Class {c}: {n} images < {k} folds")

        # Shuffle and split into K folds
        perm = torch.from_numpy(rng.permutation(n))
        fold_size = n // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size
            query_pos = perm[start:end]
            support_pos = torch.cat([perm[:start], perm[end:]])
            query_idx = idx[query_pos]
            support_idx = idx[support_pos]

            # Build prototypes
            proto_sums = features.new_zeros(num_classes, dim)
            proto_counts = features.new_zeros(num_classes, 1)
            proto_sums.index_add_(0, labels[support_idx], features[support_idx])
            proto_counts.index_add_(0, labels[support_idx],
                                    torch.ones(support_idx.size(0), 1))
            prototypes = F.normalize(proto_sums / proto_counts.clamp_min(1.0), dim=1)

            # Query (no centering needed since support and query are from same distribution)
            query_norm = F.normalize(features[query_idx], dim=1)
            probs = F.softmax(score_scale * query_norm @ prototypes.t(), dim=1)
            all_probs[query_idx] = probs.cpu()

    return all_probs


def write_score(task, num_classes, probs, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    csv_file = os.path.join(result_dir, f"{task}_score.csv")
    index = [f"{task}_{i:05d}" for i in range(probs.size(0))]
    columns = [f"score_{i}" for i in range(num_classes)]
    pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(csv_file)
    return csv_file


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load frozen teacher
    model = timm.create_model(args.teacher, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)

    selected = TASKS if args.task == "all" else {args.task: TASKS[args.task]}

    for task, num_classes in selected.items():
        train_dir = os.path.join(args.data_root, task, "train")
        dataset = ImageFolder(train_dir, transform=transform)
        features, labels = extract_features(model, dataset, args.batch_size,
                                            args.workers, args.cuda, device)

        probs = kfold_soft_labels(features, labels, num_classes, args.k,
                                  args.score_scale, args.seed)
        csv_file = write_score(task, num_classes, probs, args.result)
        # Print entropy stats to verify soft labels aren't one-hot
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        top1 = (probs.argmax(dim=1) == labels).float().mean() * 100
        print(json.dumps({
            "task": task,
            "score_file": csv_file,
            "samples": probs.size(0),
            "teacher_self_accuracy": round(top1.item(), 2),
            "mean_entropy": round(entropy.item(), 4),
        }))


if __name__ == "__main__":
    main()
