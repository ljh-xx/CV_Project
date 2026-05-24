import argparse
import json
import os
import time

import timm
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.data import create_transform, resolve_model_data_config
from utils.test_imagefolder import TestImageFolder


TASKS = {
    "Task1": 102,
    "Task2": 100,
    "Task3": 101,
    "Task4": 37,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype upper-bound probe with a public pretrained teacher."
    )
    parser.add_argument("--teacher", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--result", default="Score_teacher")
    parser.add_argument("--write-score", action="store_true")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--score-scale", default=30.0, type=float)
    parser.add_argument("--center", default="support", choices=["none", "support"])
    parser.add_argument("--transductive-iters", default=1, type=int)
    parser.add_argument("--query-weight", default=0.5, type=float)
    parser.add_argument("--tta", default="none", choices=["none", "hflip"])
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def load_teacher(args, device):
    model = timm.create_model(args.teacher, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    if args.tta == "hflip":
        return model, [
            transform,
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transform]),
        ]
    return model, [transform]


def make_dataset(root, split, transform):
    if split == "test":
        return TestImageFolder(root, transform=transform)
    return ImageFolder(root, transform=transform)


def extract(model, root, split, transform_list, args, device):
    all_features = None
    labels = None
    paths = []
    for transform in transform_list:
        dataset = make_dataset(root, split, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.cuda,
        )
        feats = []
        cur_labels = []
        with torch.no_grad():
            for images, target in loader:
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
                cur_labels.append(target.cpu())
        features = torch.cat(feats, dim=0)
        all_features = features if all_features is None else all_features + features
        if labels is None:
            labels = torch.cat(cur_labels, dim=0)
            if hasattr(dataset, "samples"):
                paths = [
                    sample[0] if isinstance(sample, (tuple, list)) else sample
                    for sample in dataset.samples
                ]
    return all_features / len(transform_list), labels, paths


def build_prototypes(features, labels, num_classes):
    dim = features.size(1)
    sums = features.new_zeros(num_classes, dim)
    counts = features.new_zeros(num_classes, 1)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones(labels.size(0), 1, dtype=features.dtype))
    prototypes = sums / counts.clamp_min(1.0)
    return F.normalize(prototypes, dim=1), sums, counts


def classify(support_raw, support_labels, query_raw, num_classes, args):
    if args.center == "support":
        center = support_raw.mean(dim=0, keepdim=True)
    else:
        center = torch.zeros(1, support_raw.size(1))
    support = F.normalize(support_raw - center, dim=1)
    query = F.normalize(query_raw - center, dim=1)
    prototypes, support_sums, support_counts = build_prototypes(
        support, support_labels, num_classes
    )
    for _ in range(args.transductive_iters):
        probs = F.softmax(args.score_scale * query @ prototypes.t(), dim=1)
        query_sums = probs.t() @ query
        query_counts = probs.sum(dim=0, keepdim=True).t()
        prototypes = (support_sums + args.query_weight * query_sums) / (
            support_counts + args.query_weight * query_counts
        ).clamp_min(1e-6)
        prototypes = F.normalize(prototypes, dim=1)
    return F.softmax(args.score_scale * query @ prototypes.t(), dim=1)


def write_score(args, task, num_classes, probs):
    os.makedirs(args.result, exist_ok=True)
    csv_file = os.path.join(args.result, f"{task}_score.csv")
    index = [f"{task}_{i:05d}" for i in range(probs.size(0))]
    columns = [f"score_{i}" for i in range(num_classes)]
    pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(csv_file)
    return csv_file


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    start = time.time()
    model, transform = load_teacher(args, device)
    selected = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    results = {}
    for task, num_classes in selected.items():
        data = os.path.join(args.data_root, task)
        support_raw, support_labels, _ = extract(
            model, os.path.join(data, "train"), "train", transform, args, device
        )
        query_raw, query_labels, _ = extract(
            model, os.path.join(data, args.split), args.split, transform, args, device
        )
        probs = classify(support_raw, support_labels, query_raw, num_classes, args)
        if args.split == "val":
            acc = (probs.argmax(dim=1) == query_labels).float().mean().item() * 100.0
            item = {"accuracy": acc}
            if args.write_score:
                item["score_file"] = write_score(args, task, num_classes, probs)
            results[task] = item if args.write_score else acc
            print(json.dumps({"task": task, **item}, sort_keys=True))
        else:
            csv_file = write_score(args, task, num_classes, probs)
            results[task] = csv_file
            print(json.dumps({"task": task, "score_file": csv_file}, sort_keys=True))

    summary = {
        "teacher": args.teacher,
        "center": args.center,
        "score_scale": args.score_scale,
        "transductive_iters": args.transductive_iters,
        "query_weight": args.query_weight,
        "tta": args.tta,
        "per_task": results,
        "elapsed_sec": round(time.time() - start, 3),
    }
    if args.split == "val":
        if args.write_score:
            summary["avg_accuracy"] = sum(
                item["accuracy"] for item in results.values()
            ) / len(results)
        else:
            summary["avg_accuracy"] = sum(results.values()) / len(results)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
