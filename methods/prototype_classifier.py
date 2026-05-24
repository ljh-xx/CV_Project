import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import model_dict
from utils.test_imagefolder import TestImageFolder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-shot prototype classifier for downstream tasks."
    )
    parser.add_argument("--data", default="data/Task1", type=str)
    parser.add_argument("--dataset", default="Task1", type=str)
    parser.add_argument("--arch", default="resnet18", choices=sorted(model_dict))
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--base-num-classes", default=200, type=int)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--result", default="./Score", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--score-scale", default=30.0, type=float)
    parser.add_argument("--center", default="support", choices=["none", "support", "base"])
    parser.add_argument("--base-data", default="data/pretrain/train", type=str)
    parser.add_argument("--transductive-iters", default=0, type=int)
    parser.add_argument("--query-weight", default=0.5, type=float)
    parser.add_argument("--tta", default="none", choices=["none", "hflip"])
    parser.add_argument("--cuda", default=torch.cuda.is_available(), type=bool)
    return parser.parse_args()


def eval_transforms(tta):
    base = [
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if tta == "none":
        return [transforms.Compose(base)]
    return [
        transforms.Compose(base),
        transforms.Compose(
            [
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    ]


def load_backbone(args, device):
    model = model_dict[args.arch](num_classes=args.base_num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_dataset(root, split, transform):
    if split == "test":
        return TestImageFolder(root, transform=transform)
    return ImageFolder(root, transform=transform)


def extract_features(model, root, split, transform_list, args, device):
    all_features = None
    labels = None
    paths = None

    for transform in transform_list:
        dataset = make_dataset(root, split, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.cuda,
        )

        features = []
        cur_labels = []
        with torch.no_grad():
            for images, target in loader:
                images = images.to(device, non_blocking=True)
                feats = model.forward_features(images)
                features.append(feats.cpu())
                cur_labels.append(target.cpu())

        features = torch.cat(features, dim=0)
        cur_labels = torch.cat(cur_labels, dim=0)
        all_features = features if all_features is None else all_features + features

        if labels is None:
            labels = cur_labels
            if hasattr(dataset, "samples"):
                paths = [
                    sample[0] if isinstance(sample, (tuple, list)) else sample
                    for sample in dataset.samples
                ]

    all_features = all_features / len(transform_list)
    return all_features, labels, paths


def compute_center(args, model, support_raw, support_labels, device):
    if args.center == "none":
        return torch.zeros(1, support_raw.size(1))
    if args.center == "support":
        return support_raw.mean(dim=0, keepdim=True)

    transforms_list = eval_transforms("none")
    base_raw, _, _ = extract_features(
        model, args.base_data, "val", transforms_list, args, device
    )
    return base_raw.mean(dim=0, keepdim=True)


def build_prototypes(features, labels, num_classes):
    dim = features.size(1)
    sums = features.new_zeros(num_classes, dim)
    counts = features.new_zeros(num_classes, 1)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones(labels.size(0), 1, dtype=features.dtype))
    prototypes = sums / counts.clamp_min(1.0)
    return F.normalize(prototypes, dim=1), sums, counts


def classify(support_raw, support_labels, query_raw, args, center):
    support = F.normalize(support_raw - center, dim=1)
    query = F.normalize(query_raw - center, dim=1)
    prototypes, support_sums, support_counts = build_prototypes(
        support, support_labels, args.num_classes
    )

    for _ in range(args.transductive_iters):
        probs = F.softmax(args.score_scale * query @ prototypes.t(), dim=1)
        query_sums = probs.t() @ query
        query_counts = probs.sum(dim=0, keepdim=True).t()
        prototypes = (support_sums + args.query_weight * query_sums) / (
            support_counts + args.query_weight * query_counts
        ).clamp_min(1e-6)
        prototypes = F.normalize(prototypes, dim=1)

    logits = args.score_scale * query @ prototypes.t()
    return F.softmax(logits, dim=1)


def write_score(args, probs):
    os.makedirs(args.result, exist_ok=True)
    csv_file = os.path.join(args.result, f"{args.dataset}_score.csv")
    index = [f"{args.dataset}_{i:05d}" for i in range(probs.size(0))]
    columns = [f"score_{i}" for i in range(args.num_classes)]
    pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(csv_file)
    return csv_file


def main():
    args = parse_args()
    start = time.time()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_backbone(args, device)
    transforms_list = eval_transforms(args.tta)

    support_root = os.path.join(args.data, "train")
    query_root = os.path.join(args.data, args.split)

    support_raw, support_labels, _ = extract_features(
        model, support_root, "train", transforms_list, args, device
    )
    query_raw, query_labels, query_paths = extract_features(
        model, query_root, args.split, transforms_list, args, device
    )
    center = compute_center(args, model, support_raw, support_labels, device)
    probs = classify(support_raw, support_labels, query_raw, args, center)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "arch": args.arch,
        "checkpoint": args.checkpoint,
        "support_images": int(support_raw.size(0)),
        "query_images": int(query_raw.size(0)),
        "num_classes": args.num_classes,
        "score_scale": args.score_scale,
        "center": args.center,
        "transductive_iters": args.transductive_iters,
        "query_weight": args.query_weight,
        "tta": args.tta,
        "elapsed_sec": round(time.time() - start, 3),
    }

    if args.split == "val":
        pred = probs.argmax(dim=1)
        metrics["accuracy"] = float((pred == query_labels).float().mean().item() * 100.0)
    else:
        metrics["score_file"] = write_score(args, probs)
        metrics["first_path"] = query_paths[0] if query_paths else None

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
