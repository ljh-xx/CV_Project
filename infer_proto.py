"""
Inductive prototype inference using a distilled student backbone.
1. Load val-distilled student checkpoint, strip FC → backbone
2. Extract features from train (support) and test (query) using backbone
3. Prototype classification + optional transductive refinement
4. Write Score CSV files
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import model_dict
from utils.test_imagefolder import TestImageFolder


TASKS = {"Task1": 102, "Task2": 100, "Task3": 101, "Task4": 37}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint-dir", default="./results/distill_val",
                        help="Directory containing Task1-4/model_best.pth")
    parser.add_argument("--result", default="./Score_student")
    parser.add_argument("--task", default="all")
    parser.add_argument("--arch", default="se_resnet18")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--score-scale", default=30.0, type=float)
    parser.add_argument("--center", default="support", choices=["none", "support"])
    parser.add_argument("--transductive-iters", default=1, type=int)
    parser.add_argument("--query-weight", default=0.5, type=float)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def load_backbone(checkpoint_path, arch, device):
    """Load checkpoint and return backbone (no FC)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Create a dummy model with any num_classes, we only need backbone
    model = model_dict[arch](num_classes=1000)
    # Filter out FC keys
    backbone_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    model.load_state_dict(backbone_state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, root, transform, batch_size, workers, pin_memory, device, is_test=False):
    if is_test:
        dataset = TestImageFolder(root, transform=transform)
    else:
        dataset = ImageFolder(root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=pin_memory)
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        f = model.forward_features(images)
        feats.append(f.cpu())
        labels.append(targets.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def prototype_classify(support_feat, support_labels, query_feat, num_classes, args):
    if args.center == "support":
        center = support_feat.mean(dim=0, keepdim=True)
    else:
        center = support_feat.new_zeros(1, support_feat.size(1))
    support_norm = F.normalize(support_feat - center, dim=1)
    query_norm = F.normalize(query_feat - center, dim=1)

    dim = support_norm.size(1)
    proto_sums = support_norm.new_zeros(num_classes, dim)
    proto_counts = support_norm.new_zeros(num_classes, 1)
    proto_sums.index_add_(0, support_labels, support_norm)
    proto_counts.index_add_(0, support_labels, torch.ones(support_labels.size(0), 1))
    prototypes = proto_sums / proto_counts.clamp_min(1.0)
    prototypes = F.normalize(prototypes, dim=1)

    for _ in range(args.transductive_iters):
        probs = F.softmax(args.score_scale * query_norm @ prototypes.t(), dim=1)
        query_sums = probs.t() @ query_norm
        query_counts = probs.sum(dim=0, keepdim=True).t()
        prototypes = (proto_sums + args.query_weight * query_sums) / (
            proto_counts + args.query_weight * query_counts).clamp_min(1e-6)
        prototypes = F.normalize(prototypes, dim=1)

    return F.softmax(args.score_scale * query_norm @ prototypes.t(), dim=1)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    selected = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)

    for task, num_classes in selected.items():
        ckpt_path = os.path.join(args.checkpoint_dir, task, "model_best.pth")
        backbone = load_backbone(ckpt_path, args.arch, device)

        support_feat, support_labels = extract_features(
            backbone, os.path.join(args.data_root, task, "train"), transform,
            args.batch_size, args.workers, args.cuda, device, is_test=False)
        query_feat, _ = extract_features(
            backbone, os.path.join(args.data_root, task, "test"), transform,
            args.batch_size, args.workers, args.cuda, device, is_test=True)

        probs = prototype_classify(support_feat, support_labels, query_feat, num_classes, args)

        csv_file = result_dir / f"{task}_score.csv"
        index = [f"{task}_{i:05d}" for i in range(probs.size(0))]
        columns = [f"score_{i}" for i in range(probs.size(1))]
        pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(csv_file)
        print(json.dumps({"task": task, "score_file": str(csv_file), "samples": probs.size(0)}))


if __name__ == "__main__":
    main()
