"""
Inductive inference: load a task-specific distilled checkpoint and run a single
forward pass on the test set.  No test images are used during training.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models import model_dict
from utils.test_imagefolder import TestImageFolder


TASKS = {
    "Task1": 102,
    "Task2": 100,
    "Task3": 101,
    "Task4": 37,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint", required=True, help="Path to val-distilled model_best.pth")
    parser.add_argument("--result", default="./Score_student")
    parser.add_argument("--arch", default="se_resnet18")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    num_classes = TASKS[args.task]

    # Build model and load checkpoint
    model = model_dict[args.arch](num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Test dataset
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_root = Path(args.data_root) / args.task / "test"
    dataset = TestImageFolder(str(test_root), transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.cuda)

    # Single forward pass — no training, no teacher labels on test
    outputs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            outputs.append(F.softmax(model(images), dim=1).cpu())
    probs = torch.cat(outputs, dim=0)

    # Write score CSV
    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_file = result_dir / f"{args.task}_score.csv"
    index = [f"{args.task}_{i:05d}" for i in range(probs.size(0))]
    columns = [f"score_{i}" for i in range(probs.size(1))]
    pd.DataFrame(probs.numpy(), index=index, columns=columns).to_csv(csv_file)

    print(json.dumps({"task": args.task, "score_file": str(csv_file), "samples": probs.size(0)}))


if __name__ == "__main__":
    main()
